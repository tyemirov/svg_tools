#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10"
# ]
# ///
"""Render animated word-by-word text into a MOV with alpha."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from domain.text_video import (
    EMPTY_TEXT_CODE,
    BACKGROUND_IMAGE_CODE,
    FONT_DIR_CODE,
    FONT_LOAD_CODE,
    INPUT_FILE_CODE,
    INVALID_CONFIG_CODE,
    INVALID_COLOR_CODE,
    RenderConfig,
    RenderValidationError,
    SubtitleWindow,
    SRT_TIME_RANGE_PATTERN,
    tokenize_words,
    parse_srt,
)
from service.render_plan import RenderPlan, build_render_plan


DIRECTIONS = ("L2R", "R2L", "T2B", "B2T")
HORIZONTAL_DIRECTIONS = ("L2R", "R2L")
VERTICAL_DIRECTIONS = ("T2B", "B2T")
LETTER_ORDER_BY_DIRECTION = {
    "L2R": "reverse",
    "R2L": "forward",
    "T2B": "reverse",
    "B2T": "forward",
}
LETTER_STAGGER_RATIO = 0.3
LETTER_TRACKING_RATIO = 0.15
MIN_TRACKING_PIXELS = 2
LOGGER = logging.getLogger("render_text_video")

FFMPEG_NOT_FOUND_CODE = "render_text_video.ffmpeg.not_found"
FFMPEG_EXEC_CODE = "render_text_video.ffmpeg.exec_error"
FFMPEG_UNSUPPORTED_CODE = "render_text_video.ffmpeg.unsupported"
FFMPEG_PROCESS_CODE = "render_text_video.ffmpeg.process_failed"
INTERNAL_DIRECTION_CODE = "render_text_video.internal.invalid_direction"


class RenderPipelineError(RuntimeError):
    """Runtime error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WordStyle:
    """Style definition for a single word."""

    font: ImageFont.FreeTypeFont
    color_rgba: Tuple[int, int, int, int]


@dataclass(frozen=True)
class WordToken:
    """Word text paired with its rendering style."""

    text: str
    style: WordStyle
    letters: Tuple["LetterToken", ...]


@dataclass(frozen=True)
class LetterToken:
    """Single letter layout for a word token."""

    text: str
    bbox: Tuple[int, int, int, int]
    image: Image.Image


@dataclass(frozen=True)
class RenderRequest:
    """Parsed CLI request and runtime options."""

    config: RenderConfig
    direction_seed: int | None
    emit_directions: bool
    remove_punctuation: bool
    background_image: Image.Image | None


def configure_logging() -> None:
    """Configure logging for CLI output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_hex_color_to_rgba(color_value: str) -> Tuple[int, int, int, int]:
    """Parse a color token into an RGBA tuple."""
    normalized = color_value.strip()
    if normalized.lower() == "transparent":
        return (0, 0, 0, 0)

    match_value = re.fullmatch(r"#([0-9a-fA-F]{6})", normalized)
    if not match_value:
        raise RenderValidationError(
            INVALID_COLOR_CODE,
            f"invalid color value: {color_value!r}",
        )

    rgb_hex = match_value.group(1)
    red_value = int(rgb_hex[0:2], 16)
    green_value = int(rgb_hex[2:4], 16)
    blue_value = int(rgb_hex[4:6], 16)
    return (red_value, green_value, blue_value, 255)


def ensure_ffmpeg_available() -> None:
    """Ensure ffmpeg is installed and executable."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RenderPipelineError(FFMPEG_NOT_FOUND_CODE, "ffmpeg not on PATH")
    try:
        subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as exc:
        raise RenderPipelineError(
            FFMPEG_EXEC_CODE, "ffmpeg exists but could not be executed"
        ) from exc


def read_utf8_text_strict(file_path: str) -> str:
    """Read a UTF-8 file with strict decoding."""
    try:
        with open(file_path, "rb") as file_handle:
            file_bytes = file_handle.read()
    except FileNotFoundError as exc:
        raise RenderValidationError(
            INPUT_FILE_CODE, f"input text file not found: {file_path}"
        ) from exc

    try:
        return file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RenderValidationError(
            INPUT_FILE_CODE,
            f"input text file is not valid UTF-8 at byte offset {exc.start}",
        ) from exc


def list_font_files(fonts_dir: str) -> list[str]:
    """List font files from the fonts directory."""
    if not os.path.isdir(fonts_dir):
        raise RenderValidationError(
            FONT_DIR_CODE, f"fonts directory does not exist: {fonts_dir}"
        )

    font_files: list[str] = []
    for entry_name in sorted(os.listdir(fonts_dir)):
        lower_name = entry_name.lower()
        if lower_name.endswith(".ttf") or lower_name.endswith(".otf"):
            font_files.append(os.path.join(fonts_dir, entry_name))

    if not font_files:
        raise RenderValidationError(
            FONT_DIR_CODE,
            f"no font files found in {fonts_dir}",
        )
    return font_files


def load_background_image(image_path: str) -> Image.Image:
    """Load a background image as RGBA."""
    try:
        image = Image.open(image_path)
    except FileNotFoundError as exc:
        raise RenderValidationError(
            BACKGROUND_IMAGE_CODE, f"background image not found: {image_path}"
        ) from exc
    except Exception as exc:
        raise RenderValidationError(
            BACKGROUND_IMAGE_CODE, f"failed to read background image: {image_path}"
        ) from exc
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def filter_loadable_fonts(font_files: Sequence[str], sample_size: int) -> list[str]:
    """Filter font files to those loadable at the sample size."""
    loadable_fonts: list[str] = []
    for font_file_path in font_files:
        try:
            ImageFont.truetype(
                font_file_path, size=sample_size, layout_engine=ImageFont.Layout.BASIC
            )
        except Exception as exc:
            LOGGER.warning(
                "%s: skipped font %s (%s)",
                FONT_LOAD_CODE,
                font_file_path,
                str(exc).strip(),
            )
            continue
        loadable_fonts.append(font_file_path)

    if not loadable_fonts:
        raise RenderValidationError(
            FONT_LOAD_CODE, "failed to load any fonts from fonts directory"
        )

    return loadable_fonts


def build_palette_rgba() -> list[Tuple[int, int, int, int]]:
    """Build the default RGBA palette."""
    return [
        (255, 255, 255, 255),
        (255, 80, 80, 255),
        (80, 255, 140, 255),
        (80, 160, 255, 255),
        (255, 220, 80, 255),
        (200, 120, 255, 255),
        (255, 140, 220, 255),
        (120, 240, 255, 255),
    ]


def compute_font_size_range(width: int, height: int) -> Tuple[int, int]:
    """Compute the minimum and maximum font size for the frame."""
    min_dimension = min(width, height)
    base_size = max(24, min_dimension // 8)
    min_size = max(32, base_size + 4, min_dimension // 5)
    max_size = max(min_size * 2, int(min_dimension * 2.0))
    return min_size, max_size


def select_font_sizes(
    word_count: int, config: RenderConfig, rng: random.Random
) -> Tuple[int, ...]:
    """Select randomized font sizes for each word."""
    min_size, max_size = compute_font_size_range(config.width, config.height)
    return tuple(rng.randint(min_size, max_size) for _ in range(word_count))


def load_font_cached(
    font_file_path: str,
    font_size: int,
    cache: dict[Tuple[str, int], ImageFont.FreeTypeFont],
) -> ImageFont.FreeTypeFont:
    """Load a font and cache by path and size."""
    cache_key = (font_file_path, font_size)
    cached_font = cache.get(cache_key)
    if cached_font is not None:
        return cached_font
    try:
        font = ImageFont.truetype(
            font_file_path, size=font_size, layout_engine=ImageFont.Layout.BASIC
        )
    except Exception as exc:
        raise RenderValidationError(
            FONT_LOAD_CODE, f"failed to load font {font_file_path} at size {font_size}"
        ) from exc
    cache[cache_key] = font
    return font


def render_letter_image(
    character: str,
    font: ImageFont.FreeTypeFont,
    color_rgba: Tuple[int, int, int, int],
    letter_bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Render a single letter into an RGBA image aligned to its bounding box."""
    left, top, right, bottom = letter_bbox
    letter_width = max(1, right - left)
    letter_height = max(1, bottom - top)
    letter_image = Image.new("RGBA", (letter_width, letter_height), (0, 0, 0, 0))
    letter_draw = ImageDraw.Draw(letter_image)
    letter_draw.text((-left, -top), character, font=font, fill=color_rgba)
    return letter_image


def build_letter_layout(
    word_text: str,
    font: ImageFont.FreeTypeFont,
    color_rgba: Tuple[int, int, int, int],
    draw_context: ImageDraw.ImageDraw,
) -> Tuple[LetterToken, ...]:
    """Build per-letter layout information for a word."""
    letters: list[LetterToken] = []
    for character in word_text:
        letter_bbox = draw_context.textbbox(
            (0, 0), character, font=font, stroke_width=0
        )
        letter_image = render_letter_image(
            character, font, color_rgba, letter_bbox
        )
        letters.append(
            LetterToken(text=character, bbox=letter_bbox, image=letter_image)
        )
    return tuple(letters)


def build_tokens(
    words: Sequence[str],
    font_files: Sequence[str],
    palette: Sequence[Tuple[int, int, int, int]],
    font_sizes: Sequence[int],
) -> list[WordToken]:
    """Create styled tokens for each word."""
    tokens: list[WordToken] = []
    font_cache: dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
    layout_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    for index_value, word_text in enumerate(words):
        font_file_path = font_files[index_value % len(font_files)]
        font_size = font_sizes[index_value]
        font_value = load_font_cached(font_file_path, font_size, font_cache)
        color_value = palette[index_value % len(palette)]
        letters = build_letter_layout(word_text, font_value, color_value, layout_draw)
        tokens.append(
            WordToken(
                text=word_text,
                style=WordStyle(font=font_value, color_rgba=color_value),
                letters=letters,
            )
        )
    return tokens


def compute_letter_position(
    direction: str,
    progress_value: float,
    frame_width: int,
    frame_height: int,
    letter_bbox: Tuple[int, int, int, int],
    band_position: int,
) -> Tuple[int, int]:
    """Compute a letter position given a motion direction and progress."""
    left, top, right, bottom = letter_bbox
    letter_width = max(1, right - left)
    letter_height = max(1, bottom - top)
    center_offset_x = (left + right) / 2.0
    center_offset_y = (top + bottom) / 2.0
    clamped_progress = min(1.0, max(0.0, progress_value))

    if direction == "L2R":
        start_center_x = -letter_width / 2.0
        end_center_x = frame_width + letter_width / 2.0
        center_x = (
            start_center_x + (end_center_x - start_center_x) * clamped_progress
        ) + band_position
        center_y = frame_height / 2.0
        x_value = int(round(center_x - center_offset_x))
        y_value = int(round(center_y - center_offset_y))
        return (x_value, y_value)

    if direction == "R2L":
        start_center_x = frame_width + letter_width / 2.0
        end_center_x = -letter_width / 2.0
        center_x = (
            start_center_x + (end_center_x - start_center_x) * clamped_progress
        ) + band_position
        center_y = frame_height / 2.0
        x_value = int(round(center_x - center_offset_x))
        y_value = int(round(center_y - center_offset_y))
        return (x_value, y_value)

    if direction == "T2B":
        start_center_y = -letter_height / 2.0
        end_center_y = frame_height + letter_height / 2.0
        center_y = (
            start_center_y + (end_center_y - start_center_y) * clamped_progress
        ) + band_position
        center_x = frame_width / 2.0
        y_value = int(round(center_y - center_offset_y))
        x_value = int(round(center_x - center_offset_x))
        return (x_value, y_value)

    if direction == "B2T":
        start_center_y = frame_height + letter_height / 2.0
        end_center_y = -letter_height / 2.0
        center_y = (
            start_center_y + (end_center_y - start_center_y) * clamped_progress
        ) + band_position
        center_x = frame_width / 2.0
        y_value = int(round(center_y - center_offset_y))
        x_value = int(round(center_x - center_offset_x))
        return (x_value, y_value)

    raise RenderPipelineError(INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}")


def compute_letter_offsets(letter_count: int, direction: str) -> Tuple[float, ...]:
    """Compute per-letter stagger offsets for a direction."""
    if letter_count <= 1 or direction not in VERTICAL_DIRECTIONS:
        return tuple(0.0 for _ in range(letter_count))
    offsets: list[float] = []
    for index_value in range(letter_count):
        normalized = index_value / float(max(1, letter_count - 1))
        offsets.append((0.5 - normalized) * LETTER_STAGGER_RATIO)
    return tuple(offsets)


def should_reverse_letter_order(direction: str) -> bool:
    """Return True when letter order should be reversed for the direction."""
    order = LETTER_ORDER_BY_DIRECTION.get(direction)
    if order is None:
        raise RenderPipelineError(
            INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}"
        )
    return order == "reverse"


def compute_letter_band_sizes(
    letters: Sequence[LetterToken], direction: str
) -> Tuple[int, ...]:
    """Compute per-letter band sizes based on glyph metrics."""
    sizes: list[int] = []
    for letter in letters:
        width = letter.bbox[2] - letter.bbox[0]
        height = letter.bbox[3] - letter.bbox[1]
        if direction in VERTICAL_DIRECTIONS:
            size = height
        elif direction in HORIZONTAL_DIRECTIONS:
            size = width
        else:
            raise RenderPipelineError(
                INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}"
            )
        sizes.append(max(1, int(size)))
    return tuple(sizes)


def compute_letter_band_positions(
    letter_band_sizes: Sequence[int],
    reverse_order: bool,
) -> Tuple[int, ...]:
    """Compute centered band offsets for letters."""
    if not letter_band_sizes:
        return ()
    sizes = list(reversed(letter_band_sizes)) if reverse_order else list(letter_band_sizes)
    tracking_sizes = [
        max(MIN_TRACKING_PIXELS, int(round(size * LETTER_TRACKING_RATIO)))
        for size in sizes
    ]
    total_span = sum(sizes) + sum(tracking_sizes[:-1])
    cursor = -total_span / 2.0
    positions: list[int] = []
    for index_value, size in enumerate(sizes):
        positions.append(int(round(cursor + size / 2.0)))
        cursor += size
        if index_value < len(sizes) - 1:
            cursor += tracking_sizes[index_value]
    if reverse_order:
        positions.reverse()
    return tuple(positions)


def apply_letter_progress(base_progress: float, offset: float) -> float:
    """Apply a stagger offset to the base progress."""
    return min(1.0, max(0.0, base_progress + offset))


def open_ffmpeg_process(config: RenderConfig) -> subprocess.Popen[bytes]:
    """Start ffmpeg for a raw RGBA frame stream."""
    ensure_ffmpeg_available()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{config.width}x{config.height}",
        "-r",
        str(config.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "qtrle",
        "-pix_fmt",
        "argb",
        "-movflags",
        "+faststart",
        config.output_video_file,
    ]

    try:
        return subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RenderPipelineError(FFMPEG_NOT_FOUND_CODE, "ffmpeg not found") from exc


def build_subtitle_windows(
    config: RenderConfig, text_value: str, remove_punctuation: bool
) -> Tuple[SubtitleWindow, ...]:
    """Build subtitle windows from plain text or SRT input."""
    is_srt_extension = config.input_text_file.lower().endswith(".srt")
    is_srt_content = any(
        SRT_TIME_RANGE_PATTERN.fullmatch(line.strip())
        for line in text_value.splitlines()
        if line.strip()
    )
    if is_srt_extension or is_srt_content:
        return parse_srt(text_value, remove_punctuation)

    words = tokenize_words(text_value, remove_punctuation)
    return (
        SubtitleWindow(
            start_seconds=0.0,
            end_seconds=config.duration_seconds,
            words=words,
        ),
    )


def build_render_plan_from_input(
    config: RenderConfig, remove_punctuation: bool
) -> RenderPlan:
    """Load input text and build a render plan."""
    input_text = read_utf8_text_strict(config.input_text_file)
    windows = build_subtitle_windows(config, input_text, remove_punctuation)
    return build_render_plan(
        windows=windows,
        fps=config.fps,
        duration_seconds=config.duration_seconds,
    )


def build_render_tokens(
    plan: RenderPlan, config: RenderConfig, font_sizes: Sequence[int]
) -> list[WordToken]:
    """Build word tokens from a render plan."""
    palette = build_palette_rgba()
    font_files = list_font_files(config.fonts_dir)
    font_files = filter_loadable_fonts(font_files, min(font_sizes))
    return build_tokens(
        plan.words, font_files=font_files, palette=palette, font_sizes=font_sizes
    )


def select_directions(word_count: int, rng: random.Random) -> Tuple[str, ...]:
    """Select random directions for each word."""
    return tuple(rng.choice(DIRECTIONS) for _ in range(word_count))


def emit_directions(
    directions: Sequence[str],
    font_sizes: Sequence[int],
    words: Sequence[str],
    letter_offsets: Sequence[Sequence[float]],
    letter_bands: Sequence[Sequence[int]],
    letter_band_sizes: Sequence[Sequence[int]],
) -> None:
    """Emit direction choices to stdout."""
    payload = {
        "directions": list(directions),
        "font_sizes": list(font_sizes),
        "words": list(words),
        "letter_offsets": [list(offsets) for offsets in letter_offsets],
        "letter_bands": [list(bands) for bands in letter_bands],
        "letter_band_sizes": [list(sizes) for sizes in letter_band_sizes],
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))


def render_video(
    config: RenderConfig,
    plan: RenderPlan,
    tokens: Sequence[WordToken],
    directions: Sequence[str],
    background_image: Image.Image | None,
) -> None:
    """Render frames based on the render plan."""
    ffmpeg_process = open_ffmpeg_process(config)
    if not ffmpeg_process.stdin:
        raise RenderPipelineError(FFMPEG_PROCESS_CODE, "ffmpeg stdin unavailable")

    schedule_index = 0
    current_schedule = None
    current_token_index: int | None = None
    current_token: WordToken | None = None
    current_letter_offsets: Tuple[float, ...] = ()
    current_letter_bands: Tuple[int, ...] = ()

    try:
        for frame_index in range(plan.total_frames):
            if schedule_index < len(plan.scheduled_words):
                schedule = plan.scheduled_words[schedule_index]
                schedule_end = schedule.start_frame + schedule.frame_count
                if frame_index >= schedule_end:
                    schedule_index += 1
                    current_schedule = None
                    current_token_index = None
                    current_token = None
                    current_letter_offsets = ()
                    current_letter_bands = ()
                if schedule_index < len(plan.scheduled_words):
                    schedule = plan.scheduled_words[schedule_index]
                    if frame_index >= schedule.start_frame:
                        current_schedule = schedule

            if background_image is None:
                frame_image = Image.new(
                    "RGBA", (config.width, config.height), color=config.background_rgba
                )
            else:
                frame_image = background_image.copy()

            if current_schedule is not None:
                token_index = current_schedule.token_index
                if token_index != current_token_index:
                    current_token_index = token_index
                    current_token = tokens[token_index]
                    direction = directions[token_index]
                    current_letter_offsets = compute_letter_offsets(
                        len(current_token.letters), direction
                    )
                    current_letter_bands = compute_letter_band_positions(
                        compute_letter_band_sizes(current_token.letters, direction),
                        should_reverse_letter_order(direction),
                    )
                token = current_token
                if token is None:
                    raise RenderPipelineError(
                        FFMPEG_PROCESS_CODE, "render token missing"
                    )
                within_word_index = frame_index - current_schedule.start_frame
                progress = within_word_index / float(
                    max(1, current_schedule.frame_count - 1)
                )
                direction = directions[token_index]

                for letter_index, letter in enumerate(token.letters):
                    letter_progress = apply_letter_progress(
                        progress, current_letter_offsets[letter_index]
                    )
                    pos_x, pos_y = compute_letter_position(
                        direction=direction,
                        progress_value=letter_progress,
                        frame_width=config.width,
                        frame_height=config.height,
                        letter_bbox=letter.bbox,
                        band_position=current_letter_bands[letter_index],
                    )
                    frame_image.paste(letter.image, (pos_x, pos_y), letter.image)

            ffmpeg_process.stdin.write(frame_image.tobytes())

        ffmpeg_process.stdin.close()
        stderr_bytes = ffmpeg_process.stderr.read() if ffmpeg_process.stderr else b""
        return_code = ffmpeg_process.wait()

        if return_code != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RenderPipelineError(
                FFMPEG_PROCESS_CODE,
                f"ffmpeg failed with exit code {return_code}. {stderr_text}",
            )

    finally:
        try:
            if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                ffmpeg_process.stdin.close()
        except Exception:
            pass
        try:
            if ffmpeg_process.poll() is None:
                ffmpeg_process.kill()
        except Exception:
            pass


def parse_args(argv: Sequence[str]) -> RenderRequest:
    """Parse CLI arguments into a RenderRequest."""
    parser = argparse.ArgumentParser(prog="render_text_video.py", add_help=True)
    parser.add_argument("--input-text-file", required=True)
    parser.add_argument("--output-video-file", default="video.mov")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--background", default="transparent", help="transparent (default) or #RRGGBB"
    )
    parser.add_argument("--background-image", default=None)
    parser.add_argument("--fonts-dir", default="fonts")
    parser.add_argument("--direction-seed", type=int, default=None)
    parser.add_argument("--emit-directions", action="store_true")
    parser.add_argument("--remove-punctuation", action="store_true")

    parsed = parser.parse_args(argv)
    background_rgba = parse_hex_color_to_rgba(parsed.background)
    background_image = None

    if parsed.background_image:
        if parsed.width is not None or parsed.height is not None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE,
                "width/height cannot be used with background-image",
            )
        background_image = load_background_image(parsed.background_image)
        width, height = background_image.size
    else:
        if parsed.width is None or parsed.height is None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE,
                "width and height are required without background-image",
            )
        width = parsed.width
        height = parsed.height

    config = RenderConfig(
        input_text_file=parsed.input_text_file,
        output_video_file=parsed.output_video_file,
        width=width,
        height=height,
        duration_seconds=parsed.duration_seconds,
        fps=parsed.fps,
        background_rgba=background_rgba,
        fonts_dir=parsed.fonts_dir,
        background_image_path=parsed.background_image,
    )

    return RenderRequest(
        config=config,
        direction_seed=parsed.direction_seed,
        emit_directions=parsed.emit_directions,
        remove_punctuation=parsed.remove_punctuation,
        background_image=background_image,
    )


def validate_ffmpeg_capabilities() -> None:
    """Validate ffmpeg encoders and pixel formats required for alpha output."""
    try:
        version_result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except Exception as exc:
        raise RenderPipelineError(
            FFMPEG_NOT_FOUND_CODE, "ffmpeg is not available"
        ) from exc

    if "ffmpeg version" not in version_result.stdout.lower():
        raise RenderPipelineError(
            FFMPEG_EXEC_CODE, "ffmpeg version output is unexpected"
        )

    encoders_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    if "qtrle" not in encoders_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            "ffmpeg does not support qtrle encoder",
        )

    pixfmts_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-pix_fmts"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    if "argb" not in pixfmts_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            "ffmpeg does not support argb pixel format",
        )


def main() -> int:
    """CLI entrypoint."""
    configure_logging()

    try:
        request = parse_args(sys.argv[1:])
        plan = build_render_plan_from_input(
            request.config, request.remove_punctuation
        )
        rng = random.Random(request.direction_seed)
        directions = select_directions(len(plan.words), rng)
        font_sizes = select_font_sizes(len(plan.words), request.config, rng)
        tokens = build_render_tokens(plan, request.config, font_sizes)
        letter_offsets = [
            compute_letter_offsets(len(token.letters), direction)
            for token, direction in zip(tokens, directions)
        ]
        letter_band_sizes = [
            compute_letter_band_sizes(token.letters, direction)
            for token, direction in zip(tokens, directions)
        ]
        letter_bands = [
            compute_letter_band_positions(sizes, should_reverse_letter_order(direction))
            for sizes, direction in zip(letter_band_sizes, directions)
        ]
        if request.emit_directions:
            emit_directions(
                directions,
                font_sizes,
                plan.words,
                letter_offsets,
                letter_bands,
                letter_band_sizes,
            )
            return 0
        validate_ffmpeg_capabilities()
        render_video(request.config, plan, tokens, directions, request.background_image)
        return 0
    except RenderValidationError as exc:
        LOGGER.error("%s: %s", exc.code, str(exc).strip())
        return 1
    except RenderPipelineError as exc:
        LOGGER.error("%s: %s", exc.code, str(exc).strip())
        return 1
    except Exception as exc:
        LOGGER.error("render_text_video.unhandled_error: %s", str(exc).strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
