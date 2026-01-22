#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10"
# ]
# ///
"""Render animated word-by-word text into a MOV (alpha only when needed)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from domain.text_video import (
    EMPTY_TEXT_CODE,
    BACKGROUND_IMAGE_CODE,
    FONT_DIR_CODE,
    FONT_LOAD_CODE,
    AUDIO_FILE_CODE,
    INPUT_FILE_CODE,
    INVALID_CONFIG_CODE,
    INVALID_COLOR_CODE,
    SubtitleRenderer,
    RenderConfig,
    RenderValidationError,
    SubtitleWindow,
    SRT_TIME_RANGE_PATTERN,
    parse_subtitle_renderer,
    tokenize_words,
    split_trailing_punctuation,
    parse_srt,
)
from service.render_plan import RenderPlan, build_render_plan, build_rsvp_render_plan


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
RSVP_ANCHOR_X_RATIO = 0.50
RSVP_ANCHOR_Y_RATIO = 0.80
RSVP_FONT_RATIO = 0.060
RSVP_FONT_MIN = 28
RSVP_FONT_MAX = 96
RSVP_STROKE_RATIO = 0.10
RSVP_STROKE_MIN = 2
RSVP_STROKE_MAX = 10
RSVP_MARGIN_RATIO = 0.60
RSVP_BASE_COLOR = (235, 235, 235, 255)
RSVP_HIGHLIGHT_COLOR = (255, 255, 255, 255)
RSVP_STROKE_COLOR = (0, 0, 0, 255)
LOGGER = logging.getLogger("render_text_video")

FFMPEG_NOT_FOUND_CODE = "render_text_video.ffmpeg.not_found"
FFMPEG_EXEC_CODE = "render_text_video.ffmpeg.exec_error"
FFMPEG_UNSUPPORTED_CODE = "render_text_video.ffmpeg.unsupported"
FFMPEG_PROCESS_CODE = "render_text_video.ffmpeg.process_failed"
FFMPEG_PROBE_CODE = "render_text_video.ffmpeg.probe_error"
INTERNAL_DIRECTION_CODE = "render_text_video.internal.invalid_direction"
PRORES_PROFILE = "4444"
PRORES_PIXEL_FORMAT = "yuva444p10le"
PRORES_QSCALE_BASE = 15
PRORES_QSCALE_MAX = 28
PRORES_QSCALE_REFERENCE_PIXELS = 1920 * 1080
PRORES_ALPHA_BITS = "8"
H264_CODEC = "libx264"
H264_PIXEL_FORMAT = "yuv420p"
H264_CRF = "20"
H264_PRESET = "veryfast"
H264_TUNE = "stillimage"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_PAD_FILTER = "apad"


class RenderPipelineError(RuntimeError):
    """Runtime error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class VideoAlphaMode(str, Enum):
    """Alpha handling mode for video output."""

    ALPHA = "alpha"
    OPAQUE = "opaque"


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
class RsvpToken:
    """Pre-rendered RSVP word sprite and layout data."""

    text: str
    word_core: str
    orp_index: int
    prefix_width: float
    orp_width: float
    anchor_x: float
    anchor_y: float
    base_image: Image.Image
    base_offset: Tuple[int, int]
    orp_image: Image.Image
    orp_offset: Tuple[int, int]


@dataclass(frozen=True)
class RenderRequest:
    """Parsed CLI request and runtime options."""

    config: RenderConfig
    direction_seed: int | None
    emit_directions: bool
    remove_punctuation: bool
    background_image: Image.Image | None
    alpha_mode: VideoAlphaMode
    audio_track: str | None
    duration_override: bool
    input_text: str | None


@dataclass(frozen=True)
class VideoEncodingSpec:
    """Encoder settings for a specific alpha mode."""

    codec: str
    pix_fmt: str
    args_builder: Callable[[RenderConfig], Tuple[str, ...]]
    encoder_name: str
    alpha_bits: str | None


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


def is_srt_input(file_path: str, text_value: str) -> bool:
    """Return True when input should be treated as SRT."""
    if file_path.lower().endswith(".srt"):
        return True
    return any(
        SRT_TIME_RANGE_PATTERN.fullmatch(line.strip())
        for line in text_value.splitlines()
        if line.strip()
    )


def compute_srt_duration_seconds(
    text_value: str, remove_punctuation: bool
) -> float:
    """Compute the maximum end time for SRT content."""
    windows = parse_srt(text_value, remove_punctuation)
    return max(window.end_seconds for window in windows)


def ensure_ffprobe_available() -> None:
    """Ensure ffprobe is installed and executable."""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RenderPipelineError(FFMPEG_NOT_FOUND_CODE, "ffprobe not on PATH")
    try:
        subprocess.run(
            [ffprobe_path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as exc:
        raise RenderPipelineError(
            FFMPEG_EXEC_CODE, "ffprobe exists but could not be executed"
        ) from exc


def get_audio_duration_seconds(audio_path: str) -> float:
    """Return the audio duration in seconds for the provided file."""
    if not os.path.isfile(audio_path):
        raise RenderValidationError(
            AUDIO_FILE_CODE, f"audio track not found: {audio_path}"
        )
    ensure_ffprobe_available()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.strip()
        raise RenderPipelineError(
            FFMPEG_PROBE_CODE,
            f"ffprobe failed for audio track: {stderr_text}",
        )
    try:
        duration_seconds = float(result.stdout.strip())
    except ValueError as exc:
        raise RenderValidationError(
            AUDIO_FILE_CODE, f"audio track duration unavailable: {audio_path}"
        ) from exc
    if duration_seconds <= 0:
        raise RenderValidationError(
            AUDIO_FILE_CODE, f"audio track duration invalid: {audio_path}"
        )
    return duration_seconds


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


def select_alpha_mode(
    background_rgba: Tuple[int, int, int, int],
    background_image: Image.Image | None,
) -> VideoAlphaMode:
    """Select alpha output mode based on the requested background."""
    if background_image is None and background_rgba[3] == 0:
        return VideoAlphaMode.ALPHA
    return VideoAlphaMode.OPAQUE


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
    return tuple(
        rng.randint(config.font_size_min, config.font_size_max)
        for _ in range(word_count)
    )


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
    letter_draw.text(
        (-left, -top), character, font=font, fill=color_rgba, anchor="ls"
    )
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
            (0, 0), character, font=font, stroke_width=0, anchor="ls"
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


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    """Clamp an integer between min and max."""
    return max(min_value, min(max_value, value))


def compute_rsvp_font_size(frame_height: int) -> int:
    """Compute RSVP font size based on frame height."""
    return clamp_int(
        int(round(frame_height * RSVP_FONT_RATIO)), RSVP_FONT_MIN, RSVP_FONT_MAX
    )


def compute_rsvp_stroke_width(font_size: int) -> int:
    """Compute RSVP stroke width based on font size."""
    return clamp_int(
        int(round(font_size * RSVP_STROKE_RATIO)),
        RSVP_STROKE_MIN,
        RSVP_STROKE_MAX,
    )


def compute_rsvp_margin(font_size: int) -> int:
    """Compute RSVP margin based on font size."""
    return int(round(font_size * RSVP_MARGIN_RATIO))


def compute_orp_index(word_core: str) -> int:
    """Compute the ORP index for a word core."""
    if len(word_core) <= 1:
        return 0
    return max(0, int(math.floor(len(word_core) * 0.35) - 1))


def measure_text_width(
    draw_context: ImageDraw.ImageDraw,
    text_value: str,
    font: ImageFont.FreeTypeFont,
) -> float:
    """Measure text width using font metrics."""
    if not text_value:
        return 0.0
    try:
        return float(draw_context.textlength(text_value, font=font))
    except Exception:
        bbox = draw_context.textbbox((0, 0), text_value, font=font)
        return float(bbox[2] - bbox[0])


def measure_text_bbox(
    draw_context: ImageDraw.ImageDraw,
    text_value: str,
    font: ImageFont.FreeTypeFont,
    stroke_width: int,
) -> Tuple[int, int, int, int]:
    """Measure text bounding box relative to a left baseline anchor."""
    return draw_context.textbbox(
        (0, 0),
        text_value,
        font=font,
        stroke_width=stroke_width,
        anchor="la",
    )


def render_text_sprite(
    text_value: str,
    font: ImageFont.FreeTypeFont,
    fill_rgba: Tuple[int, int, int, int],
    stroke_rgba: Tuple[int, int, int, int],
    stroke_width: int,
    draw_context: ImageDraw.ImageDraw,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Render a text sprite and return the image plus its anchor offset."""
    left, top, right, bottom = measure_text_bbox(
        draw_context, text_value, font, stroke_width
    )
    sprite_width = max(1, right - left)
    sprite_height = max(1, bottom - top)
    sprite = Image.new("RGBA", (sprite_width, sprite_height), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite)
    sprite_draw.text(
        (-left, -top),
        text_value,
        font=font,
        fill=fill_rgba,
        stroke_width=stroke_width,
        stroke_fill=stroke_rgba,
        anchor="la",
    )
    return sprite, (left, top)


def build_rsvp_tokens(
    words: Sequence[str],
    config: RenderConfig,
) -> list[RsvpToken]:
    """Create RSVP tokens with ORP anchoring."""
    font_size = compute_rsvp_font_size(config.height)
    stroke_width = compute_rsvp_stroke_width(font_size)
    margin_px = compute_rsvp_margin(font_size)
    font_files = list_font_files(config.fonts_dir)
    font_files = filter_loadable_fonts(font_files, font_size)
    font_value = load_font_cached(font_files[0], font_size, {})

    layout_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    anchor_x = config.width * RSVP_ANCHOR_X_RATIO
    anchor_y = config.height * RSVP_ANCHOR_Y_RATIO

    tokens: list[RsvpToken] = []
    for word_text in words:
        word_core, _ = split_trailing_punctuation(word_text)
        orp_index = compute_orp_index(word_core)
        prefix = word_core[:orp_index]
        orp_char = word_core[orp_index]
        orp_advance = measure_text_width(layout_draw, orp_char, font_value)
        if prefix:
            prefix_width = (
                measure_text_width(layout_draw, prefix + orp_char, font_value)
                - orp_advance
            )
        else:
            prefix_width = 0.0
        orp_left, _, orp_right, _ = measure_text_bbox(
            layout_draw, orp_char, font_value, stroke_width
        )
        orp_width = float(orp_right - orp_left)
        text_left, _, text_right, _ = measure_text_bbox(
            layout_draw, word_text, font_value, stroke_width
        )
        orp_center_offset = orp_left + (orp_width / 2.0)
        x_anchor = anchor_x - prefix_width - orp_center_offset
        min_anchor = margin_px - text_left
        max_anchor = config.width - margin_px - text_right
        if min_anchor <= max_anchor:
            x_anchor = min(max(x_anchor, min_anchor), max_anchor)

        base_image, base_offset = render_text_sprite(
            word_text,
            font_value,
            RSVP_BASE_COLOR,
            RSVP_STROKE_COLOR,
            stroke_width,
            layout_draw,
        )
        orp_image, orp_offset = render_text_sprite(
            orp_char,
            font_value,
            RSVP_HIGHLIGHT_COLOR,
            RSVP_STROKE_COLOR,
            stroke_width,
            layout_draw,
        )

        tokens.append(
            RsvpToken(
                text=word_text,
                word_core=word_core,
                orp_index=orp_index,
                prefix_width=prefix_width,
                orp_width=orp_width,
                anchor_x=x_anchor,
                anchor_y=anchor_y,
                base_image=base_image,
                base_offset=base_offset,
                orp_image=orp_image,
                orp_offset=orp_offset,
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
    baseline_y: float,
) -> Tuple[int, int]:
    """Compute a letter position given a motion direction and progress."""
    left, top, right, bottom = letter_bbox
    letter_width = max(1, right - left)
    letter_height = max(1, bottom - top)
    center_offset_x = letter_width / 2.0
    center_offset_y = letter_height / 2.0
    clamped_progress = min(1.0, max(0.0, progress_value))

    if direction == "L2R":
        start_center_x = -letter_width / 2.0
        end_center_x = frame_width + letter_width / 2.0
        center_x = (
            start_center_x + (end_center_x - start_center_x) * clamped_progress
        ) + band_position
        x_value = int(round(center_x - center_offset_x))
        y_value = int(round(baseline_y + top))
        return (x_value, y_value)

    if direction == "R2L":
        start_center_x = frame_width + letter_width / 2.0
        end_center_x = -letter_width / 2.0
        center_x = (
            start_center_x + (end_center_x - start_center_x) * clamped_progress
        ) + band_position
        x_value = int(round(center_x - center_offset_x))
        y_value = int(round(baseline_y + top))
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


def compute_baseline_y(font: ImageFont.FreeTypeFont, frame_height: int) -> float:
    """Compute a baseline Y position using font metrics."""
    default_baseline = frame_height / 2.0
    try:
        ascent, descent = font.getmetrics()
    except Exception:
        return default_baseline
    min_baseline = float(ascent)
    max_baseline = float(frame_height - descent)
    if min_baseline <= max_baseline:
        return min(max(default_baseline, min_baseline), max_baseline)
    return min_baseline


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


def adjust_letter_band_positions(
    band_positions: Sequence[int],
    letters: Sequence[LetterToken],
    direction: str,
) -> Tuple[int, ...]:
    """Return band positions (cropped glyphs do not need bearing adjustments)."""
    if not band_positions:
        return ()
    return tuple(band_positions)


def compute_entry_edge_position(
    direction: str,
    band_position: int,
    letter_bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> float:
    """Compute the entry-edge coordinate at progress zero for a letter."""
    left, top, right, bottom = letter_bbox
    letter_width = max(1, right - left)
    letter_height = max(1, bottom - top)
    center_offset_x = letter_width / 2.0
    center_offset_y = letter_height / 2.0

    if direction == "L2R":
        start_center_x = -letter_width / 2.0
        entry_offset = letter_width - center_offset_x
        return start_center_x + band_position + entry_offset

    if direction == "R2L":
        start_center_x = frame_width + letter_width / 2.0
        entry_offset = -center_offset_x
        return start_center_x + band_position + entry_offset

    if direction == "T2B":
        start_center_y = -letter_height / 2.0
        entry_offset = letter_height - center_offset_y
        return start_center_y + band_position + entry_offset

    if direction == "B2T":
        start_center_y = frame_height + letter_height / 2.0
        entry_offset = -center_offset_y
        return start_center_y + band_position + entry_offset

    raise RenderPipelineError(INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}")


def compute_band_position_limits(
    direction: str,
    letter_bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float]:
    """Return min/max band offsets that keep a letter path intersecting the frame."""
    left, top, right, bottom = letter_bbox
    letter_width = max(1, right - left)
    letter_height = max(1, bottom - top)
    center_offset_x = letter_width / 2.0
    center_offset_y = letter_height / 2.0

    if direction == "L2R":
        start_center = -letter_width / 2.0
        end_center = frame_width + letter_width / 2.0
        min_center = center_offset_x - letter_width
        max_center = frame_width + center_offset_x
    elif direction == "R2L":
        start_center = frame_width + letter_width / 2.0
        end_center = -letter_width / 2.0
        min_center = center_offset_x - letter_width
        max_center = frame_width + center_offset_x
    elif direction == "T2B":
        start_center = -letter_height / 2.0
        end_center = frame_height + letter_height / 2.0
        min_center = center_offset_y - letter_height
        max_center = frame_height + center_offset_y
    elif direction == "B2T":
        start_center = frame_height + letter_height / 2.0
        end_center = -letter_height / 2.0
        min_center = center_offset_y - letter_height
        max_center = frame_height + center_offset_y
    else:
        raise RenderPipelineError(
            INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}"
        )

    min_allowed = min_center - max(start_center, end_center)
    max_allowed = max_center - min(start_center, end_center)
    return (min_allowed, max_allowed)


def normalize_letter_band_positions(
    band_positions: Sequence[int],
    letters: Sequence[LetterToken],
    direction: str,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, ...]:
    """Normalize band positions to keep all letters visible."""
    if not band_positions:
        return ()
    if not letters:
        return tuple(band_positions)

    limits = [
        compute_band_position_limits(direction, letter.bbox, frame_width, frame_height)
        for letter in letters
    ]
    min_limit = max(limit[0] for limit in limits)
    max_limit = min(limit[1] for limit in limits)
    if min_limit > max_limit:
        return tuple(band_positions)

    min_pos = min(band_positions)
    max_pos = max(band_positions)
    if min_pos == max_pos:
        target = min(max(min_pos, min_limit), max_limit)
        shift = target - min_pos
        return tuple(int(round(position + shift)) for position in band_positions)

    span = max_pos - min_pos
    limit_span = max_limit - min_limit
    scale = min(1.0, limit_span / span) if span else 1.0
    center = (min_pos + max_pos) / 2.0
    target_center = (min_limit + max_limit) / 2.0
    return tuple(
        int(round(target_center + (position - center) * scale))
        for position in band_positions
    )


def align_letter_band_positions_to_entry(
    band_positions: Sequence[int],
    letters: Sequence[LetterToken],
    direction: str,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, ...]:
    """Shift band positions so the first letter touches the entry edge."""
    if not band_positions:
        return ()
    if not letters:
        return tuple(band_positions)

    entry_edge = compute_entry_edge_position(
        direction,
        band_positions[0],
        letters[0].bbox,
        frame_width,
        frame_height,
    )
    if direction in ("L2R", "T2B"):
        desired_edge = 0.0
    elif direction == "R2L":
        desired_edge = float(frame_width)
    elif direction == "B2T":
        desired_edge = float(frame_height)
    else:
        raise RenderPipelineError(
            INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}"
        )

    shift = desired_edge - entry_edge
    min_shift = float("-inf")
    max_shift = float("inf")
    for position, letter in zip(band_positions, letters):
        min_allowed, max_allowed = compute_band_position_limits(
            direction, letter.bbox, frame_width, frame_height
        )
        min_shift = max(min_shift, min_allowed - position)
        max_shift = min(max_shift, max_allowed - position)
    if min_shift <= max_shift:
        if not (min_shift <= shift <= max_shift) and direction in HORIZONTAL_DIRECTIONS:
            shift = desired_edge - entry_edge
        else:
            shift = min(max(shift, min_shift), max_shift)
    elif direction in HORIZONTAL_DIRECTIONS:
        shift = desired_edge - entry_edge
    else:
        shift = 0.0
    return tuple(int(round(position + shift)) for position in band_positions)


def apply_letter_progress(base_progress: float, offset: float) -> float:
    """Apply a stagger offset to the base progress."""
    return min(1.0, max(0.0, base_progress + offset))


def compute_prores_qscale(width: int, height: int) -> int:
    """Compute a ProRes quantizer based on the frame size."""
    pixel_count = width * height
    scale = pixel_count / PRORES_QSCALE_REFERENCE_PIXELS
    qscale = int(round(PRORES_QSCALE_BASE * math.sqrt(scale)))
    return max(PRORES_QSCALE_BASE, min(PRORES_QSCALE_MAX, qscale))


def build_prores_args(config: RenderConfig) -> Tuple[str, ...]:
    """Build ProRes codec arguments."""
    qscale = compute_prores_qscale(config.width, config.height)
    return (
        "-profile:v",
        PRORES_PROFILE,
        "-qscale:v",
        str(qscale),
        "-alpha_bits",
        PRORES_ALPHA_BITS,
    )


def build_h264_args(config: RenderConfig) -> Tuple[str, ...]:
    """Build H.264 codec arguments."""
    return (
        "-crf",
        H264_CRF,
        "-preset",
        H264_PRESET,
        "-tune",
        H264_TUNE,
    )


ENCODING_SPECS = {
    VideoAlphaMode.ALPHA: VideoEncodingSpec(
        codec="prores_ks",
        pix_fmt=PRORES_PIXEL_FORMAT,
        args_builder=build_prores_args,
        encoder_name="prores_ks",
        alpha_bits=PRORES_ALPHA_BITS,
    ),
    VideoAlphaMode.OPAQUE: VideoEncodingSpec(
        codec=H264_CODEC,
        pix_fmt=H264_PIXEL_FORMAT,
        args_builder=build_h264_args,
        encoder_name=H264_CODEC,
        alpha_bits=None,
    ),
}


def open_ffmpeg_process(
    config: RenderConfig,
    alpha_mode: VideoAlphaMode,
    audio_track: str | None,
) -> subprocess.Popen[bytes]:
    """Start ffmpeg for a raw RGBA frame stream."""
    ensure_ffmpeg_available()

    encoding = ENCODING_SPECS[alpha_mode]
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
    ]
    if audio_track:
        ffmpeg_cmd.extend(["-i", audio_track, "-map", "0:v:0", "-map", "1:a:0"])
    else:
        ffmpeg_cmd.append("-an")
    ffmpeg_cmd.extend(["-c:v", encoding.codec])
    ffmpeg_cmd.extend(encoding.args_builder(config))
    if audio_track:
        ffmpeg_cmd.extend(
            [
                "-c:a",
                AUDIO_CODEC,
                "-b:a",
                AUDIO_BITRATE,
                "-af",
                AUDIO_PAD_FILTER,
                "-shortest",
            ]
        )
    ffmpeg_cmd.extend(
        [
            "-pix_fmt",
            encoding.pix_fmt,
            "-movflags",
            "+faststart",
            config.output_video_file,
        ]
    )

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
    config: RenderConfig,
    text_value: str,
    remove_punctuation: bool,
    subtitle_renderer: SubtitleRenderer,
) -> Tuple[SubtitleWindow, ...]:
    """Build subtitle windows from plain text or SRT input."""
    is_srt = is_srt_input(config.input_text_file, text_value)

    if subtitle_renderer == SubtitleRenderer.RSVP_ORP:
        if not is_srt:
            raise RenderValidationError(
                INVALID_CONFIG_CODE,
                "rsvp_orp requires SRT input",
            )
        return parse_srt(text_value, remove_punctuation)

    if is_srt:
        return parse_srt(text_value, remove_punctuation)

    words = tokenize_words(text_value, remove_punctuation)
    return (
        SubtitleWindow(
            start_seconds=0.0,
            end_seconds=config.duration_seconds,
            words=words,
        ),
    )


def trim_subtitle_windows(
    windows: Sequence[SubtitleWindow], duration_seconds: float
) -> Tuple[SubtitleWindow, ...]:
    """Trim subtitle windows that exceed the provided duration."""
    trimmed: list[SubtitleWindow] = []
    for window in windows:
        if window.start_seconds >= duration_seconds:
            continue
        if window.end_seconds <= duration_seconds:
            trimmed.append(window)
            continue
        trimmed.append(
            SubtitleWindow(
                start_seconds=window.start_seconds,
                end_seconds=duration_seconds,
                words=window.words,
            )
        )
    return tuple(trimmed)


def build_render_plan_from_input(
    config: RenderConfig, remove_punctuation: bool, input_text: str, duration_override: bool
) -> RenderPlan:
    """Load input text and build a render plan."""
    windows = build_subtitle_windows(
        config, input_text, remove_punctuation, config.subtitle_renderer
    )
    if duration_override:
        windows = trim_subtitle_windows(windows, config.duration_seconds)
    if config.subtitle_renderer == SubtitleRenderer.RSVP_ORP:
        return build_rsvp_render_plan(
            windows=windows,
            fps=config.fps,
            duration_seconds=config.duration_seconds,
        )
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


def compute_total_frames(duration_seconds: float, fps: int) -> int:
    """Compute total frames for a video duration."""
    total_frames = int(round(duration_seconds * fps))
    if total_frames <= 0:
        raise RenderValidationError(
            INVALID_CONFIG_CODE, "duration and fps produce zero frames"
        )
    return total_frames


def render_background_video(
    config: RenderConfig,
    background_image: Image.Image | None,
    alpha_mode: VideoAlphaMode,
    audio_track: str | None,
) -> None:
    """Render a background-only video for the configured duration."""
    ffmpeg_process = open_ffmpeg_process(config, alpha_mode, audio_track)
    if not ffmpeg_process.stdin:
        raise RenderPipelineError(FFMPEG_PROCESS_CODE, "ffmpeg stdin unavailable")

    total_frames = compute_total_frames(config.duration_seconds, config.fps)
    if background_image is None:
        frame_image = Image.new(
            "RGBA", (config.width, config.height), color=config.background_rgba
        )
    else:
        frame_image = background_image
    frame_bytes = frame_image.tobytes()

    try:
        for _ in range(total_frames):
            ffmpeg_process.stdin.write(frame_bytes)

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


def render_video(
    config: RenderConfig,
    plan: RenderPlan,
    tokens: Sequence[WordToken],
    directions: Sequence[str],
    background_image: Image.Image | None,
    alpha_mode: VideoAlphaMode,
    audio_track: str | None,
) -> None:
    """Render frames based on the render plan."""
    ffmpeg_process = open_ffmpeg_process(config, alpha_mode, audio_track)
    if not ffmpeg_process.stdin:
        raise RenderPipelineError(FFMPEG_PROCESS_CODE, "ffmpeg stdin unavailable")

    schedule_index = 0
    current_schedule = None
    current_token_index: int | None = None
    current_token: WordToken | None = None
    current_letter_offsets: Tuple[float, ...] = ()
    current_letter_bands: Tuple[int, ...] = ()
    current_baseline_y = config.height / 2.0

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
                    current_letter_bands = adjust_letter_band_positions(
                        current_letter_bands, current_token.letters, direction
                    )
                    current_letter_bands = normalize_letter_band_positions(
                        current_letter_bands,
                        current_token.letters,
                        direction,
                        config.width,
                        config.height,
                    )
                    current_letter_bands = align_letter_band_positions_to_entry(
                        current_letter_bands,
                        current_token.letters,
                        direction,
                        config.width,
                        config.height,
                    )
                    if direction in HORIZONTAL_DIRECTIONS:
                        current_baseline_y = compute_baseline_y(
                            current_token.style.font, config.height
                        )
                    else:
                        current_baseline_y = config.height / 2.0
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
                        baseline_y=current_baseline_y,
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


def render_rsvp_video(
    config: RenderConfig,
    plan: RenderPlan,
    tokens: Sequence[RsvpToken],
    background_image: Image.Image | None,
    alpha_mode: VideoAlphaMode,
    audio_track: str | None,
) -> None:
    """Render frames for RSVP subtitles."""
    ffmpeg_process = open_ffmpeg_process(config, alpha_mode, audio_track)
    if not ffmpeg_process.stdin:
        raise RenderPipelineError(FFMPEG_PROCESS_CODE, "ffmpeg stdin unavailable")

    schedule_index = 0
    current_schedule = None
    current_token_index: int | None = None
    current_token: RsvpToken | None = None

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
                token = current_token
                if token is None:
                    raise RenderPipelineError(
                        FFMPEG_PROCESS_CODE, "render token missing"
                    )
                base_x = int(round(token.anchor_x + token.base_offset[0]))
                base_y = int(round(token.anchor_y + token.base_offset[1]))
                frame_image.paste(
                    token.base_image, (base_x, base_y), token.base_image
                )

                orp_anchor_x = token.anchor_x + token.prefix_width
                orp_x = int(round(orp_anchor_x + token.orp_offset[0]))
                orp_y = int(round(token.anchor_y + token.orp_offset[1]))
                frame_image.paste(token.orp_image, (orp_x, orp_y), token.orp_image)

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
    parser.add_argument("--input-text-file", required=False)
    parser.add_argument("--output-video-file", default="video.mov")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--background", default="transparent", help="transparent (default) or #RRGGBB"
    )
    parser.add_argument("--background-image", default=None)
    parser.add_argument("--fonts-dir", default="fonts")
    parser.add_argument("--audio-track", default=None)
    parser.add_argument("--direction-seed", type=int, default=None)
    parser.add_argument("--emit-directions", action="store_true")
    punctuation_group = parser.add_mutually_exclusive_group()
    punctuation_group.add_argument("--remove-punctuation", action="store_true")
    punctuation_group.add_argument("--keep-punctuation", action="store_true")
    parser.add_argument("--subtitle-renderer", default="motion")
    parser.add_argument("--font-min", type=int, default=None)
    parser.add_argument("--font-max", type=int, default=None)

    parsed = parser.parse_args(argv)
    subtitle_renderer = parse_subtitle_renderer(parsed.subtitle_renderer)
    remove_punctuation = not parsed.keep_punctuation
    background_rgba = parse_hex_color_to_rgba(parsed.background)
    background_image = None
    input_text_file = parsed.input_text_file
    input_text = None
    is_srt = False
    srt_duration = None
    if input_text_file is not None:
        input_text = read_utf8_text_strict(input_text_file)
        is_srt = is_srt_input(input_text_file, input_text)
        if is_srt:
            srt_duration = compute_srt_duration_seconds(
                input_text, remove_punctuation
            )
    audio_track = parsed.audio_track
    audio_duration = (
        get_audio_duration_seconds(audio_track) if audio_track else None
    )

    if input_text_file is None:
        if subtitle_renderer == SubtitleRenderer.RSVP_ORP:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "rsvp_orp requires input-text-file"
            )
        if parsed.emit_directions:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "emit-directions requires input-text-file"
            )
        if parsed.direction_seed is not None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "direction-seed requires input-text-file"
            )
        if parsed.font_min is not None or parsed.font_max is not None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "font-min/font-max require input-text-file"
            )

    if subtitle_renderer == SubtitleRenderer.RSVP_ORP:
        if parsed.direction_seed is not None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "direction-seed is not supported for rsvp_orp"
            )
        if parsed.emit_directions:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "emit-directions is not supported for rsvp_orp"
            )
        if parsed.font_min is not None or parsed.font_max is not None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE,
                "font-min/font-max are not supported for rsvp_orp",
            )
    elif subtitle_renderer != SubtitleRenderer.CRISS_CROSS:
        if parsed.font_min is not None or parsed.font_max is not None:
            raise RenderValidationError(
                INVALID_CONFIG_CODE,
                "font-min/font-max require subtitle-renderer criss_cross",
            )

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

    duration_override = parsed.duration_seconds is not None
    if duration_override:
        duration_seconds = parsed.duration_seconds
        if audio_track or srt_duration is not None:
            LOGGER.warning(
                "render_text_video.input.duration_override: using duration-seconds; "
                "audio will be trimmed or padded when supplied"
            )
    else:
        duration_candidates = []
        if srt_duration is not None:
            duration_candidates.append(srt_duration)
        if audio_duration is not None:
            duration_candidates.append(audio_duration)
        if not duration_candidates:
            raise RenderValidationError(
                INVALID_CONFIG_CODE,
                "duration-seconds is required without audio track or SRT timing",
            )
        duration_seconds = max(duration_candidates)

    alpha_mode = select_alpha_mode(background_rgba, background_image)
    if alpha_mode == VideoAlphaMode.OPAQUE and (width % 2 or height % 2):
        raise RenderValidationError(
            INVALID_CONFIG_CODE,
            "width and height must be even for opaque output",
        )
    min_font_size, max_font_size = compute_font_size_range(width, height)
    if parsed.font_min is not None and parsed.font_max is not None:
        min_font_size = parsed.font_min
        max_font_size = parsed.font_max
    elif parsed.font_min is not None:
        min_font_size = parsed.font_min
        max_font_size = max(max_font_size, min_font_size)
    elif parsed.font_max is not None:
        max_font_size = parsed.font_max
        min_font_size = min(min_font_size, max_font_size)
    if min_font_size > max_font_size:
        raise RenderValidationError(
            INVALID_CONFIG_CODE, "font-min exceeds font-max"
        )

    config = RenderConfig(
        input_text_file=input_text_file,
        output_video_file=parsed.output_video_file,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
        fps=parsed.fps,
        background_rgba=background_rgba,
        fonts_dir=parsed.fonts_dir,
        background_image_path=parsed.background_image,
        subtitle_renderer=subtitle_renderer,
        font_size_min=min_font_size,
        font_size_max=max_font_size,
    )

    return RenderRequest(
        config=config,
        direction_seed=parsed.direction_seed,
        emit_directions=parsed.emit_directions,
        remove_punctuation=remove_punctuation,
        background_image=background_image,
        alpha_mode=alpha_mode,
        audio_track=audio_track,
        duration_override=duration_override,
        input_text=input_text,
    )


def validate_ffmpeg_capabilities(
    alpha_mode: VideoAlphaMode, audio_track: str | None
) -> None:
    """Validate ffmpeg encoders and pixel formats for output."""
    encoding = ENCODING_SPECS[alpha_mode]
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
    if encoding.encoder_name not in encoders_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            f"ffmpeg does not support {encoding.encoder_name} encoder",
        )

    if encoding.alpha_bits is not None:
        encoder_help = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"encoder={encoding.encoder_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if "alpha_bits" not in encoder_help.stdout:
            raise RenderPipelineError(
                FFMPEG_UNSUPPORTED_CODE,
                f"ffmpeg {encoding.encoder_name} encoder does not support alpha_bits",
            )

    if audio_track and AUDIO_CODEC not in encoders_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            f"ffmpeg does not support {AUDIO_CODEC} encoder",
        )

    pixfmts_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-pix_fmts"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    if encoding.pix_fmt not in pixfmts_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            f"ffmpeg does not support {encoding.pix_fmt} pixel format",
        )


def main() -> int:
    """CLI entrypoint."""
    configure_logging()

    try:
        request = parse_args(sys.argv[1:])
        if request.input_text is None:
            validate_ffmpeg_capabilities(request.alpha_mode, request.audio_track)
            render_background_video(
                request.config,
                request.background_image,
                request.alpha_mode,
                request.audio_track,
            )
            return 0
        plan = build_render_plan_from_input(
            request.config,
            request.remove_punctuation,
            request.input_text,
            request.duration_override,
        )
        if request.config.subtitle_renderer == SubtitleRenderer.RSVP_ORP:
            tokens = build_rsvp_tokens(plan.words, request.config)
            validate_ffmpeg_capabilities(request.alpha_mode, request.audio_track)
            render_rsvp_video(
                request.config,
                plan,
                tokens,
                request.background_image,
                request.alpha_mode,
                request.audio_track,
            )
            return 0
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
            align_letter_band_positions_to_entry(
                normalize_letter_band_positions(
                    adjust_letter_band_positions(
                        compute_letter_band_positions(
                            sizes, should_reverse_letter_order(direction)
                        ),
                        token.letters,
                        direction,
                    ),
                    token.letters,
                    direction,
                    request.config.width,
                    request.config.height,
                ),
                token.letters,
                direction,
                request.config.width,
                request.config.height,
            )
            for sizes, token, direction in zip(letter_band_sizes, tokens, directions)
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
        validate_ffmpeg_capabilities(request.alpha_mode, request.audio_track)
        render_video(
            request.config,
            plan,
            tokens,
            directions,
            request.background_image,
            request.alpha_mode,
            request.audio_track,
        )
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
