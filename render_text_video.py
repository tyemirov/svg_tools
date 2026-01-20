#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10"
# ]
# ///
"""Render animated word-by-word text into a ProRes MOV."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from domain.text_video import (
    EMPTY_TEXT_CODE,
    FONT_DIR_CODE,
    FONT_LOAD_CODE,
    INPUT_FILE_CODE,
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


def compute_font_size(width: int, height: int) -> int:
    """Compute a font size based on frame dimensions."""
    base_value = max(24, min(width, height) // 8)
    return int(base_value)


def load_fonts(font_files: Sequence[str], font_size: int) -> list[ImageFont.FreeTypeFont]:
    """Load fonts from paths at the requested size."""
    loaded_fonts: list[ImageFont.FreeTypeFont] = []
    last_error: Exception | None = None
    for font_file_path in font_files:
        try:
            loaded_fonts.append(
                ImageFont.truetype(
                    font_file_path, size=font_size, layout_engine=ImageFont.Layout.BASIC
                )
            )
        except Exception as exc:
            last_error = exc

    if not loaded_fonts:
        raise RenderValidationError(
            FONT_LOAD_CODE, "failed to load any fonts from fonts directory"
        ) from last_error
    return loaded_fonts


def build_tokens(
    words: Sequence[str],
    fonts: Sequence[ImageFont.FreeTypeFont],
    palette: Sequence[Tuple[int, int, int, int]],
) -> list[WordToken]:
    """Create styled tokens for each word."""
    tokens: list[WordToken] = []
    for index_value, word_text in enumerate(words):
        font_value = fonts[index_value % len(fonts)]
        color_value = palette[index_value % len(palette)]
        tokens.append(
            WordToken(
                text=word_text, style=WordStyle(font=font_value, color_rgba=color_value)
            )
        )
    return tokens


def measure_text(
    draw_context: ImageDraw.ImageDraw, token: WordToken
) -> Tuple[int, int, int, int]:
    """Measure a token bounding box."""
    return draw_context.textbbox(
        (0, 0), token.text, font=token.style.font, stroke_width=0
    )


def compute_position_for_frame(
    direction: str,
    progress_value: float,
    frame_width: int,
    frame_height: int,
    text_width: int,
    text_height: int,
) -> Tuple[int, int]:
    """Compute a text position given a motion direction and progress."""
    clamped_progress = min(1.0, max(0.0, progress_value))
    center_x = (frame_width - text_width) // 2
    center_y = (frame_height - text_height) // 2

    if direction == "L2R":
        start_x = -text_width
        end_x = frame_width
        x_value = int(round(start_x + (end_x - start_x) * clamped_progress))
        return (x_value, center_y)

    if direction == "R2L":
        start_x = frame_width
        end_x = -text_width
        x_value = int(round(start_x + (end_x - start_x) * clamped_progress))
        return (x_value, center_y)

    if direction == "T2B":
        start_y = -text_height
        end_y = frame_height
        y_value = int(round(start_y + (end_y - start_y) * clamped_progress))
        return (center_x, y_value)

    if direction == "B2T":
        start_y = frame_height
        end_y = -text_height
        y_value = int(round(start_y + (end_y - start_y) * clamped_progress))
        return (center_x, y_value)

    raise RenderPipelineError(INTERNAL_DIRECTION_CODE, f"unsupported direction: {direction}")


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
        "prores_ks",
        "-profile:v",
        "4444",
        "-pix_fmt",
        "yuva444p10le",
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
    config: RenderConfig, text_value: str
) -> Tuple[SubtitleWindow, ...]:
    """Build subtitle windows from plain text or SRT input."""
    is_srt_extension = config.input_text_file.lower().endswith(".srt")
    is_srt_content = any(
        SRT_TIME_RANGE_PATTERN.fullmatch(line.strip())
        for line in text_value.splitlines()
        if line.strip()
    )
    if is_srt_extension or is_srt_content:
        return parse_srt(text_value)

    words = tokenize_words(text_value)
    return (
        SubtitleWindow(
            start_seconds=0.0,
            end_seconds=config.duration_seconds,
            words=words,
        ),
    )


def build_render_inputs(config: RenderConfig) -> Tuple[RenderPlan, list[WordToken]]:
    """Load input text, build a render plan, and generate tokens."""
    input_text = read_utf8_text_strict(config.input_text_file)
    windows = build_subtitle_windows(config, input_text)
    plan = build_render_plan(
        windows=windows,
        fps=config.fps,
        duration_seconds=config.duration_seconds,
    )
    palette = build_palette_rgba()
    font_files = list_font_files(config.fonts_dir)
    font_size = compute_font_size(config.width, config.height)
    fonts = load_fonts(font_files, font_size=font_size)
    tokens = build_tokens(plan.words, fonts=fonts, palette=palette)
    return plan, tokens


def render_video(config: RenderConfig, plan: RenderPlan, tokens: Sequence[WordToken]) -> None:
    """Render frames based on the render plan."""
    ffmpeg_process = open_ffmpeg_process(config)
    if not ffmpeg_process.stdin:
        raise RenderPipelineError(FFMPEG_PROCESS_CODE, "ffmpeg stdin unavailable")

    schedule_index = 0
    current_schedule = None

    try:
        for frame_index in range(plan.total_frames):
            if schedule_index < len(plan.scheduled_words):
                schedule = plan.scheduled_words[schedule_index]
                schedule_end = schedule.start_frame + schedule.frame_count
                if frame_index >= schedule_end:
                    schedule_index += 1
                    current_schedule = None
                if schedule_index < len(plan.scheduled_words):
                    schedule = plan.scheduled_words[schedule_index]
                    if frame_index >= schedule.start_frame:
                        current_schedule = schedule

            frame_image = Image.new(
                "RGBA", (config.width, config.height), color=config.background_rgba
            )
            draw_context = ImageDraw.Draw(frame_image)

            if current_schedule is not None:
                token = tokens[current_schedule.token_index]
                within_word_index = frame_index - current_schedule.start_frame
                progress = within_word_index / float(
                    max(1, current_schedule.frame_count - 1)
                )
                direction = DIRECTIONS[current_schedule.token_index % len(DIRECTIONS)]

                bbox = measure_text(draw_context, token)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                pos_x, pos_y = compute_position_for_frame(
                    direction=direction,
                    progress_value=progress,
                    frame_width=config.width,
                    frame_height=config.height,
                    text_width=text_width,
                    text_height=text_height,
                )

                draw_context.text(
                    (pos_x, pos_y),
                    token.text,
                    font=token.style.font,
                    fill=token.style.color_rgba,
                )

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


def parse_args(argv: Sequence[str]) -> RenderConfig:
    """Parse CLI arguments into a RenderConfig."""
    parser = argparse.ArgumentParser(prog="render_text_video.py", add_help=True)
    parser.add_argument("--input-text-file", required=True)
    parser.add_argument("--output-video-file", default="video.mov")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--background", default="transparent", help="transparent (default) or #RRGGBB"
    )
    parser.add_argument("--fonts-dir", default="fonts")

    parsed = parser.parse_args(argv)
    background_rgba = parse_hex_color_to_rgba(parsed.background)

    return RenderConfig(
        input_text_file=parsed.input_text_file,
        output_video_file=parsed.output_video_file,
        width=parsed.width,
        height=parsed.height,
        duration_seconds=parsed.duration_seconds,
        fps=parsed.fps,
        background_rgba=background_rgba,
        fonts_dir=parsed.fonts_dir,
    )


def validate_ffmpeg_capabilities() -> None:
    """Validate ffmpeg encoders and pixel formats required for ProRes."""
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
    if "prores_ks" not in encoders_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            "ffmpeg does not support prores_ks encoder",
        )

    pixfmts_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-pix_fmts"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    if "yuva444p10le" not in pixfmts_result.stdout:
        raise RenderPipelineError(
            FFMPEG_UNSUPPORTED_CODE,
            "ffmpeg does not support yuva444p10le pixel format",
        )


def main() -> int:
    """CLI entrypoint."""
    configure_logging()

    try:
        config = parse_args(sys.argv[1:])
        validate_ffmpeg_capabilities()
        plan, tokens = build_render_inputs(config)
        render_video(config, plan, tokens)
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
