#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10"
# ]
# ///
import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RenderConfig:
    input_text_file: str
    output_video_file: str
    width: int
    height: int
    duration_seconds: float
    fps: int
    background_rgba: Tuple[int, int, int, int]
    fonts_dir: str


@dataclass(frozen=True)
class WordStyle:
    font: ImageFont.FreeTypeFont
    color_rgba: Tuple[int, int, int, int]


@dataclass(frozen=True)
class WordToken:
    text: str
    style: WordStyle


DIRECTIONS = ("L2R", "R2L", "T2B", "B2T")


def parse_hex_color_to_rgba(color_value: str) -> Tuple[int, int, int, int]:
    normalized = color_value.strip()
    if normalized.lower() == "transparent":
        return (0, 0, 0, 0)

    match_value = re.fullmatch(r"#([0-9a-fA-F]{6})", normalized)
    if not match_value:
        raise ValueError(
            f"Invalid color value: {color_value!r}. Use 'transparent' or '#RRGGBB'."
        )

    rgb_hex = match_value.group(1)
    red_value = int(rgb_hex[0:2], 16)
    green_value = int(rgb_hex[2:4], 16)
    blue_value = int(rgb_hex[4:6], 16)
    return (red_value, green_value, blue_value, 255)


def ensure_ffmpeg_available() -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    try:
        subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError("ffmpeg exists but could not be executed.") from exc


def read_utf8_text_strict(file_path: str) -> str:
    try:
        with open(file_path, "rb") as file_handle:
            file_bytes = file_handle.read()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input text file not found: {file_path}") from exc

    try:
        return file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Input text file is not valid UTF-8. "
            f"Decode error at byte offset {exc.start}: {exc.reason}."
        ) from exc


def tokenize_words(text_value: str) -> List[str]:
    stripped_text = text_value.replace("\ufeff", "").strip()
    if not stripped_text:
        return []
    return stripped_text.split()


def list_font_files(fonts_dir: str) -> List[str]:
    if not os.path.isdir(fonts_dir):
        raise RuntimeError(f"Fonts directory does not exist: {fonts_dir}")

    font_files: List[str] = []
    for entry_name in sorted(os.listdir(fonts_dir)):
        lower_name = entry_name.lower()
        if lower_name.endswith(".ttf") or lower_name.endswith(".otf"):
            font_files.append(os.path.join(fonts_dir, entry_name))

    if not font_files:
        raise RuntimeError(
            f"No font files found in {fonts_dir}. "
            "Place open-source .ttf/.otf fonts there (bold variants recommended)."
        )
    return font_files


def build_palette_rgba() -> List[Tuple[int, int, int, int]]:
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
    base_value = max(24, min(width, height) // 8)
    return int(base_value)


def load_fonts(font_files: List[str], font_size: int) -> List[ImageFont.FreeTypeFont]:
    loaded_fonts: List[ImageFont.FreeTypeFont] = []
    last_error: Optional[Exception] = None
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
        raise RuntimeError(
            "Failed to load any fonts from fonts directory."
        ) from last_error
    return loaded_fonts


def build_tokens(
    words: List[str],
    fonts: List[ImageFont.FreeTypeFont],
    palette: List[Tuple[int, int, int, int]],
) -> List[WordToken]:
    tokens: List[WordToken] = []
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

    raise RuntimeError(f"Unsupported direction: {direction}")


def open_ffmpeg_process(config: RenderConfig) -> subprocess.Popen:
    if not config.output_video_file.lower().endswith(".mov"):
        raise RuntimeError("Output must be .mov (MOV only).")

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
        raise RuntimeError("ffmpeg not found on PATH.") from exc


def render_video(config: RenderConfig) -> None:
    input_text = read_utf8_text_strict(config.input_text_file)
    words = tokenize_words(input_text)
    if not words:
        raise RuntimeError("Input text file contains no words after trimming.")

    palette = build_palette_rgba()
    font_files = list_font_files(config.fonts_dir)
    font_size = compute_font_size(config.width, config.height)
    fonts = load_fonts(font_files, font_size=font_size)
    tokens = build_tokens(words, fonts=fonts, palette=palette)

    total_frames = int(round(config.duration_seconds * config.fps))
    if total_frames <= 0:
        raise RuntimeError(
            "Duration and FPS produce zero frames. Increase duration or FPS."
        )

    word_count = len(tokens)
    frames_per_word = max(1, total_frames // word_count)
    effective_total_frames = frames_per_word * word_count

    ffmpeg_process = open_ffmpeg_process(config)
    if not ffmpeg_process.stdin:
        raise RuntimeError("Failed to open ffmpeg stdin pipe.")

    try:
        for frame_index in range(effective_total_frames):
            token_index = frame_index // frames_per_word
            token = tokens[min(token_index, word_count - 1)]

            within_word_index = frame_index % frames_per_word
            progress = within_word_index / float(max(1, frames_per_word - 1))

            direction = DIRECTIONS[token_index % len(DIRECTIONS)]

            frame_image = Image.new(
                "RGBA", (config.width, config.height), color=config.background_rgba
            )
            draw_context = ImageDraw.Draw(frame_image)

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
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code}.\n{stderr_text}"
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


def parse_args(argv: List[str]) -> RenderConfig:
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

    if parsed.width <= 0 or parsed.height <= 0:
        raise RuntimeError("Width and height must be positive integers.")
    if parsed.fps <= 0:
        raise RuntimeError("FPS must be a positive integer.")
    if parsed.duration_seconds <= 0:
        raise RuntimeError("Duration must be > 0 seconds.")

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
    try:
        version_result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError("ffmpeg is not available or not executable.") from exc

    if "ffmpeg version" not in version_result.stdout.lower():
        raise RuntimeError("ffmpeg version output is unexpected.")

    encoders_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    if "prores_ks" not in encoders_result.stdout:
        raise RuntimeError(
            "ffmpeg does not support prores_ks encoder (required for ProRes 4444)."
        )

    pixfmts_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-pix_fmts"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    if "yuva444p10le" not in pixfmts_result.stdout:
        raise RuntimeError(
            "ffmpeg does not support yuva444p10le pixel format (alpha required)."
        )


def main() -> int:
    try:
        config = parse_args(sys.argv[1:])
        validate_ffmpeg_capabilities()
        render_video(config)
        return 0
    except Exception as exc:
        sys.stderr.write(str(exc).strip() + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
