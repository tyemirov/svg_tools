"""Domain types and parsing for render_text_video."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Tuple

INVALID_COLOR_CODE = "render_text_video.input.invalid_color"
INVALID_CONFIG_CODE = "render_text_video.input.invalid_config"
INVALID_SRT_CODE = "render_text_video.input.invalid_srt"
INVALID_WINDOW_CODE = "render_text_video.input.invalid_window"
EMPTY_TEXT_CODE = "render_text_video.input.empty_text"
INPUT_FILE_CODE = "render_text_video.input.file_error"
FONT_DIR_CODE = "render_text_video.input.fonts_missing"
FONT_LOAD_CODE = "render_text_video.input.fonts_unloadable"
BACKGROUND_IMAGE_CODE = "render_text_video.input.background_image"
AUDIO_FILE_CODE = "render_text_video.input.audio_track"
INVALID_RENDERER_CODE = "render_text_video.input.invalid_renderer"

SRT_TIME_RANGE_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})$"
)
SRT_TIMECODE_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
TRAILING_PUNCTUATION = ".,!?:;"


class RenderValidationError(ValueError):
    """Validation error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SubtitleRenderer(str, Enum):
    """Supported subtitle render modes."""

    MOTION = "motion"
    CRISS_CROSS = "criss_cross"
    RSVP_ORP = "rsvp_orp"


@dataclass(frozen=True)
class RenderConfig:
    """Validated configuration for render_text_video."""

    input_text_file: str | None
    output_video_file: str
    width: int
    height: int
    duration_seconds: float
    fps: int
    background_rgba: Tuple[int, int, int, int]
    fonts_dir: str
    background_image_path: str | None
    subtitle_renderer: SubtitleRenderer
    font_size_min: int
    font_size_max: int

    def __post_init__(self) -> None:
        if self.input_text_file is not None and not self.input_text_file.strip():
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "input_text_file must be non-empty"
            )
        if self.width <= 0 or self.height <= 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "width and height must be positive"
            )
        if self.fps <= 0:
            raise RenderValidationError(INVALID_CONFIG_CODE, "fps must be positive")
        if self.duration_seconds <= 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "duration_seconds must be positive"
            )
        if not self.output_video_file.lower().endswith(".mov"):
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "output_video_file must end with .mov"
            )
        if len(self.background_rgba) != 4:
            raise RenderValidationError(INVALID_CONFIG_CODE, "background_rgba is invalid")
        for channel in self.background_rgba:
            if channel < 0 or channel > 255:
                raise RenderValidationError(
                    INVALID_CONFIG_CODE, "background_rgba channel out of range"
                )
        if self.background_image_path is not None and not self.background_image_path.strip():
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "background_image_path must be non-empty"
            )
        if not isinstance(self.subtitle_renderer, SubtitleRenderer):
            raise RenderValidationError(
                INVALID_RENDERER_CODE, "subtitle_renderer is invalid"
            )
        if self.font_size_min <= 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "font_size_min must be positive"
            )
        if self.font_size_max <= 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "font_size_max must be positive"
            )
        if self.font_size_min > self.font_size_max:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "font_size_min exceeds font_size_max"
            )


@dataclass(frozen=True)
class SubtitleWindow:
    """Time-bounded subtitle words."""

    start_seconds: float
    end_seconds: float
    words: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle start time must be non-negative"
            )
        if self.end_seconds <= self.start_seconds:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle end time must be after start time"
            )
        if not self.words:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle window contains no words"
            )


def strip_punctuation(text_value: str) -> str:
    """Remove unicode punctuation characters from text."""
    return "".join(
        character
        for character in text_value
        if not unicodedata.category(character).startswith("P")
    )


def split_trailing_punctuation(token: str) -> Tuple[str, str]:
    """Split a token into its core and trailing punctuation."""
    stripped = token.rstrip(TRAILING_PUNCTUATION)
    if not stripped:
        return token, ""
    return stripped, token[len(stripped) :]


def tokenize_words(text_value: str, remove_punctuation: bool) -> Tuple[str, ...]:
    """Split text into whitespace-delimited words."""
    stripped_text = text_value.replace("\ufeff", "").strip()
    if not stripped_text:
        raise RenderValidationError(EMPTY_TEXT_CODE, "input text contains no words")

    raw_words = stripped_text.split()
    if remove_punctuation:
        cleaned_words = [strip_punctuation(word) for word in raw_words]
        words = [word for word in cleaned_words if word]
    else:
        words = raw_words

    if not words:
        raise RenderValidationError(EMPTY_TEXT_CODE, "input text contains no words")

    return tuple(words)


def parse_timecode(timecode_value: str) -> float:
    """Parse an SRT timecode into seconds."""
    match = SRT_TIMECODE_PATTERN.fullmatch(timecode_value.strip())
    if not match:
        raise RenderValidationError(
            INVALID_SRT_CODE, f"invalid timecode: {timecode_value!r}"
        )
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_subtitle_renderer(value: str) -> SubtitleRenderer:
    """Parse a subtitle renderer name into a SubtitleRenderer."""
    normalized = value.strip().lower()
    try:
        return SubtitleRenderer(normalized)
    except ValueError as exc:
        raise RenderValidationError(
            INVALID_RENDERER_CODE, f"invalid subtitle renderer: {value!r}"
        ) from exc


def parse_srt(text_value: str, remove_punctuation: bool) -> Tuple[SubtitleWindow, ...]:
    """Parse SRT content into subtitle windows."""
    normalized = text_value.replace("\ufeff", "").strip()
    if not normalized:
        raise RenderValidationError(EMPTY_TEXT_CODE, "SRT input is empty")

    blocks = re.split(r"\n\s*\n", normalized)
    windows: list[SubtitleWindow] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        if not lines:
            raise RenderValidationError(INVALID_SRT_CODE, "SRT block missing timecode")

        time_line = lines[0]
        match = SRT_TIME_RANGE_PATTERN.fullmatch(time_line)
        if not match:
            raise RenderValidationError(
                INVALID_SRT_CODE, f"invalid time range: {time_line!r}"
            )

        start_seconds = parse_timecode(match.group("start"))
        end_seconds = parse_timecode(match.group("end"))
        subtitle_lines = lines[1:]
        if not subtitle_lines:
            raise RenderValidationError(INVALID_SRT_CODE, "SRT block missing text")

        words = tokenize_words(" ".join(subtitle_lines), remove_punctuation)
        windows.append(
            SubtitleWindow(
                start_seconds=start_seconds, end_seconds=end_seconds, words=words
            )
        )

    if not windows:
        raise RenderValidationError(EMPTY_TEXT_CODE, "SRT contains no subtitles")

    return tuple(windows)
