#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "whisperx==3.3.0",
#   "matplotlib<4",
#   "numpy<2",
# ]
# ///
"""Force-align audio or video to input text and emit SRT."""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import whisperx

LOGGER = logging.getLogger("audio_to_text")

INPUT_AUDIO_CODE = "audio_to_text.input.audio_file"
INPUT_TEXT_CODE = "audio_to_text.input.text_file"
OUTPUT_SRT_CODE = "audio_to_text.output.srt_file"
INVALID_CONFIG_CODE = "audio_to_text.input.invalid_config"
ALIGN_MODEL_CODE = "audio_to_text.align.model"
ALIGNMENT_CODE = "audio_to_text.align.failed"
ALIGNMENT_TIMESTAMP_CODE = "audio_to_text.align.missing_timestamps"
DEVICE_UNAVAILABLE_CODE = "audio_to_text.device.unavailable"
DEVICE_AUTO = "auto"
SUPPORTED_DEVICES = {DEVICE_AUTO, "cpu", "cuda"}
SRT_TIME_RANGE_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$"
)


class AlignmentValidationError(ValueError):
    """Validation error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AlignmentPipelineError(RuntimeError):
    """Runtime error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AlignmentRequest:
    """Parsed CLI request for alignment."""

    input_audio: str
    input_text: str
    output_srt: str
    language: str
    device: str
    align_model: str | None

    def __post_init__(self) -> None:
        if not self.input_audio.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "input-audio must be non-empty"
            )
        if not self.input_text.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "input-text must be non-empty"
            )
        if not self.output_srt.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "output-srt must be non-empty"
            )
        if not self.language.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "language must be non-empty"
            )
        if self.device not in SUPPORTED_DEVICES:
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, f"invalid device: {self.device!r}"
            )
        if not self.output_srt.lower().endswith(".srt"):
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "output-srt must end with .srt"
            )


@dataclass(frozen=True)
class AlignedWord:
    """Aligned word with time bounds."""

    text: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not self.text:
            raise AlignmentPipelineError(
                ALIGNMENT_TIMESTAMP_CODE, "aligned word text is empty"
            )
        if self.start_seconds < 0:
            raise AlignmentPipelineError(
                ALIGNMENT_TIMESTAMP_CODE, "aligned word start is negative"
            )
        if self.end_seconds <= self.start_seconds:
            raise AlignmentPipelineError(
                ALIGNMENT_TIMESTAMP_CODE,
                "aligned word end is not after start",
            )


def configure_logging() -> None:
    """Configure logging for CLI output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def read_utf8_text_strict(file_path: str) -> str:
    """Read a UTF-8 file with strict decoding."""
    try:
        with open(file_path, "rb") as file_handle:
            file_bytes = file_handle.read()
    except FileNotFoundError as exc:
        raise AlignmentValidationError(
            INPUT_TEXT_CODE, f"input text file not found: {file_path}"
        ) from exc
    except OSError as exc:
        raise AlignmentValidationError(
            INPUT_TEXT_CODE, f"input text file error: {file_path}"
        ) from exc

    try:
        return file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AlignmentValidationError(
            INPUT_TEXT_CODE,
            f"input text file is not valid UTF-8 at byte offset {exc.start}",
        ) from exc


def ensure_audio_file_exists(file_path: str) -> None:
    """Ensure the input audio file exists."""
    if not os.path.isfile(file_path):
        raise AlignmentValidationError(
            INPUT_AUDIO_CODE, f"input audio file not found: {file_path}"
        )


def is_srt_text(file_path: str, text_value: str) -> bool:
    """Return True when the input text appears to be SRT."""
    if file_path.lower().endswith(".srt"):
        return True
    return any(
        SRT_TIME_RANGE_PATTERN.fullmatch(line.strip())
        for line in text_value.splitlines()
        if line.strip()
    )


def sanitize_srt_text(text_value: str) -> str:
    """Remove SRT indices and time ranges from input text."""
    cleaned_lines: list[str] = []
    for line in text_value.replace("\ufeff", "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if SRT_TIME_RANGE_PATTERN.fullmatch(stripped):
            continue
        cleaned_lines.append(stripped)
    return " ".join(cleaned_lines)


def normalize_transcript(text_value: str, input_text_path: str) -> str:
    """Normalize transcript text for alignment."""
    sanitized = (
        sanitize_srt_text(text_value)
        if is_srt_text(input_text_path, text_value)
        else text_value
    )
    normalized = " ".join(sanitized.replace("\ufeff", "").split())
    if not normalized:
        raise AlignmentValidationError(
            INVALID_CONFIG_CODE, "input text contains no words"
        )
    return normalized


def default_output_path(input_audio: str) -> str:
    """Derive the default SRT output path from the audio path."""
    return str(Path(input_audio).with_suffix(".srt"))


def parse_args(argv: Sequence[str]) -> AlignmentRequest:
    """Parse CLI arguments into an AlignmentRequest."""
    parser = argparse.ArgumentParser(prog="audio_to_text.py", add_help=True)
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--output-srt", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default=DEVICE_AUTO)
    parser.add_argument("--align-model", default=None)
    parsed = parser.parse_args(argv)

    output_srt = parsed.output_srt or default_output_path(parsed.input_audio)
    device_value = str(parsed.device).strip().lower()
    return AlignmentRequest(
        input_audio=parsed.input_audio,
        input_text=parsed.input_text,
        output_srt=output_srt,
        language=parsed.language,
        device=device_value,
        align_model=parsed.align_model,
    )


def load_alignment_model(
    language: str, device: str, model_name: str | None
) -> tuple[object, dict[str, object]]:
    """Load the alignment model and metadata."""
    try:
        return whisperx.load_align_model(
            language_code=language, device=device, model_name=model_name
        )
    except Exception as exc:
        raise AlignmentPipelineError(
            ALIGN_MODEL_CODE, f"align model load failed: {exc}"
        ) from exc


def resolve_device(device_value: str) -> str:
    """Resolve auto device selection to a concrete device."""
    if device_value != DEVICE_AUTO:
        return device_value
    try:
        import torch
    except Exception as exc:
        raise AlignmentPipelineError(
            DEVICE_UNAVAILABLE_CODE, f"torch is unavailable: {exc}"
        ) from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def align_words(
    audio_path: str,
    transcript_text: str,
    language: str,
    device: str,
    align_model_name: str | None,
) -> tuple[AlignedWord, ...]:
    """Align transcript text to the audio and return word timings."""
    audio = whisperx.load_audio(audio_path)
    audio_duration = float(len(audio)) / float(whisperx.audio.SAMPLE_RATE)
    segments = [{"start": 0.0, "end": audio_duration, "text": transcript_text}]

    align_model, metadata = load_alignment_model(language, device, align_model_name)
    try:
        result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, f"alignment failed: {exc}"
        ) from exc

    return extract_aligned_words(result.get("segments", []))


def extract_aligned_words(
    segments: Iterable[dict[str, object]],
) -> tuple[AlignedWord, ...]:
    """Extract aligned words from alignment output."""
    words: list[AlignedWord] = []
    for segment in segments:
        for word in segment.get("words", []):
            text_value = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")
            if text_value == "":
                raise AlignmentPipelineError(
                    ALIGNMENT_TIMESTAMP_CODE, "aligned word text is empty"
                )
            if start is None or end is None:
                raise AlignmentPipelineError(
                    ALIGNMENT_TIMESTAMP_CODE,
                    f"aligned word is missing timestamps: {text_value}",
                )
            words.append(
                AlignedWord(
                    text=text_value,
                    start_seconds=float(start),
                    end_seconds=float(end),
                )
            )

    if not words:
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, "alignment produced no words"
        )

    return tuple(words)


def srt_timestamp_from_seconds(seconds: float, rounding: str) -> int:
    """Convert seconds to milliseconds for SRT output."""
    if rounding == "floor":
        return int(math.floor(seconds * 1000))
    if rounding == "ceil":
        return int(math.ceil(seconds * 1000))
    raise AlignmentPipelineError(
        ALIGNMENT_TIMESTAMP_CODE, f"invalid rounding mode: {rounding}"
    )


def format_srt_timestamp(milliseconds: int) -> str:
    """Format milliseconds into an SRT timestamp."""
    if milliseconds < 0:
        raise AlignmentPipelineError(
            ALIGNMENT_TIMESTAMP_CODE, "timestamp milliseconds must be non-negative"
        )
    total_seconds, millis = divmod(milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(words: Sequence[AlignedWord]) -> str:
    """Build SRT content from aligned words."""
    lines: list[str] = []
    for index, word in enumerate(words, start=1):
        start_ms = srt_timestamp_from_seconds(word.start_seconds, "floor")
        end_ms = srt_timestamp_from_seconds(word.end_seconds, "ceil")
        if end_ms <= start_ms:
            raise AlignmentPipelineError(
                ALIGNMENT_TIMESTAMP_CODE,
                f"aligned word window is empty: {word.text}",
            )
        lines.append(str(index))
        lines.append(
            f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}"
        )
        lines.append(word.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_srt_file(file_path: str, content: str) -> None:
    """Write the SRT content to disk."""
    output_path = Path(file_path)
    if output_path.parent and not output_path.parent.exists():
        raise AlignmentValidationError(
            OUTPUT_SRT_CODE,
            f"output directory does not exist: {output_path.parent}",
        )
    try:
        output_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise AlignmentValidationError(
            OUTPUT_SRT_CODE, f"failed to write srt file: {file_path}"
        ) from exc


def main() -> int:
    """CLI entrypoint."""
    configure_logging()
    try:
        request = parse_args(sys.argv[1:])
        resolved_device = resolve_device(request.device)
        ensure_audio_file_exists(request.input_audio)
        transcript_text = normalize_transcript(
            read_utf8_text_strict(request.input_text),
            request.input_text,
        )
        words = align_words(
            request.input_audio,
            transcript_text,
            request.language,
            resolved_device,
            request.align_model,
        )
        srt_content = build_srt(words)
        write_srt_file(request.output_srt, srt_content)
        LOGGER.info("audio_to_text.output.srt_written: %s", request.output_srt)
        return 0
    except AlignmentValidationError as exc:
        LOGGER.error("%s: %s", exc.code, str(exc).strip())
        return 1
    except AlignmentPipelineError as exc:
        LOGGER.error("%s: %s", exc.code, str(exc).strip())
        return 1
    except Exception as exc:
        LOGGER.error("audio_to_text.unhandled_error: %s", str(exc).strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
