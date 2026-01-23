#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "whisperx==3.3.0",
#   "matplotlib<4",
#   "numpy<2",
#   "safetensors",
# ]
# ///
"""Force-align audio or video to input text and emit SRT."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from types import ModuleType
from typing import Iterable, NamedTuple, Sequence
from urllib.parse import urlparse


LOGGER = logging.getLogger("audio_to_text")

INPUT_AUDIO_CODE = "audio_to_text.input.audio_file"
INPUT_TEXT_CODE = "audio_to_text.input.text_file"
OUTPUT_SRT_CODE = "audio_to_text.output.srt_file"
INVALID_CONFIG_CODE = "audio_to_text.input.invalid_config"
INVALID_LANGUAGE_CODE = "audio_to_text.input.invalid_language"
ALIGN_MODEL_CODE = "audio_to_text.align.model"
ALIGNMENT_CODE = "audio_to_text.align.failed"
ALIGNMENT_TIMESTAMP_CODE = "audio_to_text.align.missing_timestamps"
DEVICE_UNAVAILABLE_CODE = "audio_to_text.device.unavailable"
TORCH_VERSION_CODE = "audio_to_text.dependency.torch_version"
INVALID_PROGRESS_CODE = "audio_to_text.job.invalid_progress"
DEVICE_AUTO = "auto"
DEVICE_LABELS = {
    DEVICE_AUTO: "Auto (GPU if available)",
    "cpu": "CPU",
    "cuda": "CUDA",
}
SUPPORTED_DEVICES = set(DEVICE_LABELS.keys())
SUPPORTED_ALIGNMENT_LANGUAGES = (
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("zh", "Chinese"),
    ("nl", "Dutch"),
    ("uk", "Ukrainian"),
    ("pt", "Portuguese"),
    ("ar", "Arabic"),
    ("cs", "Czech"),
    ("ru", "Russian"),
    ("pl", "Polish"),
    ("hu", "Hungarian"),
    ("fi", "Finnish"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("tr", "Turkish"),
    ("da", "Danish"),
    ("he", "Hebrew"),
    ("vi", "Vietnamese"),
    ("ko", "Korean"),
    ("ur", "Urdu"),
    ("te", "Telugu"),
    ("hi", "Hindi"),
    ("ca", "Catalan"),
    ("ml", "Malayalam"),
    ("no", "Norwegian Bokmal"),
    ("nn", "Norwegian Nynorsk"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("hr", "Croatian"),
    ("ro", "Romanian"),
    ("eu", "Basque"),
    ("gl", "Galician"),
    ("ka", "Georgian"),
)
SUPPORTED_LANGUAGE_CODES = {code for code, _ in SUPPORTED_ALIGNMENT_LANGUAGES}
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 7860
MAX_PORT = 65535
TORCH_MIN_VERSION = (2, 6)
TORCH_MIN_VERSION_TEXT = "2.6"
TORCHAUDIO_ALIGNMENT_LANGUAGES = {"en", "fr", "de", "es", "it"}
ALIGNMENT_MODEL_OVERRIDES = {
    "ru": "UrukHan/wav2vec2-russian",
}
SRT_TIME_RANGE_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$"
)
TORCH_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)")


class RequestMode(str, Enum):
    """Supported runtime modes."""

    CLI = "cli"
    UI = "ui"


class JobStatus(str, Enum):
    """Lifecycle states for UI jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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

    mode: RequestMode
    input_audio: str | None
    input_text: str | None
    output_srt: str | None
    language: str
    device: str
    ui_host: str
    ui_port: int

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "language must be non-empty"
            )
        if self.language not in SUPPORTED_LANGUAGE_CODES:
            raise AlignmentValidationError(
                INVALID_LANGUAGE_CODE, f"unsupported language: {self.language!r}"
            )
        if self.device not in SUPPORTED_DEVICES:
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, f"invalid device: {self.device!r}"
            )
        if self.mode == RequestMode.CLI:
            if self.input_audio is None or not self.input_audio.strip():
                raise AlignmentValidationError(
                    INPUT_AUDIO_CODE, "input-audio is required"
                )
            if self.input_text is None or not self.input_text.strip():
                raise AlignmentValidationError(
                    INPUT_TEXT_CODE, "input-text is required"
                )
            if self.output_srt is None or not self.output_srt.strip():
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "output-srt must be non-empty"
                )
            if not self.output_srt.lower().endswith(".srt"):
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "output-srt must end with .srt"
                )
        else:
            if not self.ui_host.strip():
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "ui-host must be non-empty"
                )
            if self.ui_port <= 0 or self.ui_port > MAX_PORT:
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "ui-port is invalid"
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


@dataclass(frozen=True)
class UiDefaults:
    """Default configuration values for the UI."""

    language: str
    device: str


@dataclass(frozen=True)
class AlignmentJob:
    """State snapshot for a background alignment job."""

    job_id: str
    status: JobStatus
    message: str | None
    output_srt: str | None
    progress: float

    def __post_init__(self) -> None:
        if self.progress < 0.0 or self.progress > 1.0:
            raise AlignmentPipelineError(
                INVALID_PROGRESS_CODE,
                f"progress must be between 0 and 1: {self.progress}",
            )


@dataclass
class JobStore:
    """Thread-safe store for background UI jobs."""

    root_dir: Path
    jobs: dict[str, AlignmentJob] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def create_job(self) -> AlignmentJob:
        """Create a new queued job."""
        job_id = uuid.uuid4().hex
        job = AlignmentJob(job_id, JobStatus.QUEUED, None, None, 0.0)
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> AlignmentJob | None:
        """Fetch a job by ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        message: str | None = None,
        output_srt: str | None = None,
        progress: float | None = None,
    ) -> AlignmentJob:
        """Update a job's status."""
        with self.lock:
            current = self.jobs.get(job_id)
        progress_value = progress
        if progress_value is None:
            progress_value = current.progress if current else 0.0
        job = AlignmentJob(job_id, status, message, output_srt, progress_value)
        with self.lock:
            self.jobs[job_id] = job
        return job

    def job_dir(self, job_id: str) -> Path:
        """Return the directory for job artifacts."""
        return self.root_dir / job_id


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


def normalize_device_value(raw_value: str, default_device: str) -> str:
    """Normalize a device string for UI usage."""
    normalized = raw_value.strip().lower()
    if not normalized:
        return default_device
    if normalized not in SUPPORTED_DEVICES:
        raise AlignmentValidationError(
            INVALID_CONFIG_CODE, f"invalid device: {raw_value!r}"
        )
    return normalized


def normalize_language_value(raw_value: str, default_language: str) -> str:
    """Normalize a language string for UI usage."""
    normalized = raw_value.strip().lower()
    if not normalized:
        return default_language
    if normalized not in SUPPORTED_LANGUAGE_CODES:
        raise AlignmentValidationError(
            INVALID_LANGUAGE_CODE, f"unsupported language: {normalized!r}"
        )
    return normalized


def parse_torch_version(version: str) -> tuple[int, int] | None:
    """Parse the major and minor torch version."""
    match = TORCH_VERSION_PATTERN.match(version.strip())
    if match is None:
        return None
    return int(match.group("major")), int(match.group("minor"))


def ensure_torch_version(torch_module: ModuleType, language: str) -> None:
    """Ensure torch meets the minimum supported version for HF .bin weights."""
    version = str(getattr(torch_module, "__version__", "")).strip()
    parsed = parse_torch_version(version)
    if parsed is None:
        raise AlignmentPipelineError(
            TORCH_VERSION_CODE, f"torch version is invalid: {version!r}"
        )
    if parsed < TORCH_MIN_VERSION:
        raise AlignmentPipelineError(
            TORCH_VERSION_CODE,
            (
                f"torch >= {TORCH_MIN_VERSION_TEXT} is required for "
                f"{language!r} alignment models (installed: {version})"
            ),
        )


def load_torch_module() -> ModuleType:
    """Import torch."""
    try:
        import torch
    except Exception as exc:
        raise AlignmentPipelineError(
            DEVICE_UNAVAILABLE_CODE, f"torch is unavailable: {exc}"
        ) from exc
    return torch


def ensure_torchaudio_metadata() -> None:
    """Ensure torchaudio exposes AudioMetaData for pyannote imports."""
    try:
        import torchaudio
    except Exception as exc:
        raise AlignmentPipelineError(
            DEVICE_UNAVAILABLE_CODE, f"torchaudio is unavailable: {exc}"
        ) from exc
    if hasattr(torchaudio, "AudioMetaData"):
        return
    audio_metadata_type = None
    for module_name in ("torchaudio.backend.common", "torchaudio._backend.common"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        audio_metadata_type = getattr(module, "AudioMetaData", None)
        if audio_metadata_type is not None:
            break
    if audio_metadata_type is None:
        class AudioMetaData(NamedTuple):
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        audio_metadata_type = AudioMetaData
        LOGGER.warning(
            "audio_to_text.dependency.torchaudio_metadata: using fallback AudioMetaData"
        )
    setattr(torchaudio, "AudioMetaData", audio_metadata_type)


def load_whisperx_module() -> ModuleType:
    """Import whisperx after verifying torchaudio metadata."""
    ensure_torchaudio_metadata()
    try:
        import whisperx
    except Exception as exc:
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, f"whisperx import failed: {exc}"
        ) from exc
    return whisperx


def normalize_form_value(raw_value: object, default_value: str) -> str:
    """Normalize a form value into a string."""
    if raw_value is None:
        return default_value
    if isinstance(raw_value, list):
        if not raw_value:
            return default_value
        return str(raw_value[0])
    return str(raw_value)


def load_cgi_module() -> ModuleType:
    """Import cgi without emitting deprecation warnings."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="cgi"
        )
        import cgi
    return cgi


def parse_args(argv: Sequence[str]) -> AlignmentRequest:
    """Parse CLI arguments into an AlignmentRequest."""
    parser = argparse.ArgumentParser(prog="audio_to_text.py", add_help=True)
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui-host", default=DEFAULT_UI_HOST)
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--input-audio")
    parser.add_argument("--input-text")
    parser.add_argument("--output-srt", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default=DEVICE_AUTO)
    parsed = parser.parse_args(argv)

    mode = RequestMode.UI if parsed.ui else RequestMode.CLI
    input_audio = parsed.input_audio
    input_text = parsed.input_text
    output_srt = None
    if mode == RequestMode.CLI:
        if input_audio is None:
            raise AlignmentValidationError(
                INPUT_AUDIO_CODE, "input-audio is required"
            )
        if input_text is None:
            raise AlignmentValidationError(
                INPUT_TEXT_CODE, "input-text is required"
            )
        output_srt = parsed.output_srt or default_output_path(input_audio)

    language_value = str(parsed.language).strip().lower()
    device_value = str(parsed.device).strip().lower() or DEVICE_AUTO
    return AlignmentRequest(
        mode=mode,
        input_audio=input_audio,
        input_text=input_text,
        output_srt=output_srt,
        language=language_value,
        device=device_value,
        ui_host=str(parsed.ui_host),
        ui_port=int(parsed.ui_port),
    )


def load_alignment_model(
    language: str, device: str
) -> tuple[object, dict[str, object]]:
    """Load the alignment model and metadata."""
    whisperx = load_whisperx_module()
    model_name = ALIGNMENT_MODEL_OVERRIDES.get(language)
    if (
        language not in TORCHAUDIO_ALIGNMENT_LANGUAGES
        and model_name is None
    ):
        torch_module = load_torch_module()
        ensure_torch_version(torch_module, language)
    try:
        return whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=model_name,
        )
    except Exception as exc:
        raise AlignmentPipelineError(
            ALIGN_MODEL_CODE, f"align model load failed: {exc}"
        ) from exc


def resolve_device(device_value: str) -> str:
    """Resolve auto device selection to a concrete device."""
    torch_module = load_torch_module()
    if device_value != DEVICE_AUTO:
        return device_value
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def align_words(
    audio_path: str,
    transcript_text: str,
    language: str,
    device: str,
) -> tuple[AlignedWord, ...]:
    """Align transcript text to the audio and return word timings."""
    whisperx = load_whisperx_module()
    audio = whisperx.load_audio(audio_path)
    audio_duration = float(len(audio)) / float(whisperx.audio.SAMPLE_RATE)
    segments = [{"start": 0.0, "end": audio_duration, "text": transcript_text}]

    align_model, metadata = load_alignment_model(language, device)
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


def build_ui_html(defaults: UiDefaults) -> str:
    """Render the UI HTML with default values."""
    device_options = []
    for device_key, label in DEVICE_LABELS.items():
        selected = " selected" if device_key == defaults.device else ""
        device_options.append(
            f'<option value="{escape(device_key)}"{selected}>{escape(label)}</option>'
        )
    language_options = []
    for language_code, language_label in SUPPORTED_ALIGNMENT_LANGUAGES:
        selected = " selected" if language_code == defaults.language else ""
        label_text = f"{language_label} ({language_code})"
        language_options.append(
            f'<option value="{escape(language_code)}"{selected}>{escape(label_text)}</option>'
        )
    template = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Audio to Text Alignment</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Fraunces:wght@600&display=swap");
    :root {
      --bg-start: #f7efe4;
      --bg-end: #dfe8e9;
      --ink: #1d1b19;
      --muted: #5f5b55;
      --accent: #ff7a59;
      --accent-strong: #f1542d;
      --accent-cool: #2b7a78;
      --panel: rgba(255, 255, 255, 0.85);
      --shadow: 0 24px 60px rgba(29, 27, 25, 0.14);
      --stroke: rgba(29, 27, 25, 0.1);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Space Grotesk", "Helvetica Neue", "Segoe UI", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #ffffff 0%, var(--bg-start) 35%, var(--bg-end) 100%);
      overflow-x: hidden;
    }
    .ambient {
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        radial-gradient(circle at 15% 20%, rgba(255, 122, 89, 0.25), transparent 45%),
        radial-gradient(circle at 80% 10%, rgba(43, 122, 120, 0.2), transparent 50%),
        radial-gradient(circle at 20% 80%, rgba(250, 205, 140, 0.3), transparent 55%);
      mix-blend-mode: multiply;
      z-index: 0;
    }
    .shell {
      position: relative;
      z-index: 1;
      max-width: 980px;
      margin: 0 auto;
      padding: 56px 24px 80px;
      display: grid;
      gap: 36px;
    }
    .hero {
      display: grid;
      gap: 16px;
      animation: floatIn 0.8s ease-out;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(29, 27, 25, 0.08);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero h1 {
      font-family: "Fraunces", "Georgia", serif;
      font-size: clamp(2rem, 4vw, 3.2rem);
      margin: 0;
      line-height: 1.1;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
      max-width: 560px;
    }
    .panel {
      background: var(--panel);
      border-radius: 24px;
      padding: 28px;
      border: 1px solid var(--stroke);
      box-shadow: var(--shadow);
      display: grid;
      gap: 22px;
      animation: floatIn 1s ease-out;
    }
    .drop-grid {
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .dropzone {
      border: 2px dashed rgba(29, 27, 25, 0.2);
      border-radius: 18px;
      padding: 24px;
      min-height: 160px;
      background: rgba(255, 255, 255, 0.6);
      display: grid;
      gap: 8px;
      justify-items: start;
      align-content: center;
      transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
      cursor: pointer;
      position: relative;
    }
    .dropzone input[type="file"] {
      display: none;
    }
    .dropzone.is-dragging {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 12px 24px rgba(255, 122, 89, 0.2);
    }
    .dropzone.is-filled {
      border-color: rgba(43, 122, 120, 0.6);
      background: rgba(43, 122, 120, 0.08);
    }
    .drop-title {
      font-weight: 600;
      font-size: 1rem;
    }
    .drop-meta {
      color: var(--muted);
      font-size: 0.9rem;
    }
    .options {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .option {
      display: grid;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .option input,
    .option select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      font-family: "Space Grotesk", "Helvetica Neue", "Segoe UI", sans-serif;
      font-size: 0.95rem;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
    }
    .actions {
      display: grid;
      gap: 12px;
    }
    .run-button {
      border: none;
      border-radius: 14px;
      padding: 14px 18px;
      font-size: 1rem;
      font-weight: 600;
      color: #fff;
      background: linear-gradient(120deg, var(--accent), var(--accent-strong));
      box-shadow: 0 14px 30px rgba(241, 84, 45, 0.25);
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .run-button:hover {
      transform: translateY(-1px);
      box-shadow: 0 18px 34px rgba(241, 84, 45, 0.3);
    }
    .run-button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }
    .progress {
      height: 6px;
      border-radius: 999px;
      background: rgba(29, 27, 25, 0.08);
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, rgba(43, 122, 120, 0.2), rgba(43, 122, 120, 0.7), rgba(43, 122, 120, 0.2));
      background-size: 200% 100%;
      transition: width 0.3s ease;
      opacity: 0.6;
    }
    body.running .progress-bar {
      animation: shimmer 1.6s linear infinite;
    }
    .status {
      display: grid;
      gap: 6px;
      font-size: 0.95rem;
    }
    .status-main {
      font-weight: 600;
    }
    .status-sub {
      color: var(--muted);
    }
    .download {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--accent-cool);
      text-decoration: none;
      font-weight: 600;
      margin-top: 8px;
    }
    .download:hover {
      color: #206362;
    }
    .error {
      padding: 12px 14px;
      background: rgba(241, 84, 45, 0.12);
      border: 1px solid rgba(241, 84, 45, 0.25);
      border-radius: 12px;
      color: #8a2a16;
      font-size: 0.9rem;
    }
    .hidden {
      display: none;
    }
    @keyframes shimmer {
      0% { background-position: 0% 0; }
      100% { background-position: 200% 0; }
    }
    @keyframes floatIn {
      0% { opacity: 0; transform: translateY(12px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 640px) {
      .panel {
        padding: 22px;
      }
      .run-button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="ambient"></div>
  <main class="shell">
    <section class="hero">
      <div class="badge">Audio to Text</div>
      <h1>Forced alignment, ready for subtitles</h1>
      <p>Drop audio or video plus your transcript. A background job aligns each word and builds an SRT you can download.</p>
    </section>
    <section class="panel">
      <div class="drop-grid">
        <div class="dropzone" id="audio-zone">
          <input id="audio-file" type="file" accept="audio/*,video/*">
          <div class="drop-title">Audio or video file</div>
          <div class="drop-meta" id="audio-meta">Drop a file or click to browse</div>
        </div>
        <div class="dropzone" id="text-zone">
          <input id="text-file" type="file" accept=".txt,.md,.srt,.sbv">
          <div class="drop-title">Transcript text file</div>
          <div class="drop-meta" id="text-meta">Drop a file or click to browse</div>
        </div>
      </div>
      <div class="options">
        <label class="option">
          <span>Language</span>
          <select id="language">
            $language_options
          </select>
        </label>
        <label class="option">
          <span>Device</span>
          <select id="device">
            $device_options
          </select>
        </label>
      </div>
      <div class="actions">
        <button id="run-button" class="run-button">Align and Build SRT</button>
        <div class="progress">
          <div class="progress-bar"></div>
        </div>
      </div>
      <div class="status">
        <div class="status-main" id="status-line">Ready to align.</div>
        <div class="status-sub" id="status-sub">Upload files to begin.</div>
        <a id="download-link" class="download hidden" href="#">Download SRT</a>
      </div>
      <div class="error hidden" id="error-line"></div>
    </section>
  </main>
  <script>
    const audioZone = document.getElementById("audio-zone");
    const textZone = document.getElementById("text-zone");
    const audioInput = document.getElementById("audio-file");
    const textInput = document.getElementById("text-file");
    const audioMeta = document.getElementById("audio-meta");
    const textMeta = document.getElementById("text-meta");
    const runButton = document.getElementById("run-button");
    const statusLine = document.getElementById("status-line");
    const statusSub = document.getElementById("status-sub");
    const downloadLink = document.getElementById("download-link");
    const errorLine = document.getElementById("error-line");
    const languageInput = document.getElementById("language");
    const deviceSelect = document.getElementById("device");
    const progressBar = document.querySelector(".progress-bar");
    let audioFile = null;
    let textFile = null;
    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    function setStatus(mainText, subText) {
      statusLine.textContent = mainText;
      statusSub.textContent = subText;
    }

    function setError(message) {
      errorLine.textContent = message;
      errorLine.classList.remove("hidden");
    }

    function clearError() {
      errorLine.classList.add("hidden");
      errorLine.textContent = "";
    }

    function setProgress(value) {
      const clamped = Math.max(0, Math.min(1, value));
      progressBar.style.width = Math.round(clamped * 100) + "%";
    }

    function markZone(zone, meta, file) {
      zone.classList.add("is-filled");
      meta.textContent = file.name;
    }

    function wireDropzone(zone, input, meta, assignFile) {
      zone.addEventListener("click", () => input.click());
      zone.addEventListener("dragover", (event) => {
        event.preventDefault();
        zone.classList.add("is-dragging");
      });
      zone.addEventListener("dragleave", () => zone.classList.remove("is-dragging"));
      zone.addEventListener("drop", (event) => {
        event.preventDefault();
        zone.classList.remove("is-dragging");
        if (event.dataTransfer.files.length === 0) {
          return;
        }
        const file = event.dataTransfer.files[0];
        assignFile(file);
        markZone(zone, meta, file);
      });
      input.addEventListener("change", () => {
        if (!input.files || input.files.length === 0) {
          return;
        }
        const file = input.files[0];
        assignFile(file);
        markZone(zone, meta, file);
      });
    }

    wireDropzone(audioZone, audioInput, audioMeta, (file) => { audioFile = file; });
    wireDropzone(textZone, textInput, textMeta, (file) => { textFile = file; });

    async function startJob() {
      clearError();
      downloadLink.classList.add("hidden");
      setProgress(0);
      if (!audioFile || !textFile) {
        setError("Select both an audio file and a transcript file.");
        return;
      }
      runButton.disabled = true;
      document.body.classList.add("running");
      setStatus("Queued.", "Uploading files and preparing alignment.");
      const formData = new FormData();
      formData.append("audio", audioFile, audioFile.name);
      formData.append("text", textFile, textFile.name);
      formData.append("language", languageInput.value.trim() || "en");
      formData.append("device", deviceSelect.value);
      let response = await fetch("/api/jobs", { method: "POST", body: formData });
      let payload = await response.json();
      if (!response.ok) {
        setError(payload.error || "Failed to start alignment.");
        runButton.disabled = false;
        document.body.classList.remove("running");
        return;
      }
      await pollStatus(payload.job_id);
    }

    async function pollStatus(jobId) {
      while (true) {
        let response = await fetch(`/api/jobs/$${jobId}`);
        if (!response.ok) {
          setError("Failed to check job status.");
          break;
        }
        let payload = await response.json();
        const progressValue = typeof payload.progress === "number" ? payload.progress : 0;
        setProgress(progressValue);
        if (payload.status === "completed") {
          setStatus("Complete.", payload.message || "SRT is ready to download.");
          downloadLink.href = `/api/jobs/$${jobId}/srt`;
          downloadLink.classList.remove("hidden");
          break;
        }
        if (payload.status === "failed") {
          setError(payload.message || "Alignment failed.");
          break;
        }
        if (payload.status === "queued") {
          setStatus("Queued.", payload.message || "Waiting for worker.");
        } else {
          setStatus("Running.", payload.message || "Aligning words to audio.");
        }
        await delay(1000);
      }
      runButton.disabled = false;
      document.body.classList.remove("running");
    }

    runButton.addEventListener("click", () => startJob());
  </script>
</body>
</html>
"""
    )
    return template.substitute(
        language_options="\n            ".join(language_options),
        device_options="\n            ".join(device_options),
    )


def run_ui_server(request: AlignmentRequest) -> int:
    """Run the web UI server."""
    defaults = UiDefaults(
        language=request.language,
        device=request.device,
    )
    root_dir = Path(tempfile.mkdtemp(prefix="audio_to_text_ui_"))
    job_store = JobStore(root_dir=root_dir)
    executor = ThreadPoolExecutor(max_workers=2)
    handler = build_ui_handler(job_store, executor, defaults)
    server = ThreadingHTTPServer((request.ui_host, request.ui_port), handler)
    LOGGER.info(
        "audio_to_text.ui.ready: http://%s:%s",
        request.ui_host,
        request.ui_port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("audio_to_text.ui.shutdown: received interrupt")
    finally:
        server.server_close()
        executor.shutdown(wait=False)
        shutil.rmtree(root_dir, ignore_errors=True)
    return 0


def build_ui_handler(
    store: JobStore,
    executor: ThreadPoolExecutor,
    defaults: UiDefaults,
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler for the UI server."""

    class UiHandler(BaseHTTPRequestHandler):
        """HTTP handler for the audio_to_text UI."""

        def log_message(self, format_string: str, *args: object) -> None:
            """Route HTTP logs through the logger."""
            LOGGER.info("audio_to_text.ui: %s", format_string % args)

        def do_GET(self) -> None:
            """Serve UI pages and job status."""
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_html(build_ui_html(defaults))
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[3] if parsed.path.count("/") >= 3 else ""
                if parsed.path.endswith("/srt"):
                    self.send_srt(job_id)
                    return
                self.send_job_status(job_id)
                return
            self.send_error_response(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            """Handle UI job creation."""
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                self.handle_create_job()
                return
            self.send_error_response(HTTPStatus.NOT_FOUND, "Not found")

        def send_html(self, content: str) -> None:
            """Send HTML content."""
            payload = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            """Send a JSON payload."""
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_error_response(self, status: HTTPStatus, message: str) -> None:
            """Send an error JSON payload."""
            self.send_json(status, {"error": message})

        def send_job_status(self, job_id: str) -> None:
            """Return the current job status."""
            job = store.get_job(job_id)
            if job is None:
                self.send_error_response(HTTPStatus.NOT_FOUND, "Job not found")
                return
            self.send_json(
                HTTPStatus.OK,
                {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "message": job.message,
                    "output_ready": bool(job.output_srt),
                    "progress": job.progress,
                },
            )

        def send_srt(self, job_id: str) -> None:
            """Return the generated SRT file."""
            job = store.get_job(job_id)
            if job is None or job.output_srt is None:
                self.send_error_response(HTTPStatus.NOT_FOUND, "SRT not available")
                return
            output_path = Path(job.output_srt)
            try:
                content = output_path.read_text(encoding="utf-8")
            except OSError:
                self.send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "SRT read failed")
                return
            payload = content.encode("utf-8")
            filename = f"{job_id}.srt"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-subrip; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def handle_create_job(self) -> None:
            """Accept uploads and queue a background alignment job."""
            cgi = load_cgi_module()

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_error_response(HTTPStatus.BAD_REQUEST, "Expected multipart form")
                return
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            }
            content_length = self.headers.get("Content-Length")
            if content_length:
                environ["CONTENT_LENGTH"] = content_length
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ=environ,
            )
            try:
                audio_field = form["audio"]
                text_field = form["text"]
            except KeyError:
                self.send_error_response(HTTPStatus.BAD_REQUEST, "Missing file fields")
                return
            if isinstance(audio_field, list) or isinstance(text_field, list):
                self.send_error_response(HTTPStatus.BAD_REQUEST, "Invalid file upload")
                return
            if not getattr(audio_field, "filename", "") or not getattr(text_field, "filename", ""):
                self.send_error_response(HTTPStatus.BAD_REQUEST, "Invalid file upload")
                return

            job = store.create_job()
            job_dir = store.job_dir(job.job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            audio_suffix = Path(audio_field.filename).suffix or ".bin"
            text_suffix = Path(text_field.filename).suffix or ".txt"
            audio_path = job_dir / f"audio{audio_suffix}"
            text_path = job_dir / f"text{text_suffix}"
            output_path = job_dir / "alignment.srt"

            try:
                with open(audio_path, "wb") as audio_target:
                    shutil.copyfileobj(audio_field.file, audio_target)
                with open(text_path, "wb") as text_target:
                    shutil.copyfileobj(text_field.file, text_target)
            except OSError:
                store.update_job(job.job_id, JobStatus.FAILED, "Upload write failed")
                self.send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "Upload failed")
                return

            language_raw = normalize_form_value(
                form.getvalue("language", defaults.language), defaults.language
            )
            try:
                language_value = normalize_language_value(
                    language_raw, defaults.language
                )
                device_value = normalize_device_value(
                    normalize_form_value(
                        form.getvalue("device", defaults.device), defaults.device
                    ),
                    defaults.device,
                )
            except AlignmentValidationError as exc:
                store.update_job(job.job_id, JobStatus.FAILED, f"{exc.code}: {exc}")
                self.send_error_response(HTTPStatus.BAD_REQUEST, str(exc))
                return

            executor.submit(
                run_alignment_job,
                store,
                job.job_id,
                str(audio_path),
                str(text_path),
                str(output_path),
                language_value,
                device_value,
            )
            self.send_json(HTTPStatus.OK, {"job_id": job.job_id})

    return UiHandler


def run_alignment_job(
    store: JobStore,
    job_id: str,
    audio_path: str,
    text_path: str,
    output_path: str,
    language: str,
    device: str,
) -> None:
    """Process a background alignment job."""
    store.update_job(
        job_id,
        JobStatus.RUNNING,
        message="Preparing input files",
        progress=0.05,
    )
    try:
        ensure_audio_file_exists(audio_path)
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Reading transcript text",
            progress=0.15,
        )
        transcript_text = normalize_transcript(
            read_utf8_text_strict(text_path),
            text_path,
        )
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Resolving device",
            progress=0.3,
        )
        resolved_device = resolve_device(device)
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Aligning words to audio",
            progress=0.45,
        )
        words = align_words(
            audio_path,
            transcript_text,
            language,
            resolved_device,
        )
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Building subtitle output",
            progress=0.85,
        )
        srt_content = build_srt(words)
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Writing subtitle file",
            progress=0.95,
        )
        write_srt_file(output_path, srt_content)
        store.update_job(
            job_id,
            JobStatus.COMPLETED,
            message="Complete",
            output_srt=output_path,
            progress=1.0,
        )
    except AlignmentValidationError as exc:
        store.update_job(
            job_id,
            JobStatus.FAILED,
            f"{exc.code}: {exc}",
            progress=1.0,
        )
    except AlignmentPipelineError as exc:
        store.update_job(
            job_id,
            JobStatus.FAILED,
            f"{exc.code}: {exc}",
            progress=1.0,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            JobStatus.FAILED,
            f"audio_to_text.unhandled_error: {str(exc).strip()}",
            progress=1.0,
        )


def main() -> int:
    """CLI entrypoint."""
    configure_logging()
    try:
        request = parse_args(sys.argv[1:])
        if request.mode == RequestMode.UI:
            return run_ui_server(request)
        ensure_audio_file_exists(request.input_audio or "")
        transcript_text = normalize_transcript(
            read_utf8_text_strict(request.input_text or ""),
            request.input_text or "",
        )
        resolved_device = resolve_device(request.device)
        words = align_words(
            request.input_audio or "",
            transcript_text,
            request.language,
            resolved_device,
        )
        srt_content = build_srt(words)
        write_srt_file(request.output_srt or "", srt_content)
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
