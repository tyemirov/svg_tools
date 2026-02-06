#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "whisperx==3.3.0",
#   "matplotlib<4",
#   "numpy<2",
#   "safetensors",
#   "torch>=2.6,<2.7; platform_system == 'Linux'",
#   "torchaudio>=2.6,<2.7; platform_system == 'Linux'",
# ]
# ///
"""Force-align audio or video to input text and emit SRT."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import math
import numbers
import os
import platform
import re
import shutil
import sys
import threading
import warnings
import unicodedata
from dataclasses import dataclass, field
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Callable, Iterable, Sequence, cast
from urllib.parse import quote


LOGGER = logging.getLogger("audio_to_text")

INPUT_AUDIO_CODE = "audio_to_text.input.audio_file"
INPUT_TEXT_CODE = "audio_to_text.input.text_file"
INPUT_ALIGNMENT_JSON_CODE = "audio_to_text.input.alignment_json_file"
OUTPUT_SRT_CODE = "audio_to_text.output.srt_file"
INVALID_CONFIG_CODE = "audio_to_text.input.invalid_config"
INVALID_LANGUAGE_CODE = "audio_to_text.input.invalid_language"
ALIGN_MODEL_CODE = "audio_to_text.align.model"
ALIGNMENT_CODE = "audio_to_text.align.failed"
ALIGNMENT_TIMESTAMP_CODE = "audio_to_text.align.missing_timestamps"
ALIGNMENT_INFERRED_TIMESTAMPS_CODE = "audio_to_text.align.inferred_timestamps"
DEVICE_UNAVAILABLE_CODE = "audio_to_text.device.unavailable"
TORCH_VERSION_CODE = "audio_to_text.dependency.torch_version"
TORCHAUDIO_METADATA_CODE = "audio_to_text.dependency.torchaudio_metadata"
PLATFORM_CODE = "audio_to_text.dependency.platform"
UI_STORAGE_CODE = "audio_to_text.ui.storage"
UI_UPLOAD_CODE = "audio_to_text.ui.upload.invalid"
UI_UPLOAD_BODY_CODE = "audio_to_text.ui.upload.body"
INVALID_JOB_INPUT_CODE = "audio_to_text.job.invalid_input"
INVALID_JOB_RESULT_CODE = "audio_to_text.job.invalid_result"
INVALID_PROGRESS_CODE = "audio_to_text.job.invalid_progress"
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
TORCH_MIN_VERSION = (2, 6)
TORCH_MIN_VERSION_TEXT = "2.6"
TORCHAUDIO_ALIGNMENT_LANGUAGES = {"en", "fr", "de", "es", "it"}
DEFAULT_MISSING_TOKEN_SECONDS = 0.25
PLATFORM_OVERRIDE_ENV = "AUDIO_TO_TEXT_PLATFORM_OVERRIDE"
SRT_TIME_RANGE_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$"
)
TORCH_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)")


class JobStatus(str, Enum):
    """Lifecycle states for alignment jobs."""

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

    input_audio: str | None
    input_text: str | None
    input_alignment_json: str | None
    output_srt: str | None
    language: str

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "language must be non-empty"
            )
        if self.language not in SUPPORTED_LANGUAGE_CODES:
            raise AlignmentValidationError(
                INVALID_LANGUAGE_CODE, f"unsupported language: {self.language!r}"
            )
        if self.input_alignment_json is None:
            if self.input_audio is None or not self.input_audio.strip():
                raise AlignmentValidationError(
                    INPUT_AUDIO_CODE, "input-audio is required"
                )
            if self.input_text is None or not self.input_text.strip():
                raise AlignmentValidationError(
                    INPUT_TEXT_CODE, "input-text is required"
                )
        else:
            if not self.input_alignment_json.strip():
                raise AlignmentValidationError(
                    INPUT_ALIGNMENT_JSON_CODE,
                    "input-alignment-json must be non-empty",
                )
            if self.input_audio is not None or self.input_text is not None:
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE,
                    "input-alignment-json cannot be combined with input-audio/input-text",
                )

        if self.output_srt is None or not self.output_srt.strip():
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE, "output-srt must be non-empty"
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


@dataclass(frozen=True)
class UiDefaults:
    """Default configuration values for the UI."""

    language: str
    remove_punctuation: bool


@dataclass(frozen=True)
class UploadFile:
    """Uploaded file payload and filename."""

    filename: str
    payload: bytes

@dataclass(frozen=True)
class UploadForm:
    """Parsed upload form payload."""

    audio: UploadFile
    text: UploadFile
    language: str
    remove_punctuation: bool
    client_job_id: str | None

@dataclass(frozen=True)
class AlignmentJobInput:
    """Captured inputs for a background alignment job."""

    audio_filename: str
    text_filename: str
    language: str
    remove_punctuation: bool
    audio_path: str
    text_path: str
    output_path: str
    client_job_id: str | None

    def __post_init__(self) -> None:
        if not self.audio_filename.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "audio filename is required"
            )
        if not self.text_filename.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "text filename is required"
            )
        if not self.language.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "language is required"
            )
        if not self.audio_path.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "audio path is required"
            )
        if not self.text_path.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "text path is required"
            )
        if not self.output_path.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "output path is required"
            )
        if self.client_job_id is not None and not self.client_job_id.strip():
            raise AlignmentPipelineError(
                INVALID_JOB_INPUT_CODE, "client job id must be non-empty"
            )


@dataclass(frozen=True)
class AlignmentJobResult:
    """Outcome and progress for an alignment job."""

    status: JobStatus
    message: str | None
    output_srt: str | None
    progress: float
    started_at: float | None
    completed_at: float | None

    def __post_init__(self) -> None:
        if self.progress < 0.0 or self.progress > 1.0:
            raise AlignmentPipelineError(
                INVALID_PROGRESS_CODE,
                f"progress must be between 0 and 1: {self.progress}",
            )
        if self.started_at is not None and self.started_at < 0:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "started_at must be non-negative"
            )
        if self.completed_at is not None and self.completed_at < 0:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "completed_at must be non-negative"
            )
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE,
                "completed_at must be after started_at",
            )
        if self.status == JobStatus.QUEUED:
            if self.started_at is not None or self.completed_at is not None:
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE,
                    "queued jobs cannot have timestamps",
                )
        if self.status == JobStatus.RUNNING:
            if self.started_at is None or self.completed_at is not None:
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE,
                    "running jobs must have started_at and no completed_at",
                )
        if self.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            if self.started_at is None or self.completed_at is None:
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE,
                    "completed jobs must have start and completion times",
                )
        if self.output_srt is not None and self.status != JobStatus.COMPLETED:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE,
                "output path is only valid for completed jobs",
            )
        if self.status == JobStatus.COMPLETED and self.output_srt is None:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE,
                "completed jobs must include output path",
            )


@dataclass(frozen=True)
class AlignmentJob:
    """State snapshot for a background alignment job."""

    job_id: str
    created_at: float
    job_input: AlignmentJobInput
    result: AlignmentJobResult

    def __post_init__(self) -> None:
        if not self.job_id:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "job id is required"
            )
        if self.created_at < 0:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE,
                "job created_at must be non-negative",
            )


def parse_job_string(value: object, label: str) -> str:
    """Parse a required string value."""
    if not isinstance(value, str):
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"{label} must be a string"
        )
    return value


def parse_job_optional_string(value: object, label: str) -> str | None:
    """Parse an optional string value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"{label} must be a string"
        )
    return value


def parse_job_float(value: object, label: str) -> float:
    """Parse a required numeric value."""
    if not isinstance(value, (int, float)):
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"{label} must be a number"
        )
    return float(value)


def parse_job_optional_float(value: object, label: str) -> float | None:
    """Parse an optional numeric value."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"{label} must be a number"
        )
    return float(value)

def parse_job_bool(value: object, label: str) -> bool:
    """Parse a required boolean value."""
    if not isinstance(value, bool):
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"{label} must be a boolean"
        )
    return value


def parse_job_status(value: object) -> JobStatus:
    """Parse a job status value."""
    status_text = parse_job_string(value, "status")
    try:
        return JobStatus(status_text)
    except ValueError as exc:
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"status is invalid: {status_text}"
        ) from exc


def parse_job_dict(value: object, label: str) -> dict[str, object]:
    """Parse a dictionary value."""
    if not isinstance(value, dict):
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE, f"{label} must be a dictionary"
        )
    return value


def parse_alignment_job(job_id: str, payload: object) -> AlignmentJob:
    """Parse a persisted job payload."""
    data = parse_job_dict(payload, "job")
    raw_job_id = data.get("job_id")
    if raw_job_id is not None and raw_job_id != job_id:
        raise AlignmentPipelineError(
            INVALID_JOB_RESULT_CODE,
            f"job id mismatch: {job_id} != {raw_job_id}",
        )
    created_at = parse_job_float(data.get("created_at"), "created_at")
    input_payload = parse_job_dict(data.get("input"), "input")
    result_payload = parse_job_dict(data.get("result"), "result")
    remove_punctuation = parse_job_bool(
        input_payload.get("remove_punctuation"), "remove_punctuation"
    )
    client_job_id = parse_job_optional_string(
        input_payload.get("client_job_id"), "client_job_id"
    )
    job_input = AlignmentJobInput(
        audio_filename=parse_job_string(
            input_payload.get("audio_filename"), "audio_filename"
        ),
        text_filename=parse_job_string(
            input_payload.get("text_filename"), "text_filename"
        ),
        language=parse_job_string(input_payload.get("language"), "language"),
        remove_punctuation=remove_punctuation,
        audio_path=parse_job_string(input_payload.get("audio_path"), "audio_path"),
        text_path=parse_job_string(input_payload.get("text_path"), "text_path"),
        output_path=parse_job_string(
            input_payload.get("output_path"), "output_path"
        ),
        client_job_id=client_job_id,
    )
    job_result = AlignmentJobResult(
        status=parse_job_status(result_payload.get("status")),
        message=parse_job_optional_string(
            result_payload.get("message"), "message"
        ),
        output_srt=parse_job_optional_string(
            result_payload.get("output_srt"), "output_srt"
        ),
        progress=parse_job_float(result_payload.get("progress"), "progress"),
        started_at=parse_job_optional_float(
            result_payload.get("started_at"), "started_at"
        ),
        completed_at=parse_job_optional_float(
            result_payload.get("completed_at"), "completed_at"
        ),
    )
    return AlignmentJob(job_id, created_at, job_input, job_result)


def serialize_alignment_job(job: AlignmentJob) -> dict[str, object]:
    """Serialize a job for persistence."""
    return {
        "job_id": job.job_id,
        "created_at": job.created_at,
        "input": {
            "audio_filename": job.job_input.audio_filename,
            "text_filename": job.job_input.text_filename,
            "language": job.job_input.language,
            "remove_punctuation": job.job_input.remove_punctuation,
            "audio_path": job.job_input.audio_path,
            "text_path": job.job_input.text_path,
            "output_path": job.job_input.output_path,
            "client_job_id": job.job_input.client_job_id,
        },
        "result": {
            "status": job.result.status.value,
            "message": job.result.message,
            "output_srt": job.result.output_srt,
            "progress": job.result.progress,
            "started_at": job.result.started_at,
            "completed_at": job.result.completed_at,
        },
    }


@dataclass
class JobStore:
    """Thread-safe store for background UI jobs."""

    root_dir: Path
    clock: Callable[[], float]
    id_factory: Callable[[], str]
    state_path: Path = field(init=False)
    jobs: dict[str, AlignmentJob] = field(default_factory=dict)
    job_order: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)
    change_id: int = 0

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)
        self.state_path = self.root_dir / "jobs.json"
        self.load_state()

    def new_job_id(self) -> str:
        """Generate a new job id."""
        return self.id_factory()

    def load_state(self) -> None:
        """Load persisted job state."""
        if not self.state_path.exists():
            return
        try:
            raw_state = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise AlignmentPipelineError(
                UI_STORAGE_CODE, f"job store load failed: {exc}"
            ) from exc
        state = parse_job_dict(raw_state, "job store")
        raw_order = state.get("job_order", [])
        raw_jobs = state.get("jobs", {})
        raw_change_id = state.get("change_id", 0)
        if not isinstance(raw_order, list):
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "job_order must be a list"
            )
        if not isinstance(raw_jobs, dict):
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "jobs must be a dictionary"
            )
        if not isinstance(raw_change_id, int):
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "change_id must be an integer"
            )
        if raw_change_id < 0:
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "change_id must be non-negative"
            )
        jobs: dict[str, AlignmentJob] = {}
        job_order: list[str] = []
        for job_id in raw_order:
            if not isinstance(job_id, str):
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE, "job ids must be strings"
                )
            if job_id not in raw_jobs:
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE,
                    f"job missing from store: {job_id}",
                )
            job = parse_alignment_job(job_id, raw_jobs[job_id])
            jobs[job_id] = job
            job_order.append(job_id)
        if set(raw_jobs.keys()) != set(job_order):
            raise AlignmentPipelineError(
                INVALID_JOB_RESULT_CODE, "job store entries are inconsistent"
            )
        with self.condition:
            self.jobs = jobs
            self.job_order = job_order
            self.change_id = raw_change_id

    def build_state_payload(self) -> dict[str, object]:
        """Build the job state payload."""
        return {
            "change_id": self.change_id,
            "job_order": list(self.job_order),
            "jobs": {
                job_id: serialize_alignment_job(job)
                for job_id, job in self.jobs.items()
            },
        }

    def save_state(self, payload: dict[str, object]) -> None:
        """Persist job state to disk."""
        temp_path = self.state_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(payload), encoding="utf-8")
            temp_path.replace(self.state_path)
        except OSError as exc:
            raise AlignmentPipelineError(
                UI_STORAGE_CODE, f"job store write failed: {exc}"
            ) from exc

    def create_job(
        self,
        job_id: str,
        job_input: AlignmentJobInput,
    ) -> AlignmentJob:
        """Create a new queued job."""
        initial_result = AlignmentJobResult(
            status=JobStatus.QUEUED,
            message="Queued",
            output_srt=None,
            progress=0.0,
            started_at=None,
            completed_at=None,
        )
        created_at = self.clock()
        job = AlignmentJob(job_id, created_at, job_input, initial_result)
        with self.condition:
            if job_id in self.jobs:
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE,
                    f"job already exists: {job_id}",
                )
            self.jobs[job_id] = job
            self.job_order.append(job_id)
            self.change_id += 1
            payload = self.build_state_payload()
            self.save_state(payload)
            self.condition.notify_all()
        return job

    def get_job(self, job_id: str) -> AlignmentJob | None:
        """Fetch a job by ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[AlignmentJob]:
        """Return jobs in creation order."""
        with self.lock:
            return [self.jobs[job_id] for job_id in self.job_order]

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        message: str,
        progress: float,
        output_srt: str | None = None,
    ) -> AlignmentJob:
        """Update a job's status."""
        with self.condition:
            current = self.jobs[job_id]
            output_value = (
                output_srt if output_srt is not None else current.result.output_srt
            )
            started_at = current.result.started_at
            completed_at = current.result.completed_at
            if status == JobStatus.RUNNING and started_at is None:
                started_at = self.clock()
                completed_at = None
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if started_at is None:
                    started_at = self.clock()
                completed_at = self.clock()
            result = AlignmentJobResult(
                status=status,
                message=message,
                output_srt=output_value,
                progress=progress,
                started_at=started_at,
                completed_at=completed_at,
            )
            job = AlignmentJob(job_id, current.created_at, current.job_input, result)
            self.jobs[job_id] = job
            self.change_id += 1
            payload = self.build_state_payload()
            self.save_state(payload)
            self.condition.notify_all()
        return job

    def wait_for_change(self, last_change_id: int, timeout: float) -> int:
        """Wait for any job change and return the latest change id."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.change_id != last_change_id, timeout=timeout
            )
            return self.change_id

    def job_dir(self, job_id: str) -> Path:
        """Return the directory for job artifacts."""
        return self.root_dir / job_id

    def delete_finished_job(self, job_id: str) -> AlignmentJob:
        """Delete a finished job and its artifacts."""
        with self.condition:
            job = self.jobs.get(job_id)
            if job is None:
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE, f"job not found: {job_id}"
                )
            if job.result.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                raise AlignmentPipelineError(
                    INVALID_JOB_RESULT_CODE,
                    "only finished jobs can be deleted",
                )
            job_dir = self.job_dir(job_id)

        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
            except OSError as exc:
                raise AlignmentPipelineError(
                    UI_STORAGE_CODE,
                    f"job artifacts delete failed: {exc}",
                ) from exc

        with self.condition:
            current = self.jobs.pop(job_id, None) or job
            self.job_order = [value for value in self.job_order if value != job_id]
            self.change_id += 1
            payload = self.build_state_payload()
            self.save_state(payload)
            self.condition.notify_all()
        return current


def configure_logging() -> None:
    """Configure logging for CLI output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def ensure_linux_runtime() -> None:
    """Ensure the tool is running on Linux."""
    override = os.environ.get(PLATFORM_OVERRIDE_ENV, "").strip().lower()
    platform_value = override or platform.system().lower()
    if platform_value != "linux":
        raise AlignmentPipelineError(
            PLATFORM_CODE, "audio_to_text requires Linux; run via Docker"
        )


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
    without_bom = sanitized.replace("\ufeff", "")
    normalized = " ".join(without_bom.split())
    if not normalized:
        raise AlignmentValidationError(
            INVALID_CONFIG_CODE, "input text contains no words"
        )
    return normalized


def normalize_transcript_for_alignment(
    text_value: str,
    input_text_path: str,
    remove_punctuation: bool,
) -> str:
    """Normalize transcript text for alignment work."""
    normalized = normalize_transcript(text_value, input_text_path)
    if not remove_punctuation:
        return normalized
    stripped = " ".join(remove_punctuation_from_transcript(normalized).split())
    if not stripped:
        raise AlignmentValidationError(
            INVALID_CONFIG_CODE,
            "input text contains no words after punctuation removal",
        )
    return stripped

def remove_punctuation_from_transcript(text_value: str) -> str:
    """Replace punctuation characters with spaces."""
    replaced: list[str] = []
    for character in text_value:
        if unicodedata.category(character).startswith("P"):
            replaced.append(" ")
        else:
            replaced.append(character)
    return "".join(replaced)


def default_output_path(input_audio: str) -> str:
    """Derive the default SRT output path from the audio path."""
    return str(Path(input_audio).with_suffix(".srt"))

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
        raise AlignmentPipelineError(
            TORCHAUDIO_METADATA_CODE,
            "torchaudio is missing AudioMetaData; install torchaudio>=2.6",
        )
    setattr(torchaudio, "AudioMetaData", audio_metadata_type)


def load_whisperx_alignment_modules() -> tuple[ModuleType, ModuleType]:
    """Import whisperx alignment modules without transcribe."""
    ensure_torchaudio_metadata()
    package_spec = importlib.util.find_spec("whisperx")
    if (
        package_spec is None
        or package_spec.submodule_search_locations is None
    ):
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, "whisperx is unavailable"
        )
    try:
        alignment_module = importlib.import_module("whisperx.alignment")
        audio_module = importlib.import_module("whisperx.audio")
    except Exception as exc:
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, f"whisperx alignment import failed: {exc}"
        ) from exc
    return alignment_module, audio_module


def parse_content_length(headers: Message) -> int:
    """Parse Content-Length header."""
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "Content-Length header is required"
        )
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "Content-Length must be an integer"
        ) from exc
    if length <= 0:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "Content-Length must be positive"
        )
    return length


def read_request_body(stream: BinaryIO, length: int) -> bytes:
    """Read the HTTP request body."""
    body = stream.read(length)
    if body is None or len(body) != length:
        raise AlignmentValidationError(
            UI_UPLOAD_BODY_CODE, "request body is incomplete"
        )
    return body


def validate_multipart_content_type(content_type: str) -> None:
    """Validate multipart Content-Type header."""
    if not content_type:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "Content-Type header is required"
        )
    if "multipart/form-data" not in content_type.lower():
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "expected multipart form data"
        )
    header = Message()
    header["Content-Type"] = content_type
    boundary = header.get_param("boundary")
    if not boundary:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "multipart boundary is required"
        )


def parse_multipart_message(content_type: str, body: bytes) -> Message:
    """Parse a multipart HTTP body into a message."""
    validate_multipart_content_type(content_type)
    header_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")
    parser = BytesParser(policy=default)
    message = parser.parsebytes(header_bytes + body)
    if not message.is_multipart():
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "multipart parse failed"
        )
    return message


def decode_form_text(part: Message, payload: bytes, field_name: str) -> str:
    """Decode a form field payload."""
    charset = part.get_content_charset()
    if charset and charset.lower() != "utf-8":
        raise AlignmentValidationError(
            UI_UPLOAD_CODE,
            f"unsupported charset for {field_name}: {charset}",
        )
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, f"invalid UTF-8 in {field_name}"
        ) from exc


def require_form_field(fields: dict[str, str], name: str) -> str:
    """Return a required form field value."""
    if name not in fields:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, f"missing form field: {name}"
        )
    return fields[name]


def require_upload(files: dict[str, UploadFile], name: str) -> UploadFile:
    """Return a required file upload."""
    if name not in files:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, f"missing upload field: {name}"
        )
    return files[name]


def parse_upload_form(
    content_type: str,
    body: bytes,
    defaults: UiDefaults,
) -> UploadForm:
    """Parse the upload form into a structured payload."""
    message = parse_multipart_message(content_type, body)
    allowed_fields = {
        "audio",
        "text",
        "language",
        "remove_punctuation",
        "client_job_id",
    }
    fields: dict[str, str] = {}
    files: dict[str, UploadFile] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="Content-Disposition")
        if not name:
            raise AlignmentValidationError(
                UI_UPLOAD_CODE, "multipart field name is required"
            )
        if name not in allowed_fields:
            raise AlignmentValidationError(
                UI_UPLOAD_CODE, f"unexpected form field: {name}"
            )
        payload = part.get_payload(decode=True)
        assert payload is not None
        filename = part.get_filename()
        if filename:
            if name in files:
                raise AlignmentValidationError(
                    UI_UPLOAD_CODE, f"duplicate upload field: {name}"
                )
            files[name] = UploadFile(filename=filename, payload=payload)
        else:
            if name in fields:
                raise AlignmentValidationError(
                    UI_UPLOAD_CODE, f"duplicate form field: {name}"
                )
            fields[name] = decode_form_text(part, payload, name)

    audio = require_upload(files, "audio")
    text = require_upload(files, "text")
    language_raw = require_form_field(fields, "language")
    remove_punctuation_raw = require_form_field(fields, "remove_punctuation")
    client_job_id_raw = fields.get("client_job_id")
    if client_job_id_raw is not None and not client_job_id_raw.strip():
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "client_job_id must be non-empty"
        )
    if not language_raw.strip():
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "language must be provided"
        )
    language_value = normalize_language_value(
        language_raw, defaults.language
    )
    token = remove_punctuation_raw.strip().lower()
    if token in ("1", "true", "on", "yes"):
        remove_punctuation_value = True
    elif token in ("0", "false", "off", "no"):
        remove_punctuation_value = False
    else:
        raise AlignmentValidationError(
            UI_UPLOAD_CODE,
            f"remove_punctuation must be a boolean: {remove_punctuation_raw!r}",
        )
    return UploadForm(
        audio=audio,
        text=text,
        language=language_value,
        remove_punctuation=remove_punctuation_value,
        client_job_id=client_job_id_raw.strip() if client_job_id_raw else None,
    )


def parse_args(argv: Sequence[str]) -> AlignmentRequest:
    """Parse CLI arguments into an AlignmentRequest."""
    parser = argparse.ArgumentParser(prog="audio_to_text.py", add_help=True)
    parser.add_argument("--input-audio")
    parser.add_argument("--input-text")
    parser.add_argument("--input-alignment-json")
    parser.add_argument("--output-srt", default=None)
    parser.add_argument("--language", default="en")
    parsed = parser.parse_args(argv)

    input_audio = parsed.input_audio
    input_text = parsed.input_text
    input_alignment_json = parsed.input_alignment_json
    output_srt = parsed.output_srt
    if input_alignment_json is not None:
        if output_srt is None and input_alignment_json.strip():
            output_srt = str(Path(input_alignment_json).with_suffix(".srt"))
    else:
        if output_srt is None and input_audio is not None and input_audio.strip():
            output_srt = default_output_path(input_audio)

    language_value = str(parsed.language).strip().lower()
    return AlignmentRequest(
        input_audio=input_audio,
        input_text=input_text,
        input_alignment_json=input_alignment_json,
        output_srt=output_srt,
        language=language_value,
    )


def load_alignment_model(
    language: str,
    device: str,
    alignment_module: ModuleType,
) -> tuple[object, dict[str, object]]:
    """Load the alignment model and metadata."""
    if language not in TORCHAUDIO_ALIGNMENT_LANGUAGES:
        torch_module = load_torch_module()
        ensure_torch_version(torch_module, language)
    try:
        return alignment_module.load_align_model(
            language_code=language,
            device=device,
        )
    except Exception as exc:
        raise AlignmentPipelineError(
            ALIGN_MODEL_CODE, f"align model load failed: {exc}"
        ) from exc


def resolve_device() -> str:
    """Resolve auto device selection to a concrete device."""
    torch_module = load_torch_module()
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def align_words(
    audio_path: str,
    transcript_text: str,
    language: str,
    device: str,
    remove_punctuation: bool = False,
) -> tuple[AlignedWord, ...]:
    """Align transcript text to the audio and return word timings."""
    alignment_module, audio_module = load_whisperx_alignment_modules()
    audio = audio_module.load_audio(audio_path)
    audio_duration = float(len(audio)) / float(audio_module.SAMPLE_RATE)
    segments = [{"start": 0.0, "end": audio_duration, "text": transcript_text}]

    try:
        align_model, metadata = load_alignment_model(
            language, device, alignment_module
        )
        result = alignment_module.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        error_message = str(exc)
        error_code = ALIGNMENT_TIMESTAMP_CODE if "missing timestamps" in error_message else ALIGNMENT_CODE
        raise AlignmentPipelineError(
            error_code, f"alignment failed: {error_message}"
        ) from exc

    return extract_aligned_words(
        result.get("segments", []),
        remove_punctuation=remove_punctuation,
    )


def is_punctuation_token(text_value: str) -> bool:
    """Return True when text contains no alphanumeric characters."""
    return bool(text_value) and not any(
        character.isalnum() for character in text_value
    )


def strip_punctuation_from_token(text_value: str) -> str:
    """Remove punctuation characters from a token."""
    kept: list[str] = []
    for character in text_value:
        if unicodedata.category(character).startswith("P"):
            continue
        kept.append(character)
    return "".join(kept).strip()


def merge_token_suffix(
    words: list[AlignedWord],
    token_text: str,
) -> None:
    """Merge token text into the previous word."""
    previous = words[-1]
    merged = AlignedWord(
        text=f"{previous.text} {token_text}",
        start_seconds=previous.start_seconds,
        end_seconds=previous.end_seconds,
    )
    words[-1] = merged


def merge_token_suffix_token(
    tokens: list[dict[str, object]],
    words: list[AlignedWord],
    token_text: str,
) -> None:
    """Merge token text into the most recent token."""
    if tokens:
        previous_text = str(tokens[-1].get("text", "")).strip()
        tokens[-1]["text"] = f"{previous_text} {token_text}".strip()
        return
    merge_token_suffix(words, token_text)


def coerce_timestamp(value: object) -> float | None:
    """Coerce a numeric timestamp into a finite float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        candidate = float(value)
        if math.isfinite(candidate):
            return candidate
    return None


def default_segment_bounds(
    token_count: int,
    last_known_end: float | None,
) -> tuple[float, float]:
    """Build fallback segment bounds when timestamps are missing."""
    start_seconds = last_known_end if last_known_end is not None else 0.0
    window_seconds = DEFAULT_MISSING_TOKEN_SECONDS * float(token_count)
    return start_seconds, start_seconds + window_seconds


def segment_bounds(
    segment: dict[str, object],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Resolve segment start and end bounds."""
    start_seconds = coerce_timestamp(segment.get("start"))
    end_seconds = coerce_timestamp(segment.get("end"))
    if start_seconds is not None and end_seconds is not None:
        if end_seconds > start_seconds:
            return start_seconds, end_seconds
    return fallback


def segment_bounds_from_tokens(
    tokens: list[dict[str, object]],
) -> tuple[float, float] | None:
    """Derive segment bounds from token timings."""
    starts: list[float] = []
    ends: list[float] = []
    for token in tokens:
        start_value = coerce_timestamp(token.get("start"))
        end_value = coerce_timestamp(token.get("end"))
        if start_value is not None and end_value is not None:
            if end_value > start_value:
                starts.append(start_value)
                ends.append(end_value)
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def token_weight(text_value: str) -> int:
    """Return the distribution weight for a token."""
    compact = "".join(part for part in text_value.split() if part)
    return max(1, len(compact))


def infer_missing_timings(
    tokens: list[dict[str, object]],
    segment_start: float,
    segment_end: float,
) -> None:
    """Fill missing token timestamps in place."""
    missing_texts: list[str] = []
    for token in tokens:
        start_value = coerce_timestamp(token.get("start"))
        end_value = coerce_timestamp(token.get("end"))
        if start_value is not None and end_value is not None:
            token["start"] = start_value
            token["end"] = end_value
            if end_value > start_value:
                continue
            continue
        token["start"] = None
        token["end"] = None
        missing_texts.append(str(token.get("text", "")).strip())

    if not missing_texts:
        return

    preview = ", ".join(missing_texts[:8])
    extra = "" if len(missing_texts) <= 8 else f" (+{len(missing_texts) - 8} more)"
    LOGGER.warning(
        "%s: inferring timestamps for %d token(s): %s%s",
        ALIGNMENT_INFERRED_TIMESTAMPS_CODE,
        len(missing_texts),
        preview,
        extra,
    )

    index = 0
    while index < len(tokens):
        if isinstance(tokens[index].get("start"), float):
            index += 1
            continue
        run_start = index
        while index < len(tokens) and tokens[index].get("start") is None:
            index += 1
        run_end = index

        left_bound = segment_start
        if run_start > 0:
            left_bound = cast(float, tokens[run_start - 1].get("end"))
        right_bound = segment_end
        if run_end < len(tokens):
            right_bound = cast(float, tokens[run_end].get("start"))
        if right_bound <= left_bound:
            right_bound = left_bound + (0.001 * float(run_end - run_start))

        window_seconds = right_bound - left_bound
        weights = [
            token_weight(str(tokens[i].get("text", "")).strip())
            for i in range(run_start, run_end)
        ]
        total_weight = float(sum(weights)) or 1.0
        cursor = left_bound
        for offset, weight in enumerate(weights):
            share = window_seconds * (float(weight) / total_weight)
            start_seconds = cursor
            end_seconds = cursor + share
            token = tokens[run_start + offset]
            token["start"] = start_seconds
            token["end"] = end_seconds
            cursor = end_seconds


def extract_aligned_words(
    segments: Iterable[dict[str, object]],
    remove_punctuation: bool = False,
) -> tuple[AlignedWord, ...]:
    """Extract aligned words from alignment output."""
    words: list[AlignedWord] = []
    pending_prefix_tokens: list[str] = []
    last_known_end: float | None = None
    for segment in segments:
        if not isinstance(segment, dict):
            raise AlignmentPipelineError(
                ALIGNMENT_CODE, "alignment segment payload must be an object"
            )
        raw_words = segment.get("words", [])
        if not isinstance(raw_words, list):
            raise AlignmentPipelineError(
                ALIGNMENT_CODE, "alignment segment words must be a list"
            )

        tokens: list[dict[str, object]] = []
        for word in raw_words:
            if not isinstance(word, dict):
                raise AlignmentPipelineError(
                    ALIGNMENT_CODE, "alignment word payload must be an object"
                )
            raw_text = str(word.get("word", "")).strip()
            text_value = raw_text
            if remove_punctuation:
                text_value = strip_punctuation_from_token(raw_text)
                if not text_value:
                    if is_punctuation_token(raw_text):
                        LOGGER.info(
                            "audio_to_text.align.dropped_punctuation: %s",
                            raw_text,
                        )
                        continue
                    raise AlignmentPipelineError(
                        ALIGNMENT_TIMESTAMP_CODE,
                        "aligned word text is empty after punctuation removal",
                    )
            start_value = coerce_timestamp(word.get("start"))
            end_value = coerce_timestamp(word.get("end"))
            if start_value is None or end_value is None:
                if remove_punctuation and is_punctuation_token(text_value):
                    LOGGER.warning(
                        "%s: dropping punctuation with missing timestamps: %s",
                        ALIGNMENT_TIMESTAMP_CODE,
                        text_value,
                    )
                    continue
                if is_punctuation_token(text_value):
                    if tokens:
                        LOGGER.warning(
                            "%s: merging punctuation with missing timestamps: %s",
                            ALIGNMENT_TIMESTAMP_CODE,
                            text_value,
                        )
                        merge_token_suffix_token(tokens, words, text_value)
                    elif words:
                        LOGGER.warning(
                            "%s: merging punctuation with missing timestamps: %s",
                            ALIGNMENT_TIMESTAMP_CODE,
                            text_value,
                        )
                        merge_token_suffix_token(tokens, words, text_value)
                    else:
                        LOGGER.warning(
                            "%s: carrying punctuation with missing timestamps: %s",
                            ALIGNMENT_TIMESTAMP_CODE,
                            text_value,
                        )
                        pending_prefix_tokens.append(text_value)
                    continue

            if remove_punctuation and is_punctuation_token(text_value):
                LOGGER.info(
                    "audio_to_text.align.dropped_punctuation: %s",
                    text_value,
                )
                continue

            if pending_prefix_tokens:
                text_value = " ".join([*pending_prefix_tokens, text_value])
                pending_prefix_tokens.clear()
            tokens.append(
                {"text": text_value, "start": start_value, "end": end_value}
            )

        if not tokens:
            continue

        fallback = segment_bounds_from_tokens(tokens)
        if fallback is None:
            LOGGER.warning(
                "%s: inferring segment bounds for %d token(s)",
                ALIGNMENT_INFERRED_TIMESTAMPS_CODE,
                len(tokens),
            )
            fallback = default_segment_bounds(len(tokens), last_known_end)
        segment_start, segment_end = segment_bounds(segment, fallback)
        infer_missing_timings(tokens, segment_start, segment_end)
        segment_has_words = False
        for token in tokens:
            token_text = str(token.get("text", "")).strip()
            start_value = coerce_timestamp(token.get("start"))
            end_value = coerce_timestamp(token.get("end"))
            words.append(
                AlignedWord(
                    text=token_text,
                    start_seconds=start_value,
                    end_seconds=end_value,
                )
            )
            last_known_end = end_value
            segment_has_words = True

    if not words:
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, "alignment produced no words"
        )

    return tuple(words)


def read_alignment_result(file_path: str) -> dict[str, object]:
    """Read a whisperx alignment result JSON file."""
    try:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AlignmentValidationError(
            INPUT_ALIGNMENT_JSON_CODE,
            f"input alignment json not found: {file_path}",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AlignmentValidationError(
            INPUT_ALIGNMENT_JSON_CODE,
            f"input alignment json is invalid: {file_path}",
        ) from exc
    if not isinstance(payload, dict):
        raise AlignmentValidationError(
            INPUT_ALIGNMENT_JSON_CODE, "input alignment json must be an object"
        )
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise AlignmentValidationError(
            INPUT_ALIGNMENT_JSON_CODE, "input alignment json segments must be a list"
        )
    for segment in segments:
        if not isinstance(segment, dict):
            raise AlignmentValidationError(
                INPUT_ALIGNMENT_JSON_CODE,
                "input alignment json segments must contain objects",
            )
    return payload


def srt_timestamp_from_seconds(seconds: float, rounding: str) -> int:
    """Convert seconds to milliseconds for SRT output."""
    if rounding == "ceil":
        return int(math.ceil(seconds * 1000))
    return int(math.floor(seconds * 1000))


def format_srt_timestamp(milliseconds: int) -> str:
    """Format milliseconds into an SRT timestamp."""
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


def srt_filename_from_audio_filename(audio_filename: str) -> str:
    """Derive an SRT filename from the original audio filename."""
    base_name = Path(audio_filename).name
    if not base_name:
        return "alignment.srt"
    return Path(base_name).with_suffix(".srt").name


def content_disposition_attachment(filename: str) -> str:
    """Build a Content-Disposition attachment header value."""
    sanitized = filename.replace("\r", " ").replace("\n", " ").strip()
    ascii_fallback = "".join(
        char if 32 <= ord(char) < 127 and char not in {'"', "\\"} else "_"
        for char in sanitized
    ).strip()
    encoded = quote(sanitized, safe="")
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{encoded}"
    )


def main() -> int:
    """CLI entrypoint."""
    configure_logging()
    try:
        request = parse_args(sys.argv[1:])
        ensure_linux_runtime()
        if request.input_alignment_json is not None:
            result = read_alignment_result(request.input_alignment_json)
            words = extract_aligned_words(
                result.get("segments", []),
            )
            srt_content = build_srt(words)
            write_srt_file(request.output_srt or "", srt_content)
            LOGGER.info("audio_to_text.output.srt_written: %s", request.output_srt)
            return 0
        ensure_audio_file_exists(request.input_audio or "")
        transcript_text = normalize_transcript_for_alignment(
            read_utf8_text_strict(request.input_text or ""),
            request.input_text or "",
            remove_punctuation=False,
        )
        resolved_device = resolve_device()
        words = align_words(
            request.input_audio or "",
            transcript_text,
            request.language,
            resolved_device,
            remove_punctuation=False,
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
