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
import time
import uuid
import warnings
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from enum import Enum
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from types import ModuleType
from typing import BinaryIO, Callable, Iterable, Sequence, cast
from urllib.parse import quote, urlparse


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
DEFAULT_UI_ROOT_DIR = ""
MAX_PORT = 65535
TORCH_MIN_VERSION = (2, 6)
TORCH_MIN_VERSION_TEXT = "2.6"
TORCHAUDIO_ALIGNMENT_LANGUAGES = {"en", "fr", "de", "es", "it"}
ALIGNMENT_PROGRESS_START = 0.45
ALIGNMENT_PROGRESS_MAX = 0.84
ALIGNMENT_PROGRESS_INTERVAL_SECONDS = 0.5
ALIGNMENT_MIN_SECONDS = 3.0
ALIGNMENT_TIME_SCALE = {
    "cpu": 1.25,
    "cuda": 0.75,
}
DEFAULT_ALIGNMENT_TIME_SCALE = 1.0
DEFAULT_MISSING_TOKEN_SECONDS = 0.25
SSE_KEEPALIVE_SECONDS = 5.0
UI_SSE_FAILURE_MODE_ENV = "AUDIO_TO_TEXT_UI_SSE_FAILURE_MODE"
UI_SSE_FAILURE_JOBS_SNAPSHOT = "jobs_snapshot"
UI_SSE_FAILURE_JOBS_KEEPALIVE = "jobs_keepalive"
UI_SSE_FAILURE_JOBS_UPDATE = "jobs_update"
UI_SSE_FAILURE_JOB_SNAPSHOT = "job_snapshot"
UI_SSE_FAILURE_JOB_KEEPALIVE = "job_keepalive"
UI_SSE_FAILURE_JOB_UPDATE = "job_update"
UI_SSE_FAILURE_MODES = {
    UI_SSE_FAILURE_JOBS_SNAPSHOT,
    UI_SSE_FAILURE_JOBS_KEEPALIVE,
    UI_SSE_FAILURE_JOBS_UPDATE,
    UI_SSE_FAILURE_JOB_SNAPSHOT,
    UI_SSE_FAILURE_JOB_KEEPALIVE,
    UI_SSE_FAILURE_JOB_UPDATE,
}
PLATFORM_OVERRIDE_ENV = "AUDIO_TO_TEXT_PLATFORM_OVERRIDE"
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
    input_alignment_json: str | None
    output_srt: str | None
    language: str
    device: str
    ui_host: str
    ui_port: int
    ui_root_dir: str

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
        if self.device != DEVICE_AUTO:
            raise AlignmentValidationError(
                INVALID_CONFIG_CODE,
                "device is fixed to auto; omit --device",
            )
        if self.mode == RequestMode.CLI:
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
                if (
                    self.input_audio is not None
                    or self.input_text is not None
                ):
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
        else:
            if not self.ui_host.strip():
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "ui-host must be non-empty"
                )
            if self.ui_port <= 0 or self.ui_port > MAX_PORT:
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "ui-port is invalid"
                )
            if self.ui_root_dir and not self.ui_root_dir.strip():
                raise AlignmentValidationError(
                    INVALID_CONFIG_CODE, "ui-root-dir must be non-empty"
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

def parse_job_optional_bool(value: object, label: str) -> bool | None:
    """Parse an optional boolean value."""
    if value is None:
        return None
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
    remove_punctuation = parse_job_optional_bool(
        input_payload.get("remove_punctuation"), "remove_punctuation"
    )
    if remove_punctuation is None:
        remove_punctuation = True
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

    def wait_for_job_update(
        self,
        job_id: str,
        last_seen: AlignmentJob,
        timeout: float,
    ) -> AlignmentJob:
        """Wait for a job update or return the latest snapshot."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.jobs[job_id] != last_seen, timeout=timeout
            )
            return self.jobs[job_id]

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


def estimate_alignment_seconds(audio_duration: float, device: str) -> float:
    """Estimate alignment runtime for progress updates."""
    scale = ALIGNMENT_TIME_SCALE.get(device, DEFAULT_ALIGNMENT_TIME_SCALE)
    estimated = audio_duration * scale
    if estimated < ALIGNMENT_MIN_SECONDS:
        return ALIGNMENT_MIN_SECONDS
    return estimated


@dataclass
class AlignmentProgressTracker:
    """Progress updates for alignment work."""

    store: JobStore
    job_id: str
    clock: Callable[[], float]
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self, audio_duration: float, device: str) -> None:
        """Begin emitting progress updates."""
        expected_seconds = estimate_alignment_seconds(audio_duration, device)

        def run() -> None:
            start_time = self.clock()
            while not self.stop_event.wait(ALIGNMENT_PROGRESS_INTERVAL_SECONDS):
                elapsed = self.clock() - start_time
                fraction = min(max(elapsed / expected_seconds, 0.0), 0.99)
                progress = ALIGNMENT_PROGRESS_START + (
                    (ALIGNMENT_PROGRESS_MAX - ALIGNMENT_PROGRESS_START) * fraction
                )
                self.store.update_job(
                    self.job_id,
                    JobStatus.RUNNING,
                    message="Aligning words to audio",
                    progress=progress,
                )

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop emitting progress updates."""
        self.stop_event.set()
        assert self.thread is not None
        self.thread.join(timeout=ALIGNMENT_PROGRESS_INTERVAL_SECONDS)


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


def resolve_ui_root_dir_override(ui_root_dir: str) -> Path:
    """Resolve the UI uploads directory override."""
    root_dir = Path(ui_root_dir).expanduser()
    try:
        root_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AlignmentPipelineError(
            UI_STORAGE_CODE, f"failed to create upload directory: {root_dir}"
        ) from exc
    return root_dir


def parse_ui_sse_failure_mode(env: dict[str, str]) -> str | None:
    """Parse the optional UI SSE failure mode."""
    raw_value = env.get(UI_SSE_FAILURE_MODE_ENV, "").strip()
    if not raw_value:
        return None
    if raw_value not in UI_SSE_FAILURE_MODES:
        raise AlignmentValidationError(
            INVALID_CONFIG_CODE,
            f"ui sse failure mode is invalid: {raw_value}",
        )
    return raw_value


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
    allowed_fields = {"audio", "text", "language", "device", "remove_punctuation"}
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
    device_raw = fields.get("device", "").strip().lower()
    remove_punctuation_raw = fields.get("remove_punctuation", "")
    if not language_raw.strip():
        raise AlignmentValidationError(
            UI_UPLOAD_CODE, "language must be provided"
        )
    if device_raw and device_raw != DEVICE_AUTO:
        LOGGER.warning(
            "audio_to_text.ui.upload.ignored_device: %s",
            device_raw,
        )
    language_value = normalize_language_value(
        language_raw, defaults.language
    )
    remove_punctuation_value = defaults.remove_punctuation
    if remove_punctuation_raw.strip():
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
    )


def parse_args(argv: Sequence[str]) -> AlignmentRequest:
    """Parse CLI arguments into an AlignmentRequest."""
    parser = argparse.ArgumentParser(prog="audio_to_text.py", add_help=True)
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui-host", default=DEFAULT_UI_HOST)
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--ui-root-dir", default=DEFAULT_UI_ROOT_DIR)
    parser.add_argument("--input-audio")
    parser.add_argument("--input-text")
    parser.add_argument("--input-alignment-json")
    parser.add_argument("--output-srt", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default=DEVICE_AUTO)
    parsed = parser.parse_args(argv)

    mode = RequestMode.UI if parsed.ui else RequestMode.CLI
    input_audio = parsed.input_audio
    input_text = parsed.input_text
    input_alignment_json = parsed.input_alignment_json
    output_srt = parsed.output_srt
    if mode == RequestMode.CLI:
        if input_alignment_json is not None:
            if output_srt is None and input_alignment_json.strip():
                output_srt = str(Path(input_alignment_json).with_suffix(".srt"))
        else:
            if output_srt is None and input_audio is not None and input_audio.strip():
                output_srt = default_output_path(input_audio)

    language_value = str(parsed.language).strip().lower()
    device_value = str(parsed.device).strip().lower() or DEVICE_AUTO
    return AlignmentRequest(
        mode=mode,
        input_audio=input_audio,
        input_text=input_text,
        input_alignment_json=input_alignment_json,
        output_srt=output_srt,
        language=language_value,
        device=device_value,
        ui_host=str(parsed.ui_host),
        ui_port=int(parsed.ui_port),
        ui_root_dir=str(parsed.ui_root_dir),
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
    progress_tracker: AlignmentProgressTracker | None = None,
    remove_punctuation: bool = False,
) -> tuple[AlignedWord, ...]:
    """Align transcript text to the audio and return word timings."""
    alignment_module, audio_module = load_whisperx_alignment_modules()
    audio = audio_module.load_audio(audio_path)
    audio_duration = float(len(audio)) / float(audio_module.SAMPLE_RATE)
    segments = [{"start": 0.0, "end": audio_duration, "text": transcript_text}]

    try:
        if progress_tracker is not None:
            progress_tracker.start(audio_duration, device)
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
        raise AlignmentPipelineError(
            ALIGNMENT_CODE, f"alignment failed: {exc}"
        ) from exc
    finally:
        if progress_tracker is not None:
            progress_tracker.stop()

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


def build_ui_html(defaults: UiDefaults) -> str:
    """Render the UI HTML with default values."""
    language_options = []
    for language_code, language_label in SUPPORTED_ALIGNMENT_LANGUAGES:
        selected = " selected" if language_code == defaults.language else ""
        label_text = f"{language_label} ({language_code})"
        language_options.append(
            f'<option value="{escape(language_code)}"{selected}>{escape(label_text)}</option>'
        )
    remove_punctuation_checked = (
        " checked" if defaults.remove_punctuation else ""
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
    .option input:not([type="checkbox"]),
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
    .toggle {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      font-size: 0.95rem;
    }
    .toggle input[type="checkbox"] {
      margin: 0;
      width: auto;
      height: 16px;
      accent-color: var(--accent-cool);
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
    .jobs {
      display: grid;
      gap: 12px;
    }
    .job-list {
      max-height: min(52vh, 420px);
      overflow-y: auto;
      padding-right: 6px;
      scrollbar-gutter: stable;
    }
    .job-list::-webkit-scrollbar {
      width: 10px;
    }
    .job-list::-webkit-scrollbar-track {
      background: transparent;
    }
    .job-list::-webkit-scrollbar-thumb {
      background: rgba(29, 27, 25, 0.12);
      border-radius: 999px;
      border: 2px solid rgba(255, 255, 255, 0.6);
    }
    .job-list::-webkit-scrollbar-thumb:hover {
      background: rgba(29, 27, 25, 0.18);
    }
    .jobs-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .jobs-count {
      font-weight: 600;
      color: var(--ink);
    }
    .job-list {
      display: grid;
      gap: 12px;
    }
    .job-card {
      border-radius: 18px;
      border: 1px solid var(--stroke);
      background: rgba(255, 255, 255, 0.7);
      padding: 16px;
      display: grid;
      gap: 10px;
    }
    .job-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .job-title {
      font-weight: 600;
      font-size: 1rem;
    }
    .job-status {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(43, 122, 120, 0.12);
      color: var(--accent-cool);
      font-weight: 600;
    }
    .job-status.is-queued {
      background: rgba(95, 91, 85, 0.12);
      color: var(--muted);
    }
    .job-status.is-running {
      background: rgba(43, 122, 120, 0.18);
      color: var(--accent-cool);
    }
    .job-status.is-complete {
      background: rgba(43, 122, 120, 0.2);
      color: var(--accent-cool);
    }
    .job-status.is-failed {
      background: rgba(241, 84, 45, 0.12);
      color: #8a2a16;
    }
    .job-meta,
    .job-message {
      color: var(--muted);
      font-size: 0.85rem;
    }
    .job-progress {
      height: 6px;
      border-radius: 999px;
      background: rgba(29, 27, 25, 0.08);
      overflow: hidden;
    }
    .job-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, rgba(43, 122, 120, 0.2), rgba(43, 122, 120, 0.7), rgba(43, 122, 120, 0.2));
      background-size: 200% 100%;
      transition: width 0.3s ease;
    }
    .job-actions {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .job-delete {
      margin-left: auto;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      background: rgba(255, 255, 255, 0.6);
      cursor: pointer;
      transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
    }
    .job-delete svg {
      width: 18px;
      height: 18px;
      stroke: #8a2a16;
    }
    .job-delete:hover {
      background: rgba(241, 84, 45, 0.12);
      border-color: rgba(241, 84, 45, 0.25);
      transform: translateY(-1px);
    }
    .job-delete:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }
    .job-download {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--accent-cool);
      text-decoration: none;
      font-weight: 600;
    }
    .job-download:hover {
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
          <input id="audio-file" type="file" accept="audio/*,video/*,.wav,.wave,.mp3,.m4a,.aac,.flac,.ogg,.mp4,.mov,.m4v">
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
          <span>Remove punctuation</span>
          <div class="toggle">
            <input id="remove-punctuation" type="checkbox"$remove_punctuation_checked>
            <span>Enabled</span>
          </div>
        </label>
      </div>
      <div class="actions">
        <button id="run-button" class="run-button" disabled>Align and Build SRT</button>
      </div>
      <div class="status">
        <div class="status-main" id="status-line">Ready to align.</div>
        <div class="status-sub" id="status-sub">Upload files to begin.</div>
      </div>
      <div class="jobs">
        <div class="jobs-header">
          <span>Session jobs</span>
          <span class="jobs-count" id="job-count">0</span>
        </div>
        <div class="job-list" id="job-list"></div>
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
    const errorLine = document.getElementById("error-line");
    const languageInput = document.getElementById("language");
    const removePunctuationInput = document.getElementById("remove-punctuation");
    const jobList = document.getElementById("job-list");
    const jobCount = document.getElementById("job-count");
    const jobEntries = new Map();
    let jobStream = null;
    let audioFile = null;
    let textFile = null;
    let jobSubmitting = false;
    let pendingSubmission = null;
    let languageTouched = false;
    const defaultAudioMeta = audioMeta.textContent;
    const defaultTextMeta = textMeta.textContent;

    function isOptimisticJobId(jobId) {
      return typeof jobId === "string" && jobId.startsWith("local_");
    }

    function updateRunButtonState() {
      runButton.disabled = jobSubmitting || !audioFile || !textFile;
    }

    function applyStoredLanguage() {
      try {
        const stored = window.localStorage ? window.localStorage.getItem("audio_to_text.language") : null;
        if (!stored) {
          return;
        }
        if (stored === languageInput.value) {
          return;
        }
        languageInput.value = stored;
        languageTouched = true;
      } catch (error) {
        return;
      }
    }

    function persistLanguage(value) {
      try {
        if (!window.localStorage) {
          return;
        }
        window.localStorage.setItem("audio_to_text.language", value);
      } catch (error) {
        return;
      }
    }

    async function detectLanguageFromTextFile(file) {
      if (!file || typeof file.slice !== "function" || typeof window.FileReader === "undefined") {
        return null;
      }
      const maxBytes = 8192;
      const slice = file.slice(0, maxBytes);
      const reader = new FileReader();
      const buffer = await new Promise((resolve) => {
        reader.onerror = () => resolve(null);
        reader.onload = () => resolve(reader.result || null);
        reader.readAsArrayBuffer(slice);
      });
      if (!buffer || !(buffer instanceof ArrayBuffer)) {
        return null;
      }
      let text = "";
      try {
        const decoder = new TextDecoder("utf-8", { fatal: false });
        text = decoder.decode(buffer);
      } catch (error) {
        return null;
      }
      const hasCyrillic = /[\u0400-\u04FF]/.test(text);
      if (hasCyrillic) {
        return "ru";
      }
      return null;
    }

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

    function updateJobCount() {
      jobCount.textContent = String(jobEntries.size);
    }

    function resetDropzone(zone, input, meta, defaultText) {
      zone.classList.remove("is-filled");
      input.value = "";
      meta.textContent = defaultText;
    }

    function resetInputs() {
      audioFile = null;
      textFile = null;
      resetDropzone(audioZone, audioInput, audioMeta, defaultAudioMeta);
      resetDropzone(textZone, textInput, textMeta, defaultTextMeta);
      updateRunButtonState();
    }

    function statusLabel(statusValue) {
      if (statusValue === "queued") {
        return "Queued";
      }
      if (statusValue === "running") {
        return "Running";
      }
      if (statusValue === "completed") {
        return "Complete";
      }
      if (statusValue === "failed") {
        return "Failed";
      }
      return "Unknown";
    }

    function setStatusBadge(element, statusValue) {
      element.textContent = statusLabel(statusValue);
      element.classList.remove(
        "is-queued",
        "is-running",
        "is-complete",
        "is-failed",
      );
      if (statusValue === "queued") {
        element.classList.add("is-queued");
      } else if (statusValue === "running") {
        element.classList.add("is-running");
      } else if (statusValue === "completed") {
        element.classList.add("is-complete");
      } else if (statusValue === "failed") {
        element.classList.add("is-failed");
      }
    }

    function createJobEntry(job) {
      const card = document.createElement("div");
      card.className = "job-card";
      card.dataset.jobId = String(job.job_id || "");
      const header = document.createElement("div");
      header.className = "job-header";
      const title = document.createElement("div");
      title.className = "job-title";
      const status = document.createElement("div");
      status.className = "job-status";
      header.appendChild(title);
      header.appendChild(status);
      const meta = document.createElement("div");
      meta.className = "job-meta";
      const message = document.createElement("div");
      message.className = "job-message";
      const progress = document.createElement("div");
      progress.className = "job-progress";
      const progressBar = document.createElement("div");
      progressBar.className = "job-progress-bar";
      progress.appendChild(progressBar);
      const actions = document.createElement("div");
      actions.className = "job-actions";
      const download = document.createElement("a");
      download.className = "job-download hidden";
      download.textContent = "Download SRT";
      download.href = "#";
      actions.appendChild(download);
      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.className = "job-delete hidden";
      removeButton.title = "Delete finished job";
      removeButton.setAttribute("aria-label", "Delete finished job");
      removeButton.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M3 6h18"></path>
          <path d="M8 6V4h8v2"></path>
          <path d="M6 6l1 16h10l1-16"></path>
          <path d="M10 11v6"></path>
          <path d="M14 11v6"></path>
        </svg>
      `;
      actions.appendChild(removeButton);
      card.appendChild(header);
      card.appendChild(meta);
      card.appendChild(message);
      card.appendChild(progress);
      card.appendChild(actions);
      return {
        card,
        title,
        status,
        meta,
        message,
        progressBar,
        download,
        removeButton,
        optimistic: Boolean(job.is_optimistic),
      };
    }

    function removeJobEntry(jobId) {
      if (!jobId) {
        return;
      }
      const entry = jobEntries.get(jobId);
      if (!entry) {
        return;
      }
      entry.card.remove();
      jobEntries.delete(jobId);
      updateJobCount();
    }

    function rekeyJobEntry(oldJobId, job) {
      const entry = jobEntries.get(oldJobId);
      if (!entry || !job || !job.job_id) {
        return false;
      }
      const existing = jobEntries.get(job.job_id);
      if (existing && existing !== entry) {
        existing.card.remove();
        jobEntries.delete(job.job_id);
      }
      jobEntries.delete(oldJobId);
      entry.card.dataset.jobId = String(job.job_id);
      entry.optimistic = false;
      jobEntries.set(job.job_id, entry);
      updateJobEntry(entry, job);
      updateJobCount();
      return true;
    }

    function claimPendingSubmission(job) {
      if (!pendingSubmission || !job || !job.job_id || isOptimisticJobId(job.job_id)) {
        return false;
      }
      if (job.audio_filename !== pendingSubmission.audio_filename) {
        return false;
      }
      if (job.text_filename !== pendingSubmission.text_filename) {
        return false;
      }
      if (job.language !== pendingSubmission.language) {
        return false;
      }
      if (Boolean(job.remove_punctuation) !== pendingSubmission.remove_punctuation) {
        return false;
      }
      const createdAt = typeof job.created_at === "number" ? job.created_at : null;
      if (createdAt !== null && Math.abs(createdAt - pendingSubmission.created_at_seconds) > 90) {
        return false;
      }
      if (rekeyJobEntry(pendingSubmission.optimisticJobId, job)) {
        pendingSubmission = null;
        return true;
      }
      return false;
    }

    async function deleteJob(jobId) {
      clearError();
      if (!jobId) {
        return;
      }
      if (!confirm("Delete this job?")) {
        return;
      }
      let response = null;
      try {
        response = await fetch("/api/jobs/" + jobId, { method: "DELETE" });
      } catch (error) {
        setError("Failed to delete the job.");
        return;
      }
      let payload = {};
      try {
        payload = await response.json();
      } catch (error) {
        payload = {};
      }
      if (!response.ok) {
        setError(payload.error || "Failed to delete the job.");
        return;
      }
      loadJobs();
    }

    function updateJobEntry(entry, job) {
      const statusValue = String(job.status || "queued");
      const audioName = job.audio_filename || "Audio alignment";
      const textName = job.text_filename || "unknown text";
      const languageLabel = job.language ? job.language.toUpperCase() : "";
      const removePunctuation = typeof job.remove_punctuation === "boolean" ? job.remove_punctuation : null;
      const metaParts = ["Text: " + textName];
      if (languageLabel) {
        metaParts.push("Lang: " + languageLabel);
      }
      if (removePunctuation !== null) {
        metaParts.push(removePunctuation ? "Punct: removed" : "Punct: kept");
      }
      entry.title.textContent = audioName;
      entry.meta.textContent = metaParts.join(" • ");
      setStatusBadge(entry.status, statusValue);
      entry.message.textContent = job.message || statusLabel(statusValue);
      const progressValue = typeof job.progress === "number" ? job.progress : 0;
      const clamped = Math.max(0, Math.min(1, progressValue));
      entry.progressBar.style.width = Math.round(clamped * 100) + "%";
      if (job.output_ready || statusValue === "completed") {
        entry.download.href = "/api/jobs/" + job.job_id + "/srt";
        entry.download.classList.remove("hidden");
      } else {
        entry.download.classList.add("hidden");
      }
      if (statusValue === "completed" || statusValue === "failed") {
        entry.removeButton.classList.remove("hidden");
        entry.removeButton.disabled = false;
        entry.removeButton.onclick = () => deleteJob(job.job_id);
      } else {
        entry.removeButton.classList.add("hidden");
        entry.removeButton.disabled = true;
        entry.removeButton.onclick = null;
      }
    }

    function applyJobUpdate(job) {
      if (!job || !job.job_id) {
        return;
      }
      if (claimPendingSubmission(job)) {
        return;
      }
      let entry = jobEntries.get(job.job_id);
      if (!entry) {
        entry = createJobEntry(job);
        jobEntries.set(job.job_id, entry);
        jobList.prepend(entry.card);
      }
      if (!entry.card.isConnected) {
        jobList.prepend(entry.card);
      }
      updateJobEntry(entry, job);
      updateJobCount();
    }

    function applyJobList(jobs) {
      const sorted = Array.from(jobs).sort((left, right) => {
        const leftTime = typeof left.created_at === "number" ? left.created_at : 0;
        const rightTime = typeof right.created_at === "number" ? right.created_at : 0;
        if (rightTime === leftTime) {
          return String(right.job_id || "").localeCompare(String(left.job_id || ""));
        }
        return rightTime - leftTime;
      });
      const seen = new Set();
      sorted.forEach((job) => {
        applyJobUpdate(job);
        seen.add(job.job_id);
      });
      for (const [jobId, entry] of jobEntries) {
        if (!seen.has(jobId) && !entry.optimistic) {
          entry.card.remove();
          jobEntries.delete(jobId);
        }
      }
      jobList.innerHTML = "";
      sorted.forEach((job) => {
        const entry = jobEntries.get(job.job_id);
        if (entry) {
          jobList.appendChild(entry.card);
        }
      });
      for (const [jobId, entry] of jobEntries) {
        if (entry.optimistic) {
          jobList.prepend(entry.card);
        }
      }
      updateJobCount();
    }

    async function loadJobs() {
      try {
        const response = await fetch("/api/jobs");
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        if (Array.isArray(payload.jobs)) {
          applyJobList(payload.jobs);
        }
      } catch (error) {
        setError("Failed to load job history.");
      }
    }

    function startJobStream() {
      if (jobStream || !window.EventSource) {
        return;
      }
      const stream = new EventSource("/api/jobs/events");
      jobStream = stream;
      stream.addEventListener("message", (event) => {
        clearError();
        let payload = null;
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          setError("Failed to parse job updates.");
          return;
        }
        if (payload && Array.isArray(payload.jobs)) {
          applyJobList(payload.jobs);
          return;
        }
        if (payload && payload.job_id) {
          applyJobUpdate(payload);
        }
      });
      stream.addEventListener("error", () => {
        setError("Connection lost while streaming job updates.");
      });
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

    wireDropzone(audioZone, audioInput, audioMeta, (file) => {
      audioFile = file;
      updateRunButtonState();
    });
    wireDropzone(textZone, textInput, textMeta, (file) => {
      textFile = file;
      removePunctuationInput.checked = true;
      updateRunButtonState();
      if (languageTouched) {
        return;
      }
      detectLanguageFromTextFile(file).then((detected) => {
        if (!detected || languageTouched) {
          return;
        }
        languageInput.value = detected;
        persistLanguage(detected);
      });
    });

    languageInput.addEventListener("change", () => {
      languageTouched = true;
      persistLanguage(languageInput.value.trim());
    });

    async function startJob() {
      clearError();
      if (jobSubmitting) {
        return;
      }
      if (!audioFile || !textFile) {
        setError("Select both an audio file and a transcript file.");
        return;
      }
      const optimisticJobId = "local_" + Date.now().toString(16) + "_" + Math.random().toString(16).slice(2);
      const languageValue = languageInput.value.trim();
      const removePunctuationValue = removePunctuationInput.checked;
      const optimisticJob = {
        job_id: optimisticJobId,
        status: "queued",
        message: "Uploading files",
        output_ready: false,
        progress: 0.02,
        audio_filename: audioFile.name,
        text_filename: textFile.name,
        language: languageValue,
        remove_punctuation: removePunctuationValue,
        created_at: Date.now() / 1000,
        is_optimistic: true,
      };
      applyJobUpdate(optimisticJob);
      pendingSubmission = {
        optimisticJobId,
        audio_filename: audioFile.name,
        text_filename: textFile.name,
        language: languageValue,
        remove_punctuation: removePunctuationValue,
        created_at_seconds: optimisticJob.created_at,
      };
      jobSubmitting = true;
      updateRunButtonState();
      setStatus("Queued.", "Uploading files and preparing alignment.");
      const formData = new FormData();
      formData.append("audio", audioFile, audioFile.name);
      formData.append("text", textFile, textFile.name);
      formData.append("language", languageValue);
      formData.append("remove_punctuation", removePunctuationValue ? "1" : "0");
      let response = null;
      try {
        response = await fetch("/api/jobs", { method: "POST", body: formData });
      } catch (error) {
        setError("Failed to submit the job.");
        removeJobEntry(optimisticJobId);
        pendingSubmission = null;
        jobSubmitting = false;
        updateRunButtonState();
        return;
      }
      let payload = {};
      try {
        payload = await response.json();
      } catch (error) {
        payload = {};
      }
      if (!response.ok) {
        setError(payload.error || "Failed to start alignment.");
        removeJobEntry(optimisticJobId);
        pendingSubmission = null;
        jobSubmitting = false;
        updateRunButtonState();
        return;
      }
      if (!claimPendingSubmission(payload) && !rekeyJobEntry(optimisticJobId, payload)) {
        removeJobEntry(optimisticJobId);
        applyJobUpdate(payload);
      }
      pendingSubmission = null;
      setStatus("Queued.", "Job added to the session.");
      resetInputs();
      jobSubmitting = false;
      updateRunButtonState();
    }

    runButton.addEventListener("click", () => startJob());
    loadJobs();
    startJobStream();
    applyStoredLanguage();
    updateRunButtonState();
  </script>
</body>
</html>
"""
    )
    return template.substitute(
        language_options="\n            ".join(language_options),
        remove_punctuation_checked=remove_punctuation_checked,
    )


def run_ui_server(request: AlignmentRequest, env: dict[str, str]) -> int:
    """Run the web UI server."""
    defaults = UiDefaults(
        language=request.language,
        remove_punctuation=True,
    )
    sse_failure_mode = parse_ui_sse_failure_mode(env)
    default_root_dir = (
        Path(__file__).resolve().parent / "data" / "audio_to_text_uploads"
    )
    root_value = request.ui_root_dir or str(default_root_dir)
    root_dir = resolve_ui_root_dir_override(root_value)
    job_store = JobStore(
        root_dir=root_dir,
        clock=time.time,
        id_factory=lambda: uuid.uuid4().hex,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    handler = build_ui_handler(job_store, executor, defaults, sse_failure_mode)
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
    return 0


def build_ui_handler(
    store: JobStore,
    executor: ThreadPoolExecutor,
    defaults: UiDefaults,
    sse_failure_mode: str | None,
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
            if parsed.path == "/api/jobs":
                self.send_jobs_list()
                return
            if parsed.path == "/api/jobs/events":
                self.send_jobs_events()
                return
            if parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.split("/")
                job_id = parts[3] if len(parts) > 3 else ""
                if parsed.path.endswith("/srt"):
                    self.send_srt(job_id)
                    return
                if parsed.path.endswith("/events"):
                    self.send_job_events(job_id)
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

        def do_DELETE(self) -> None:
            """Handle completed job deletion."""
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.split("/")
                job_id = parts[3] if len(parts) > 3 else ""
                if not job_id:
                    self.send_error_response(
                        HTTPStatus.BAD_REQUEST, "Job id is required"
                    )
                    return
                try:
                    store.delete_finished_job(job_id)
                except AlignmentPipelineError as exc:
                    if exc.code == INVALID_JOB_RESULT_CODE:
                        message = str(exc).strip()
                        if "job not found" in message:
                            self.send_error_response(
                                HTTPStatus.NOT_FOUND, "Job not found"
                            )
                            return
                        self.send_error_response(
                            HTTPStatus.CONFLICT, f"{exc.code}: {message}"
                        )
                        return
                    self.send_error_response(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"{exc.code}: {str(exc).strip()}",
                    )
                    return
                self.send_json(
                    HTTPStatus.OK,
                    {"deleted": True, "job_id": job_id},
                )
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

        def send_sse_headers(self) -> None:
            """Send headers for an SSE response."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

        def send_sse_event(
            self,
            payload: dict[str, object],
            event_kind: str | None = None,
        ) -> bool:
            """Send a single SSE event."""
            body = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            try:
                if event_kind is not None and event_kind == sse_failure_mode:
                    raise OSError("sse event failure")
                self.wfile.write(body)
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                return False
            return True

        def send_sse_keepalive(self, event_kind: str | None = None) -> bool:
            """Send a keepalive SSE comment."""
            try:
                if event_kind is not None and event_kind == sse_failure_mode:
                    raise OSError("sse keepalive failure")
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                return False
            return True

        def send_error_response(self, status: HTTPStatus, message: str) -> None:
            """Send an error JSON payload."""
            self.send_json(status, {"error": message})

        def build_job_payload(self, job: AlignmentJob) -> dict[str, object]:
            """Build the job status payload."""
            return {
                "job_id": job.job_id,
                "status": job.result.status.value,
                "message": job.result.message,
                "output_ready": bool(job.result.output_srt),
                "progress": job.result.progress,
                "audio_filename": job.job_input.audio_filename,
                "text_filename": job.job_input.text_filename,
                "language": job.job_input.language,
                "remove_punctuation": job.job_input.remove_punctuation,
                "created_at": job.created_at,
                "started_at": job.result.started_at,
                "completed_at": job.result.completed_at,
            }

        def send_jobs_list(self) -> None:
            """Return the current job list."""
            jobs = store.list_jobs()
            payload = {"jobs": [self.build_job_payload(job) for job in jobs]}
            self.send_json(HTTPStatus.OK, payload)

        def send_jobs_events(self) -> None:
            """Stream job updates as SSE events."""
            self.send_sse_headers()
            change_id = store.change_id
            jobs = store.list_jobs()
            if not self.send_sse_event(
                {
                    "type": "snapshot",
                    "change_id": change_id,
                    "jobs": [self.build_job_payload(job) for job in jobs],
                },
                event_kind=UI_SSE_FAILURE_JOBS_SNAPSHOT,
            ):
                return
            while True:
                next_change = store.wait_for_change(
                    change_id, SSE_KEEPALIVE_SECONDS
                )
                if next_change == change_id:
                    if not self.send_sse_keepalive(
                        event_kind=UI_SSE_FAILURE_JOBS_KEEPALIVE
                    ):
                        return
                    continue
                change_id = next_change
                jobs = store.list_jobs()
                if not self.send_sse_event(
                    {
                        "type": "snapshot",
                        "change_id": change_id,
                        "jobs": [self.build_job_payload(job) for job in jobs],
                    },
                    event_kind=UI_SSE_FAILURE_JOBS_UPDATE,
                ):
                    return

        def send_job_status(self, job_id: str) -> None:
            """Return the current job status."""
            job = store.get_job(job_id)
            if job is None:
                self.send_error_response(HTTPStatus.NOT_FOUND, "Job not found")
                return
            self.send_json(HTTPStatus.OK, self.build_job_payload(job))

        def send_job_events(self, job_id: str) -> None:
            """Stream job updates as SSE events."""
            if not job_id:
                self.send_error_response(
                    HTTPStatus.BAD_REQUEST, "Job id is required"
                )
                return
            job = store.get_job(job_id)
            if job is None:
                self.send_error_response(HTTPStatus.NOT_FOUND, "Job not found")
                return
            self.send_sse_headers()
            if not self.send_sse_event(
                self.build_job_payload(job),
                event_kind=UI_SSE_FAILURE_JOB_SNAPSHOT,
            ):
                return
            last_seen = job
            while job.result.status not in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            ):
                updated = store.wait_for_job_update(
                    job_id, last_seen, SSE_KEEPALIVE_SECONDS
                )
                if updated == last_seen:
                    if not self.send_sse_keepalive(
                        event_kind=UI_SSE_FAILURE_JOB_KEEPALIVE
                    ):
                        return
                    continue
                job = updated
                if not self.send_sse_event(
                    self.build_job_payload(job),
                    event_kind=UI_SSE_FAILURE_JOB_UPDATE,
                ):
                    return
                last_seen = job

        def send_srt(self, job_id: str) -> None:
            """Return the generated SRT file."""
            job = store.get_job(job_id)
            if job is None or job.result.output_srt is None:
                self.send_error_response(HTTPStatus.NOT_FOUND, "SRT not available")
                return
            output_path = Path(job.result.output_srt)
            try:
                content = output_path.read_text(encoding="utf-8")
            except OSError:
                self.send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "SRT read failed")
                return
            payload = content.encode("utf-8")
            filename = srt_filename_from_audio_filename(job.job_input.audio_filename)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-subrip; charset=utf-8")
            self.send_header(
                "Content-Disposition",
                content_disposition_attachment(filename),
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def handle_create_job(self) -> None:
            """Accept uploads and queue a background alignment job."""
            try:
                content_length = parse_content_length(self.headers)
                body = read_request_body(self.rfile, content_length)
                upload = parse_upload_form(
                    self.headers.get("Content-Type", ""),
                    body,
                    defaults,
                )
            except AlignmentValidationError as exc:
                self.send_error_response(
                    HTTPStatus.BAD_REQUEST, f"{exc.code}: {exc}"
                )
                return

            job_id = store.new_job_id()
            job_dir = store.job_dir(job_id)
            audio_suffix = Path(upload.audio.filename).suffix or ".bin"
            text_suffix = Path(upload.text.filename).suffix or ".txt"
            audio_path = job_dir / f"audio{audio_suffix}"
            text_path = job_dir / f"text{text_suffix}"
            output_path = job_dir / "alignment.srt"
            job_input = AlignmentJobInput(
                audio_filename=upload.audio.filename,
                text_filename=upload.text.filename,
                language=upload.language,
                remove_punctuation=upload.remove_punctuation,
                audio_path=str(audio_path),
                text_path=str(text_path),
                output_path=str(output_path),
            )
            try:
                job = store.create_job(job_id, job_input)
                job_dir.mkdir(parents=True, exist_ok=True)
            except (AlignmentPipelineError, OSError) as exc:
                self.send_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"job setup failed: {exc}",
                )
                return

            try:
                audio_path.write_bytes(upload.audio.payload)
                text_path.write_bytes(upload.text.payload)
            except OSError:
                store.update_job(
                    job.job_id,
                    JobStatus.FAILED,
                    "Upload write failed",
                    progress=1.0,
                )
                self.send_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR, "Upload failed"
                )
                return

            executor.submit(run_alignment_job, store, job.job_id)
            self.send_json(HTTPStatus.OK, self.build_job_payload(job))

    return UiHandler


def run_alignment_job(store: JobStore, job_id: str) -> None:
    """Process a background alignment job."""
    job = store.get_job(job_id)
    assert job is not None
    job_input = job.job_input
    store.update_job(
        job_id,
        JobStatus.RUNNING,
        message="Preparing input files",
        progress=0.05,
    )
    try:
        ensure_audio_file_exists(job_input.audio_path)
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Reading transcript text",
            progress=0.15,
        )
        transcript_text = normalize_transcript_for_alignment(
            read_utf8_text_strict(job_input.text_path),
            job_input.text_path,
            remove_punctuation=job_input.remove_punctuation,
        )
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Resolving device",
            progress=0.3,
        )
        resolved_device = resolve_device()
        store.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Aligning words to audio",
            progress=ALIGNMENT_PROGRESS_START,
        )
        progress_tracker = AlignmentProgressTracker(
            store=store,
            job_id=job_id,
            clock=time.monotonic,
        )
        words = align_words(
            job_input.audio_path,
            transcript_text,
            job_input.language,
            resolved_device,
            progress_tracker=progress_tracker,
            remove_punctuation=job_input.remove_punctuation,
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
        write_srt_file(job_input.output_path, srt_content)
        store.update_job(
            job_id,
            JobStatus.COMPLETED,
            message="Complete",
            output_srt=job_input.output_path,
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
    env = dict(os.environ)
    try:
        request = parse_args(sys.argv[1:])
        ensure_linux_runtime()
        if request.mode == RequestMode.UI:
            return run_ui_server(request, env)
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
