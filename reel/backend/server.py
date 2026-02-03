"""HTTP backend service for audio_to_text orchestration."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from concurrent import futures
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import grpc

from reel import audio_to_text
from reel.audio_grpc import audio_to_text_pb2
from reel.audio_grpc import audio_to_text_pb2_grpc

LOGGER = logging.getLogger("audio_to_text_backend")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_DATA_DIR = "data/audio_to_text_backend"
DEFAULT_GRPC_TARGET = "127.0.0.1:50051"
DEFAULT_GRPC_TIMEOUT_SECONDS = 300.0
DEFAULT_GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_WORKERS = 4
DEFAULT_KEEPALIVE_SECONDS = 5.0
DEFAULT_LANGUAGE = "en"

HOST_ENV = "AUDIO_TO_TEXT_BACKEND_HOST"
PORT_ENV = "AUDIO_TO_TEXT_BACKEND_PORT"
DATA_DIR_ENV = "AUDIO_TO_TEXT_BACKEND_DATA_DIR"
GRPC_TARGET_ENV = "AUDIO_TO_TEXT_BACKEND_GRPC_TARGET"
GRPC_TIMEOUT_ENV = "AUDIO_TO_TEXT_BACKEND_GRPC_TIMEOUT_SECONDS"
GRPC_MAX_MESSAGE_BYTES_ENV = "AUDIO_TO_TEXT_BACKEND_GRPC_MAX_MESSAGE_BYTES"
GRPC_AUTH_TOKEN_ENV = "AUDIO_TO_TEXT_BACKEND_GRPC_AUTH_TOKEN"
GRPC_TLS_ENV = "AUDIO_TO_TEXT_BACKEND_GRPC_USE_TLS"
MAX_UPLOAD_BYTES_ENV = "AUDIO_TO_TEXT_BACKEND_MAX_UPLOAD_BYTES"
ALLOWED_ORIGINS_ENV = "AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS"
FFMPEG_PATH_ENV = "AUDIO_TO_TEXT_BACKEND_FFMPEG_PATH"
MAX_WORKERS_ENV = "AUDIO_TO_TEXT_BACKEND_MAX_WORKERS"
LOG_LEVEL_ENV = "AUDIO_TO_TEXT_BACKEND_LOG_LEVEL"
KEEPALIVE_SECONDS_ENV = "AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS"
SSE_FAILURE_MODE_ENV = "AUDIO_TO_TEXT_BACKEND_SSE_FAILURE_MODE"

BACKEND_CONFIG_CODE = "audio_to_text_backend.config.invalid"
BACKEND_UPLOAD_CODE = "audio_to_text_backend.upload.invalid"
BACKEND_ALIGN_CODE = "audio_to_text_backend.align.failed"
BACKEND_AUDIO_CODE = "audio_to_text_backend.audio.extract_failed"
BACKEND_JOB_NOT_FOUND_CODE = "audio_to_text_backend.job.not_found"
BACKEND_JOB_NOT_READY_CODE = "audio_to_text_backend.job.not_ready"
BACKEND_JOB_SETUP_CODE = "audio_to_text_backend.job.setup_failed"
BACKEND_UPLOAD_WRITE_CODE = "audio_to_text_backend.upload.write_failed"
BACKEND_OUTPUT_NOT_FOUND_CODE = "audio_to_text_backend.output.not_found"
BACKEND_OUTPUT_READ_CODE = "audio_to_text_backend.output.read_failed"
BACKEND_NOT_FOUND_CODE = "audio_to_text_backend.path.not_found"
SSE_FAILURE_SNAPSHOT = "snapshot"
SSE_FAILURE_KEEPALIVE = "keepalive"
SSE_FAILURE_UPDATE = "update"
SSE_FAILURE_MODES = {
    SSE_FAILURE_SNAPSHOT,
    SSE_FAILURE_KEEPALIVE,
    SSE_FAILURE_UPDATE,
}


class BackendError(RuntimeError):
    """Backend error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclasses.dataclass(frozen=True)
class BackendConfig:
    """Backend configuration."""

    host: str
    port: int
    data_dir: Path
    grpc_target: str
    grpc_timeout_seconds: float
    grpc_max_message_bytes: int
    grpc_auth_token: str | None
    grpc_use_tls: bool
    max_upload_bytes: int
    allowed_origins: tuple[str, ...]
    allow_any_origin: bool
    ffmpeg_path: str
    max_workers: int
    keepalive_seconds: float
    sse_failure_mode: str | None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must be non-empty")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if not self.grpc_target.strip():
            raise ValueError("grpc-target must be non-empty")
        if self.grpc_timeout_seconds < 0:
            raise ValueError("grpc-timeout-seconds must be non-negative")
        if self.grpc_max_message_bytes <= 0:
            raise ValueError("grpc-max-message-bytes must be positive")
        if self.max_upload_bytes <= 0:
            raise ValueError("max-upload-bytes must be positive")
        if not self.ffmpeg_path.strip():
            raise ValueError("ffmpeg-path must be non-empty")
        if self.max_workers <= 0:
            raise ValueError("max-workers must be positive")
        if self.keepalive_seconds <= 0:
            raise ValueError("keepalive-seconds must be positive")
        if self.sse_failure_mode is not None:
            if self.sse_failure_mode not in SSE_FAILURE_MODES:
                raise ValueError("sse-failure-mode is invalid")


def parse_positive_int(raw_value: str, label: str) -> int:
    """Parse a positive integer from a string."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def parse_non_negative_float(raw_value: str, label: str) -> float:
    """Parse a non-negative float from a string."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def parse_sse_failure_mode(env: dict[str, str]) -> str | None:
    """Parse the optional SSE failure mode."""
    raw_value = env.get(SSE_FAILURE_MODE_ENV, "").strip()
    if not raw_value:
        return None
    return raw_value


def parse_bool(raw_value: str) -> bool:
    """Parse a boolean from a string."""
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def read_env_int(env: dict[str, str], key: str, label: str, fallback: int) -> int:
    """Read a positive integer from the environment."""
    raw_value = env.get(key, "").strip()
    if not raw_value:
        return fallback
    return parse_positive_int(raw_value, label)


def read_env_float(
    env: dict[str, str], key: str, label: str, fallback: float
) -> float:
    """Read a non-negative float from the environment."""
    raw_value = env.get(key, "").strip()
    if not raw_value:
        return fallback
    return parse_non_negative_float(raw_value, label)


def parse_allowed_origins(raw_value: str) -> tuple[tuple[str, ...], bool]:
    """Parse allowed origins from a comma-delimited string."""
    trimmed = raw_value.strip()
    if not trimmed:
        return tuple(), True
    if trimmed == "*":
        return tuple(), True
    values = tuple(value.strip() for value in trimmed.split(",") if value.strip())
    return values, False


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse backend CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="audio_to_text_backend.py", add_help=True
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--grpc-target", default=None)
    parser.add_argument("--grpc-timeout-seconds", type=float, default=None)
    parser.add_argument("--grpc-max-message-bytes", type=int, default=None)
    parser.add_argument("--grpc-auth-token", default=None)
    parser.add_argument("--grpc-use-tls", action="store_true")
    parser.add_argument("--max-upload-bytes", type=int, default=None)
    parser.add_argument("--allowed-origins", default=None)
    parser.add_argument("--ffmpeg-path", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--keepalive-seconds", type=float, default=None)
    return parser.parse_args(list(argv))


def configure_logging(env: dict[str, str]) -> None:
    """Configure logging from environment."""
    level_name = env.get(LOG_LEVEL_ENV, "INFO").strip().upper()
    level = logging.INFO
    if level_name == "DEBUG":
        level = logging.DEBUG
    elif level_name == "WARNING":
        level = logging.WARNING
    elif level_name == "ERROR":
        level = logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_config(args: argparse.Namespace, env: dict[str, str]) -> BackendConfig:
    """Load backend configuration from args and environment."""
    host = env.get(HOST_ENV, DEFAULT_HOST)
    if args.host:
        host = args.host
    port = read_env_int(env, PORT_ENV, "port", DEFAULT_PORT)
    if args.port is not None:
        port = args.port
    data_dir = Path(env.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))
    if args.data_dir:
        data_dir = Path(args.data_dir)
    grpc_target = env.get(GRPC_TARGET_ENV, DEFAULT_GRPC_TARGET)
    if args.grpc_target:
        grpc_target = args.grpc_target
    grpc_timeout = read_env_float(
        env, GRPC_TIMEOUT_ENV, "grpc-timeout-seconds", DEFAULT_GRPC_TIMEOUT_SECONDS
    )
    if args.grpc_timeout_seconds is not None:
        grpc_timeout = args.grpc_timeout_seconds
    grpc_max_message_bytes = read_env_int(
        env,
        GRPC_MAX_MESSAGE_BYTES_ENV,
        "grpc-max-message-bytes",
        DEFAULT_GRPC_MAX_MESSAGE_BYTES,
    )
    if args.grpc_max_message_bytes is not None:
        grpc_max_message_bytes = args.grpc_max_message_bytes
    grpc_auth_token = args.grpc_auth_token
    if grpc_auth_token is None:
        grpc_auth_token = env.get(GRPC_AUTH_TOKEN_ENV, "").strip() or None
    grpc_use_tls = parse_bool(env.get(GRPC_TLS_ENV, "")) or bool(args.grpc_use_tls)
    max_upload_bytes = read_env_int(
        env,
        MAX_UPLOAD_BYTES_ENV,
        "max-upload-bytes",
        DEFAULT_MAX_UPLOAD_BYTES,
    )
    if args.max_upload_bytes is not None:
        max_upload_bytes = args.max_upload_bytes
    allowed_raw = env.get(ALLOWED_ORIGINS_ENV, "").strip()
    if args.allowed_origins is not None:
        allowed_raw = args.allowed_origins
    allowed_origins, allow_any = parse_allowed_origins(allowed_raw)
    ffmpeg_path = env.get(FFMPEG_PATH_ENV, "ffmpeg")
    if args.ffmpeg_path is not None:
        ffmpeg_path = args.ffmpeg_path
    max_workers = read_env_int(
        env, MAX_WORKERS_ENV, "max-workers", DEFAULT_MAX_WORKERS
    )
    if args.max_workers is not None:
        max_workers = args.max_workers
    keepalive = read_env_float(
        env,
        KEEPALIVE_SECONDS_ENV,
        "keepalive-seconds",
        DEFAULT_KEEPALIVE_SECONDS,
    )
    if args.keepalive_seconds is not None:
        keepalive = args.keepalive_seconds
    sse_failure_mode = parse_sse_failure_mode(env)
    return BackendConfig(
        host=str(host),
        port=int(port),
        data_dir=data_dir,
        grpc_target=str(grpc_target),
        grpc_timeout_seconds=float(grpc_timeout),
        grpc_max_message_bytes=int(grpc_max_message_bytes),
        grpc_auth_token=grpc_auth_token,
        grpc_use_tls=bool(grpc_use_tls),
        max_upload_bytes=int(max_upload_bytes),
        allowed_origins=allowed_origins,
        allow_any_origin=allow_any,
        ffmpeg_path=str(ffmpeg_path),
        max_workers=int(max_workers),
        keepalive_seconds=float(keepalive),
        sse_failure_mode=sse_failure_mode,
    )


def validate_wav(path: Path) -> None:
    """Validate that the file is a readable WAV container."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            wav_file.getnchannels()
    except wave.Error as exc:
        raise BackendError(
            BACKEND_AUDIO_CODE, f"invalid wav: {exc}"
        ) from exc


def extract_audio_to_wav(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: str,
) -> None:
    """Extract audio into a WAV file."""
    if input_path.suffix.lower() == ".wav":
        shutil.copyfile(input_path, output_path)
        validate_wav(output_path)
        return
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        message = detail or "ffmpeg failed"
        raise BackendError(
            BACKEND_AUDIO_CODE, f"ffmpeg extraction failed: {message}"
        ) from exc
    except OSError as exc:
        raise BackendError(
            BACKEND_AUDIO_CODE, f"ffmpeg execution failed: {exc}"
        ) from exc
    validate_wav(output_path)


@dataclasses.dataclass
class GrpcAligner:
    """gRPC aligner client."""

    channel: grpc.Channel
    stub: audio_to_text_pb2_grpc.AudioToTextStub
    auth_token: str | None
    timeout_seconds: float

    def align(
        self,
        wav_path: Path,
        transcript: str,
        language: str,
        remove_punctuation: bool,
        audio_filename: str,
    ) -> str:
        """Align text to audio via gRPC."""
        punctuation_mode = (
            audio_to_text_pb2.PUNCTUATION_MODE_KEEP
            if not remove_punctuation
            else audio_to_text_pb2.PUNCTUATION_MODE_REMOVE
        )
        init = audio_to_text_pb2.AlignInit(
            transcript=transcript,
            language=language,
            punctuation=punctuation_mode,
            audio_filename=audio_filename,
        )

        def stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            yield audio_to_text_pb2.AlignChunk(init=init)
            with wav_path.open("rb") as handle:
                while True:
                    chunk = handle.read(65536)
                    if not chunk:
                        break
                    yield audio_to_text_pb2.AlignChunk(wav_chunk=chunk)

        metadata = []
        if self.auth_token:
            metadata.append(("authorization", f"Bearer {self.auth_token}"))
        try:
            response = self.stub.Align(
                stream(),
                timeout=self.timeout_seconds if self.timeout_seconds > 0 else None,
                metadata=metadata or None,
            )
        except grpc.RpcError as exc:
            detail = exc.details() or exc.code().name
            raise BackendError(
                BACKEND_ALIGN_CODE, f"grpc alignment failed: {detail}"
            ) from exc
        return response.srt


def build_grpc_aligner(config: BackendConfig) -> GrpcAligner:
    """Build a gRPC aligner client."""
    options = [
        ("grpc.max_receive_message_length", config.grpc_max_message_bytes),
        ("grpc.max_send_message_length", config.grpc_max_message_bytes),
    ]
    if config.grpc_use_tls:
        channel = grpc.secure_channel(
            config.grpc_target,
            grpc.ssl_channel_credentials(),
            options=options,
        )
    else:
        channel = grpc.insecure_channel(config.grpc_target, options=options)
    stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
    return GrpcAligner(
        channel=channel,
        stub=stub,
        auth_token=config.grpc_auth_token,
        timeout_seconds=config.grpc_timeout_seconds,
    )


def build_job_payload(job: audio_to_text.AlignmentJob) -> dict[str, object]:
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
        "client_job_id": job.job_input.client_job_id,
        "created_at": job.created_at,
        "started_at": job.result.started_at,
        "completed_at": job.result.completed_at,
    }


def process_job(
    store: audio_to_text.JobStore,
    job_id: str,
    aligner: GrpcAligner,
    config: BackendConfig,
) -> None:
    """Process a background alignment job."""
    job = store.get_job(job_id)
    assert job is not None
    job_input = job.job_input
    store.update_job(
        job_id,
        audio_to_text.JobStatus.RUNNING,
        message="Preparing input files",
        progress=0.05,
    )
    try:
        audio_path = Path(job_input.audio_path)
        text_path = Path(job_input.text_path)
        wav_path = audio_path.parent / "audio.wav"
        store.update_job(
            job_id,
            audio_to_text.JobStatus.RUNNING,
            message="Extracting audio",
            progress=0.2,
        )
        extract_audio_to_wav(audio_path, wav_path, config.ffmpeg_path)
        store.update_job(
            job_id,
            audio_to_text.JobStatus.RUNNING,
            message="Reading transcript text",
            progress=0.35,
        )
        transcript_text = audio_to_text.read_utf8_text_strict(str(text_path))
        store.update_job(
            job_id,
            audio_to_text.JobStatus.RUNNING,
            message="Aligning with backend",
            progress=0.6,
        )
        srt_content = aligner.align(
            wav_path=wav_path,
            transcript=transcript_text,
            language=job_input.language,
            remove_punctuation=job_input.remove_punctuation,
            audio_filename=job_input.audio_filename,
        )
        store.update_job(
            job_id,
            audio_to_text.JobStatus.RUNNING,
            message="Writing subtitle output",
            progress=0.9,
        )
        audio_to_text.write_srt_file(job_input.output_path, srt_content)
        store.update_job(
            job_id,
            audio_to_text.JobStatus.COMPLETED,
            message="Complete",
            output_srt=job_input.output_path,
            progress=1.0,
        )
    except (audio_to_text.AlignmentValidationError, audio_to_text.AlignmentPipelineError) as exc:
        store.update_job(
            job_id,
            audio_to_text.JobStatus.FAILED,
            f"{exc.code}: {exc}",
            progress=1.0,
        )
    except BackendError as exc:
        store.update_job(
            job_id,
            audio_to_text.JobStatus.FAILED,
            f"{exc.code}: {exc}",
            progress=1.0,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            audio_to_text.JobStatus.FAILED,
            f"audio_to_text_backend.unhandled_error: {str(exc).strip()}",
            progress=1.0,
        )


def serve(config: BackendConfig) -> None:
    """Run the backend HTTP server."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = audio_to_text.JobStore(
        root_dir=config.data_dir,
        clock=time.time,
        id_factory=lambda: uuid.uuid4().hex,
    )
    aligner = build_grpc_aligner(config)
    executor = futures.ThreadPoolExecutor(max_workers=config.max_workers)
    shutdown_event = threading.Event()

    class BackendHandler(BaseHTTPRequestHandler):
        """HTTP request handler for the backend service."""

        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.client_address[0], format % args)

        def send_cors_headers(self) -> None:
            origin = self.headers.get("Origin")
            if config.allow_any_origin:
                self.send_header("Access-Control-Allow-Origin", "*")
                return
            if not origin:
                return
            if origin in config.allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)

        def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_error_response(self, status: HTTPStatus, message: str) -> None:
            self.send_json(status, {"error": message})

        def send_sse_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_cors_headers()
            self.send_header(
                "Content-Type", "text/event-stream; charset=utf-8"
            )
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.wfile.flush()
            self.close_connection = False

        def send_sse_event(
            self,
            payload: dict[str, object],
            event_kind: str | None = None,
            event_id: int | None = None,
        ) -> bool:
            lines: list[str] = []
            if event_id is not None:
                lines.append(f"id: {event_id}")
            lines.append(f"data: {json.dumps(payload)}")
            body = f"{'\n'.join(lines)}\n\n".encode("utf-8")
            try:
                if event_kind is not None and event_kind == config.sse_failure_mode:
                    raise OSError("sse event failure")
                self.wfile.write(body)
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                return False
            return True

        def send_sse_keepalive(
            self,
            event_kind: str | None = None,
            event_id: int | None = None,
        ) -> bool:
            payload = {"type": "keepalive"}
            return self.send_sse_event(
                payload,
                event_kind=event_kind,
                event_id=event_id,
            )

        def parse_job_id(self, path: str, suffix: str | None = None) -> str | None:
            prefix = "/api/jobs/"
            if not path.startswith(prefix):
                return None
            job_fragment = path[len(prefix):]
            if suffix:
                if not job_fragment.endswith(suffix):
                    return None
                job_fragment = job_fragment[: -len(suffix)]
            return job_fragment or None

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if parsed.path == "/api/jobs":
                jobs = store.list_jobs()
                payload = {"jobs": [build_job_payload(job) for job in jobs]}
                self.send_json(HTTPStatus.OK, payload)
                return
            if parsed.path == "/api/jobs/events":
                self.send_sse_headers()
                change_id = store.change_id
                jobs = store.list_jobs()
                if not self.send_sse_event(
                    {
                        "type": "snapshot",
                        "change_id": change_id,
                        "jobs": [build_job_payload(job) for job in jobs],
                    },
                    event_kind=SSE_FAILURE_SNAPSHOT,
                    event_id=change_id,
                ):
                    return
                while not shutdown_event.is_set():
                    next_change = store.wait_for_change(
                        change_id, timeout=config.keepalive_seconds
                    )
                    if next_change == change_id:
                        if not self.send_sse_keepalive(
                            event_kind=SSE_FAILURE_KEEPALIVE,
                        ):
                            return
                        continue
                    change_id = next_change
                    jobs = store.list_jobs()
                    if not self.send_sse_event(
                        {
                            "type": "snapshot",
                            "change_id": change_id,
                            "jobs": [build_job_payload(job) for job in jobs],
                        },
                        event_kind=SSE_FAILURE_UPDATE,
                        event_id=change_id,
                    ):
                        return
                return
            job_id = self.parse_job_id(parsed.path, suffix="/srt")
            if job_id:
                job = store.get_job(job_id)
                if job is None:
                    self.send_error_response(
                        HTTPStatus.NOT_FOUND,
                        f"{BACKEND_JOB_NOT_FOUND_CODE}: job not found",
                    )
                    return
                if job.result.status != audio_to_text.JobStatus.COMPLETED:
                    self.send_error_response(
                        HTTPStatus.BAD_REQUEST,
                        f"{BACKEND_JOB_NOT_READY_CODE}: job is not complete",
                    )
                    return
                output_path = Path(job.result.output_srt or "")
                if not output_path.exists():
                    self.send_error_response(
                        HTTPStatus.NOT_FOUND,
                        f"{BACKEND_OUTPUT_NOT_FOUND_CODE}: output not found",
                    )
                    return
                try:
                    payload = output_path.read_bytes()
                except OSError:
                    self.send_error_response(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"{BACKEND_OUTPUT_READ_CODE}: output read failed",
                    )
                    return
                filename = audio_to_text.srt_filename_from_audio_filename(
                    job.job_input.audio_filename
                )
                self.send_response(HTTPStatus.OK)
                self.send_cors_headers()
                self.send_header("Content-Type", "application/x-subrip")
                self.send_header(
                    "Content-Disposition",
                    audio_to_text.content_disposition_attachment(filename),
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error_response(
                HTTPStatus.NOT_FOUND,
                f"{BACKEND_NOT_FOUND_CODE}: not found",
            )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/jobs":
                self.send_error_response(
                    HTTPStatus.NOT_FOUND,
                    f"{BACKEND_NOT_FOUND_CODE}: not found",
                )
                return
            try:
                audio_to_text.validate_multipart_content_type(
                    self.headers.get("Content-Type", "")
                )
                content_length = audio_to_text.parse_content_length(self.headers)
                if content_length > config.max_upload_bytes:
                    raise BackendError(
                        BACKEND_UPLOAD_CODE, "upload exceeds max size"
                    )
                body = audio_to_text.read_request_body(self.rfile, content_length)
                defaults = audio_to_text.UiDefaults(
                    language=DEFAULT_LANGUAGE,
                    remove_punctuation=True,
                )
                upload = audio_to_text.parse_upload_form(
                    self.headers.get("Content-Type", ""),
                    body,
                    defaults,
                )
            except (audio_to_text.AlignmentValidationError, BackendError) as exc:
                code = exc.code if isinstance(exc, BackendError) else exc.code
                self.send_error_response(
                    HTTPStatus.BAD_REQUEST, f"{code}: {exc}"
                )
                return
            job_id = store.new_job_id()
            job_dir = store.job_dir(job_id)
            audio_suffix = Path(upload.audio.filename).suffix or ".bin"
            text_suffix = Path(upload.text.filename).suffix or ".txt"
            raw_audio_path = job_dir / f"upload{audio_suffix}"
            text_path = job_dir / f"text{text_suffix}"
            output_name = audio_to_text.srt_filename_from_audio_filename(
                upload.audio.filename
            )
            output_path = job_dir / output_name
            job_input = audio_to_text.AlignmentJobInput(
                audio_filename=upload.audio.filename,
                text_filename=upload.text.filename,
                language=upload.language,
                remove_punctuation=upload.remove_punctuation,
                audio_path=str(raw_audio_path),
                text_path=str(text_path),
                output_path=str(output_path),
                client_job_id=upload.client_job_id,
            )
            try:
                job_dir.mkdir(parents=True, exist_ok=True)
                job = store.create_job(job_id, job_input)
            except audio_to_text.AlignmentPipelineError as exc:
                self.send_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{exc.code}: {exc}",
                )
                return
            except OSError as exc:
                self.send_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{BACKEND_JOB_SETUP_CODE}: job setup failed: {exc}",
                )
                return
            try:
                raw_audio_path.write_bytes(upload.audio.payload)
                text_path.write_bytes(upload.text.payload)
            except OSError as exc:
                store.update_job(
                    job.job_id,
                    audio_to_text.JobStatus.FAILED,
                    f"{BACKEND_UPLOAD_WRITE_CODE}: upload write failed",
                    progress=1.0,
                )
                self.send_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{BACKEND_UPLOAD_WRITE_CODE}: upload write failed: {exc}",
                )
                return
            executor.submit(process_job, store, job.job_id, aligner, config)
            self.send_json(HTTPStatus.OK, build_job_payload(job))

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            job_id = self.parse_job_id(parsed.path)
            if not job_id:
                self.send_error_response(
                    HTTPStatus.NOT_FOUND,
                    f"{BACKEND_NOT_FOUND_CODE}: not found",
                )
                return
            try:
                job = store.delete_finished_job(job_id)
            except audio_to_text.AlignmentPipelineError as exc:
                self.send_error_response(
                    HTTPStatus.BAD_REQUEST, f"{exc.code}: {exc}"
                )
                return
            self.send_json(HTTPStatus.OK, build_job_payload(job))

    server = ThreadingHTTPServer((config.host, config.port), BackendHandler)
    LOGGER.info("audio_to_text_backend.server.started address=%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("audio_to_text_backend.server.shutdown: received interrupt")
    finally:
        shutdown_event.set()
        server.server_close()
        executor.shutdown(wait=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the backend server."""
    env = dict(os.environ)
    configure_logging(env)
    try:
        args = parse_args(list(argv) if argv is not None else sys.argv[1:])
        config = load_config(args, env)
    except ValueError as exc:
        LOGGER.error("%s: %s", BACKEND_CONFIG_CODE, exc)
        return 1
    serve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
