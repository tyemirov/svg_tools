"""audio_to_text gRPC server (forced alignment for WAV + transcript)."""

from __future__ import annotations

import argparse
import math
import dataclasses
import functools
import logging
import os
import tempfile
import threading
import time
import uuid
import wave
import sys
from concurrent import futures
from pathlib import Path
from typing import Callable, Iterator, Sequence

import grpc
from grpc_health.v1 import health as grpc_health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

import audio_to_text
from audio_to_text_grpc import audio_to_text_grpc_pb2
from audio_to_text_grpc import audio_to_text_grpc_pb2_grpc

LOGGER = logging.getLogger("audio_to_text_grpc")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50051
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_AUDIO_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TRANSCRIPT_CHARS = 200_000
DEFAULT_MAX_TRANSCRIPT_WORDS = 50_000
DEFAULT_MAX_WORKERS = 4
DEFAULT_ALIGNMENT_TIMEOUT_SECONDS = 300.0

TEST_MODE_ENV = "AUDIO_TO_TEXT_GRPC_TEST_MODE"
TEST_DELAY_ENV = "AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS"
TEST_CRASH_ENV = "AUDIO_TO_TEXT_GRPC_TEST_CRASH"
AUTH_TOKEN_ENV = "AUDIO_TO_TEXT_GRPC_AUTH_TOKEN"
MAX_AUDIO_BYTES_ENV = "AUDIO_TO_TEXT_GRPC_MAX_AUDIO_BYTES"
MAX_TRANSCRIPT_CHARS_ENV = "AUDIO_TO_TEXT_GRPC_MAX_TRANSCRIPT_CHARS"
MAX_TRANSCRIPT_WORDS_ENV = "AUDIO_TO_TEXT_GRPC_MAX_TRANSCRIPT_WORDS"
MAX_INFLIGHT_ENV = "AUDIO_TO_TEXT_GRPC_MAX_INFLIGHT"
ALIGNMENT_TIMEOUT_ENV = "AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS"
TLS_CERT_ENV = "AUDIO_TO_TEXT_GRPC_TLS_CERT"
TLS_KEY_ENV = "AUDIO_TO_TEXT_GRPC_TLS_KEY"
LOG_LEVEL_ENV = "AUDIO_TO_TEXT_GRPC_LOG_LEVEL"

INVALID_ARGUMENT_CODE = "audio_to_text_grpc.input.invalid_argument"
MISSING_INIT_CODE = "audio_to_text_grpc.input.missing_init"
INVALID_WAV_CODE = "audio_to_text_grpc.input.invalid_wav"
INPUT_AUDIO_TOO_LARGE_CODE = "audio_to_text_grpc.input.audio_too_large"
INPUT_TEXT_TOO_LARGE_CODE = "audio_to_text_grpc.input.text_too_large"
INPUT_TEXT_TOO_LONG_CODE = "audio_to_text_grpc.input.text_too_long"
AUTH_REQUIRED_CODE = "audio_to_text_grpc.auth.required"
INFLIGHT_LIMIT_CODE = "audio_to_text_grpc.request.inflight_limit"
DEADLINE_EXCEEDED_CODE = "audio_to_text_grpc.request.deadline_exceeded"
REQUEST_CANCELLED_CODE = "audio_to_text_grpc.request.cancelled"
ALIGNMENT_TIMEOUT_CODE = "audio_to_text_grpc.align.timeout"
ALIGNMENT_FAILED_CODE = "audio_to_text_grpc.align.failed"


@dataclasses.dataclass(frozen=True)
class AlignRequest:
    """Validated alignment request."""

    wav_path: Path
    transcript: str
    language: str
    remove_punctuation: bool
    audio_filename: str
    audio_bytes: int


AlignmentRunner = Callable[[AlignRequest], list[audio_to_text.AlignedWord]]


@dataclasses.dataclass(frozen=True)
class MetricsSnapshot:
    """Snapshot of server metrics."""

    requests_total: int
    requests_succeeded: int
    requests_failed: int
    inflight: int
    bytes_received: int
    uptime_seconds: float
    average_latency_seconds: float
    max_latency_seconds: float


@dataclasses.dataclass
class MetricsRegistry:
    """Thread-safe metrics registry."""

    started_at: float
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    inflight: int = 0
    bytes_received: int = 0
    total_latency_seconds: float = 0.0
    max_latency_seconds: float = 0.0

    def record_start(self) -> None:
        """Record a new request start."""
        with self.lock:
            self.requests_total += 1
            self.inflight += 1

    def record_bytes(self, audio_bytes: int) -> None:
        """Record bytes received for a request."""
        with self.lock:
            self.bytes_received += audio_bytes

    def record_finish(self, success: bool, latency_seconds: float) -> None:
        """Record a request completion."""
        with self.lock:
            self.inflight -= 1
            if success:
                self.requests_succeeded += 1
            else:
                self.requests_failed += 1
            self.total_latency_seconds += latency_seconds
            if latency_seconds > self.max_latency_seconds:
                self.max_latency_seconds = latency_seconds

    def snapshot(self, clock: Callable[[], float]) -> MetricsSnapshot:
        """Return a consistent snapshot of metrics."""
        with self.lock:
            uptime = max(0.0, clock() - self.started_at)
            average_latency = (
                self.total_latency_seconds / self.requests_total
                if self.requests_total
                else 0.0
            )
            return MetricsSnapshot(
                requests_total=self.requests_total,
                requests_succeeded=self.requests_succeeded,
                requests_failed=self.requests_failed,
                inflight=self.inflight,
                bytes_received=self.bytes_received,
                uptime_seconds=uptime,
                average_latency_seconds=average_latency,
                max_latency_seconds=self.max_latency_seconds,
            )


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    """Server configuration."""

    host: str
    port: int
    max_message_bytes: int
    max_workers: int
    max_audio_bytes: int
    max_transcript_chars: int
    max_transcript_words: int
    max_inflight: int
    alignment_timeout_seconds: float
    auth_token: str | None
    test_mode: bool
    test_delay_seconds: float
    test_crash: bool
    tls_cert_path: Path | None
    tls_key_path: Path | None


class GrpcRequestError(RuntimeError):
    """Request-level error with gRPC status."""

    def __init__(self, status: grpc.StatusCode, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI args for the gRPC server."""
    parser = argparse.ArgumentParser(prog="audio_to_text_grpc.py", add_help=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-message-bytes", type=int, default=DEFAULT_MAX_MESSAGE_BYTES)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-audio-bytes", type=int, default=None)
    parser.add_argument("--max-transcript-chars", type=int, default=None)
    parser.add_argument("--max-transcript-words", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--alignment-timeout-seconds", type=float, default=None)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--tls-cert", default=None)
    parser.add_argument("--tls-key", default=None)
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


def parse_positive_int(raw_value: str, field_name: str) -> int:
    """Parse a positive integer from a string."""
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def parse_non_negative_float(raw_value: str, field_name: str) -> float:
    """Parse a non-negative float from a string."""
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def parse_bool(raw_value: str) -> bool:
    """Parse a boolean from a string."""
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def read_env_int(
    env: dict[str, str], key: str, field_name: str, fallback: int | None
) -> int | None:
    """Read an integer from the environment."""
    raw_value = env.get(key, "").strip()
    if not raw_value:
        return fallback
    return parse_positive_int(raw_value, field_name)


def read_env_float(
    env: dict[str, str], key: str, field_name: str, fallback: float | None
) -> float | None:
    """Read a float from the environment."""
    raw_value = env.get(key, "").strip()
    if not raw_value:
        return fallback
    return parse_non_negative_float(raw_value, field_name)


def resolve_optional_path(raw_value: str | None, field_name: str) -> Path | None:
    """Resolve a filesystem path from an optional value."""
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed:
        raise ValueError(f"{field_name} must be non-empty when set")
    return Path(trimmed)


def is_test_mode(env: dict[str, str]) -> bool:
    """Return True when test-mode is enabled."""
    raw = env.get(TEST_MODE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_default_language(transcript: str) -> str:
    """Detect a default language from the transcript text."""
    if any("\u0400" <= character <= "\u04ff" for character in transcript):
        return "ru"
    return "en"


def resolve_language(raw_value: str, transcript: str) -> str:
    """Resolve and validate a language code."""
    default_language = detect_default_language(transcript)
    try:
        return audio_to_text.normalize_language_value(raw_value, default_language)
    except audio_to_text.AlignmentValidationError as exc:
        raise GrpcRequestError(grpc.StatusCode.INVALID_ARGUMENT, exc.code, str(exc))


def resolve_remove_punctuation(
    punctuation_mode: audio_to_text_grpc_pb2.PunctuationMode.ValueType,
) -> bool:
    """Map punctuation mode to a boolean removal flag."""
    if punctuation_mode == audio_to_text_grpc_pb2.PUNCTUATION_MODE_KEEP:
        return False
    return True


def validate_wav(path: Path) -> None:
    """Validate that the file is a readable WAV container."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            wav_file.getnchannels()
    except wave.Error as exc:
        raise GrpcRequestError(
            grpc.StatusCode.INVALID_ARGUMENT, INVALID_WAV_CODE, f"invalid wav: {exc}"
        ) from exc


def load_config(args: argparse.Namespace, env: dict[str, str]) -> ServerConfig:
    """Load server configuration from args and environment."""
    max_audio_bytes = read_env_int(
        env, MAX_AUDIO_BYTES_ENV, "max-audio-bytes", DEFAULT_MAX_AUDIO_BYTES
    )
    if args.max_audio_bytes is not None:
        max_audio_bytes = args.max_audio_bytes
    if max_audio_bytes is None or max_audio_bytes <= 0:
        raise ValueError("max-audio-bytes must be positive")

    max_transcript_chars = read_env_int(
        env,
        MAX_TRANSCRIPT_CHARS_ENV,
        "max-transcript-chars",
        DEFAULT_MAX_TRANSCRIPT_CHARS,
    )
    if args.max_transcript_chars is not None:
        max_transcript_chars = args.max_transcript_chars
    if max_transcript_chars is None or max_transcript_chars <= 0:
        raise ValueError("max-transcript-chars must be positive")

    max_transcript_words = read_env_int(
        env,
        MAX_TRANSCRIPT_WORDS_ENV,
        "max-transcript-words",
        DEFAULT_MAX_TRANSCRIPT_WORDS,
    )
    if args.max_transcript_words is not None:
        max_transcript_words = args.max_transcript_words
    if max_transcript_words is None or max_transcript_words <= 0:
        raise ValueError("max-transcript-words must be positive")

    max_inflight = read_env_int(env, MAX_INFLIGHT_ENV, "max-inflight", None)
    if args.max_inflight is not None:
        max_inflight = args.max_inflight
    if max_inflight is None:
        max_inflight = args.max_workers
    if max_inflight <= 0:
        raise ValueError("max-inflight must be positive")

    alignment_timeout = read_env_float(
        env,
        ALIGNMENT_TIMEOUT_ENV,
        "alignment-timeout-seconds",
        DEFAULT_ALIGNMENT_TIMEOUT_SECONDS,
    )
    if args.alignment_timeout_seconds is not None:
        alignment_timeout = args.alignment_timeout_seconds
    if alignment_timeout is None or alignment_timeout < 0:
        raise ValueError("alignment-timeout-seconds must be non-negative")

    auth_token = args.auth_token
    if auth_token is None:
        auth_token = env.get(AUTH_TOKEN_ENV, "").strip() or None

    test_mode = is_test_mode(env)
    test_delay_ms = read_env_float(env, TEST_DELAY_ENV, "test-delay-ms", 0.0) or 0.0
    test_delay_seconds = test_delay_ms / 1000.0
    test_crash = parse_bool(env.get(TEST_CRASH_ENV, ""))
    if test_crash and not test_mode:
        raise ValueError("test-crash requires test mode")

    tls_cert_path = resolve_optional_path(
        args.tls_cert or env.get(TLS_CERT_ENV), "tls-cert"
    )
    tls_key_path = resolve_optional_path(
        args.tls_key or env.get(TLS_KEY_ENV), "tls-key"
    )
    if (tls_cert_path is None) ^ (tls_key_path is None):
        raise ValueError("tls-cert and tls-key must be provided together")

    return ServerConfig(
        host=str(args.host),
        port=int(args.port),
        max_message_bytes=int(args.max_message_bytes),
        max_workers=int(args.max_workers),
        max_audio_bytes=int(max_audio_bytes),
        max_transcript_chars=int(max_transcript_chars),
        max_transcript_words=int(max_transcript_words),
        max_inflight=int(max_inflight),
        alignment_timeout_seconds=float(alignment_timeout),
        auth_token=auth_token,
        test_mode=test_mode,
        test_delay_seconds=float(test_delay_seconds),
        test_crash=test_crash,
        tls_cert_path=tls_cert_path,
        tls_key_path=tls_key_path,
    )


def build_server_credentials(config: ServerConfig) -> grpc.ServerCredentials | None:
    """Build TLS credentials when configured."""
    if config.tls_cert_path is None or config.tls_key_path is None:
        return None
    certificate_bytes = config.tls_cert_path.read_bytes()
    key_bytes = config.tls_key_path.read_bytes()
    return grpc.ssl_server_credentials(((key_bytes, certificate_bytes),))


def resolve_time_remaining(
    context: grpc.ServicerContext, test_mode: bool
) -> float | None:
    """Resolve request time remaining with optional test overrides."""
    remaining = context.time_remaining()
    if not test_mode:
        return remaining
    metadata: dict[str, str] = {}
    for key, value in context.invocation_metadata():
        metadata[str(key).lower()] = str(value)
    override = metadata.get("x-test-remaining")
    if override is None:
        return remaining
    token = override.strip().lower()
    if token == "inf":
        return math.inf
    if token == "none":
        return None
    try:
        return float(token)
    except ValueError:
        return remaining


def ensure_deadline(context: grpc.ServicerContext, test_mode: bool = False) -> None:
    """Ensure the request deadline has not expired."""
    if not context.is_active():
        raise GrpcRequestError(
            grpc.StatusCode.CANCELLED,
            REQUEST_CANCELLED_CODE,
            "request cancelled",
        )
    remaining = resolve_time_remaining(context, test_mode)
    if remaining is not None and remaining <= 0:
        raise GrpcRequestError(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            DEADLINE_EXCEEDED_CODE,
            "request deadline exceeded",
        )


def collect_request(
    request_iterator: Iterator[audio_to_text_grpc_pb2.AlignChunk],
    config: ServerConfig,
    context: grpc.ServicerContext,
    temp_root: Path,
) -> AlignRequest:
    """Collect a streaming request into a temporary WAV file."""
    ensure_deadline(context, config.test_mode)
    first = next(request_iterator, None)
    if first is None or first.WhichOneof("payload") != "init":
        raise GrpcRequestError(
            grpc.StatusCode.INVALID_ARGUMENT,
            MISSING_INIT_CODE,
            "first message must contain init payload",
        )

    init = first.init
    transcript_raw = init.transcript
    if not transcript_raw.strip():
        raise GrpcRequestError(
            grpc.StatusCode.INVALID_ARGUMENT,
            INVALID_ARGUMENT_CODE,
            "transcript is required",
        )
    remove_punctuation = resolve_remove_punctuation(init.punctuation)
    try:
        normalized_transcript = audio_to_text.normalize_transcript_for_alignment(
            transcript_raw,
            "input.txt",
            remove_punctuation=remove_punctuation,
        )
    except audio_to_text.AlignmentValidationError as exc:
        raise GrpcRequestError(grpc.StatusCode.INVALID_ARGUMENT, exc.code, str(exc))

    if len(normalized_transcript) > config.max_transcript_chars:
        raise GrpcRequestError(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            INPUT_TEXT_TOO_LARGE_CODE,
            "transcript exceeds max length",
        )

    normalized_word_count = len(normalized_transcript.split())
    if normalized_word_count > config.max_transcript_words:
        raise GrpcRequestError(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            INPUT_TEXT_TOO_LONG_CODE,
            "transcript exceeds max word count",
        )

    language = resolve_language(init.language, normalized_transcript)

    wav_path = temp_root / "input.wav"
    audio_bytes = 0
    with wav_path.open("wb") as wav_file:
        for message in request_iterator:
            ensure_deadline(context)
            if message.WhichOneof("payload") != "wav_chunk":
                raise GrpcRequestError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    INVALID_ARGUMENT_CODE,
                    "unexpected message in stream",
                )
            chunk = bytes(message.wav_chunk)
            if not chunk:
                continue
            audio_bytes += len(chunk)
            if audio_bytes > config.max_audio_bytes:
                raise GrpcRequestError(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    INPUT_AUDIO_TOO_LARGE_CODE,
                    "audio exceeds max bytes",
                )
            wav_file.write(chunk)

    if audio_bytes <= 0:
        raise GrpcRequestError(
            grpc.StatusCode.INVALID_ARGUMENT,
            INVALID_WAV_CODE,
            "audio stream contained no bytes",
        )

    validate_wav(wav_path)
    return AlignRequest(
        wav_path=wav_path,
        transcript=normalized_transcript,
        language=language,
        remove_punctuation=remove_punctuation,
        audio_filename=init.audio_filename.strip(),
        audio_bytes=audio_bytes,
    )


def align_in_test_mode(
    request: AlignRequest,
    delay_seconds: float,
    sleep: Callable[[float], None],
    crash: bool,
) -> list[audio_to_text.AlignedWord]:
    """Return deterministic fake alignment for integration tests."""
    if crash:
        raise RuntimeError("test alignment crash")
    if delay_seconds > 0:
        sleep(delay_seconds)
    aligned: list[audio_to_text.AlignedWord] = []
    cursor = 0.0
    step = 0.25
    for token in request.transcript.split():
        aligned.append(
            audio_to_text.AlignedWord(
                text=token,
                start_seconds=cursor,
                end_seconds=cursor + step,
            )
        )
        cursor += step
    return aligned


def align_in_process(request: AlignRequest) -> list[audio_to_text.AlignedWord]:
    """Run whisperx alignment in-process."""
    device = audio_to_text.resolve_device()
    return audio_to_text.align_words(
        str(request.wav_path),
        request.transcript,
        request.language,
        device,
        remove_punctuation=request.remove_punctuation,
    )


def build_alignment_runner(
    config: ServerConfig,
    sleep: Callable[[float], None],
) -> AlignmentRunner:
    """Build an alignment runner for the server."""
    if config.test_mode:
        return functools.partial(
            align_in_test_mode,
            delay_seconds=config.test_delay_seconds,
            sleep=sleep,
            crash=config.test_crash,
        )
    return align_in_process


class AudioToTextService(audio_to_text_grpc_pb2_grpc.AudioToTextServicer):
    """gRPC service implementation."""

    def __init__(
        self,
        config: ServerConfig,
        metrics: MetricsRegistry,
        clock: Callable[[], float],
        alignment_runner: AlignmentRunner,
        alignment_executor: futures.Executor,
    ) -> None:
        self._config = config
        self._metrics = metrics
        self._clock = clock
        self._alignment_runner = alignment_runner
        self._alignment_executor = alignment_executor
        self._inflight_semaphore = threading.BoundedSemaphore(config.max_inflight)

    def Align(
        self,
        request_iterator: Iterator[audio_to_text_grpc_pb2.AlignChunk],
        context: grpc.ServicerContext,
    ) -> audio_to_text_grpc_pb2.AlignResponse:
        """Align a transcript to a streamed WAV."""
        if not self._inflight_semaphore.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"{INFLIGHT_LIMIT_CODE}: too many concurrent requests",
            )

        request_id = uuid.uuid4().hex
        start_time = self._clock()
        success = False
        try:
            self._metrics.record_start()
            self._authorize(context)
            with tempfile.TemporaryDirectory(prefix="audio_to_text_grpc_") as temp_dir:
                temp_root = Path(temp_dir)
                request = collect_request(
                    request_iterator, self._config, context, temp_root
                )
                self._metrics.record_bytes(request.audio_bytes)
                LOGGER.info(
                    "audio_to_text_grpc.request.started id=%s bytes=%s language=%s",
                    request_id,
                    request.audio_bytes,
                    request.language,
                )
                ensure_deadline(context, self._config.test_mode)
                aligned_words = self._run_alignment(request, context)
                ensure_deadline(context, self._config.test_mode)
                srt_text = audio_to_text.build_srt(aligned_words)
            response_words = [
                audio_to_text_grpc_pb2.AlignedWord(
                    text=word.text,
                    start_seconds=float(word.start_seconds),
                    end_seconds=float(word.end_seconds),
                )
                for word in aligned_words
            ]
            success = True
            return audio_to_text_grpc_pb2.AlignResponse(
                words=response_words,
                srt=srt_text,
                audio_filename=request.audio_filename,
            )
        except GrpcRequestError as exc:
            LOGGER.warning(
                "audio_to_text_grpc.request.failed id=%s code=%s status=%s",
                request_id,
                exc.code,
                exc.status.name,
            )
            context.abort(exc.status, f"{exc.code}: {exc}")
        except audio_to_text.AlignmentPipelineError as exc:
            LOGGER.error(
                "audio_to_text_grpc.request.error id=%s code=%s",
                request_id,
                exc.code,
            )
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"{exc.code}: {exc}")
        except Exception as exc:
            LOGGER.exception("audio_to_text_grpc.request.crash id=%s", request_id)
            context.abort(grpc.StatusCode.INTERNAL, f"{ALIGNMENT_FAILED_CODE}: {exc}")
        finally:
            latency = max(0.0, self._clock() - start_time)
            self._metrics.record_finish(success, latency)
            self._inflight_semaphore.release()

    def GetStats(
        self,
        request: audio_to_text_grpc_pb2.StatsRequest,
        context: grpc.ServicerContext,
    ) -> audio_to_text_grpc_pb2.StatsResponse:
        """Return a snapshot of metrics."""
        self._authorize(context)
        snapshot = self._metrics.snapshot(self._clock)
        return audio_to_text_grpc_pb2.StatsResponse(
            requests_total=snapshot.requests_total,
            requests_succeeded=snapshot.requests_succeeded,
            requests_failed=snapshot.requests_failed,
            inflight=snapshot.inflight,
            bytes_received=snapshot.bytes_received,
            uptime_seconds=snapshot.uptime_seconds,
            average_latency_seconds=snapshot.average_latency_seconds,
            max_latency_seconds=snapshot.max_latency_seconds,
        )

    def _authorize(self, context: grpc.ServicerContext) -> None:
        """Authorize the request when a token is configured."""
        if self._config.auth_token is None:
            return
        metadata: dict[str, str] = {}
        for key, value in context.invocation_metadata():
            metadata[str(key).lower()] = str(value)
        auth_header = metadata.get("authorization", "")
        token_value = ""
        if auth_header.lower().startswith("bearer "):
            token_value = auth_header.split(" ", 1)[1].strip()
        if not token_value:
            token_value = metadata.get("x-api-key", "")
        if token_value != self._config.auth_token:
            raise GrpcRequestError(
                grpc.StatusCode.UNAUTHENTICATED,
                AUTH_REQUIRED_CODE,
                "authorization token is required",
            )

    def _run_alignment(
        self, request: AlignRequest, context: grpc.ServicerContext
    ) -> list[audio_to_text.AlignedWord]:
        """Run alignment with timeout handling."""
        return self._run_with_timeout(
            lambda: self._alignment_runner(request),
            context,
        )

    def _run_with_timeout(
        self,
        task: Callable[[], list[audio_to_text.AlignedWord]],
        context: grpc.ServicerContext,
    ) -> list[audio_to_text.AlignedWord]:
        """Run an alignment task with timeout handling."""
        timeout_seconds = self._effective_timeout(context)
        future = self._alignment_executor.submit(task)
        try:
            if timeout_seconds is None:
                return future.result()
            return future.result(timeout=timeout_seconds)
        except futures.TimeoutError as exc:
            future.cancel()
            raise GrpcRequestError(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                ALIGNMENT_TIMEOUT_CODE,
                "alignment timed out",
            ) from exc

    def _effective_timeout(
        self, context: grpc.ServicerContext
    ) -> float | None:
        """Compute the effective alignment timeout."""
        remaining = resolve_time_remaining(context, self._config.test_mode)
        if remaining is not None and math.isinf(remaining):
            remaining = None
        if self._config.alignment_timeout_seconds <= 0:
            return remaining
        if remaining is None:
            return self._config.alignment_timeout_seconds
        return min(self._config.alignment_timeout_seconds, remaining)


def serve(
    config: ServerConfig,
    metrics: MetricsRegistry,
    clock: Callable[[], float],
    alignment_runner: AlignmentRunner,
) -> None:
    """Run the gRPC server."""
    alignment_executor = futures.ThreadPoolExecutor(max_workers=config.max_inflight)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        options=[
            ("grpc.max_receive_message_length", config.max_message_bytes),
            ("grpc.max_send_message_length", config.max_message_bytes),
        ],
    )
    service = AudioToTextService(
        config=config,
        metrics=metrics,
        clock=clock,
        alignment_runner=alignment_runner,
        alignment_executor=alignment_executor,
    )
    audio_to_text_grpc_pb2_grpc.add_AudioToTextServicer_to_server(service, server)
    health_service = grpc_health.HealthServicer()
    health_service.set(
        "svg_tools.audio_to_text.v1.AudioToText",
        health_pb2.HealthCheckResponse.SERVING,
    )
    health_service.set("", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)

    credentials = build_server_credentials(config)
    address = f"{config.host}:{config.port}"
    if credentials is None:
        server.add_insecure_port(address)
    else:
        server.add_secure_port(address, credentials)

    server.start()
    LOGGER.info("audio_to_text_grpc.server.started address=%s", address)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("audio_to_text_grpc.server.shutdown: received interrupt")
    finally:
        server.stop(grace=None)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the gRPC server."""
    env = dict(os.environ)
    configure_logging(env)
    try:
        args = parse_args(list(argv) if argv is not None else sys.argv[1:])
        config = load_config(args, env)
    except ValueError as exc:
        LOGGER.error("audio_to_text_grpc.config.invalid: %s", exc)
        return 1

    metrics = MetricsRegistry(started_at=time.monotonic())
    alignment_runner = build_alignment_runner(config=config, sleep=time.sleep)
    serve(
        config=config,
        metrics=metrics,
        clock=time.monotonic,
        alignment_runner=alignment_runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
