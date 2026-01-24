"""audio_to_text gRPC server (forced alignment for WAV + transcript)."""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import tempfile
import unicodedata
import wave
from concurrent import futures
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import grpc

from audio_to_text_grpc import audio_to_text_grpc_pb2
from audio_to_text_grpc import audio_to_text_grpc_pb2_grpc

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50051
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_WORKERS = 4

TEST_MODE_ENV = "AUDIO_TO_TEXT_GRPC_TEST_MODE"

INVALID_ARGUMENT_CODE = "audio_to_text_grpc.input.invalid_argument"
MISSING_INIT_CODE = "audio_to_text_grpc.input.missing_init"
INVALID_WAV_CODE = "audio_to_text_grpc.input.invalid_wav"
ALIGNMENT_FAILED_CODE = "audio_to_text_grpc.align.failed"

SRT_TIME_RANGE_PATTERN = re.compile(
    r"^(?P<start>\\d{2}:\\d{2}:\\d{2},\\d{3})\\s*-->\\s*(?P<end>\\d{2}:\\d{2}:\\d{2},\\d{3})$"
)


@dataclasses.dataclass(frozen=True)
class AlignRequest:
    """Validated alignment request."""

    wav_path: Path
    transcript: str
    language: str
    punctuation_mode: audio_to_text_grpc_pb2.PunctuationMode.ValueType
    audio_filename: str


@dataclasses.dataclass(frozen=True)
class AlignedWord:
    """Single aligned word with timestamps."""

    text: str
    start_seconds: float
    end_seconds: float


class GrpcValidationError(ValueError):
    """Validation error for gRPC input."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI args for the gRPC server."""
    parser = argparse.ArgumentParser(prog="audio_to_text_grpc.py", add_help=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-message-bytes", type=int, default=DEFAULT_MAX_MESSAGE_BYTES)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args(list(argv))


def is_test_mode() -> bool:
    """Return True when test-mode is enabled."""
    raw = os.environ.get(TEST_MODE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_language(raw_value: str, transcript: str) -> str:
    """Normalize an alignment language code, defaulting by transcript heuristic."""
    normalized = raw_value.strip().lower()
    if normalized:
        return normalized
    if any("\u0400" <= character <= "\u04ff" for character in transcript):
        return "ru"
    return "en"


def remove_punctuation_from_transcript(text_value: str) -> str:
    """Replace punctuation characters with spaces."""
    replaced: list[str] = []
    for character in text_value:
        if unicodedata.category(character).startswith("P"):
            replaced.append(" ")
        else:
            replaced.append(character)
    return "".join(replaced)


def normalize_transcript(text_value: str) -> str:
    """Normalize transcript text for alignment work."""
    without_bom = text_value.replace("\ufeff", "")
    normalized = " ".join(without_bom.split())
    if not normalized:
        raise GrpcValidationError(INVALID_ARGUMENT_CODE, "input text contains no words")
    return normalized


def normalize_transcript_for_alignment(
    text_value: str, punctuation_mode: audio_to_text_grpc_pb2.PunctuationMode.ValueType
) -> str:
    """Normalize transcript text and optionally strip punctuation."""
    normalized = normalize_transcript(text_value)
    if punctuation_mode == audio_to_text_grpc_pb2.PUNCTUATION_MODE_KEEP:
        return normalized
    stripped = " ".join(remove_punctuation_from_transcript(normalized).split())
    if not stripped:
        raise GrpcValidationError(
            INVALID_ARGUMENT_CODE,
            "input text contains no words after punctuation removal",
        )
    return stripped


def validate_wav(path: Path) -> None:
    """Validate that the file is a readable WAV container."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            wav_file.getnchannels()
    except wave.Error as exc:
        raise GrpcValidationError(INVALID_WAV_CODE, f"invalid wav: {exc}") from exc


def srt_timestamp_to_seconds(timestamp: str) -> float:
    """Convert an SRT timestamp (HH:MM:SS,mmm) to seconds."""
    hours_str, minutes_str, rest = timestamp.split(":")
    seconds_str, milliseconds_str = rest.split(",")
    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    milliseconds = int(milliseconds_str)
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def parse_word_level_srt(srt_text: str) -> list[AlignedWord]:
    """Parse an SRT produced by audio_to_text.py into aligned words."""
    words: list[AlignedWord] = []
    blocks = re.split(r"(?:\\r?\\n){2,}", srt_text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        time_match = SRT_TIME_RANGE_PATTERN.fullmatch(lines[1])
        if time_match is None:
            continue
        start_seconds = srt_timestamp_to_seconds(time_match.group("start"))
        end_seconds = srt_timestamp_to_seconds(time_match.group("end"))
        text_value = " ".join(lines[2:])
        words.append(
            AlignedWord(
                text=text_value,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return words


def build_word_level_srt(words: Sequence[AlignedWord]) -> str:
    """Build a word-level SRT string from aligned words."""
    lines: list[str] = []
    for index, word in enumerate(words, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_time(word.start_seconds)} --> {format_srt_time(word.end_seconds)}")
        lines.append(word.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_srt_time(seconds: float) -> str:
    """Format seconds into an SRT timestamp string."""
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def split_transcript_words(transcript: str) -> list[str]:
    """Split transcript into word tokens."""
    return [token for token in transcript.split() if token]


def align_in_test_mode(
    transcript: str,
    audio_filename: str,
) -> tuple[list[AlignedWord], str]:
    """Return a deterministic fake alignment for integration tests."""
    words = split_transcript_words(transcript)
    aligned: list[AlignedWord] = []
    cursor = 0.0
    step = 0.25
    for token in words:
        aligned.append(
            AlignedWord(text=token, start_seconds=cursor, end_seconds=cursor + step)
        )
        cursor += step
    return aligned, build_word_level_srt(aligned)


def run_audio_to_text_cli(
    wav_path: Path,
    transcript_path: Path,
    output_srt_path: Path,
    language: str,
) -> str:
    """Invoke audio_to_text.py CLI to produce an SRT file."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    process = subprocess.run(
        [
            str(script_path),
            "--input-audio",
            str(wav_path),
            "--input-text",
            str(transcript_path),
            "--output-srt",
            str(output_srt_path),
            "--language",
            language,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(stderr or "audio_to_text.py failed")
    return output_srt_path.read_text(encoding="utf-8")


def collect_request(
    request_iterator: Iterator[audio_to_text_grpc_pb2.AlignChunk],
    temp_root: Path,
) -> AlignRequest:
    """Collect a streaming request into temporary files and validated fields."""
    first = next(request_iterator, None)
    if first is None or first.WhichOneof("payload") != "init":
        raise GrpcValidationError(
            MISSING_INIT_CODE, "first message must contain init payload"
        )
    init = first.init
    punctuation_mode = init.punctuation
    if punctuation_mode == audio_to_text_grpc_pb2.PUNCTUATION_MODE_UNSPECIFIED:
        punctuation_mode = audio_to_text_grpc_pb2.PUNCTUATION_MODE_REMOVE

    transcript_raw = init.transcript
    if not transcript_raw.strip():
        raise GrpcValidationError(INVALID_ARGUMENT_CODE, "transcript is required")

    audio_filename = init.audio_filename.strip()
    wav_path = temp_root / "input.wav"
    transcript_path = temp_root / "transcript.txt"
    with wav_path.open("wb") as wav_file:
        byte_count = 0
        for message in request_iterator:
            if message.WhichOneof("payload") != "wav_chunk":
                raise GrpcValidationError(
                    INVALID_ARGUMENT_CODE, "unexpected message in stream"
                )
            chunk = bytes(message.wav_chunk)
            if not chunk:
                continue
            wav_file.write(chunk)
            byte_count += len(chunk)

    if byte_count <= 0:
        raise GrpcValidationError(INVALID_WAV_CODE, "audio stream contained no bytes")
    validate_wav(wav_path)
    normalized_transcript = normalize_transcript_for_alignment(
        transcript_raw, punctuation_mode
    )
    transcript_path.write_text(normalized_transcript, encoding="utf-8")
    language = normalize_language(init.language, normalized_transcript)
    return AlignRequest(
        wav_path=wav_path,
        transcript=normalized_transcript,
        language=language,
        punctuation_mode=punctuation_mode,
        audio_filename=audio_filename,
    )


class AudioToTextService(audio_to_text_grpc_pb2_grpc.AudioToTextServicer):
    """gRPC service implementation."""

    def Align(
        self,
        request_iterator: Iterator[audio_to_text_grpc_pb2.AlignChunk],
        context: grpc.ServicerContext,
    ) -> audio_to_text_grpc_pb2.AlignResponse:
        """Align a transcript to a streamed WAV."""
        with tempfile.TemporaryDirectory(prefix="audio_to_text_grpc_") as temp_dir:
            temp_root = Path(temp_dir)
            try:
                request = collect_request(request_iterator, temp_root=temp_root)
            except GrpcValidationError as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{exc.code}: {exc}")
            except Exception as exc:
                context.abort(
                    grpc.StatusCode.INTERNAL, f"{INVALID_ARGUMENT_CODE}: {exc}"
                )

            try:
                if is_test_mode():
                    aligned_words, srt_text = align_in_test_mode(
                        request.transcript, request.audio_filename
                    )
                else:
                    output_srt_path = temp_root / "alignment.srt"
                    transcript_path = temp_root / "transcript.txt"
                    transcript_path.write_text(request.transcript, encoding="utf-8")
                    srt_text = run_audio_to_text_cli(
                        request.wav_path,
                        transcript_path,
                        output_srt_path,
                        request.language,
                    )
                    aligned_words = parse_word_level_srt(srt_text)
            except RuntimeError as exc:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"{ALIGNMENT_FAILED_CODE}: {exc}",
                )
            except Exception as exc:
                context.abort(
                    grpc.StatusCode.INTERNAL, f"{ALIGNMENT_FAILED_CODE}: {exc}"
                )

            response_words = [
                audio_to_text_grpc_pb2.AlignedWord(
                    text=word.text,
                    start_seconds=float(word.start_seconds),
                    end_seconds=float(word.end_seconds),
                )
                for word in aligned_words
            ]
            return audio_to_text_grpc_pb2.AlignResponse(
                words=response_words,
                srt=srt_text,
                audio_filename=request.audio_filename,
            )


def serve(host: str, port: int, max_message_bytes: int, max_workers: int) -> None:
    """Run the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        ],
    )
    audio_to_text_grpc_pb2_grpc.add_AudioToTextServicer_to_server(
        AudioToTextService(), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the gRPC server."""
    args = parse_args(list(argv or ()))
    serve(
        host=str(args.host),
        port=int(args.port),
        max_message_bytes=int(args.max_message_bytes),
        max_workers=int(args.max_workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
