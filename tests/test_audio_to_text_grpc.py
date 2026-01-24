"""Integration tests for the audio_to_text gRPC server."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterator

import grpc
import pytest

from audio_to_text_grpc import audio_to_text_grpc_pb2
from audio_to_text_grpc import audio_to_text_grpc_pb2_grpc


def free_local_port() -> int:
    """Reserve a free port for local servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_seconds: float) -> None:
    """Wait until a TCP port is accepting connections."""
    start = time.monotonic()
    while True:
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("server did not become ready in time")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(0.05)


def build_silent_wav_bytes(duration_seconds: float) -> bytes:
    """Build a silent 16-bit PCM WAV."""
    sample_rate = 48_000
    frame_count = max(1, int(round(duration_seconds * sample_rate)))
    silence = b"\x00\x00" * frame_count
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence)
    return buffer.getvalue()


def stream_request(
    init: audio_to_text_grpc_pb2.AlignInit, wav_bytes: bytes, chunk_size: int = 4096
) -> Iterator[audio_to_text_grpc_pb2.AlignChunk]:
    """Yield an init message followed by WAV chunks."""
    yield audio_to_text_grpc_pb2.AlignChunk(init=init)
    for offset in range(0, len(wav_bytes), chunk_size):
        yield audio_to_text_grpc_pb2.AlignChunk(wav_chunk=wav_bytes[offset : offset + chunk_size])


def start_server(repo_root: Path, port: int) -> subprocess.Popen[str]:
    """Start the gRPC server subprocess in test mode."""
    env = os.environ.copy()
    env["AUDIO_TO_TEXT_GRPC_TEST_MODE"] = "1"
    script_path = repo_root / "audio_to_text_grpc.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a subprocess and wait."""
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def test_audio_to_text_grpc_align_defaults_to_remove_punctuation(tmp_path: Path) -> None:
    """Default punctuation mode strips punctuation before alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_grpc_pb2.AlignInit(
            transcript="Hello, world!",
            punctuation=audio_to_text_grpc_pb2.PUNCTUATION_MODE_UNSPECIFIED,
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert [word.text for word in response.words] == ["Hello", "world"]
        assert response.srt
    finally:
        stop_process(process)


def test_audio_to_text_grpc_align_keeps_punctuation_when_requested(tmp_path: Path) -> None:
    """Punctuation keep mode preserves punctuation tokens."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_grpc_pb2.AlignInit(
            transcript="Hello, world!",
            punctuation=audio_to_text_grpc_pb2.PUNCTUATION_MODE_KEEP,
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert [word.text for word in response.words] == ["Hello,", "world!"]
        assert response.srt
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_invalid_wav(tmp_path: Path) -> None:
    """Reject non-WAV uploads."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, b"not-a-wav"))
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.invalid_wav" in exc_info.value.details()
    finally:
        stop_process(process)


def test_audio_to_text_grpc_requires_init(tmp_path: Path) -> None:
    """Reject streams that do not start with init."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)

        def invalid_stream() -> Iterator[audio_to_text_grpc_pb2.AlignChunk]:
            yield audio_to_text_grpc_pb2.AlignChunk(wav_chunk=b"\x00\x01")

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(invalid_stream())
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.missing_init" in exc_info.value.details()
    finally:
        stop_process(process)
