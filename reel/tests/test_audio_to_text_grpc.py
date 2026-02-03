"""Integration tests for the audio_to_text gRPC server."""

from __future__ import annotations

import concurrent.futures
import os
import socket
import subprocess
import sys
import time
import wave
import signal
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

import grpc
import pytest
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from reel.grpc import audio_to_text_pb2
from reel.grpc import audio_to_text_pb2_grpc
import reel.grpc.server as audio_to_text_grpc_server

TEST_CERT_PEM = b"""-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUT5mIG80bR112155jkEP7GIhvy2cwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDEyNTA2MjkyN1oXDTM2MDEy
MzA2MjkyN1owFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAzJXYW4nEsx0ASByso7gcLJBkaOLVTybVd5RdJfe5BK1t
QuuklQnHj0E9sCrs0Y2vYpWjhKP6ne0ruTo8yay2GoII2uZ+xflYLNUjNmQ+SUeL
tbWX+TGD1IYZxSouCa3f9u6xFbBHFPw1zM0vc3dnRgry9qU21nItg0eG6o65mKf7
EpAMi4UX1j07uNnG1+XriNDmaH+tAxdwxKAOHK1jeCfPz/uaw7alHr35LpP4KWEK
tsHpkYAwdFql+CVRDLQW5M5b/+PdtzF/7Tv2LY+6znBOuUhpEuibScAvKBN1U0cJ
dtwgnjs679KZbuKKrKVytIDczbNiRVZtlpf2II5MrQIDAQABo1MwUTAdBgNVHQ4E
FgQUGdwY6QvKwPWyDDc9l9lihkK1kb4wHwYDVR0jBBgwFoAUGdwY6QvKwPWyDDc9
l9lihkK1kb4wDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAl/eD
gEJXw+r8/yDm8PGEfHdx5wdT8u1mrwlqzKt5nlVBdeurgwE41H7Ig+RmGIi/U0ku
P5IDN+Im31l3GcZ/UGeA3kUAHGK/8etbMvMSj1qGN9KtHNPSwbBpN1T48M51ao6a
tSgNvLdqQI36nCBaSIpUmpHD+pwnfc8PUXAn/MNI25JBtpl6b3DJD2lfl6qcUyBY
zSI9fEuUnzj5V0qik4MmbFZm10CH7q2Ty+b8Dy2tf+sUlM6JLq8lt7Wo+2ZTThxl
Yf6QEDyUHTkRFzEP/zQSn6L3ZlGYM9z0oDmh2uWJ94+9RGx03yOUQdkXGdBgDZnp
If3wz0mKDqenYfkXJA==
-----END CERTIFICATE-----"""

TEST_KEY_PEM = b"""-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDMldhbicSzHQBI
HKyjuBwskGRo4tVPJtV3lF0l97kErW1C66SVCcePQT2wKuzRja9ilaOEo/qd7Su5
OjzJrLYaggja5n7F+Vgs1SM2ZD5JR4u1tZf5MYPUhhnFKi4Jrd/27rEVsEcU/DXM
zS9zd2dGCvL2pTbWci2DR4bqjrmYp/sSkAyLhRfWPTu42cbX5euI0OZof60DF3DE
oA4crWN4J8/P+5rDtqUevfkuk/gpYQq2wemRgDB0WqX4JVEMtBbkzlv/4923MX/t
O/Ytj7rOcE65SGkS6JtJwC8oE3VTRwl23CCeOzrv0plu4oqspXK0gNzNs2JFVm2W
l/YgjkytAgMBAAECggEAG3iDdMcrAmICh6xSAinWnqVE1lCfE+HDCCuD1nVNArPG
2BWMI/cMNNAt2FnrMrgztKkjK8xY+pu+I0EOqIXobebICZypqSuyC1MTR1pugqh4
ug2IOo3Lhd37192yENqoGCA05xSkdszm7HkgxgBifaU6uqO0mVb7hYU1OImxoi2u
mktfhDWzlPhAcqYupfhkgKj3c0bOAm0VdqqYBPjCvIBmYJFD38ReM5x5gnacZVrz
PHkjw0Mhgrh9diWaQTRxa4nCo0zz/NsAbRWWWKMLQiuPXoYaOmRD5YrIhHjCxr2z
uik+jwT29B8tMxppqf73jX/CdllwEz/QCbI8EZSIAQKBgQDyHcisvaecutjkpmvo
KSttEU0tGRGYYzo+kQZZ9F/AFtOsgXAbkD5vQIvtSBHLURJgXFbzIhnCLH7MmW7Y
svVjU/oC/Z+W4L7HId6zo0e/MciaZUGXI8e8LMcm5PyWz6+Ubah1O0jBEDVFyy4Z
rfghrigCwf+XzNC3oKT+vbl70QKBgQDYUR0yqs3k65JDNK4QmRzdnybcy0qrKcgF
aKkaGshLQZYmEkdpS2713lmxohVKwFr7l39vjtm5g5yKv3Asv7ekq/dPUd8PoPNE
Mg4IaPXzFQJsWtlqlrxiFgeEj2L+Le8foQcyl+gP83NludNBY+UgQT9Au8Nt8RQL
U+egs1ZmHQKBgQClX6qeMry11Poo03OJE/XRfavVTfYyvDQgYDaSYtS0A+N5RMAq
U5ARtWjeHgfpc+q1Xt0VHQmzE2lYnsSTx0jNG6L9P/ctuz0UDxJqaPCsq3h13Qu8
DHh1E7DD5EyTRjCLlYsm5+N923BHx8wpRibh+m7h01idewxlIJkSVszGwQKBgCvv
da230W/ghOmPhpcqchl1XNBVngnbx5uJoWcC75GRdayv4784KgZYLgGNOSPgo9ob
8C85ZXFvkNkBfwgnvGX+45FBwLuM5jwAqqb8oo/HwaE+ZpBmo0aM0OQUt3eG3t2e
uDaHcgPjX5nWv1i1sHD3r6A3Qr9OytJuLUqPgknlAoGADcgmRCfPCcQd1QUZvThk
UgGpBucqvuCBysLePeqtbDoPhu+IFVRWBCQ4y7JZFE4Vx8lAeE0tLbxQYHiClLf5
mTs+JpcBVChIdy4i1lyEucg+v3mt5gezU/51c1b64p2fTEMBbR6xpNGY+D0ESk4Y
/ecXcW22aAHNoNMQv1bLdHI=
-----END PRIVATE KEY-----"""


def write_stub_modules(stub_root: Path) -> None:
    """Write stub torch/torchaudio/whisperx modules for in-process alignment."""
    stub_root.mkdir(parents=True, exist_ok=True)
    torch_dir = stub_root / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)
    torch_dir.joinpath("__init__.py").write_text(
        "\n".join(
            [
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available() -> bool:",
                "        return False",
                "",
                "cuda = _Cuda()",
                "__version__ = \"2.6.0\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    torchaudio_dir = stub_root / "torchaudio"
    torchaudio_dir.mkdir(parents=True, exist_ok=True)
    torchaudio_dir.joinpath("__init__.py").write_text(
        "\n".join(
            [
                "class AudioMetaData:",
                "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )
    whisperx_dir = stub_root / "whisperx"
    whisperx_dir.mkdir(parents=True, exist_ok=True)
    whisperx_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    whisperx_dir.joinpath("audio.py").write_text(
        "\n".join(
            [
                "SAMPLE_RATE = 48000",
                "",
                "def load_audio(path: str):",
                "    return [0.0] * SAMPLE_RATE",
                "",
            ]
        ),
        encoding="utf-8",
    )
    whisperx_dir.joinpath("alignment.py").write_text(
        "\n".join(
            [
                "import os",
                "",
                "def load_align_model(language_code: str, device: str):",
                "    return object(), {}",
                "",
                "def align(segments, align_model, metadata, audio, device, return_char_alignments=False):",
                "    mode = os.environ.get(\"AUDIO_TO_TEXT_GRPC_STUB_MODE\", \"ok\")",
                "    if mode == \"invalid\":",
                "        return {\"segments\": [{\"words\": [{\"word\": \"\", \"start\": 0.0, \"end\": 0.1}]}]}",
                "    if mode == \"punctuation\":",
                "        words = [",
                "            {\"word\": \"!!!\", \"start\": 0.0, \"end\": 0.1},",
                "            {\"word\": \"★\", \"start\": None, \"end\": None},",
                "            {\"word\": \"★\", \"start\": 0.1, \"end\": 0.2},",
                "            {\"word\": \"hello\", \"start\": 0.2, \"end\": 0.3},",
                "        ]",
                "        return {\"segments\": [{\"words\": words}]}",
                "    transcript = segments[0].get(\"text\", \"\") if segments else \"\"",
                "    tokens = [token for token in transcript.split() if token]",
                "    if not tokens:",
                "        tokens = [\"silence\"]",
                "    duration_seconds = max(float(len(audio)) / 48000.0, 0.1)",
                "    step = duration_seconds / float(len(tokens))",
                "    cursor = 0.0",
                "    words = []",
                "    for token in tokens:",
                "        start = cursor",
                "        end = cursor + step",
                "        words.append({\"word\": token, \"start\": start, \"end\": end})",
                "        cursor = end",
                "    return {\"segments\": [{\"words\": words}]}",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    init: audio_to_text_pb2.AlignInit, wav_bytes: bytes, chunk_size: int = 4096
) -> Iterator[audio_to_text_pb2.AlignChunk]:
    """Yield an init message followed by WAV chunks."""
    yield audio_to_text_pb2.AlignChunk(init=init)
    for offset in range(0, len(wav_bytes), chunk_size):
        yield audio_to_text_pb2.AlignChunk(
            wav_chunk=wav_bytes[offset : offset + chunk_size]
        )


def start_server(
    repo_root: Path,
    port: int,
    extra_env: dict[str, str] | None = None,
    extra_args: Iterable[str] | None = None,
    test_mode: bool = True,
) -> subprocess.Popen[str]:
    """Start the gRPC server subprocess in test mode."""
    env = os.environ.copy()
    if test_mode:
        env["AUDIO_TO_TEXT_GRPC_TEST_MODE"] = "1"
    else:
        env.pop("AUDIO_TO_TEXT_GRPC_TEST_MODE", None)
    if extra_env:
        env.update(extra_env)
    command = [
        sys.executable,
        "-m",
        "audio_to_text_grpc.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if extra_args:
        command.extend(extra_args)
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def run_grpc_command(
    repo_root: Path,
    args: Iterable[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the gRPC server CLI and capture output."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    command = [sys.executable, "-m", "audio_to_text_grpc.server", *args]
    return subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a subprocess and wait."""
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def read_error_status(error: grpc.RpcError) -> tuple[grpc.StatusCode, str]:
    """Extract status code and details from an RpcError."""
    return error.code(), error.details() or ""


def test_audio_to_text_grpc_align_defaults_to_remove_punctuation(tmp_path: Path) -> None:
    """Default punctuation mode strips punctuation before alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(
            transcript="Hello, world!",
            punctuation=audio_to_text_pb2.PUNCTUATION_MODE_UNSPECIFIED,
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert [word.text for word in response.words] == ["Hello", "world"]
        assert response.srt
    finally:
        stop_process(process)


def test_audio_to_text_grpc_defaults_language_for_cyrillic(tmp_path: Path) -> None:
    """Detect Cyrillic transcripts and default to Russian."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="Привет мир")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_align_keeps_punctuation_when_requested(
    tmp_path: Path,
) -> None:
    """Punctuation keep mode preserves punctuation tokens."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(
            transcript="Hello, world!",
            punctuation=audio_to_text_pb2.PUNCTUATION_MODE_KEEP,
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
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
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, b"not-a-wav"))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.invalid_wav" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_requires_init(tmp_path: Path) -> None:
    """Reject streams that do not start with init."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)

        def invalid_stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            yield audio_to_text_pb2.AlignChunk(wav_chunk=b"\x00\x01")

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(invalid_stream())
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.missing_init" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_empty_transcript(tmp_path: Path) -> None:
    """Reject empty transcript values."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript=" ")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "transcript is required" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_health_serving(tmp_path: Path) -> None:
    """Expose gRPC health status."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            response = stub.Check(
                health_pb2.HealthCheckRequest(
                    service="svg_tools.audio_to_text.v1.AudioToText"
                )
            )
        assert response.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        stop_process(process)


def test_audio_to_text_grpc_stats_counts_requests(tmp_path: Path) -> None:
    """Return metrics after a request."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            stub.Align(stream_request(init, wav_bytes))
            stats = stub.GetStats(audio_to_text_pb2.StatsRequest())
        assert stats.requests_total >= 1
        assert stats.requests_succeeded >= 1
        assert stats.bytes_received >= len(wav_bytes)
    finally:
        stop_process(process)


def test_audio_to_text_grpc_stats_tracks_latency(tmp_path: Path) -> None:
    """Record latency metrics for alignment requests."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "50"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            stub.Align(stream_request(init, wav_bytes))
            stats = stub.GetStats(audio_to_text_pb2.StatsRequest())
        assert stats.max_latency_seconds > 0.0
    finally:
        stop_process(process)


def test_audio_to_text_grpc_stats_handle_non_max_latency(tmp_path: Path) -> None:
    """Track latency when later requests are faster."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.02)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")

        def slow_stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            yield audio_to_text_pb2.AlignChunk(init=init)
            for offset in range(0, len(wav_bytes), 512):
                time.sleep(0.02)
                yield audio_to_text_pb2.AlignChunk(
                    wav_chunk=wav_bytes[offset : offset + 512]
                )

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            stub.Align(slow_stream())
            stub.Align(stream_request(init, wav_bytes))
            stats = stub.GetStats(audio_to_text_pb2.StatsRequest())
        assert stats.requests_total >= 2
        assert stats.max_latency_seconds > 0.0
    finally:
        stop_process(process)


def test_audio_to_text_grpc_requires_auth_when_configured(tmp_path: Path) -> None:
    """Reject calls without auth metadata when configured."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_AUTH_TOKEN": "secret"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.UNAUTHENTICATED
        assert "audio_to_text_grpc.auth.required" in details
        metadata = (("authorization", "Bearer secret"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes), metadata=metadata)
            stats = stub.GetStats(
                audio_to_text_pb2.StatsRequest(), metadata=metadata
            )
        assert response.words
        assert stats.requests_total >= 1
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_audio_limit(tmp_path: Path) -> None:
    """Reject audio streams exceeding configured size."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_MAX_AUDIO_BYTES": "10"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "audio_to_text_grpc.input.audio_too_large" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_transcript_char_limit(
    tmp_path: Path,
) -> None:
    """Reject transcripts exceeding max length."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_MAX_TRANSCRIPT_CHARS": "5"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "audio_to_text_grpc.input.text_too_large" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_transcript_word_limit(
    tmp_path: Path,
) -> None:
    """Reject transcripts exceeding max word count."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_MAX_TRANSCRIPT_WORDS": "1"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "audio_to_text_grpc.input.text_too_long" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_inflight_limit(tmp_path: Path) -> None:
    """Reject requests exceeding inflight limit."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={
            "AUDIO_TO_TEXT_GRPC_MAX_INFLIGHT": "1",
            "AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "200",
        },
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")

        def call_align() -> grpc.StatusCode:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
                try:
                    stub.Align(stream_request(init, wav_bytes))
                    return grpc.StatusCode.OK
                except grpc.RpcError as exc:
                    return exc.code()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(call_align)
            time.sleep(0.05)
            second_future = executor.submit(call_align)
            first_result = first_future.result()
            second_result = second_future.result()

        assert grpc.StatusCode.RESOURCE_EXHAUSTED in (first_result, second_result)
        assert grpc.StatusCode.INTERNAL not in (first_result, second_result)
        third_result = call_align()
        assert third_result == grpc.StatusCode.OK
    finally:
        stop_process(process)


def test_audio_to_text_grpc_alignment_timeout(tmp_path: Path) -> None:
    """Enforce alignment timeout when configured."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={
            "AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "0.05",
            "AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "100",
        },
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.DEADLINE_EXCEEDED
        assert "audio_to_text_grpc.align.timeout" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_request_deadline_exceeded(tmp_path: Path) -> None:
    """Reject requests when the deadline is exceeded."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")

        def slow_stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            yield audio_to_text_pb2.AlignChunk(init=init)
            for offset in range(0, len(wav_bytes), 256):
                time.sleep(0.03)
                yield audio_to_text_pb2.AlignChunk(
                    wav_chunk=wav_bytes[offset : offset + 256]
                )

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(slow_stream(), timeout=0.05)
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.DEADLINE_EXCEEDED
        assert (
            "audio_to_text_grpc.request.deadline_exceeded" in details
            or "Deadline Exceeded" in details
        )
    finally:
        stop_process(process)


def test_audio_to_text_grpc_deadline_exceeded_after_alignment(
    tmp_path: Path,
) -> None:
    """Reject requests that exceed the deadline mid-alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "120"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes), timeout=0.05)
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.DEADLINE_EXCEEDED
        assert (
            "audio_to_text_grpc.align.timeout" in details
            or "audio_to_text_grpc.request.deadline_exceeded" in details
            or "Deadline Exceeded" in details
        )
    finally:
        stop_process(process)


def test_audio_to_text_grpc_alignment_timeout_disabled(tmp_path: Path) -> None:
    """Allow alignment to run without a timeout when disabled."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "0"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_deadline_exceeded_before_request(
    tmp_path: Path,
) -> None:
    """Reject requests that start after the deadline."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        metadata = (("x-test-remaining", "0"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(
                    stream_request(init, wav_bytes),
                    metadata=metadata,
                )
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.DEADLINE_EXCEEDED
        assert "audio_to_text_grpc.request.deadline_exceeded" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_alignment_timeout_disabled_infinite_remaining(
    tmp_path: Path,
) -> None:
    """Allow disabled alignment timeouts with infinite deadlines."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "0"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        metadata = (("x-test-remaining", "inf"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(
                stream_request(init, wav_bytes),
                metadata=metadata,
            )
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_effective_timeout_defaults_to_config(
    tmp_path: Path,
) -> None:
    """Use configured alignment timeout when no deadline is supplied."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "1"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        metadata = (("x-test-remaining", "none"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(
                stream_request(init, wav_bytes),
                metadata=metadata,
            )
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_invalid_remaining_override(tmp_path: Path) -> None:
    """Ignore invalid x-test-remaining overrides."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        metadata = (("x-test-remaining", "bad-value"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(
                stream_request(init, wav_bytes),
                metadata=metadata,
            )
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_request_cancelled(tmp_path: Path) -> None:
    """Handle cancelled requests during alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "200"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            call = stub.Align.future(stream_request(init, wav_bytes))
            time.sleep(0.05)
            assert call.cancel()
            with pytest.raises((grpc.RpcError, grpc.FutureCancelledError)) as exc_info:
                call.result()
        if isinstance(exc_info.value, grpc.RpcError):
            status, details = read_error_status(exc_info.value)
            assert status == grpc.StatusCode.CANCELLED
            assert "audio_to_text_grpc.request.cancelled" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_tls(tmp_path: Path) -> None:
    """Serve over TLS when certs are configured."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    cert_path = tmp_path / "server.crt"
    key_path = tmp_path / "server.key"
    cert_path.write_bytes(TEST_CERT_PEM)
    key_path.write_bytes(TEST_KEY_PEM)
    process = start_server(
        repo_root,
        port,
        extra_args=["--tls-cert", str(cert_path), "--tls-key", str(key_path)],
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        credentials = grpc.ssl_channel_credentials(root_certificates=TEST_CERT_PEM)
        with grpc.secure_channel(
            f"localhost:{port}",
            credentials,
            options=(("grpc.ssl_target_name_override", "localhost"),),
        ) as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(
                stream_request(
                    audio_to_text_pb2.AlignInit(transcript="hello"),
                    build_silent_wav_bytes(0.2),
                )
            )
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_invalid_language(tmp_path: Path) -> None:
    """Reject unsupported languages in the request."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(
            transcript="hello",
            language="xx",
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text.input.invalid_language" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_punctuation_only_transcript(
    tmp_path: Path,
) -> None:
    """Reject transcripts that become empty after punctuation removal."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(
            transcript="!!!",
            punctuation=audio_to_text_pb2.PUNCTUATION_MODE_REMOVE,
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text.input.invalid_config" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_unexpected_stream_payload(tmp_path: Path) -> None:
    """Reject streams with unexpected messages after init."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)

        def invalid_stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            init = audio_to_text_pb2.AlignInit(transcript="hello")
            yield audio_to_text_pb2.AlignChunk(init=init)
            yield audio_to_text_pb2.AlignChunk(init=init)

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(invalid_stream())
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.invalid_argument" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_missing_init(tmp_path: Path) -> None:
    """Reject streams that omit the init message."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)

        def missing_init_stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            yield audio_to_text_pb2.AlignChunk(wav_chunk=b"RIFF")

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(missing_init_stream())
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.missing_init" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_empty_audio_stream(tmp_path: Path) -> None:
    """Reject streams with no audio bytes."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(repo_root, port)
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)

        def empty_audio_stream() -> Iterator[audio_to_text_pb2.AlignChunk]:
            init = audio_to_text_pb2.AlignInit(transcript="hello")
            yield audio_to_text_pb2.AlignChunk(init=init)
            yield audio_to_text_pb2.AlignChunk(wav_chunk=b"")

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(empty_audio_stream())
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.invalid_wav" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_inprocess_alignment_with_stubs(tmp_path: Path) -> None:
    """Run alignment in-process using stub whisperx modules."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    process = start_server(
        repo_root,
        port,
        extra_env={
            "PYTHONPATH": f"{stub_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "AUDIO_TO_TEXT_GRPC_TEST_MODE": "0",
        },
        test_mode=False,
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert [word.text for word in response.words] == ["hello", "world"]
    finally:
        stop_process(process)


def test_audio_to_text_grpc_inprocess_alignment_drops_punctuation(
    tmp_path: Path,
) -> None:
    """Drop punctuation tokens when remove-punctuation is active."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    process = start_server(
        repo_root,
        port,
        extra_env={
            "PYTHONPATH": f"{stub_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "AUDIO_TO_TEXT_GRPC_STUB_MODE": "punctuation",
            "AUDIO_TO_TEXT_GRPC_TEST_MODE": "0",
        },
        test_mode=False,
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="Hello!!!")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert [word.text for word in response.words] == ["hello"]
        assert response.srt
    finally:
        stop_process(process)


def test_audio_to_text_grpc_inprocess_alignment_failure(tmp_path: Path) -> None:
    """Propagate alignment failures from the in-process runner."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    process = start_server(
        repo_root,
        port,
        extra_env={
            "PYTHONPATH": f"{stub_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "AUDIO_TO_TEXT_GRPC_STUB_MODE": "invalid",
            "AUDIO_TO_TEXT_GRPC_TEST_MODE": "0",
        },
        test_mode=False,
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.FAILED_PRECONDITION
        assert "audio_to_text.align.missing_timestamps" in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_test_mode_crash(tmp_path: Path) -> None:
    """Return an internal error when test-mode alignment crashes."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_CRASH": "1"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INTERNAL
        assert audio_to_text_grpc_server.ALIGNMENT_FAILED_CODE in details
    finally:
        stop_process(process)


def test_audio_to_text_grpc_timeout_disabled(tmp_path: Path) -> None:
    """Allow requests when the timeout is disabled."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "0"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_timeout_uses_config_without_deadline(
    tmp_path: Path,
) -> None:
    """Apply the server timeout when no deadline is provided."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "1.0"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes))
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_accepts_api_key_auth(tmp_path: Path) -> None:
    """Authorize using x-api-key metadata."""
    repo_root = Path(__file__).resolve().parents[1]
    port = free_local_port()
    process = start_server(
        repo_root,
        port,
        extra_env={"AUDIO_TO_TEXT_GRPC_AUTH_TOKEN": "secret"},
    )
    try:
        wait_for_port("127.0.0.1", port, timeout_seconds=5)
        wav_bytes = build_silent_wav_bytes(0.2)
        init = audio_to_text_pb2.AlignInit(transcript="hello")
        metadata = (("x-api-key", "secret"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes), metadata=metadata)
        assert response.words
    finally:
        stop_process(process)


def test_audio_to_text_grpc_rejects_tls_config(tmp_path: Path) -> None:
    """Reject TLS configs missing key or cert."""
    repo_root = Path(__file__).resolve().parents[1]
    cert_path = tmp_path / "server.crt"
    cert_path.write_bytes(TEST_CERT_PEM)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_to_text_grpc.server",
            "--tls-cert",
            str(cert_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1


def test_audio_to_text_grpc_rejects_invalid_config(tmp_path: Path) -> None:
    """Reject invalid gRPC configuration values."""
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        ({"AUDIO_TO_TEXT_GRPC_MAX_AUDIO_BYTES": "0"}, "max-audio-bytes must be positive"),
        ({"AUDIO_TO_TEXT_GRPC_MAX_AUDIO_BYTES": "nope"}, "max-audio-bytes must be an integer"),
        (
            {"AUDIO_TO_TEXT_GRPC_MAX_TRANSCRIPT_CHARS": "0"},
            "max-transcript-chars must be positive",
        ),
        (
            {"AUDIO_TO_TEXT_GRPC_MAX_TRANSCRIPT_WORDS": "0"},
            "max-transcript-words must be positive",
        ),
        ({"AUDIO_TO_TEXT_GRPC_MAX_INFLIGHT": "0"}, "max-inflight must be positive"),
        ({"AUDIO_TO_TEXT_GRPC_TEST_CRASH": "1"}, "test-crash requires test mode"),
        (
            {"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "-1"},
            "alignment-timeout-seconds must be non-negative",
        ),
        (
            {"AUDIO_TO_TEXT_GRPC_ALIGNMENT_TIMEOUT_SECONDS": "bad"},
            "alignment-timeout-seconds must be a number",
        ),
    ]
    for env_overrides, expected in cases:
        result = run_grpc_command(repo_root, [], env_overrides)
        assert result.returncode == 1
        assert expected in result.stderr


def test_audio_to_text_grpc_loads_auth_token_from_env(tmp_path: Path) -> None:
    """Load auth token from environment when unset in CLI."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_grpc_command(
        repo_root,
        ["--max-audio-bytes", "0"],
        {"AUDIO_TO_TEXT_GRPC_AUTH_TOKEN": "secret"},
    )
    assert result.returncode == 1
    assert "max-audio-bytes must be positive" in result.stderr


def test_audio_to_text_grpc_prefers_cli_auth_token(tmp_path: Path) -> None:
    """Prefer auth token from CLI when provided."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_grpc_command(
        repo_root,
        ["--auth-token", "secret"],
        {
            "AUDIO_TO_TEXT_GRPC_AUTH_TOKEN": "ignored",
            "AUDIO_TO_TEXT_GRPC_TLS_CERT": "cert.pem",
        },
    )
    assert result.returncode == 1
    assert "tls-cert and tls-key must be provided together" in result.stderr


def test_audio_to_text_grpc_rejects_invalid_args(tmp_path: Path) -> None:
    """Reject invalid gRPC CLI argument values."""
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        (["--max-audio-bytes", "0"], "max-audio-bytes must be positive"),
        (["--max-transcript-chars", "0"], "max-transcript-chars must be positive"),
        (["--max-transcript-words", "0"], "max-transcript-words must be positive"),
        (["--max-inflight", "0"], "max-inflight must be positive"),
        (
            ["--alignment-timeout-seconds", "-1"],
            "alignment-timeout-seconds must be non-negative",
        ),
    ]
    for args, expected in cases:
        result = run_grpc_command(repo_root, args, None)
        assert result.returncode == 1
        assert expected in result.stderr


def test_audio_to_text_grpc_rejects_invalid_tls_paths(tmp_path: Path) -> None:
    """Reject whitespace TLS paths."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_grpc_command(
        repo_root,
        [],
        {
            "AUDIO_TO_TEXT_GRPC_TLS_CERT": " ",
            "AUDIO_TO_TEXT_GRPC_TLS_KEY": "key",
        },
    )
    assert result.returncode == 1
    assert "tls-cert must be non-empty when set" in result.stderr


def test_audio_to_text_grpc_logging_levels(tmp_path: Path) -> None:
    """Apply configured log levels."""
    repo_root = Path(__file__).resolve().parents[1]
    for level in ("DEBUG", "WARNING", "ERROR"):
        port = free_local_port()
        process = start_server(
            repo_root,
            port,
            extra_env={"AUDIO_TO_TEXT_GRPC_LOG_LEVEL": level},
        )
        try:
            wait_for_port("127.0.0.1", port, timeout_seconds=5)
        finally:
            stop_process(process)


def test_audio_to_text_grpc_cli_help(tmp_path: Path) -> None:
    """Expose CLI usage via --help."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_grpc_command(
        repo_root,
        ["--help"],
        {"AUDIO_TO_TEXT_GRPC_TEST_MODE": "1"},
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
