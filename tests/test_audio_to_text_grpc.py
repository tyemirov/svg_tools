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

from audio_to_text_grpc import audio_to_text_grpc_pb2
from audio_to_text_grpc import audio_to_text_grpc_pb2_grpc

TEST_CERT_PEM = b"""-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUIs74gD9ZUUjgI1eTbLMeHkF7fqcwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDEyNDA2MjQ0MVoXDTI2MDEy
NTA2MjQ0MVowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEA2ZhkBi/rjys23w6Ch/OItTNl/d7wNF0sQMJlR/MxcPwx
ck/tb7yizOV7Mou7JkxMcH+Kwpl6U1ciiPK8yrZcxeMNlkmXILD3NUzzxneypS4Q
jF58Fafz7OQp8038Aowrk5fINkEznu0CTRgWvmj1TlHc9mX6VFoCpmmm3MxVCx+p
5ZxjCpZnuDbiM//HzaXVVU6Uv++VUmqs8yhM9LGYATis7QoQL1VXxT+HS85L4HLd
wII8ECT3XM+SUkCuN5iU7VMkyur65hO3gOibpsjL32eCKfGCWHTBeoWPTo0pVCIq
wZBGjo9gGL5f5Hg7nZYWQ8E65mqpenR7nDtOUqNf+wIDAQABo1MwUTAdBgNVHQ4E
FgQUFChcB0Miq7C+/uPbySg9OlKLxyMwHwYDVR0jBBgwFoAUFChcB0Miq7C+/uPb
ySg9OlKLxyMwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAYNBM
5JwbtU/yrdpIg1H7Edq/YTWQKCcp4F6p3unZeQP2GWXgwTduFV66vzC8wbbqRPcj
lVVLq0k8KVBonY4mHNZRybwrQQCRhLNiDvqf0PrtEdNoZ1OR6xBuEGTXYFw4+KpF
NGBEah/s+olqgt77nc6u0oB1F1i06knnEHSSjRUAGPg6pk5yzY6hOwdy197/MjO0
rIVhmbo5Na2/S6FbPzeJ5aY1bLQpKXKfTWHBeAGsHLJW6WSDFmRyM6M8xpR97/ni
EvS1QnJWqxLwVYylxWGubn8rZZHkABaUGyX83nn7vX4ITIzDvazJUbb9t0OA7KHQ
xTE0RbR94VOn0XvQKA==
-----END CERTIFICATE-----"""

TEST_KEY_PEM = b"""-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDZmGQGL+uPKzbf
DoKH84i1M2X93vA0XSxAwmVH8zFw/DFyT+1vvKLM5Xsyi7smTExwf4rCmXpTVyKI
8rzKtlzF4w2WSZcgsPc1TPPGd7KlLhCMXnwVp/Ps5CnzTfwCjCuTl8g2QTOe7QJN
GBa+aPVOUdz2ZfpUWgKmaabczFULH6nlnGMKlme4NuIz/8fNpdVVTpS/75VSaqzz
KEz0sZgBOKztChAvVVfFP4dLzkvgct3AgjwQJPdcz5JSQK43mJTtUyTK6vrmE7eA
6JumyMvfZ4Ip8YJYdMF6hY9OjSlUIirBkEaOj2AYvl/keDudlhZDwTrmaql6dHuc
O05So1/7AgMBAAECggEARhDjDHb7xAMnTRYQBBTwTWC6k8/oywPBjyzJjiyKHBap
GWURUKyOQ6oVTIZgAgoJhyQam2TuVc22CiEXl7K3FuVw55bUuishvyEDUkIS3UuC
+qAmFpcZXYtu0rCm2G8oTQNP7eB67c7lscty72c+rjSAETtvKyA8wDc/CtQlAkpc
1vLgU+4US6t75XjOBxK3fZjOYnuVMAoChj8dEPb0WcZWbQn9bC3l7/JHwnmInfax
wT99BpDP0B4iyWhgXRJ5PFjOm5u4lDGEMt2tHDBG7mm49nE9dWroLKMvjaYc9mP1
mWYtaIFDxJI/fEpYlGUwL362Npw2KsTkgpHkAMoeaQKBgQDv837GioGnI5c6IzIv
vg/em7XLM5T2l5QstPwgnD9fu5rQi5+6P/9iwB0K5bGocJHWosWdr36yclwQ9Fux
yr+dE5V/dQeD9NmzA+OjKpLlvYL6v7+E3RsltRXc0K8RPBKiGXBDxi71e+yzpD+w
c5G1o2gPblhgx4oDAGQ/1KybzwKBgQDoJhzLkiOfQoMCJpso0kEyotXkKP1THz5c
CnmHVsiynbpLHqEZ+CzkcFuXCp3R8O/La5pCqqy34cc10hIj6w9og1sIn1yjgBqn
ebvCr5rTxRUxonsFFHmUHCgDUocSOoaWXEK4TWisoZ9p3Ras7YK26veamNUAQMAk
jC+ttwnoFQKBgQDgXH5a6KGIhbferZA8oeu2y0O159pOpLbxAp2JZG/BdOqyOwu0
HCkQze+aA+IVTISg+/vpkHHYKyQ42jNuz9RQ7EHqNqQ7V1PDlkxggetvE7+H9IkA
k7nRjTwPB7XaWKBXn4+gX4wDD8foxRikRbul6k/ftd6/R3Sdj90s7hAjcwKBgFWr
Yur49i5ahDAyAQZSeP7vupElUz7ug7vBndGFBDFuJYZ9OiR0QlTW5eXaZB1KlaNz
BQ1fILgHGNXSvSEGKLx/5QNMlAo+RxfQw6p31WIbE9QqSs24HZm7vkRjmzbt5Cbu
yyxqXJuAVWziyvBsAdXw5gjmJ4ydWIrkNOAdcQl1AoGBAItlbReAIKN7sZkORKNO
NwBI3TGT7fGEKdPSPR5yjbbsJNcjq1GX46EC8cZT/LWaXCd5LXf+lNLYz0LIJP7E
NFIwu3wOxCuBwXN2w1QERrbGyszvO6KKD2eWBbJaoPeXllzKuak+VNbtrUt/OT31
smU4GhXG+Dq3wiDcMB2pak0D
-----END PRIVATE KEY-----"""


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
        yield audio_to_text_grpc_pb2.AlignChunk(
            wav_chunk=wav_bytes[offset : offset + chunk_size]
        )


def start_server(
    repo_root: Path,
    port: int,
    extra_env: dict[str, str] | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.Popen[str]:
    """Start the gRPC server subprocess in test mode."""
    env = os.environ.copy()
    env["AUDIO_TO_TEXT_GRPC_TEST_MODE"] = "1"
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

        def invalid_stream() -> Iterator[audio_to_text_grpc_pb2.AlignChunk]:
            yield audio_to_text_grpc_pb2.AlignChunk(wav_chunk=b"\x00\x01")

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(invalid_stream())
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.INVALID_ARGUMENT
        assert "audio_to_text_grpc.input.missing_init" in details
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            stub.Align(stream_request(init, wav_bytes))
            stats = stub.GetStats(audio_to_text_grpc_pb2.StatsRequest())
        assert stats.requests_total >= 1
        assert stats.requests_succeeded >= 1
        assert stats.bytes_received >= len(wav_bytes)
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.UNAUTHENTICATED
        assert "audio_to_text_grpc.auth.required" in details
        metadata = (("authorization", "Bearer secret"),)
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(stream_request(init, wav_bytes), metadata=metadata)
            stats = stub.GetStats(
                audio_to_text_grpc_pb2.StatsRequest(), metadata=metadata
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello world")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello")

        def call_align() -> grpc.StatusCode | None:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
                try:
                    stub.Align(stream_request(init, wav_bytes))
                except grpc.RpcError as exc:
                    return exc.code()
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(call_align)
            time.sleep(0.05)
            second_future = executor.submit(call_align)
            first_result = first_future.result()
            second_result = second_future.result()

        assert (
            first_result == grpc.StatusCode.RESOURCE_EXHAUSTED
            or second_result == grpc.StatusCode.RESOURCE_EXHAUSTED
        )
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
        init = audio_to_text_grpc_pb2.AlignInit(transcript="hello")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Align(stream_request(init, wav_bytes))
        status, details = read_error_status(exc_info.value)
        assert status == grpc.StatusCode.DEADLINE_EXCEEDED
        assert "audio_to_text_grpc.align.timeout" in details
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
            stub = audio_to_text_grpc_pb2_grpc.AudioToTextStub(channel)
            response = stub.Align(
                stream_request(
                    audio_to_text_grpc_pb2.AlignInit(transcript="hello"),
                    build_silent_wav_bytes(0.2),
                )
            )
        assert response.words
    finally:
        stop_process(process)
