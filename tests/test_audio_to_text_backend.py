"""Integration tests for the audio_to_text backend service."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen


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


def stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a subprocess and wait."""
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def start_grpc_server(repo_root: Path, port: int) -> subprocess.Popen[str]:
    """Start the gRPC server subprocess in test mode."""
    env = os.environ.copy()
    env["AUDIO_TO_TEXT_GRPC_TEST_MODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "audio_to_text_grpc.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def start_backend_server(
    repo_root: Path,
    port: int,
    data_dir: Path,
    grpc_target: str,
) -> subprocess.Popen[str]:
    """Start the backend server subprocess."""
    env = os.environ.copy()
    env["AUDIO_TO_TEXT_BACKEND_GRPC_TARGET"] = grpc_target
    command = [
        sys.executable,
        "-m",
        "audio_to_text_backend.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--data-dir",
        str(data_dir),
    ]
    return subprocess.Popen(
        command,
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def build_wav_bytes(duration_seconds: float) -> bytes:
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


def build_multipart(
    fields: dict[str, str],
    files: Iterable[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    """Build a multipart/form-data payload."""
    boundary = f"boundary-{int(time.time() * 1000)}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    for name, filename, content_type, payload in files:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        parts.append(header + payload + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


def read_json(url: str) -> dict[str, object]:
    """Read JSON from a URL."""
    with urlopen(url, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_job_completion(base_url: str, job_id: str) -> dict[str, object]:
    """Poll the job list until a job completes."""
    deadline = time.monotonic() + 8
    while True:
        payload = read_json(f"{base_url}/api/jobs")
        jobs = payload.get("jobs", [])
        if isinstance(jobs, list):
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                if job.get("job_id") != job_id:
                    continue
                if job.get("status") in ("completed", "failed"):
                    return job
        if time.monotonic() > deadline:
            raise TimeoutError("job did not complete in time")
        time.sleep(0.1)


def test_audio_to_text_backend_job_flow(tmp_path: Path) -> None:
    """Create a job through the backend and download the SRT."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        wav_bytes = build_wav_bytes(0.2)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", wav_bytes),
                ("text", "sample.txt", "text/plain", b"Hello world"),
            ],
        )
        request = Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        completed = wait_for_job_completion(base_url, job_id)
        assert completed.get("status") == "completed"
        with urlopen(f"{base_url}/api/jobs/{job_id}/srt", timeout=5) as response:
            srt_payload = response.read().decode("utf-8")
        assert "Hello" in srt_payload
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_sse_events(tmp_path: Path) -> None:
    """Serve job events over SSE."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        wav_bytes = build_wav_bytes(0.1)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", wav_bytes),
                ("text", "sample.txt", "text/plain", b"Hi"),
            ],
        )
        request = Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urlopen(request, timeout=5):
            pass
        with urlopen(f"{base_url}/api/jobs/events", timeout=5) as response:
            line = response.readline().decode("utf-8").strip()
        assert line.startswith("data:")
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)
