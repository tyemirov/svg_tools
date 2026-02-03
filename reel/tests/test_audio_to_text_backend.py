"""Integration tests for the audio_to_text backend service."""

from __future__ import annotations

import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import time
import wave
import signal
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import backend.server as audio_to_text_backend_server


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


def reset_sse_connection(
    host: str,
    port: int,
    path: str,
    read_bytes: int | None = None,
) -> None:
    """Open an SSE request and reset the connection."""
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(
        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
    )
    connection.connect((host, port))
    request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode("utf-8")
    connection.sendall(request)
    if read_bytes:
        connection.recv(read_bytes)
    connection.close()


def reset_http_connection(connection: http.client.HTTPConnection) -> None:
    """Reset an HTTP connection with a TCP RST."""
    sock = connection.sock
    if sock is None:
        connection.close()
        return
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    connection.close()


def read_sse_data_line(response: http.client.HTTPResponse) -> str:
    """Read the next SSE data line from a response."""
    while True:
        line = response.readline()
        if not line:
            return ""
        decoded = line.decode("utf-8").strip()
        if not decoded:
            continue
        if decoded.startswith("data:"):
            return decoded


def read_sse_payload(response: http.client.HTTPResponse) -> dict[str, object] | None:
    """Read the next SSE payload from a response."""
    line = read_sse_data_line(response)
    if not line:
        return None
    payload_text = line.removeprefix("data:").strip()
    return json.loads(payload_text)


def start_grpc_server(
    repo_root: Path,
    port: int,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Start the gRPC server subprocess in test mode."""
    env = os.environ.copy()
    # Apply extra_env first, then ensure repo_root is in PYTHONPATH
    if extra_env:
        env.update(extra_env)
    # Ensure repo_root is in PYTHONPATH so modules can be found
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(repo_root)
    env["AUDIO_TO_TEXT_GRPC_TEST_MODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "audio_grpc.server",
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
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Start the backend server subprocess."""
    env = os.environ.copy()
    # Apply extra_env first, then ensure repo_root is in PYTHONPATH
    if extra_env:
        env.update(extra_env)
    # Ensure repo_root is in PYTHONPATH so modules can be found
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(repo_root)
    env["AUDIO_TO_TEXT_BACKEND_GRPC_TARGET"] = grpc_target
    command = [
        sys.executable,
        "-m",
        "backend.server",
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


def run_backend_command(
    repo_root: Path,
    args: Iterable[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the backend CLI and capture output."""
    env = os.environ.copy()
    # Apply extra_env first, then ensure repo_root is in PYTHONPATH
    if extra_env:
        env.update(extra_env)
    # Ensure repo_root is in PYTHONPATH so modules can be found
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(repo_root)
    command = [sys.executable, "-m", "backend.server", *args]
    return subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
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


def build_job_store_payload(job_id: str) -> dict[str, object]:
    """Build a persisted job store payload for tests."""
    job_root = f"/tmp/{job_id}"
    return {
        "change_id": 1,
        "job_order": [job_id],
        "jobs": {
            job_id: {
                "job_id": job_id,
                "created_at": 1.0,
                "input": {
                    "audio_filename": "audio.wav",
                    "text_filename": "text.txt",
                    "language": "en",
                    "remove_punctuation": True,
                    "audio_path": f"{job_root}/upload.wav",
                    "text_path": f"{job_root}/text.txt",
                    "output_path": f"{job_root}/audio.srt",
                },
                "result": {
                    "status": "queued",
                    "message": "Queued",
                    "output_srt": None,
                    "progress": 0.0,
                    "started_at": None,
                    "completed_at": None,
                },
            }
        },
    }


def build_raw_multipart(boundary: str, parts: Iterable[bytes]) -> bytes:
    """Build a raw multipart/form-data payload."""
    boundary_bytes = f"--{boundary}\r\n".encode("utf-8")
    closing_bytes = f"--{boundary}--\r\n".encode("utf-8")
    payload_parts = [boundary_bytes + part + b"\r\n" for part in parts]
    payload_parts.append(closing_bytes)
    return b"".join(payload_parts)


def multipart_field(
    name: str,
    value: str,
    extra_headers: Iterable[str] | None = None,
) -> bytes:
    """Build a multipart field part."""
    header_lines = [f'Content-Disposition: form-data; name="{name}"']
    if extra_headers:
        header_lines.extend(extra_headers)
    header = "\r\n".join(header_lines) + "\r\n\r\n"
    return header.encode("utf-8") + value.encode("utf-8")


def multipart_field_bytes(
    name: str,
    payload: bytes,
    extra_headers: Iterable[str] | None = None,
) -> bytes:
    """Build a multipart field part with raw bytes."""
    header_lines = [f'Content-Disposition: form-data; name="{name}"']
    if extra_headers:
        header_lines.extend(extra_headers)
    header = "\r\n".join(header_lines) + "\r\n\r\n"
    return header.encode("utf-8") + payload


def multipart_file(
    name: str,
    filename: str,
    content_type: str,
    payload: bytes,
) -> bytes:
    """Build a multipart file part."""
    header = (
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    )
    return header.encode("utf-8") + payload


def send_json_post(
    host: str,
    port: int,
    path: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, dict[str, object]]:
    """Send an HTTP POST and decode the JSON response."""
    connection = http.client.HTTPConnection(host, port, timeout=4)
    connection.request("POST", path, body=body, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def write_stub_ffmpeg(script_path: Path, stderr_text: str) -> None:
    """Write a stub ffmpeg executable that exits with an error."""
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "",
                "def main() -> None:",
                f"    sys.stderr.write({stderr_text!r})",
                "    sys.exit(1)",
                "",
                "if __name__ == \"__main__\":",
                "    main()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def write_stub_ffmpeg_success(script_path: Path) -> None:
    """Write a stub ffmpeg executable that emits a valid WAV."""
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import wave",
                "",
                "def main() -> None:",
                "    output_path = sys.argv[-1]",
                "    with wave.open(output_path, \"wb\") as wav_file:",
                "        wav_file.setnchannels(1)",
                "        wav_file.setsampwidth(2)",
                "        wav_file.setframerate(48000)",
                "        wav_file.writeframes(b\"\\x00\\x00\" * 10)",
                "    sys.exit(0)",
                "",
                "if __name__ == \"__main__\":",
                "    main()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def write_uuid_stub(stub_root: Path, hex_value: str) -> None:
    """Write a stub uuid module returning a fixed hex value."""
    stub_root.mkdir(parents=True, exist_ok=True)
    stub_root.joinpath("uuid.py").write_text(
        "\n".join(
            [
                "class _Uuid:",
                "    def __init__(self, value: str) -> None:",
                "        self.hex = value",
                "",
                "def uuid4() -> _Uuid:",
                f"    return _Uuid({hex_value!r})",
                "",
            ]
        ),
        encoding="utf-8",
    )


def extend_pythonpath(stub_root: Path) -> str:
    """Extend PYTHONPATH with a stub module directory."""
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return str(stub_root)
    return f"{stub_root}{os.pathsep}{existing}"


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


def test_audio_to_text_backend_accepts_remove_punctuation_false(
    tmp_path: Path,
) -> None:
    """Accept uploads that keep punctuation."""
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
            {"language": "en", "remove_punctuation": "0"},
            [
                ("audio", "sample.wav", "audio/wav", wav_bytes),
                ("text", "sample.txt", "text/plain", b"Hello, world"),
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
        assert created.get("remove_punctuation") is False
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_extracts_non_wav_upload(tmp_path: Path) -> None:
    """Extract WAV when the upload is not already a WAV."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    ffmpeg_path = tmp_path / "ffmpeg"
    write_stub_ffmpeg_success(ffmpeg_path)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_FFMPEG_PATH": str(ffmpeg_path)},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.mp4", "video/mp4", b"\x00\x00\x00\x18ftypmp42"),
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


def test_audio_to_text_backend_defaults_srt_filename_when_empty(
    tmp_path: Path,
) -> None:
    """Default SRT filename when the audio name is empty."""
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
                ("audio", "/", "audio/wav", wav_bytes),
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
        wait_for_job_completion(base_url, job_id)
        with urlopen(
            f"{base_url}/api/jobs/{job_id}/srt", timeout=5
        ) as response:
            header = response.getheader("Content-Disposition", "")
        assert "filename=\"alignment.srt\"" in header
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
        extra_env={"AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS": "*"},
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
        sse_request = Request(f"{base_url}/api/jobs/events", headers={"Origin": "https://ui.test"})
        with urlopen(sse_request, timeout=5) as response:
            assert response.headers.get("Access-Control-Allow-Origin") == "*"
            assert response.headers.get("Content-Type") == "text/event-stream; charset=utf-8"
            line = read_sse_data_line(response)
        assert line.startswith("data:")
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_sse_updates_match_client_job_id(
    tmp_path: Path,
) -> None:
    """Stream job updates that preserve client job ids."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS": "0.1"},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=5)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        first_payload = read_sse_payload(response)
        assert isinstance(first_payload, dict)
        client_job_id = "client_test_job"
        wav_bytes = build_wav_bytes(0.1)
        payload, boundary = build_multipart(
            {
                "language": "en",
                "remove_punctuation": "1",
                "client_job_id": client_job_id,
            },
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
        matched_job = None
        matched_jobs = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and matched_job is None:
            payload = read_sse_payload(response)
            if not payload:
                continue
            if payload.get("type") != "snapshot":
                continue
            jobs = payload.get("jobs", [])
            if not isinstance(jobs, list):
                continue
            for job in jobs:
                if (
                    isinstance(job, dict)
                    and job.get("client_job_id") == client_job_id
                ):
                    matched_job = job
                    matched_jobs = jobs
                    break
        assert matched_job is not None
        assert matched_jobs is not None
        matching = [
            job
            for job in matched_jobs
            if isinstance(job, dict)
            and job.get("client_job_id") == client_job_id
        ]
        assert len(matching) == 1
        assert matched_job.get("job_id") != client_job_id
        assert matched_job.get("client_job_id") == client_job_id
        connection.close()
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_options_and_cors(tmp_path: Path) -> None:
    """Serve CORS preflight responses when requested."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS": "https://example.com"},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=4)
        connection.putrequest("OPTIONS", "/api/jobs")
        connection.putheader("Origin", "https://example.com")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 204
        assert response.getheader("Access-Control-Allow-Origin") == "https://example.com"
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_upload_too_large(tmp_path: Path) -> None:
    """Reject uploads exceeding size limits."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_MAX_UPLOAD_BYTES": "10"},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", b"01234567890"),
                ("text", "sample.txt", "text/plain", b"Hello"),
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
        try:
            with urlopen(request, timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "upload exceeds max size" in payload.get("error", "")
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_rejects_invalid_upload_headers(
    tmp_path: Path,
) -> None:
    """Reject malformed upload headers."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        cases = [
            ({"Content-Length": "1"}, b"x", "Content-Type header is required"),
            (
                {"Content-Type": "application/json", "Content-Length": "1"},
                b"x",
                "expected multipart form data",
            ),
            (
                {"Content-Type": "multipart/form-data", "Content-Length": "1"},
                b"x",
                "multipart boundary is required",
            ),
            (
                {
                    "Content-Type": "multipart/form-data; boundary=stub",
                    "Content-Length": "0",
                },
                b"",
                "Content-Length must be positive",
            ),
            (
                {
                    "Content-Type": "multipart/form-data; boundary=stub",
                    "Content-Length": "nope",
                },
                b"x",
                "Content-Length must be an integer",
            ),
        ]
        for headers, body, expected in cases:
            status, payload = send_json_post(
                "127.0.0.1", backend_port, "/api/jobs", headers, body
            )
            assert status == 400
            assert expected in payload.get("error", "")
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=4)
        connection.putrequest("POST", "/api/jobs")
        connection.putheader("Content-Type", "multipart/form-data; boundary=stub")
        connection.endheaders()
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 400
        assert "Content-Length header is required" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_rejects_incomplete_body(tmp_path: Path) -> None:
    """Reject uploads with incomplete request bodies."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        with socket.create_connection(("127.0.0.1", backend_port), timeout=3) as sock:
            request = (
                "POST /api/jobs HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: multipart/form-data; boundary=stub\r\n"
                "Content-Length: 10\r\n"
                "\r\n"
                "abc"
            ).encode("utf-8")
            sock.sendall(request)
            sock.shutdown(socket.SHUT_WR)
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        assert b"request body is incomplete" in response
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_rejects_invalid_upload_payloads(
    tmp_path: Path,
) -> None:
    """Reject malformed multipart payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        boundary = "raw-boundary"
        audio_part = multipart_file("audio", "sample.wav", "audio/wav", b"RIFF")
        text_part = multipart_file("text", "sample.txt", "text/plain", b"Hello")

        def send_body(body: bytes) -> str:
            """Send a raw multipart body and return the error message."""
            status, payload = send_json_post(
                "127.0.0.1",
                backend_port,
                "/api/jobs",
                {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(body)),
                },
                body,
            )
            assert status == 400
            return str(payload.get("error", ""))

        cases = [
            (b"not-a-multipart", "multipart parse failed"),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("device", "cpu"),
                        multipart_field("language", "en"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "unexpected form field: device",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "en"),
                        audio_part,
                        text_part,
                    ],
                ),
                "missing form field: remove_punctuation",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("client_job_id", " "),
                        multipart_field("language", "en"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "client_job_id must be non-empty",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "en"),
                        multipart_field("remove_punctuation", "maybe"),
                        audio_part,
                        text_part,
                    ],
                ),
                "remove_punctuation must be a boolean",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "en"),
                        multipart_field("remove_punctuation", "1"),
                        text_part,
                    ],
                ),
                "missing upload field: audio",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "en"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                    ],
                ),
                "missing upload field: text",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "en"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        audio_part,
                        text_part,
                    ],
                ),
                "duplicate upload field: audio",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "en"),
                        multipart_field("language", "ru"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "duplicate form field: language",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "missing form field: language",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", " "),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "language must be provided",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field("language", "xx"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "unsupported language",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        b'Content-Disposition: form-data; filename="sample.wav"\r\n'
                        b"Content-Type: audio/wav\r\n\r\nRIFF",
                    ],
                ),
                "multipart field name is required",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field(
                            "language",
                            "en",
                            ["Content-Type: text/plain; charset=iso-8859-1"],
                        ),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "unsupported charset",
            ),
            (
                build_raw_multipart(
                    boundary,
                    [
                        multipart_field_bytes("language", b"\xff"),
                        multipart_field("remove_punctuation", "1"),
                        audio_part,
                        text_part,
                    ],
                ),
                "invalid UTF-8",
            ),
        ]
        for body, expected in cases:
            error = send_body(body)
            assert expected in error
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_job_not_ready(tmp_path: Path) -> None:
    """Reject SRT downloads before completion."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(
        repo_root,
        grpc_port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "200"},
    )
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
        try:
            with urlopen(f"{base_url}/api/jobs/{job_id}/srt", timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "job is not complete" in payload.get("error", "")
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_missing_output(tmp_path: Path) -> None:
    """Reject SRT downloads when output files are missing or unreadable."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "backend-data"
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
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
        wait_for_job_completion(base_url, job_id)
        output_path = data_dir / job_id / "sample.srt"
        assert output_path.exists()
        output_path.unlink()
        try:
            with urlopen(f"{base_url}/api/jobs/{job_id}/srt", timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "output not found" in payload.get("error", "")
        output_path.mkdir()
        try:
            with urlopen(f"{base_url}/api/jobs/{job_id}/srt", timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "output read failed" in payload.get("error", "")
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_srt_missing_job(tmp_path: Path) -> None:
    """Reject SRT downloads for unknown jobs."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        try:
            with urlopen(f"http://127.0.0.1:{backend_port}/api/jobs/missing/srt", timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "job not found" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_keepalive_sse(tmp_path: Path) -> None:
    """Emit keepalive SSE frames when no changes occur."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS": "0.1"},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        with urlopen(f"http://127.0.0.1:{backend_port}/api/jobs/events", timeout=5) as response:
            first_line = read_sse_data_line(response)
            assert first_line.startswith("data:")
            payload = json.loads(first_line.removeprefix("data:").strip())
            assert payload.get("type") == "snapshot"
            keepalive = read_sse_data_line(response)
            assert keepalive.startswith("data:")
            keepalive_payload = json.loads(keepalive.removeprefix("data:").strip())
            assert keepalive_payload.get("type") == "keepalive"
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_sse_snapshot_failure(tmp_path: Path) -> None:
    """Handle SSE snapshot failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        extra_env={
            audio_to_text_backend_server.SSE_FAILURE_MODE_ENV: (
                audio_to_text_backend_server.SSE_FAILURE_SNAPSHOT
            )
        },
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=4)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        line = response.readline()
        assert line == b""
        connection.close()
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_sse_keepalive_failure(tmp_path: Path) -> None:
    """Handle SSE keepalive failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        extra_env={
            audio_to_text_backend_server.SSE_FAILURE_MODE_ENV: (
                audio_to_text_backend_server.SSE_FAILURE_KEEPALIVE
            ),
            "AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS": "0.1",
        },
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        with urlopen(f"http://127.0.0.1:{backend_port}/api/jobs/events", timeout=4) as response:
            first_line = read_sse_data_line(response)
            assert first_line.startswith("data:")
            tail = read_sse_data_line(response)
            assert tail == ""
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_sse_disconnect_on_snapshot(tmp_path: Path) -> None:
    """Handle SSE clients disconnecting before the first snapshot."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        reset_sse_connection("127.0.0.1", backend_port, "/api/jobs/events")
        time.sleep(0.2)
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_sse_disconnect_on_update(tmp_path: Path) -> None:
    """Handle SSE disconnects when sending an update."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS": "0.2"},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=4)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        first_line = read_sse_data_line(response)
        assert first_line.startswith("data:")
        connection.close()
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
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
        time.sleep(0.2)
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_sse_disconnect_on_keepalive(tmp_path: Path) -> None:
    """Handle SSE disconnects before keepalive frames."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        extra_env={"AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS": "0.2"},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=5)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        first_line = read_sse_data_line(response)
        assert first_line.startswith("data:")
        reset_http_connection(connection)
        time.sleep(0.4)
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_loads_auth_token_from_env(tmp_path: Path) -> None:
    """Load gRPC auth token from environment when unset in CLI."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_backend_command(
        repo_root,
        ["--data-dir", str(tmp_path / "data"), "--port", "0"],
        extra_env={"AUDIO_TO_TEXT_BACKEND_GRPC_AUTH_TOKEN": "secret"},
    )
    assert result.returncode != 0
    assert "audio_to_text_backend.config.invalid" in result.stderr


def test_audio_to_text_backend_prefers_cli_auth_token(tmp_path: Path) -> None:
    """Prefer gRPC auth token from CLI when provided."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_backend_command(
        repo_root,
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--port",
            "0",
            "--grpc-auth-token",
            "secret",
        ],
    )
    assert result.returncode != 0
    assert "audio_to_text_backend.config.invalid" in result.stderr


def test_audio_to_text_backend_rejects_disallowed_origin(tmp_path: Path) -> None:
    """Skip CORS headers for disallowed origins."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        extra_env={"AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS": "https://allowed.test"},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=5)
        connection.request(
            "GET",
            "/api/jobs",
            headers={"Origin": "https://denied.test"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.headers.get("Access-Control-Allow-Origin") is None
        connection.close()
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_cli_help(tmp_path: Path) -> None:
    """Expose CLI usage via --help."""
    repo_root = Path(__file__).resolve().parents[1]
    result = run_backend_command(
        repo_root,
        ["--help"],
        extra_env={"AUDIO_TO_TEXT_BACKEND_DATA_DIR": str(tmp_path / "data")},
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_audio_to_text_backend_delete_paths(tmp_path: Path) -> None:
    """Delete completed jobs and reject invalid deletes."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(
        repo_root,
        grpc_port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "200"},
    )
    backend_data = tmp_path / "backend-data"
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        backend_data,
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

        missing_delete = Request(f"{base_url}/api/jobs", method="DELETE")
        try:
            with urlopen(missing_delete, timeout=3):
                raise AssertionError("delete should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert "audio_to_text_backend.path.not_found" in payload.get("error", "")

        delete_missing_job = Request(
            f"{base_url}/api/jobs/missing", method="DELETE"
        )
        try:
            with urlopen(delete_missing_job, timeout=3):
                raise AssertionError("delete should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 400
            assert "job not found" in payload.get("error", "")

        delete_request = Request(
            f"{base_url}/api/jobs/{job_id}", method="DELETE"
        )
        try:
            with urlopen(delete_request, timeout=3):
                raise AssertionError("delete should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 400
            assert "only finished jobs can be deleted" in payload.get("error", "")

        completed = wait_for_job_completion(base_url, job_id)
        assert completed.get("status") == "completed"
        delete_request = Request(
            f"{base_url}/api/jobs/{job_id}", method="DELETE"
        )
        with urlopen(delete_request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload.get("job_id") == job_id
        assert not (backend_data / job_id).exists()
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_delete_job_artifacts_failure(
    tmp_path: Path,
) -> None:
    """Reject deletes when job artifacts cannot be removed."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    data_dir = tmp_path / "backend-data"
    job_id = "job-1"
    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"] = {
        "status": "completed",
        "message": "Done",
        "output_srt": "/tmp/output.srt",
        "progress": 1.0,
        "started_at": 1.0,
        "completed_at": 2.0,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.joinpath("jobs.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    data_dir.joinpath(job_id).write_text("blocked", encoding="utf-8")
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        delete_request = Request(
            f"http://127.0.0.1:{backend_port}/api/jobs/{job_id}",
            method="DELETE",
        )
        try:
            with urlopen(delete_request, timeout=3):
                raise AssertionError("delete should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 400
            assert "job artifacts delete failed" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_delete_missing_artifacts(
    tmp_path: Path,
) -> None:
    """Delete jobs even when artifacts are already missing."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    data_dir = tmp_path / "backend-data"
    job_id = "job-1"
    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"] = {
        "status": "completed",
        "message": "Done",
        "output_srt": "/tmp/output.srt",
        "progress": 1.0,
        "started_at": 1.0,
        "completed_at": 2.0,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.joinpath("jobs.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        delete_request = Request(
            f"http://127.0.0.1:{backend_port}/api/jobs/{job_id}",
            method="DELETE",
        )
        with urlopen(delete_request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload.get("job_id") == job_id
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_rejects_duplicate_job_id(tmp_path: Path) -> None:
    """Return an error when a job id already exists."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "backend-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    job_id = "fixed-job"
    job_dir = data_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    jobs_payload = {
        "change_id": 1,
        "job_order": [job_id],
        "jobs": {
            job_id: {
                "job_id": job_id,
                "created_at": 1.0,
                "input": {
                    "audio_filename": "audio.wav",
                    "text_filename": "text.txt",
                    "language": "en",
                    "remove_punctuation": True,
                    "audio_path": str(job_dir / "upload.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "audio.srt"),
                },
                "result": {
                    "status": "queued",
                    "message": "Queued",
                    "output_srt": None,
                    "progress": 0.0,
                    "started_at": None,
                    "completed_at": None,
                },
            }
        },
    }
    (data_dir / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    stub_root = tmp_path / "uuid_stub"
    write_uuid_stub(stub_root, job_id)
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:59999",
        extra_env={"PYTHONPATH": extend_pythonpath(stub_root)},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", b"RIFF"),
                ("text", "sample.txt", "text/plain", b"Hello"),
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
        try:
            with urlopen(request, timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert "job already exists" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_job_store_write_failure(
    tmp_path: Path,
) -> None:
    """Return an error when job store persistence fails."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "backend-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "jobs.tmp").mkdir(parents=True, exist_ok=True)
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", b"RIFF"),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = Request(
            f"http://127.0.0.1:{backend_port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urlopen(request, timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert "job store write failed" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_unhandled_error_sets_job_failed(
    tmp_path: Path,
) -> None:
    """Mark jobs failed when unexpected errors occur."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    data_dir = tmp_path / "backend-data"
    grpc_process = start_grpc_server(
        repo_root,
        grpc_port,
        extra_env={"AUDIO_TO_TEXT_GRPC_TEST_DELAY_MS": "500"},
    )
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        f"127.0.0.1:{grpc_port}",
        extra_env={"AUDIO_TO_TEXT_BACKEND_MAX_WORKERS": "1"},
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
            created_first = json.loads(response.read().decode("utf-8"))
        first_job_id = created_first.get("job_id")
        assert isinstance(first_job_id, str)
        time.sleep(0.1)
        with urlopen(request, timeout=5) as response:
            created_second = json.loads(response.read().decode("utf-8"))
        second_job_id = created_second.get("job_id")
        assert isinstance(second_job_id, str)
        audio_path = data_dir / second_job_id / "upload.wav"
        if audio_path.exists():
            audio_path.unlink()
        audio_path.mkdir(parents=True, exist_ok=True)
        completed = wait_for_job_completion(base_url, second_job_id)
        assert completed.get("status") == "failed"
        assert "audio_to_text_backend.unhandled_error" in str(
            completed.get("message", "")
        )
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_not_found_route(tmp_path: Path) -> None:
    """Return 404 for unknown endpoints."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        try:
            with urlopen(f"http://127.0.0.1:{backend_port}/missing", timeout=3):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert "audio_to_text_backend.path.not_found" in payload.get("error", "")
        try:
            with urlopen(
                f"http://127.0.0.1:{backend_port}/api/jobs/missing/events",
                timeout=3,
            ):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert "audio_to_text_backend.path.not_found" in payload.get("error", "")
        request = Request(
            f"http://127.0.0.1:{backend_port}/api/unknown",
            data=b"",
            method="POST",
            headers={"Content-Length": "0"},
        )
        try:
            with urlopen(request, timeout=3):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert "audio_to_text_backend.path.not_found" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_logging_levels(tmp_path: Path) -> None:
    """Apply configured logging levels."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "backend-data"
    for level in ("DEBUG", "WARNING", "ERROR"):
        result = run_backend_command(
            repo_root,
            ["--port", "0", "--data-dir", str(data_dir)],
            {"AUDIO_TO_TEXT_BACKEND_LOG_LEVEL": level},
        )
        assert result.returncode != 0


def test_audio_to_text_backend_rejects_invalid_job_store(
    tmp_path: Path,
) -> None:
    """Reject invalid persisted job store data."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "backend-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "jobs.json").write_text("{", encoding="utf-8")
    result = run_backend_command(
        repo_root,
        [
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
            "--data-dir",
            str(data_dir),
            "--grpc-target",
            "127.0.0.1:1234",
        ],
    )
    assert result.returncode != 0
    assert "job store load failed" in result.stderr


def test_audio_to_text_backend_rejects_invalid_job_store_payloads(
    tmp_path: Path,
) -> None:
    """Reject invalid job store payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    job_id = "job-1"
    cases: list[tuple[str, object]] = []

    cases.append(("job store must be a dictionary", []))

    payload = build_job_store_payload(job_id)
    payload["job_order"] = "bad"
    cases.append(("job_order must be a list", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"] = []
    cases.append(("jobs must be a dictionary", payload))

    payload = build_job_store_payload(job_id)
    payload["change_id"] = "bad"
    cases.append(("change_id must be an integer", payload))

    payload = build_job_store_payload(job_id)
    payload["change_id"] = -1
    cases.append(("change_id must be non-negative", payload))

    payload = build_job_store_payload(job_id)
    payload["job_order"] = [123]
    cases.append(("job ids must be strings", payload))

    payload = build_job_store_payload(job_id)
    payload["job_order"] = ["missing"]
    payload["jobs"] = {}
    cases.append(("job missing from store: missing", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"]["extra"] = payload["jobs"][job_id]
    cases.append(("job store entries are inconsistent", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["job_id"] = "other"
    cases.append((f"job id mismatch: {job_id}", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["audio_filename"] = 123
    cases.append(("audio_filename must be a string", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["audio_filename"] = " "
    cases.append(("audio filename is required", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["text_filename"] = " "
    cases.append(("text filename is required", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["language"] = " "
    cases.append(("language is required", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["audio_path"] = " "
    cases.append(("audio path is required", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["text_path"] = " "
    cases.append(("text path is required", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["output_path"] = " "
    cases.append(("output path is required", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["client_job_id"] = " "
    cases.append(("client job id must be non-empty", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["input"]["remove_punctuation"] = "yes"
    cases.append(("remove_punctuation must be a boolean", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["message"] = 123
    cases.append(("message must be a string", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["started_at"] = "bad"
    cases.append(("started_at must be a number", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["created_at"] = "bad"
    cases.append(("created_at must be a number", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["status"] = "unknown"
    cases.append(("status is invalid: unknown", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["progress"] = 1.5
    cases.append(("progress must be between 0 and 1", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["started_at"] = -1.0
    cases.append(("started_at must be non-negative", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["completed_at"] = -1.0
    cases.append(("completed_at must be non-negative", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["started_at"] = 2.0
    payload["jobs"][job_id]["result"]["completed_at"] = 1.0
    cases.append(("completed_at must be after started_at", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["started_at"] = 1.0
    cases.append(("queued jobs cannot have timestamps", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["status"] = "running"
    cases.append(
        ("running jobs must have started_at and no completed_at", payload)
    )

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["status"] = "completed"
    payload["jobs"][job_id]["result"]["output_srt"] = "output.srt"
    cases.append(
        ("completed jobs must have start and completion times", payload)
    )

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["status"] = "failed"
    payload["jobs"][job_id]["result"]["output_srt"] = "output.srt"
    payload["jobs"][job_id]["result"]["started_at"] = 1.0
    payload["jobs"][job_id]["result"]["completed_at"] = 2.0
    payload["jobs"][job_id]["result"]["progress"] = 1.0
    cases.append(("output path is only valid for completed jobs", payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["result"]["status"] = "completed"
    payload["jobs"][job_id]["result"]["started_at"] = 1.0
    payload["jobs"][job_id]["result"]["completed_at"] = 2.0
    payload["jobs"][job_id]["result"]["progress"] = 1.0
    cases.append(("completed jobs must include output path", payload))

    empty_job_id_payload = build_job_store_payload("")
    cases.append(("job id is required", empty_job_id_payload))

    payload = build_job_store_payload(job_id)
    payload["jobs"][job_id]["created_at"] = -1.0
    cases.append(("job created_at must be non-negative", payload))

    for case_index, (expected, payload) in enumerate(cases, start=1):
        data_dir = tmp_path / f"job-store-{case_index}"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_dir.joinpath("jobs.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        result = run_backend_command(
            repo_root,
            [
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
                "--data-dir",
                str(data_dir),
                "--grpc-target",
                "127.0.0.1:1234",
            ],
        )
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_backend_rejects_invalid_config(tmp_path: Path) -> None:
    """Reject invalid backend configuration values."""
    repo_root = Path(__file__).resolve().parents[1]
    base_env = {"AUDIO_TO_TEXT_BACKEND_DATA_DIR": str(tmp_path / "data")}
    cases = [
        (["--host", " "], {}, "host must be non-empty"),
        (["--port", "0"], {}, "port must be between 1 and 65535"),
        ([], {"AUDIO_TO_TEXT_BACKEND_PORT": "0"}, "port must be positive"),
        ([], {"AUDIO_TO_TEXT_BACKEND_PORT": "bad"}, "port must be an integer"),
        ([], {"AUDIO_TO_TEXT_BACKEND_GRPC_TARGET": " "}, "grpc-target must be non-empty"),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_GRPC_TIMEOUT_SECONDS": "-1"},
            "grpc-timeout-seconds must be non-negative",
        ),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_GRPC_TIMEOUT_SECONDS": "bad"},
            "grpc-timeout-seconds must be a number",
        ),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_GRPC_MAX_MESSAGE_BYTES": "0"},
            "grpc-max-message-bytes must be positive",
        ),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_GRPC_MAX_MESSAGE_BYTES": "bad"},
            "grpc-max-message-bytes must be an integer",
        ),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_MAX_UPLOAD_BYTES": "0"},
            "max-upload-bytes must be positive",
        ),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_MAX_UPLOAD_BYTES": "bad"},
            "max-upload-bytes must be an integer",
        ),
        ([], {"AUDIO_TO_TEXT_BACKEND_FFMPEG_PATH": " "}, "ffmpeg-path must be non-empty"),
        ([], {"AUDIO_TO_TEXT_BACKEND_MAX_WORKERS": "0"}, "max-workers must be positive"),
        ([], {"AUDIO_TO_TEXT_BACKEND_MAX_WORKERS": "bad"}, "max-workers must be an integer"),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_KEEPALIVE_SECONDS": "0"},
            "keepalive-seconds must be positive",
        ),
        (
            [],
            {"AUDIO_TO_TEXT_BACKEND_SSE_FAILURE_MODE": "invalid"},
            "sse-failure-mode is invalid",
        ),
    ]
    for args, env_overrides, expected in cases:
        env = {**base_env, **env_overrides}
        result = run_backend_command(repo_root, args, env)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_backend_rejects_invalid_cli_values(tmp_path: Path) -> None:
    """Reject invalid backend CLI overrides."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    base_args = [
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--data-dir",
        str(data_dir),
        "--grpc-target",
        "127.0.0.1:1234",
    ]
    cases = [
        (["--grpc-timeout-seconds", "-1"], "grpc-timeout-seconds must be non-negative"),
        (["--grpc-max-message-bytes", "0"], "grpc-max-message-bytes must be positive"),
        (["--max-upload-bytes", "0"], "max-upload-bytes must be positive"),
        (["--max-workers", "0"], "max-workers must be positive"),
    ]
    for extra_args, expected in cases:
        result = run_backend_command(repo_root, [*base_args, *extra_args])
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_backend_cli_overrides_env(tmp_path: Path) -> None:
    """Apply CLI overrides for backend configuration values."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--data-dir",
        str(data_dir),
        "--grpc-target",
        " ",
        "--grpc-timeout-seconds",
        "1",
        "--grpc-max-message-bytes",
        "1024",
        "--max-upload-bytes",
        "1024",
        "--allowed-origins",
        "https://allowed.test",
        "--ffmpeg-path",
        "ffmpeg",
        "--max-workers",
        "1",
        "--keepalive-seconds",
        "1",
    ]
    result = run_backend_command(
        repo_root,
        args,
        {
            "AUDIO_TO_TEXT_BACKEND_GRPC_TARGET": "127.0.0.1:1234",
            "AUDIO_TO_TEXT_BACKEND_GRPC_MAX_MESSAGE_BYTES": "2048",
            "AUDIO_TO_TEXT_BACKEND_MAX_UPLOAD_BYTES": "2048",
        },
    )
    assert result.returncode != 0
    assert "grpc-target must be non-empty" in result.stderr


def test_audio_to_text_backend_allows_configured_origin(tmp_path: Path) -> None:
    """Apply CORS headers for an allowed origin."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    data_dir = tmp_path / "backend-data"
    process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:1234",
        extra_env={"AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS": "https://allowed.test"},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        request = Request(f"http://127.0.0.1:{backend_port}/health")
        request.add_header("Origin", "https://allowed.test")
        with urlopen(request, timeout=3) as response:
            assert response.headers.get("Access-Control-Allow-Origin") == "https://allowed.test"
    finally:
        stop_process(process)


def test_audio_to_text_backend_cors_headers(tmp_path: Path) -> None:
    """Emit CORS headers for allowed origins."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        {"AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS": "https://allowed.test"},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        request = Request(
            f"http://127.0.0.1:{backend_port}/health",
            headers={"Origin": "https://allowed.test"},
        )
        with urlopen(request, timeout=3) as response:
            assert response.headers.get("Access-Control-Allow-Origin") == "https://allowed.test"
        with urlopen(f"http://127.0.0.1:{backend_port}/health", timeout=3) as response:
            assert response.headers.get("Access-Control-Allow-Origin") is None
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_cors_any_origin(tmp_path: Path) -> None:
    """Emit wildcard CORS headers when configured."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        {"AUDIO_TO_TEXT_BACKEND_ALLOWED_ORIGINS": "*"},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        request = Request(
            f"http://127.0.0.1:{backend_port}/health",
            headers={"Origin": "https://any.test"},
        )
        with urlopen(request, timeout=3) as response:
            assert response.headers.get("Access-Control-Allow-Origin") == "*"
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_grpc_failure_sets_job_failed(tmp_path: Path) -> None:
    """Surface gRPC failures in job status."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        {
            "AUDIO_TO_TEXT_BACKEND_GRPC_TIMEOUT_SECONDS": "0.2",
            "AUDIO_TO_TEXT_BACKEND_GRPC_USE_TLS": "1",
        },
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        wav_bytes = build_wav_bytes(0.1)
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
        assert completed.get("status") == "failed"
        assert "grpc alignment failed" in str(completed.get("message", ""))
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_invalid_wav_fails_job(tmp_path: Path) -> None:
    """Fail jobs when the uploaded WAV is invalid."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", b"not-a-wav"),
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
        assert completed.get("status") == "failed"
        assert "invalid wav" in str(completed.get("message", ""))
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_ffmpeg_process_error(tmp_path: Path) -> None:
    """Fail jobs when ffmpeg returns an error."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    stub_path = tmp_path / "ffmpeg-stub"
    write_stub_ffmpeg(stub_path, "ffmpeg failed")
    grpc_process = start_grpc_server(repo_root, grpc_port)
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        {"AUDIO_TO_TEXT_BACKEND_FFMPEG_PATH": str(stub_path)},
    )
    try:
        wait_for_port("127.0.0.1", grpc_port, timeout_seconds=5)
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.mp4", "video/mp4", b"fake"),
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
        assert completed.get("status") == "failed"
        assert "ffmpeg extraction failed" in str(completed.get("message", ""))
    finally:
        stop_process(backend_process)
        stop_process(grpc_process)


def test_audio_to_text_backend_ffmpeg_failure_fails_job(tmp_path: Path) -> None:
    """Fail jobs when ffmpeg cannot run."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
        {"AUDIO_TO_TEXT_BACKEND_FFMPEG_PATH": "/missing/ffmpeg"},
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.mp4", "video/mp4", b"fake"),
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
        assert completed.get("status") == "failed"
        assert "ffmpeg execution failed" in str(completed.get("message", ""))
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_invalid_text_fails_job(tmp_path: Path) -> None:
    """Fail jobs when the transcript text is not valid UTF-8."""
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = free_local_port()
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        "127.0.0.1:59999",
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        base_url = f"http://127.0.0.1:{backend_port}"
        wav_bytes = build_wav_bytes(0.1)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", wav_bytes),
                ("text", "sample.txt", "text/plain", b"\xff"),
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
        assert completed.get("status") == "failed"
        assert "audio_to_text.input.text_file" in str(completed.get("message", ""))
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_job_setup_failure(tmp_path: Path) -> None:
    """Return errors when job directories cannot be created."""
    repo_root = Path(__file__).resolve().parents[1]
    job_id = "deadbeef"
    data_dir = tmp_path / "backend-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / job_id).write_text("not a directory", encoding="utf-8")
    stub_root = tmp_path / "stubs"
    write_uuid_stub(stub_root, job_id)
    backend_port = free_local_port()
    env = {"PYTHONPATH": f"{stub_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:59999",
        env,
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", build_wav_bytes(0.1)),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = Request(
            f"http://127.0.0.1:{backend_port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urlopen(request, timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "audio_to_text_backend.job.setup_failed" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_upload_write_failure(tmp_path: Path) -> None:
    """Return errors when uploaded files cannot be saved."""
    repo_root = Path(__file__).resolve().parents[1]
    job_id = "feedface"
    data_dir = tmp_path / "backend-data"
    job_dir = data_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "upload.wav").mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_uuid_stub(stub_root, job_id)
    backend_port = free_local_port()
    env = {"PYTHONPATH": f"{stub_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        data_dir,
        "127.0.0.1:59999",
        env,
    )
    try:
        wait_for_port("127.0.0.1", backend_port, timeout_seconds=5)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", build_wav_bytes(0.1)),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = Request(
            f"http://127.0.0.1:{backend_port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urlopen(request, timeout=5):
                raise AssertionError("request should fail")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "audio_to_text_backend.upload.write_failed" in payload.get("error", "")
    finally:
        stop_process(backend_process)


def test_audio_to_text_backend_auth_token_and_timeout(tmp_path: Path) -> None:
    """Align successfully with auth metadata and no timeout."""
    repo_root = Path(__file__).resolve().parents[1]
    grpc_port = free_local_port()
    backend_port = free_local_port()
    grpc_process = start_grpc_server(
        repo_root,
        grpc_port,
        extra_env={"AUDIO_TO_TEXT_GRPC_AUTH_TOKEN": "secret"},
    )
    backend_process = start_backend_server(
        repo_root,
        backend_port,
        tmp_path / "backend-data",
        f"127.0.0.1:{grpc_port}",
        extra_env={
            "AUDIO_TO_TEXT_BACKEND_GRPC_AUTH_TOKEN": "secret",
            "AUDIO_TO_TEXT_BACKEND_GRPC_TIMEOUT_SECONDS": "0",
        },
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
