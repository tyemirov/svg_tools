"""Integration tests for audio_to_text CLI."""

from __future__ import annotations

import http.client
import os
import platform
import subprocess
import signal
import wave
import json
import math
import socket
import struct
import time
import urllib.error
import urllib.request
import sys
from pathlib import Path
from typing import List

import pytest
import audio_to_text

if platform.system().lower() != "linux":
    pytest.skip(
        "audio_to_text is supported on Linux only; use Docker",
        allow_module_level=True,
    )


def run_audio_to_text(
    args: List[str],
    repo_root: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run audio_to_text.py with the provided arguments."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def write_wav(target_path: Path, duration_seconds: float) -> None:
    """Write a silent PCM WAV file."""
    sample_rate = 48000
    frame_count = max(1, int(round(duration_seconds * sample_rate)))
    silence = b"\x00\x00" * frame_count
    with wave.open(str(target_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence)


def parse_srt_timestamp(value: str) -> float:
    """Parse an SRT timestamp into seconds."""
    time_value = value.strip()
    hours_text, minutes_text, seconds_text = time_value.split(":")
    seconds_part, millis_text = seconds_text.split(",")
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + int(seconds_part)
        + int(millis_text) / 1000.0
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


def free_local_port() -> int:
    """Reserve a free port for local HTTP servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_ui_ready(base_url: str, timeout_seconds: float) -> None:
    """Wait until the UI server responds."""
    start = time.monotonic()
    while True:
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("UI server did not become ready in time")
        try:
            with urllib.request.urlopen(f"{base_url}/api/jobs", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(0.05)


def build_multipart(
    fields: dict[str, str],
    files: list[tuple[str, str, str, bytes]],
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


def build_raw_multipart(boundary: str, parts: list[bytes]) -> bytes:
    """Build a raw multipart body from preformatted parts."""
    boundary_bytes = boundary.encode("utf-8")
    body = b""
    for part in parts:
        body += b"--" + boundary_bytes + b"\r\n" + part + b"\r\n"
    body += b"--" + boundary_bytes + b"--\r\n"
    return body


def read_json(url: str) -> dict[str, object]:
    """Read JSON from a URL."""
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_job_status(base_url: str, job_id: str, timeout_seconds: float) -> dict[str, object]:
    """Wait until a job reaches a terminal state."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        payload = read_json(f"{base_url}/api/jobs")
        jobs = payload.get("jobs", [])
        if isinstance(jobs, list):
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                if job.get("job_id") != job_id:
                    continue
                status = job.get("status")
                if status in ("completed", "failed"):
                    return job
        if time.monotonic() > deadline:
            raise TimeoutError("job did not complete in time")
        time.sleep(0.1)


def extend_pythonpath(stub_root: Path) -> str:
    """Extend PYTHONPATH with a stub module directory."""
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return str(stub_root)
    return f"{stub_root}{os.pathsep}{existing}"


def write_torch_stub(stub_root: Path) -> None:
    """Write a minimal torch stub module."""
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


def write_failing_torch_stub(stub_root: Path) -> None:
    """Write a torch stub that raises during device resolution."""
    stub_root.mkdir(parents=True, exist_ok=True)
    torch_dir = stub_root / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)
    torch_dir.joinpath("__init__.py").write_text(
        "\n".join(
            [
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available() -> bool:",
                "        raise RuntimeError(\"cuda failure\")",
                "",
                "cuda = _Cuda()",
                "__version__ = \"2.6.0\"",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_stub_modules(
    stub_root: Path,
    include_platform: bool = False,
    torchaudio_has_metadata: bool = True,
    backend_metadata: bool = False,
) -> None:
    """Write stub torch, torchaudio, and whisperx modules for tests."""
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
    metadata_lines = []
    if torchaudio_has_metadata:
        metadata_lines = ["class AudioMetaData:", "    pass", ""]
    torchaudio_dir.joinpath("__init__.py").write_text(
        "\n".join(metadata_lines),
        encoding="utf-8",
    )
    if backend_metadata:
        backend_dir = torchaudio_dir / "backend"
        backend_dir.mkdir(parents=True, exist_ok=True)
        backend_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
        backend_dir.joinpath("common.py").write_text(
            "\n".join(["class AudioMetaData:", "    pass", ""]),
            encoding="utf-8",
        )
    whisperx_dir = stub_root / "whisperx"
    whisperx_dir.mkdir(parents=True, exist_ok=True)
    whisperx_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    whisperx_dir.joinpath("audio.py").write_text(
        "\n".join(
            [
                "import os",
                "SAMPLE_RATE = 48000",
                "",
                "def load_audio(path: str):",
                "    multiplier_text = os.environ.get(\"AUDIO_TO_TEXT_TEST_AUDIO_MULTIPLIER\", \"1\")",
                "    try:",
                "        multiplier = int(multiplier_text)",
                "    except ValueError:",
                "        multiplier = 1",
                "    return [0.0] * SAMPLE_RATE * max(1, multiplier)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    whisperx_dir.joinpath("alignment.py").write_text(
        "\n".join(
            [
                "import os",
                "import time",
                "",
                "def load_align_model(language_code: str, device: str):",
                "    mode = os.environ.get(\"AUDIO_TO_TEXT_TEST_ALIGN_MODE\", \"\")",
                "    if mode == \"model_fail\":",
                "        raise RuntimeError(\"model load failed\")",
                "    return object(), {}",
                "",
                "def align(segments, align_model, metadata, audio, device, return_char_alignments=False):",
                "    mode = os.environ.get(\"AUDIO_TO_TEXT_TEST_ALIGN_MODE\", \"\")",
                "    delay_text = os.environ.get(\"AUDIO_TO_TEXT_TEST_ALIGN_DELAY\", \"0\")",
                "    try:",
                "        delay_seconds = float(delay_text)",
                "    except ValueError:",
                "        delay_seconds = 0.0",
                "    if delay_seconds > 0:",
                "        time.sleep(delay_seconds)",
                "    if mode == \"align_fail\":",
                "        raise RuntimeError(\"alignment failed\")",
                "    if mode == \"invalid_segments\":",
                "        return {\"segments\": [\"bad\"]}",
                "    if mode == \"punctuation\":",
                "        words = [",
                "            {\"word\": \"Hello!\", \"start\": 0.0, \"end\": 0.1},",
                "            {\"word\": \"🙂\", \"start\": None, \"end\": None},",
                "            {\"word\": \"🙂\", \"start\": 0.1, \"end\": 0.2},",
                "            {\"word\": \"!!!\", \"start\": 0.2, \"end\": 0.3},",
                "            {\"word\": \"world\", \"start\": 0.3, \"end\": 0.4},",
                "        ]",
                "        return {\"segments\": [{\"words\": words}]}",
                "    if mode == \"missing_middle\":",
                "        words = [",
                "            {\"word\": \"Hello\", \"start\": 0.0, \"end\": 0.2},",
                "            {\"word\": \"gap\", \"start\": None, \"end\": None},",
                "            {\"word\": \"World\", \"start\": 0.6, \"end\": 0.8},",
                "        ]",
                "        return {\"segments\": [{\"start\": 0.0, \"end\": 1.0, \"words\": words}]}",
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
    if include_platform:
        stub_root.joinpath("platform.py").write_text(
            "\n".join(
                [
                    "def system() -> str:",
                    "    return \"Darwin\"",
                    "",
                ]
            ),
            encoding="utf-8",
        )


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


def start_ui_server(
    repo_root: Path,
    ui_root: Path,
    port: int,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Start the audio_to_text UI server."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.Popen(
        [
            sys.executable,
            str(repo_root / "audio_to_text.py"),
            "--ui",
            "--ui-host",
            "127.0.0.1",
            "--ui-port",
            str(port),
            "--ui-root-dir",
            str(ui_root),
        ],
        cwd=repo_root,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_audio_to_text_missing_audio(tmp_path: Path) -> None:
    """Fail when the input audio file does not exist."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    text_path = tmp_path / "input.txt"
    text_path.write_text("hello world", encoding="utf-8")

    args = [
        str(script_path),
        "--input-audio",
        str(tmp_path / "missing.wav"),
        "--input-text",
        str(text_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.audio_file" in result.stderr


def test_audio_to_text_missing_text(tmp_path: Path) -> None:
    """Fail when the input text file does not exist."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)

    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(tmp_path / "missing.txt"),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.text_file" in result.stderr


def test_audio_to_text_invalid_language(tmp_path: Path) -> None:
    """Fail when an unsupported language is requested."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)

    text_path = tmp_path / "input.txt"
    text_path.write_text("hello world", encoding="utf-8")

    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--language",
        "xx",
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.invalid_language" in result.stderr


def test_audio_to_text_srt_sanitizes_empty_text(tmp_path: Path) -> None:
    """Fail when SRT input contains only timestamps after sanitization."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)

    srt_path = tmp_path / "input.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\n\n", encoding="utf-8"
    )

    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(srt_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.invalid_config" in result.stderr


def test_audio_to_text_srt_sanitizes_text_lines(tmp_path: Path) -> None:
    """Strip SRT indices and time ranges before alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    srt_path = tmp_path / "input.srt"
    srt_path.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:00,200",
                "Hello world",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(srt_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode == 0
    content = output_path.read_text(encoding="utf-8")
    assert "Hello" in content


def test_audio_to_text_alignment_json_merges_punctuation_missing_timestamps(
    tmp_path: Path,
) -> None:
    """Merge punctuation tokens that have no timestamps instead of failing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {"word": "В", "start": 0.0, "end": 0.1},
                            {"word": "низовьях", "start": 0.1, "end": 0.3},
                            {"word": "—", "start": None, "end": None},
                            {"word": "реки", "start": 0.3, "end": 0.5},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert "низовьях —" in content


def test_audio_to_text_alignment_json_infers_missing_timestamps_for_words(
    tmp_path: Path,
) -> None:
    """Infer timestamps when whisperx emits word tokens without timings."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "words": [
                            {"word": "A", "start": 0.0, "end": 0.2},
                            {"word": "B", "start": None, "end": None},
                            {"word": "C", "start": 0.8, "end": 1.0},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert "\nB\n" in content


def test_audio_to_text_alignment_json_infers_missing_timestamps_without_segment_bounds(
    tmp_path: Path,
) -> None:
    """Infer timestamps when segment bounds are missing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {"word": "One", "start": None, "end": None},
                            {"word": "Two", "start": None, "end": None},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert content.index("One") < content.index("Two")


def test_audio_to_text_alignment_json_recovers_nonfinite_timestamps(
    tmp_path: Path,
) -> None:
    """Recover when alignment JSON includes non-finite timestamps."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "words": [
                            {"word": "A", "start": 0.0, "end": 0.2},
                            {"word": "B", "start": math.nan, "end": math.nan},
                            {"word": "C", "start": 0.8, "end": 1.0},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert "\nB\n" in content


def test_audio_to_text_alignment_json_infers_boolean_timestamps(
    tmp_path: Path,
) -> None:
    """Infer timestamps when boolean values appear in alignment JSON."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "words": [
                            {"word": "A", "start": 0.0, "end": 0.2},
                            {"word": "B", "start": True, "end": False},
                            {"word": "C", "start": 0.8, "end": 1.0},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert "\nB\n" in content


def test_audio_to_text_alignment_json_infers_with_overlapping_bounds(
    tmp_path: Path,
) -> None:
    """Infer timestamps when neighbor bounds overlap."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "words": [
                            {"word": "Alpha", "start": 0.0, "end": 1.0},
                            {"word": "Beta", "start": None, "end": None},
                            {"word": "Gamma", "start": 0.5, "end": 0.6},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert "\nBeta\n" in content


def test_audio_to_text_alignment_json_uses_token_bounds(
    tmp_path: Path,
) -> None:
    """Derive segment bounds from token timestamps when missing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {"word": "One", "start": 0.0, "end": 0.2},
                            {"word": "Two", "start": 0.3, "end": 0.5},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_srt = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_srt),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_srt.read_text(encoding="utf-8")
    assert "One" in content
    assert "Two" in content


def test_audio_to_text_alignment_json_invalid_word_payloads(tmp_path: Path) -> None:
    """Fail when alignment JSON contains invalid word payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    output_path = tmp_path / "output.srt"
    alignment_json = tmp_path / "alignment.json"
    cases = [
        ({"segments": [{"words": "nope"}]}, "alignment segment words must be a list"),
        (
            {"segments": [{"words": ["nope"]}]},
            "alignment word payload must be an object",
        ),
    ]
    for payload, expected in cases:
        alignment_json.write_text(json.dumps(payload), encoding="utf-8")
        args = [
            str(script_path),
            "--input-alignment-json",
            str(alignment_json),
            "--output-srt",
            str(output_path),
        ]
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_alignment_json_negative_timestamp(tmp_path: Path) -> None:
    """Fail when alignment JSON includes negative timestamps."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Hello", "start": -0.2, "end": -0.1}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.align.missing_timestamps" in result.stderr


def test_audio_to_text_alignment_json_rejects_nonincreasing_timestamps(
    tmp_path: Path,
) -> None:
    """Fail when aligned word timestamps are not increasing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Oops", "start": 0.5, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "aligned word end is not after start" in result.stderr


def test_audio_to_text_alignment_json_infers_between_neighbors(
    tmp_path: Path,
) -> None:
    """Infer missing timestamps between neighboring aligned words."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "words": [
                            {"word": "Hello", "start": 0.0, "end": 0.2},
                            {"word": "gap", "start": None, "end": None},
                            {"word": "World", "start": 0.6, "end": 0.8},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    original_argv = sys.argv
    try:
        sys.argv = [
            str(script_path),
            "--input-alignment-json",
            str(alignment_json),
            "--output-srt",
            str(output_path),
        ]
        exit_code = audio_to_text.main()
    finally:
        sys.argv = original_argv

    assert exit_code == 0
    content = output_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    gap_index = lines.index("gap")
    timing_line = lines[gap_index - 1]
    start_text, end_text = timing_line.split(" --> ")
    start_seconds = parse_srt_timestamp(start_text)
    end_seconds = parse_srt_timestamp(end_text)
    assert 0.19 <= start_seconds <= 0.21
    assert 0.59 <= end_seconds <= 0.61


def test_audio_to_text_alignment_infers_missing_with_stub(
    tmp_path: Path,
) -> None:
    """Infer missing timestamps in the alignment pipeline."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello gap World", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    original_argv = sys.argv
    original_sys_path = list(sys.path)
    existing_modules = set(sys.modules)
    original_mode = os.environ.get("AUDIO_TO_TEXT_TEST_ALIGN_MODE")
    try:
        sys.path.insert(0, str(stub_root))
        os.environ["AUDIO_TO_TEXT_TEST_ALIGN_MODE"] = "missing_middle"
        sys.argv = [
            str(script_path),
            "--input-audio",
            str(audio_path),
            "--input-text",
            str(text_path),
            "--output-srt",
            str(output_path),
        ]
        exit_code = audio_to_text.main()
    finally:
        sys.argv = original_argv
        sys.path = original_sys_path
        if original_mode is None:
            os.environ.pop("AUDIO_TO_TEXT_TEST_ALIGN_MODE", None)
        else:
            os.environ["AUDIO_TO_TEXT_TEST_ALIGN_MODE"] = original_mode
        for name in list(sys.modules):
            if name not in existing_modules and name.startswith(
                ("whisperx", "torch", "torchaudio")
            ):
                sys.modules.pop(name, None)

    assert exit_code == 0
    content = output_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    gap_index = lines.index("gap")
    timing_line = lines[gap_index - 1]
    start_text, end_text = timing_line.split(" --> ")
    start_seconds = parse_srt_timestamp(start_text)
    end_seconds = parse_srt_timestamp(end_text)
    assert 0.19 <= start_seconds <= 0.21
    assert 0.59 <= end_seconds <= 0.61


def test_audio_to_text_alignment_json_rejects_empty_word(tmp_path: Path) -> None:
    """Fail when aligned word text is empty."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": " ", "start": 0.0, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "aligned word text is empty" in result.stderr


def test_audio_to_text_alignment_json_handles_punctuation_branches(
    tmp_path: Path,
) -> None:
    """Merge punctuation tokens and pending prefixes during alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {"word": "—", "start": None, "end": None},
                            {"word": "Hello", "start": 0.0, "end": 0.2},
                            {"word": "!", "start": None, "end": None},
                        ]
                    },
                    {
                        "words": [
                            {"word": "?", "start": None, "end": None},
                            {"word": "world", "start": 0.2, "end": 0.4},
                        ]
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    content = output_path.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "world" in content


def test_audio_to_text_alignment_json_fails_without_words(tmp_path: Path) -> None:
    """Fail when alignment produces no words."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "…", "start": None, "end": None}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "alignment produced no words" in result.stderr


def test_audio_to_text_ui_rejects_invalid_job_store(tmp_path: Path) -> None:
    """Reject invalid persisted job state when loading the UI."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "bad-job"
    jobs_payload = {
        "change_id": 0,
        "job_order": [job_id],
        "jobs": {
            job_id: {
                "job_id": job_id,
                "created_at": 1.0,
                "input": {
                    "audio_filename": "audio.wav",
                    "text_filename": "text.txt",
                    "language": "en",
                    "remove_punctuation": "yes",
                    "audio_path": str(ui_root / job_id / "audio.wav"),
                    "text_path": str(ui_root / job_id / "text.txt"),
                    "output_path": str(ui_root / job_id / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    args = [
        str(script_path),
        "--ui",
        "--ui-host",
        "127.0.0.1",
        "--ui-port",
        str(port),
        "--ui-root-dir",
        str(ui_root),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "remove_punctuation must be a boolean" in result.stderr


def test_audio_to_text_ui_deletes_completed_job(tmp_path: Path) -> None:
    """Delete a completed job via the UI API."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)

    completed_job_id = "deadbeef"
    completed_dir = ui_root / completed_job_id
    completed_dir.mkdir(parents=True, exist_ok=True)
    output_srt = completed_dir / "alignment.srt"
    output_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,100\nHello\n", encoding="utf-8"
    )

    failed_job_id = "badc0ffe"
    failed_dir = ui_root / failed_job_id
    failed_dir.mkdir(parents=True, exist_ok=True)

    jobs_payload = {
        "change_id": 1,
        "job_order": [completed_job_id, failed_job_id],
        "jobs": {
            completed_job_id: {
                "job_id": completed_job_id,
                "created_at": 1.0,
                "input": {
                    "audio_filename": "audio.wav",
                    "text_filename": "text.txt",
                    "language": "en",
                    "remove_punctuation": True,
                    "audio_path": str(completed_dir / "audio.wav"),
                    "text_path": str(completed_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
            ,
            failed_job_id: {
                "job_id": failed_job_id,
                "created_at": 1.5,
                "input": {
                    "audio_filename": "audio.wav",
                    "text_filename": "text.txt",
                    "language": "en",
                    "remove_punctuation": True,
                    "audio_path": str(failed_dir / "audio.wav"),
                    "text_path": str(failed_dir / "text.txt"),
                    "output_path": str(failed_dir / "alignment.srt"),
                },
                "result": {
                    "status": "failed",
                    "message": "Failed",
                    "output_srt": None,
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            },
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")

    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        with urllib.request.urlopen(
            f"{base_url}/api/jobs/{completed_job_id}/srt", timeout=2.0
        ) as response:
            assert response.status == 200
            content_disposition = response.headers.get("Content-Disposition", "")
            assert "audio.srt" in content_disposition
        delete_failed_request = urllib.request.Request(
            f"{base_url}/api/jobs/{failed_job_id}", method="DELETE"
        )
        with urllib.request.urlopen(delete_failed_request, timeout=2.0) as response:
            assert response.status == 200
        delete_request = urllib.request.Request(
            f"{base_url}/api/jobs/{completed_job_id}", method="DELETE"
        )
        with urllib.request.urlopen(delete_request, timeout=2.0) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode("utf-8"))
            assert payload.get("deleted") is True
        with urllib.request.urlopen(f"{base_url}/api/jobs", timeout=2.0) as response:
            jobs = json.loads(response.read().decode("utf-8"))
            assert jobs.get("jobs") == []
        assert not completed_dir.exists()
        assert not failed_dir.exists()
    finally:
        stop_process(process)


def test_audio_to_text_ui_deletes_missing_artifacts(tmp_path: Path) -> None:
    """Delete completed jobs even when artifacts are missing."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "missing-artifacts"
    output_path = ui_root / job_id / "alignment.srt"
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
                    "audio_path": str(ui_root / job_id / "audio.wav"),
                    "text_path": str(ui_root / job_id / "text.txt"),
                    "output_path": str(output_path),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_path),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            },
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        delete_request = urllib.request.Request(
            f"{base_url}/api/jobs/{job_id}", method="DELETE"
        )
        with urllib.request.urlopen(delete_request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload.get("deleted") is True
        assert not (ui_root / job_id).exists()
    finally:
        stop_process(process)


def test_audio_to_text_ui_delete_artifacts_failure(tmp_path: Path) -> None:
    """Return an error when deleting job artifacts fails."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "artifact-failure"
    job_dir_path = ui_root / job_id
    job_dir_path.write_text("not a directory", encoding="utf-8")
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
                    "audio_path": str(job_dir_path / "audio.wav"),
                    "text_path": str(job_dir_path / "text.txt"),
                    "output_path": str(job_dir_path / "alignment.srt"),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(job_dir_path / "alignment.srt"),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        delete_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs/{job_id}", method="DELETE"
        )
        try:
            with urllib.request.urlopen(delete_request, timeout=3):
                raise AssertionError("delete should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert "audio_to_text.ui.storage" in payload.get("error", "")
    finally:
        stop_process(process)


def test_audio_to_text_ui_srt_uses_fallback_filename(tmp_path: Path) -> None:
    """Fall back to alignment.srt when audio filename has no basename."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "fallback-name"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_srt = job_dir / "alignment.srt"
    output_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,100\nHello\n", encoding="utf-8"
    )
    jobs_payload = {
        "change_id": 1,
        "job_order": [job_id],
        "jobs": {
            job_id: {
                "job_id": job_id,
                "created_at": 1.0,
                "input": {
                    "audio_filename": "/",
                    "text_filename": "text.txt",
                    "language": "en",
                    "remove_punctuation": True,
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/jobs/{job_id}/srt", timeout=3
        ) as response:
            content_disposition = response.headers.get("Content-Disposition", "")
            assert "alignment.srt" in content_disposition
    finally:
        stop_process(process)


def test_audio_to_text_cli_aligns_with_stub_whisperx(tmp_path: Path) -> None:
    """Run CLI alignment using stub whisperx modules."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.4)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello, world!", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_DELAY": "0.6",
    }
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
        "--language",
        "ru",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode == 0
    content = output_path.read_text(encoding="utf-8")
    assert "Hello," in content
    assert "world!" in content


def test_audio_to_text_cli_rejects_invalid_alignment_segments(
    tmp_path: Path,
) -> None:
    """Fail when whisperx returns invalid segment payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_MODE": "invalid_segments",
    }
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
        "--language",
        "en",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "alignment segment payload must be an object" in result.stderr


def test_audio_to_text_cli_uses_backend_torchaudio_metadata_module(
    tmp_path: Path,
) -> None:
    """Load AudioMetaData from torchaudio backend module."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(
        stub_root,
        torchaudio_has_metadata=False,
        backend_metadata=True,
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
        "--language",
        "en",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode == 0
    assert output_path.exists()


def test_audio_to_text_cli_rejects_missing_torch(tmp_path: Path) -> None:
    """Fail when torch cannot be imported."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Привет мир", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    stub_root.joinpath("torch", "__init__.py").write_text(
        "raise ImportError('missing torch')\n",
        encoding="utf-8",
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
        "--language",
        "ru",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "torch is unavailable" in result.stderr


def test_audio_to_text_cli_rejects_missing_torchaudio(tmp_path: Path) -> None:
    """Fail when torchaudio cannot be imported."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    stub_root.joinpath("torchaudio", "__init__.py").write_text(
        "raise ImportError('missing torchaudio')\n",
        encoding="utf-8",
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "torchaudio is unavailable" in result.stderr


def test_audio_to_text_cli_rejects_missing_torchaudio_metadata(
    tmp_path: Path,
) -> None:
    """Fail when torchaudio lacks AudioMetaData."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    stub_root.mkdir(parents=True, exist_ok=True)
    write_torch_stub(stub_root)
    torchaudio_dir = stub_root / "torchaudio"
    torchaudio_dir.mkdir(parents=True, exist_ok=True)
    torchaudio_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "torchaudio is missing AudioMetaData" in result.stderr


def test_audio_to_text_cli_uses_backend_torchaudio_metadata(
    tmp_path: Path,
) -> None:
    """Fallback to torchaudio backend metadata when missing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(
        stub_root,
        torchaudio_has_metadata=False,
        backend_metadata=True,
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode == 0
    assert output_path.exists()


def test_audio_to_text_cli_uses_backend_metadata_fallback_module(
    tmp_path: Path,
) -> None:
    """Load AudioMetaData from torchaudio fallback backend modules."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(
        stub_root,
        torchaudio_has_metadata=False,
        backend_metadata=False,
    )
    backend_dir = stub_root / "torchaudio" / "backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    backend_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    backend_dir.joinpath("common.py").write_text(
        "class NotMeta:\n    pass\n",
        encoding="utf-8",
    )
    fallback_dir = stub_root / "torchaudio" / "_backend"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    fallback_dir.joinpath("common.py").write_text(
        "class AudioMetaData:\n    pass\n",
        encoding="utf-8",
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode == 0
    assert output_path.exists()


def test_audio_to_text_cli_reports_alignment_model_failure(
    tmp_path: Path,
) -> None:
    """Report errors when alignment model loading fails."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_MODE": "model_fail",
    }
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "audio_to_text.align.failed" in result.stderr


def test_audio_to_text_cli_rejects_invalid_torch_version(tmp_path: Path) -> None:
    """Fail when torch version is invalid."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Привет мир", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    stub_root.joinpath("torch", "__init__.py").write_text(
        "\n".join(
            [
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available() -> bool:",
                "        return False",
                "",
                "cuda = _Cuda()",
                "__version__ = \"bad\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
        "--language",
        "ru",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "torch version is invalid" in result.stderr


def test_audio_to_text_cli_rejects_old_torch_version(tmp_path: Path) -> None:
    """Fail when torch is older than the required version."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Привет мир", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    stub_root.joinpath("torch", "__init__.py").write_text(
        "\n".join(
            [
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available() -> bool:",
                "        return False",
                "",
                "cuda = _Cuda()",
                "__version__ = \"2.5.0\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
        "--language",
        "ru",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "torch >=" in result.stderr


def test_audio_to_text_cli_rejects_whisperx_unavailable(tmp_path: Path) -> None:
    """Fail when whisperx is not available as a package."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    stub_root.mkdir(parents=True, exist_ok=True)
    write_torch_stub(stub_root)
    torchaudio_dir = stub_root / "torchaudio"
    torchaudio_dir.mkdir(parents=True, exist_ok=True)
    torchaudio_dir.joinpath("__init__.py").write_text(
        "class AudioMetaData:\n    pass\n",
        encoding="utf-8",
    )
    stub_root.joinpath("whisperx.py").write_text("", encoding="utf-8")
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "whisperx is unavailable" in result.stderr


def test_audio_to_text_cli_rejects_whisperx_import_failure(
    tmp_path: Path,
) -> None:
    """Fail when whisperx alignment modules are missing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    stub_root.mkdir(parents=True, exist_ok=True)
    write_torch_stub(stub_root)
    torchaudio_dir = stub_root / "torchaudio"
    torchaudio_dir.mkdir(parents=True, exist_ok=True)
    torchaudio_dir.joinpath("__init__.py").write_text(
        "class AudioMetaData:\n    pass\n",
        encoding="utf-8",
    )
    whisperx_dir = stub_root / "whisperx"
    whisperx_dir.mkdir(parents=True, exist_ok=True)
    whisperx_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "whisperx alignment import failed" in result.stderr


def test_audio_to_text_cli_reports_unhandled_exception(tmp_path: Path) -> None:
    """Report unhandled exceptions during CLI alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    output_path = tmp_path / "output.srt"
    stub_root = tmp_path / "stubs"
    write_failing_torch_stub(stub_root)
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "audio_to_text.unhandled_error" in result.stderr


def test_audio_to_text_cli_rejects_device_override(tmp_path: Path) -> None:
    """Fail when a device override is provided."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
        "--device",
        "cpu",
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.invalid_config" in result.stderr


def test_audio_to_text_cli_rejects_output_extension(tmp_path: Path) -> None:
    """Fail when output-srt does not end with .srt."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(tmp_path / "output.txt"),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.invalid_config" in result.stderr


def test_audio_to_text_cli_defaults_output_srt_for_alignment_json(
    tmp_path: Path,
) -> None:
    """Derive output SRT path from alignment JSON input."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    expected_output = alignment_json.with_suffix(".srt")
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    assert expected_output.exists()


def test_audio_to_text_cli_infers_missing_bounds_from_neighbors(
    tmp_path: Path,
) -> None:
    """Infer missing token timings using adjacent timestamps."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 2.0,
                        "end": 1.0,
                        "words": [
                            {"word": "alpha", "start": 0.0, "end": 0.5},
                            {"word": "beta", "start": None, "end": None},
                            {"word": "gamma", "start": 1.0, "end": 1.5},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode == 0
    assert output_path.exists()


def test_audio_to_text_cli_rejects_non_linux_runtime(tmp_path: Path) -> None:
    """Fail when the runtime is not Linux."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    env_overrides = {"AUDIO_TO_TEXT_PLATFORM_OVERRIDE": "darwin"}
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(tmp_path / "output.srt"),
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "audio_to_text.dependency.platform" in result.stderr


def test_audio_to_text_cli_rejects_invalid_requests(tmp_path: Path) -> None:
    """Reject invalid CLI argument combinations."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    cases = [
        (
            [
                str(script_path),
                "--input-alignment-json",
                str(alignment_json),
                "--output-srt",
                str(output_path),
                "--language",
                " ",
            ],
            "language must be non-empty",
        ),
        (
            [
                str(script_path),
                "--input-alignment-json",
                str(alignment_json),
                "--output-srt",
                str(output_path),
                "--device",
                "bogus",
            ],
            "invalid device",
        ),
        (
            [
                str(script_path),
                "--input-alignment-json",
                " ",
                "--output-srt",
                str(output_path),
            ],
            "input-alignment-json must be non-empty",
        ),
        (
            [
                str(script_path),
                "--input-alignment-json",
                str(alignment_json),
                "--input-audio",
                str(tmp_path / "audio.wav"),
                "--output-srt",
                str(output_path),
            ],
            "input-alignment-json cannot be combined",
        ),
        (
            [
                str(script_path),
                "--input-alignment-json",
                str(alignment_json),
                "--output-srt",
                " ",
            ],
            "output-srt must be non-empty",
        ),
        (
            [
                str(script_path),
                "--ui",
                "--ui-host",
                " ",
            ],
            "ui-host must be non-empty",
        ),
        (
            [
                str(script_path),
                "--ui",
                "--ui-port",
                "0",
            ],
            "ui-port is invalid",
        ),
        (
            [
                str(script_path),
                "--ui",
                "--ui-root-dir",
                " ",
            ],
            "ui-root-dir must be non-empty",
        ),
    ]
    for args, expected in cases:
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_ui_root_dir_failure(tmp_path: Path) -> None:
    """Fail when the UI root directory cannot be created."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    ui_root = tmp_path / "uploads"
    ui_root.write_text("not a directory", encoding="utf-8")
    args = [
        str(script_path),
        "--ui",
        "--ui-host",
        "127.0.0.1",
        "--ui-port",
        str(free_local_port()),
        "--ui-root-dir",
        str(ui_root),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "failed to create upload directory" in result.stderr


def test_audio_to_text_ui_rejects_invalid_sse_failure_mode(
    tmp_path: Path,
) -> None:
    """Reject invalid UI SSE failure mode values."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    env_overrides = {audio_to_text.UI_SSE_FAILURE_MODE_ENV: "invalid"}
    args = [
        str(script_path),
        "--ui",
        "--ui-host",
        "127.0.0.1",
        "--ui-port",
        "8085",
    ]

    result = run_audio_to_text(args, repo_root, env_overrides)

    assert result.returncode != 0
    assert "audio_to_text.input.invalid_config" in result.stderr
    assert "ui sse failure mode is invalid" in result.stderr


def test_audio_to_text_cli_requires_input_audio_and_text(tmp_path: Path) -> None:
    """Fail when CLI inputs are missing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    text_path = tmp_path / "input.txt"
    write_wav(audio_path, 0.2)
    text_path.write_text("Hello world", encoding="utf-8")
    cases = [
        (
            [
                str(script_path),
                "--input-audio",
                str(audio_path),
            ],
            "input-text is required",
        ),
        (
            [
                str(script_path),
                "--input-text",
                str(text_path),
            ],
            "input-audio is required",
        ),
    ]
    for args, expected in cases:
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_cli_rejects_empty_input_values(tmp_path: Path) -> None:
    """Fail when input paths are blank strings."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    text_path = tmp_path / "input.txt"
    write_wav(audio_path, 0.2)
    text_path.write_text("Hello world", encoding="utf-8")
    cases = [
        (
            [
                str(script_path),
                "--input-audio",
                "",
                "--input-text",
                str(text_path),
            ],
            "input-audio is required",
        ),
        (
            [
                str(script_path),
                "--input-audio",
                str(audio_path),
                "--input-text",
                "",
            ],
            "input-text is required",
        ),
    ]
    for args, expected in cases:
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_cli_rejects_invalid_text_utf8(tmp_path: Path) -> None:
    """Fail when the input text file is not UTF-8."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_bytes(b"\xff")
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "input text file is not valid UTF-8" in result.stderr


def test_audio_to_text_cli_rejects_unreadable_text(tmp_path: Path) -> None:
    """Fail when the input text file cannot be read."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, 0.2)
    text_path = tmp_path / "input.txt"
    text_path.write_text("Hello world", encoding="utf-8")
    text_path.chmod(0o000)
    args = [
        str(script_path),
        "--input-audio",
        str(audio_path),
        "--input-text",
        str(text_path),
    ]
    try:
        result = run_audio_to_text(args, repo_root)
    finally:
        text_path.chmod(0o600)

    assert result.returncode != 0
    assert "audio_to_text.input.text_file" in result.stderr


def test_audio_to_text_cli_alignment_json_errors(tmp_path: Path) -> None:
    """Fail on invalid alignment JSON input."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    output_path = tmp_path / "output.srt"
    missing_path = tmp_path / "missing.json"
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    non_dict_path = tmp_path / "list.json"
    non_dict_path.write_text(json.dumps(["nope"]), encoding="utf-8")
    segments_not_list = tmp_path / "segments.json"
    segments_not_list.write_text(json.dumps({"segments": "nope"}), encoding="utf-8")
    segment_not_dict = tmp_path / "segment.json"
    segment_not_dict.write_text(json.dumps({"segments": ["nope"]}), encoding="utf-8")
    cases = [
        (missing_path, "input alignment json not found"),
        (invalid_json, "input alignment json is invalid"),
        (non_dict_path, "input alignment json must be an object"),
        (segments_not_list, "input alignment json segments must be a list"),
        (segment_not_dict, "input alignment json segments must contain objects"),
    ]
    for path, expected in cases:
        args = [
            str(script_path),
            "--input-alignment-json",
            str(path),
            "--output-srt",
            str(output_path),
        ]
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_cli_rejects_missing_output_dir(tmp_path: Path) -> None:
    """Fail when the output directory does not exist."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {"segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}]}
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "missing" / "output.srt"
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "output directory does not exist" in result.stderr


def test_audio_to_text_cli_write_srt_failure(tmp_path: Path) -> None:
    """Fail when the output SRT cannot be written."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
    alignment_json = tmp_path / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {"segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.2}]}]}
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.srt"
    output_path.mkdir(parents=True, exist_ok=True)
    args = [
        str(script_path),
        "--input-alignment-json",
        str(alignment_json),
        "--output-srt",
        str(output_path),
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "failed to write srt file" in result.stderr


def test_audio_to_text_ui_serves_html(tmp_path: Path) -> None:
    """Serve the UI HTML template."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        with urllib.request.urlopen(f"{base_url}/", timeout=3) as response:
            html = response.read().decode("utf-8")
            assert "Audio to Text Alignment" in html
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_flow_with_stub_alignment(tmp_path: Path) -> None:
    """Create a UI job and download its SRT using stub alignment."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_DELAY": "0.6",
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.4)
        payload, boundary = build_multipart(
            {"language": "ru", "remove_punctuation": "0", "device": "cpu"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello, world!"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        completed = wait_for_job_status(base_url, job_id, timeout_seconds=10.0)
        assert completed.get("status") == "completed"
        with urllib.request.urlopen(
            f"{base_url}/api/jobs/{job_id}", timeout=3
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload.get("job_id") == job_id
            assert payload.get("remove_punctuation") is False
        with urllib.request.urlopen(
            f"{base_url}/api/jobs/{job_id}/srt", timeout=3
        ) as response:
            content_disposition = response.headers.get("Content-Disposition", "")
            assert "sample.srt" in content_disposition
            srt_payload = response.read().decode("utf-8")
            assert "Hello," in srt_payload
    finally:
        stop_process(process)


def test_audio_to_text_ui_drops_punctuation_tokens(tmp_path: Path) -> None:
    """Drop punctuation tokens when removal is enabled."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_MODE": "punctuation",
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.4)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello world"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        completed = wait_for_job_status(base_url, job_id, timeout_seconds=10.0)
        assert completed.get("status") == "completed"
        with urllib.request.urlopen(
            f"{base_url}/api/jobs/{job_id}/srt", timeout=3
        ) as response:
            srt_payload = response.read().decode("utf-8")
        assert "Hello" in srt_payload
        assert "world" in srt_payload
        assert "🙂" not in srt_payload
    finally:
        stop_process(process)


def test_audio_to_text_ui_alignment_failure_marks_job_failed(
    tmp_path: Path,
) -> None:
    """Mark UI jobs failed when alignment errors occur."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_MODE": "align_fail",
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello world"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        failed = wait_for_job_status(base_url, job_id, timeout_seconds=8.0)
        assert failed.get("status") == "failed"
        assert "audio_to_text.align" in str(failed.get("message", ""))
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_punctuation_only_text(
    tmp_path: Path,
) -> None:
    """Fail when punctuation removal leaves no transcript text."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "1"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"!!!"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        failed = wait_for_job_status(base_url, job_id, timeout_seconds=8.0)
        assert failed.get("status") == "failed"
        message = failed.get("message", "")
        assert "input text contains no words after punctuation removal" in message
    finally:
        stop_process(process)


def test_audio_to_text_ui_estimates_long_audio_progress(tmp_path: Path) -> None:
    """Estimate progress with audio durations above the minimum."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_AUDIO_MULTIPLIER": "5",
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello world"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        completed = wait_for_job_status(base_url, job_id, timeout_seconds=8.0)
        assert completed.get("status") == "completed"
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_invalid_upload(tmp_path: Path) -> None:
    """Reject malformed uploads and invalid flags."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                raise AssertionError("upload should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "audio_to_text.ui.upload.invalid" in payload.get("error", "")

        payload, boundary = build_multipart(
            {"language": "en", "remove_punctuation": "maybe"},
            [
                ("audio", "sample.wav", "audio/wav", b"RIFF"),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                raise AssertionError("upload should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert "remove_punctuation must be a boolean" in payload.get("error", "")
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_invalid_headers(tmp_path: Path) -> None:
    """Reject missing or invalid upload headers."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)

    def send_request(headers: dict[str, str], body: bytes) -> tuple[int, str]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request("POST", "/api/jobs", body=body, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload.get("error", "")

    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        status, error = send_request({"Content-Length": "1"}, b"x")
        assert status == 400
        assert "Content-Type header is required" in error
        status, error = send_request(
            {"Content-Type": "application/json", "Content-Length": "1"}, b"x"
        )
        assert status == 400
        assert "expected multipart form data" in error
        status, error = send_request(
            {"Content-Type": "multipart/form-data", "Content-Length": "1"}, b"x"
        )
        assert status == 400
        assert "multipart boundary is required" in error
        status, error = send_request(
            {
                "Content-Type": "multipart/form-data; boundary=stub",
                "Content-Length": "0",
            },
            b"",
        )
        assert status == 400
        assert "Content-Length must be positive" in error
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.putrequest("POST", "/api/jobs")
        connection.putheader("Content-Type", "multipart/form-data; boundary=stub")
        connection.endheaders()
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 400
        assert "Content-Length header is required" in payload.get("error", "")
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_incomplete_body(tmp_path: Path) -> None:
    """Reject requests with incomplete bodies."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
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
        stop_process(process)


def test_audio_to_text_ui_rejects_invalid_multipart(tmp_path: Path) -> None:
    """Reject malformed multipart payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)

    def field_part(name: str, value: str, extra_headers: list[str] | None = None) -> bytes:
        header_lines = [f'Content-Disposition: form-data; name="{name}"']
        if extra_headers:
            header_lines.extend(extra_headers)
        header = "\r\n".join(header_lines) + "\r\n\r\n"
        return header.encode("utf-8") + value.encode("utf-8")

    def field_part_bytes(name: str, payload: bytes, extra_headers: list[str] | None = None) -> bytes:
        header_lines = [f'Content-Disposition: form-data; name="{name}"']
        if extra_headers:
            header_lines.extend(extra_headers)
        header = "\r\n".join(header_lines) + "\r\n\r\n"
        return header.encode("utf-8") + payload

    def file_part(name: str, filename: str, content_type: str, payload: bytes) -> bytes:
        header = (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        )
        return header.encode("utf-8") + payload

    def send_body(body: bytes, boundary: str) -> tuple[int, str]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request(
            "POST",
            "/api/jobs",
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload.get("error", "")

    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        boundary = "raw-boundary"
        not_multipart = b"not-a-multipart"
        status, error = send_body(not_multipart, boundary)
        assert status == 400
        assert "multipart parse failed" in error

        parts = [
            field_part("language", "en"),
            field_part("bogus", "1"),
            file_part("audio", "sample.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "unexpected form field" in error

        parts = [
            file_part("audio", "sample.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "missing form field: language" in error

        parts = [
            field_part("language", "en"),
            field_part("language", "ru"),
            file_part("audio", "sample.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "duplicate form field" in error

        parts = [
            field_part("language", "en"),
            file_part("audio", "first.wav", "audio/wav", b"RIFF"),
            file_part("audio", "second.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "duplicate upload field" in error

        parts = [
            b'Content-Disposition: form-data; filename="sample.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\nRIFF",
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "multipart field name is required" in error

        parts = [
            field_part("language", "en"),
            file_part("audio", "", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "missing upload field: audio" in error

        parts = [
            field_part("language", "en", ["Content-Type: text/plain; charset=iso-8859-1"]),
            file_part("audio", "sample.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "unsupported charset" in error

        parts = [
            field_part_bytes("language", b"\xff"),
            file_part("audio", "sample.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "invalid UTF-8" in error

        parts = [
            b"Content-Disposition: form-data; name=\"language\"\r\n"
            b"Content-Type: multipart/mixed; boundary=sub\r\n\r\n"
            b"--sub--",
            file_part("audio", "sample.wav", "audio/wav", b"RIFF"),
            file_part("text", "sample.txt", "text/plain", b"Hello"),
        ]
        status, error = send_body(build_raw_multipart(boundary, parts), boundary)
        assert status == 400
        assert "language must be provided" in error
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_invalid_job_store_payloads(
    tmp_path: Path,
) -> None:
    """Fail fast on invalid job store payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"

    def base_payload(job_id: str, job_dir: Path) -> dict[str, object]:
        output_srt = job_dir / "alignment.srt"
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
                        "audio_path": str(job_dir / "audio.wav"),
                        "text_path": str(job_dir / "text.txt"),
                        "output_path": str(output_srt),
                    },
                    "result": {
                        "status": "completed",
                        "message": "Complete",
                        "output_srt": str(output_srt),
                        "progress": 1.0,
                        "started_at": 2.0,
                        "completed_at": 3.0,
                    },
                }
            },
        }

    cases: list[tuple[str | None, str, callable | None]] = [
        ("{", "job store load failed", None),
        (None, "job_order must be a list", lambda payload: payload.update(job_order="bad")),
        (None, "jobs must be a dictionary", lambda payload: payload.update(jobs="bad")),
        (None, "change_id must be an integer", lambda payload: payload.update(change_id="bad")),
        (None, "change_id must be non-negative", lambda payload: payload.update(change_id=-1)),
        (None, "job ids must be strings", lambda payload: payload.update(job_order=[123])),
        (
            None,
            "job missing from store",
            lambda payload: payload.update(job_order=["missing"], jobs={}),
        ),
        (
            None,
            "job store entries are inconsistent",
            lambda payload: payload["jobs"].update({"extra": {}}),
        ),
        (
            None,
            "job id mismatch",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]].update(
                job_id="other"
            ),
        ),
        (
            None,
            "input must be a dictionary",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]].update(
                input="bad"
            ),
        ),
        (
            None,
            "result must be a dictionary",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]].update(
                result="bad"
            ),
        ),
        (
            None,
            "audio_filename must be a string",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                audio_filename=123
            ),
        ),
        (
            None,
            "created_at must be a number",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]].update(
                created_at="bad"
            ),
        ),
        (
            None,
            "remove_punctuation must be a boolean",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                remove_punctuation="bad"
            ),
        ),
        (
            None,
            "status is invalid",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                status="nope"
            ),
        ),
        (
            None,
            "message must be a string",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                message=123
            ),
        ),
        (
            None,
            "progress must be a number",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                progress="bad"
            ),
        ),
        (
            None,
            "started_at must be a number",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                started_at="bad"
            ),
        ),
        (
            None,
            "completed_at must be a number",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                completed_at="bad"
            ),
        ),
        (
            None,
            "audio filename is required",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                audio_filename=""
            ),
        ),
        (
            None,
            "text filename is required",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                text_filename=""
            ),
        ),
        (
            None,
            "language is required",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                language=""
            ),
        ),
        (
            None,
            "audio path is required",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                audio_path=""
            ),
        ),
        (
            None,
            "text path is required",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                text_path=""
            ),
        ),
        (
            None,
            "output path is required",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["input"].update(
                output_path=""
            ),
        ),
        (
            None,
            "progress must be between 0 and 1",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                progress=2.0
            ),
        ),
        (
            None,
            "started_at must be non-negative",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                started_at=-1.0
            ),
        ),
        (
            None,
            "completed_at must be non-negative",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                completed_at=-1.0
            ),
        ),
        (
            None,
            "completed_at must be after started_at",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                started_at=3.0,
                completed_at=2.0,
            ),
        ),
        (
            None,
            "queued jobs cannot have timestamps",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                status="queued",
                started_at=1.0,
            ),
        ),
        (
            None,
            "running jobs must have started_at and no completed_at",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                status="running",
                started_at=None,
                completed_at=None,
            ),
        ),
        (
            None,
            "completed jobs must have start and completion times",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                status="completed",
                started_at=None,
                completed_at=None,
            ),
        ),
        (
            None,
            "output path is only valid for completed jobs",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                status="failed",
                output_srt="output.srt",
            ),
        ),
        (
            None,
            "completed jobs must include output path",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]]["result"].update(
                status="completed",
                output_srt=None,
            ),
        ),
        (
            None,
            "job id is required",
            lambda payload: payload.update(
                job_order=[""],
                jobs={
                    "": dict(
                        payload["jobs"][list(payload["jobs"].keys())[0]],
                        job_id="",
                    )
                },
            ),
        ),
        (
            None,
            "job created_at must be non-negative",
            lambda payload: payload["jobs"][list(payload["jobs"].keys())[0]].update(
                created_at=-1.0
            ),
        ),
    ]

    for index, (raw_text, expected, mutator) in enumerate(cases):
        ui_root = tmp_path / f"case-{index}"
        ui_root.mkdir(parents=True, exist_ok=True)
        job_id = "job"
        job_dir = ui_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        jobs_path = ui_root / "jobs.json"
        if raw_text is not None:
            jobs_path.write_text(raw_text, encoding="utf-8")
        else:
            payload = base_payload(job_id, job_dir)
            if mutator is not None:
                mutator(payload)
            jobs_path.write_text(json.dumps(payload), encoding="utf-8")
        port = free_local_port()
        args = [
            str(script_path),
            "--ui",
            "--ui-host",
            "127.0.0.1",
            "--ui-port",
            str(port),
            "--ui-root-dir",
            str(ui_root),
        ]
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


def test_audio_to_text_ui_defaults_remove_punctuation(tmp_path: Path) -> None:
    """Default remove_punctuation to True when missing."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "defaults"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_srt = job_dir / "alignment.srt"
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
                    "remove_punctuation": None,
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        payload = read_json(f"{base_url}/api/jobs")
        jobs = payload.get("jobs", [])
        assert isinstance(jobs, list)
        assert jobs[0].get("remove_punctuation") is True
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_job_store_write_failure(
    tmp_path: Path,
) -> None:
    """Return an error when job store writes fail."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    (ui_root / "jobs.tmp").mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", b"RIFF"),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert "job store write failed" in payload.get("error", "")
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_setup_failure(tmp_path: Path) -> None:
    """Return an error when job setup fails."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "fixed-job"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_srt = job_dir / "alignment.srt"
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    stub_root = tmp_path / "uuid_stub"
    write_uuid_stub(stub_root, job_id)
    port = free_local_port()
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", b"RIFF"),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert "job setup failed" in payload.get("error", "")
    finally:
        stop_process(process)


def test_audio_to_text_ui_upload_write_failure(tmp_path: Path) -> None:
    """Return an error when upload files cannot be written."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "write-fail"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_dir.chmod(0o500)
    stub_root = tmp_path / "uuid_stub"
    write_uuid_stub(stub_root, job_id)
    port = free_local_port()
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", b"RIFF"),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert payload.get("error") == "Upload failed"
    finally:
        job_dir.chmod(0o700)
        stop_process(process)


def test_audio_to_text_ui_keepalive_events(tmp_path: Path) -> None:
    """Emit keepalive SSE events when no changes occur."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "queued-job"
    job_dir = ui_root / job_id
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        with urllib.request.urlopen(
            f"{base_url}/api/jobs/events", timeout=8
        ) as response:
            first_line = response.readline().decode("utf-8").strip()
            assert first_line.startswith("data:")
            response.readline()
            keepalive = response.readline().decode("utf-8").strip()
            assert keepalive.startswith(":")
        with urllib.request.urlopen(
            f"{base_url}/api/jobs/{job_id}/events", timeout=8
        ) as response:
            first_line = response.readline().decode("utf-8").strip()
            assert first_line.startswith("data:")
            response.readline()
            keepalive = response.readline().decode("utf-8").strip()
            assert keepalive.startswith(":")
    finally:
        stop_process(process)


def test_audio_to_text_ui_jobs_events_fail_on_snapshot(tmp_path: Path) -> None:
    """Handle job list snapshot failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "snapshot-fail"
    job_dir = ui_root / job_id
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    env_overrides = {
        audio_to_text.UI_SSE_FAILURE_MODE_ENV: audio_to_text.UI_SSE_FAILURE_JOBS_SNAPSHOT
    }
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        line = response.readline()
        assert line == b""
        connection.close()
    finally:
        stop_process(process)


def test_audio_to_text_ui_jobs_events_fail_on_keepalive(tmp_path: Path) -> None:
    """Handle job list keepalive failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "keepalive-fail"
    job_dir = ui_root / job_id
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    env_overrides = {
        audio_to_text.UI_SSE_FAILURE_MODE_ENV: audio_to_text.UI_SSE_FAILURE_JOBS_KEEPALIVE
    }
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/jobs/events", timeout=8
        ) as response:
            first_line = response.readline().decode("utf-8").strip()
            assert first_line.startswith("data:")
            response.readline()
            tail = response.readline()
            assert tail == b""
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_fail_on_snapshot(tmp_path: Path) -> None:
    """Handle job snapshot failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "job-snapshot-fail"
    job_dir = ui_root / job_id
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    env_overrides = {
        audio_to_text.UI_SSE_FAILURE_MODE_ENV: audio_to_text.UI_SSE_FAILURE_JOB_SNAPSHOT
    }
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request("GET", f"/api/jobs/{job_id}/events")
        response = connection.getresponse()
        line = response.readline()
        assert line == b""
        connection.close()
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_fail_on_keepalive(tmp_path: Path) -> None:
    """Handle job keepalive failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "job-keepalive-fail"
    job_dir = ui_root / job_id
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    env_overrides = {
        audio_to_text.UI_SSE_FAILURE_MODE_ENV: audio_to_text.UI_SSE_FAILURE_JOB_KEEPALIVE
    }
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/jobs/{job_id}/events", timeout=8
        ) as response:
            first_line = response.readline().decode("utf-8").strip()
            assert first_line.startswith("data:")
            response.readline()
            tail = response.readline()
            assert tail == b""
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_fail_on_update(tmp_path: Path) -> None:
    """Handle job update failures deterministically."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_DELAY": "1.0",
        audio_to_text.UI_SSE_FAILURE_MODE_ENV: audio_to_text.UI_SSE_FAILURE_JOB_UPDATE,
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", f"/api/jobs/{job_id}/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        response.readline()
        tail = response.readline()
        assert tail == b""
        wait_for_job_status(base_url, job_id, timeout_seconds=8.0)
        connection.close()
    finally:
        stop_process(process)


def test_audio_to_text_ui_jobs_events_disconnect_on_update(tmp_path: Path) -> None:
    """Handle SSE clients disconnecting before an update arrives."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        connection.close()
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        time.sleep(0.2)
    finally:
        stop_process(process)


def test_audio_to_text_ui_jobs_events_disconnect_on_snapshot(
    tmp_path: Path,
) -> None:
    """Handle SSE clients disconnecting before the snapshot."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        reset_sse_connection("127.0.0.1", port, "/api/jobs/events")
        time.sleep(0.2)
    finally:
        stop_process(process)


def test_audio_to_text_ui_jobs_events_disconnect_on_keepalive(
    tmp_path: Path,
) -> None:
    """Handle SSE clients disconnecting before keepalive."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        response.readline()
        reset_http_connection(connection)
        time.sleep(6.0)
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_disconnect_on_snapshot(tmp_path: Path) -> None:
    """Handle job SSE clients disconnecting before the first event."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "snapshot-job"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_srt = job_dir / "alignment.srt"
    output_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,100\nHello\n", encoding="utf-8"
    )
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        reset_sse_connection(
            "127.0.0.1",
            port,
            f"/api/jobs/{job_id}/events",
        )
        time.sleep(0.2)
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_disconnect_on_keepalive(tmp_path: Path) -> None:
    """Handle job SSE clients disconnecting before keepalive."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "keepalive-job"
    job_dir = ui_root / job_id
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(job_dir / "alignment.srt"),
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
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", f"/api/jobs/{job_id}/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        response.readline()
        reset_http_connection(connection)
        time.sleep(6.0)
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_completed_closes_stream(
    tmp_path: Path,
) -> None:
    """Stop streaming immediately for completed jobs."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "completed-job"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_srt = job_dir / "alignment.srt"
    output_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,100\nHello\n", encoding="utf-8"
    )
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", f"/api/jobs/{job_id}/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        response.readline()
        try:
            tail = response.readline()
        except socket.timeout:
            tail = b""
        assert tail == b""
        connection.close()
    finally:
        stop_process(process)


def test_audio_to_text_ui_job_events_disconnect_on_update(tmp_path: Path) -> None:
    """Handle job SSE clients disconnecting before updates."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_DELAY": "0.4",
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", f"/api/jobs/{job_id}/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        response.readline()
        reset_http_connection(connection)
        wait_for_job_status(base_url, job_id, timeout_seconds=8.0)
    finally:
        stop_process(process)


def test_audio_to_text_ui_sse_updates_on_job_create(tmp_path: Path) -> None:
    """Stream updated job snapshots over SSE."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {"PYTHONPATH": extend_pythonpath(stub_root)}
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/api/jobs/events")
        response = connection.getresponse()
        first_line = response.readline().decode("utf-8").strip()
        assert first_line.startswith("data:")
        response.readline()
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5):
            pass
        updated_line = response.readline().decode("utf-8").strip()
        assert updated_line.startswith("data:")
        connection.close()
    finally:
        stop_process(process)


def test_audio_to_text_ui_rejects_delete_while_running(tmp_path: Path) -> None:
    """Return a conflict when deleting an unfinished job."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    stub_root = tmp_path / "stubs"
    write_stub_modules(stub_root)
    env_overrides = {
        "PYTHONPATH": extend_pythonpath(stub_root),
        "AUDIO_TO_TEXT_TEST_ALIGN_DELAY": "1.0",
    }
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, env_overrides)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        wav_path = tmp_path / "sample.wav"
        write_wav(wav_path, 0.2)
        payload, boundary = build_multipart(
            {"language": "en"},
            [
                ("audio", "sample.wav", "audio/wav", wav_path.read_bytes()),
                ("text", "sample.txt", "text/plain", b"Hello"),
            ],
        )
        request = urllib.request.Request(
            f"{base_url}/api/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created.get("job_id")
        assert isinstance(job_id, str)
        delete_request = urllib.request.Request(
            f"{base_url}/api/jobs/{job_id}", method="DELETE"
        )
        try:
            with urllib.request.urlopen(delete_request, timeout=3):
                raise AssertionError("delete should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 409
            assert "only finished jobs can be deleted" in payload.get("error", "")
    finally:
        stop_process(process)


def test_audio_to_text_ui_reports_not_found_routes(tmp_path: Path) -> None:
    """Return errors for unknown routes and missing jobs."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        try:
            with urllib.request.urlopen(
                f"{base_url}/does-not-exist", timeout=3
            ):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload.get("error") == "Not found"
        try:
            request = urllib.request.Request(
                f"{base_url}/api/unknown",
                data=b"",
                method="POST",
                headers={"Content-Length": "0"},
            )
            with urllib.request.urlopen(request, timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload.get("error") == "Not found"
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request("DELETE", "/api/unknown")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 404
        assert payload.get("error") == "Not found"
        try:
            with urllib.request.urlopen(f"{base_url}/api/jobs/missing", timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload.get("error") == "Job not found"
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request("GET", "/api/jobs//events")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 400
        assert payload.get("error") == "Job id is required"
        try:
            with urllib.request.urlopen(
                f"{base_url}/api/jobs/missing/events", timeout=3
            ):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload.get("error") == "Job not found"
        try:
            with urllib.request.urlopen(
                f"{base_url}/api/jobs/missing/srt", timeout=3
            ):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload.get("error") == "SRT not available"
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.request("DELETE", "/api/jobs/")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 400
        assert payload.get("error") == "Job id is required"
        delete_request = urllib.request.Request(
            f"{base_url}/api/jobs/missing", method="DELETE"
        )
        try:
            with urllib.request.urlopen(delete_request, timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload.get("error") == "Job not found"
    finally:
        stop_process(process)


def test_audio_to_text_ui_srt_read_failure(tmp_path: Path) -> None:
    """Return errors when SRT output cannot be read."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    job_id = "broken"
    job_dir = ui_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_srt = job_dir / "alignment.srt"
    output_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,100\nHello\n", encoding="utf-8"
    )
    output_srt.chmod(0o000)
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
                    "audio_path": str(job_dir / "audio.wav"),
                    "text_path": str(job_dir / "text.txt"),
                    "output_path": str(output_srt),
                },
                "result": {
                    "status": "completed",
                    "message": "Complete",
                    "output_srt": str(output_srt),
                    "progress": 1.0,
                    "started_at": 2.0,
                    "completed_at": 3.0,
                },
            }
        },
    }
    (ui_root / "jobs.json").write_text(json.dumps(jobs_payload), encoding="utf-8")
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(base_url, timeout_seconds=6.0)
        try:
            with urllib.request.urlopen(f"{base_url}/api/jobs/{job_id}/srt", timeout=3):
                raise AssertionError("request should fail")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 500
            assert payload.get("error") == "SRT read failed"
    finally:
        output_srt.chmod(0o600)
        stop_process(process)


def test_audio_to_text_ui_missing_content_length(tmp_path: Path) -> None:
    """Reject requests with invalid content length."""
    repo_root = Path(__file__).resolve().parents[1]
    ui_root = tmp_path / "uploads"
    ui_root.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    process = start_ui_server(repo_root, ui_root, port, None)
    try:
        wait_for_ui_ready(f"http://127.0.0.1:{port}", timeout_seconds=6.0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        connection.putrequest("POST", "/api/jobs")
        cases = [
            ("nope", "Content-Length must be an integer"),
            ("-5", "Content-Length must be positive"),
        ]
        for value, expected in cases:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
            connection.putrequest("POST", "/api/jobs")
            connection.putheader("Content-Type", "multipart/form-data; boundary=stub")
            connection.putheader("Content-Length", value)
            connection.endheaders()
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            assert response.status == 400
            assert expected in payload.get("error", "")
    finally:
        stop_process(process)
