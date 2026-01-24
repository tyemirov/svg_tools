"""Integration tests for audio_to_text CLI."""

from __future__ import annotations

import platform
import subprocess
import wave
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import pytest

if platform.system().lower() != "linux":
    pytest.skip(
        "audio_to_text is supported on Linux only; use Docker",
        allow_module_level=True,
    )


def run_audio_to_text(args: List[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    """Run audio_to_text.py with the provided arguments."""
    return subprocess.run(
        args,
        cwd=repo_root,
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


def test_audio_to_text_ui_deletes_completed_job(tmp_path: Path) -> None:
    """Delete a completed job via the UI API."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "audio_to_text.py"
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
    process = subprocess.Popen(
        [
            str(script_path),
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)
