"""Integration tests for audio_to_text CLI."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import List


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
    audio_path = repo_root / "video.mov"

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
        "--device",
        "cpu",
    ]

    result = run_audio_to_text(args, repo_root)

    assert result.returncode != 0
    assert "audio_to_text.input.invalid_config" in result.stderr
