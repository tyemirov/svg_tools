"""Integration tests for render_text_video CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def run_render_text_video(args: List[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    """Run render_text_video.py with the provided arguments."""
    return subprocess.run(
        args,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def write_srt_file(target_path: Path, content: str) -> None:
    """Write SRT content to disk."""
    target_path.write_text(content, encoding="utf-8")


def build_common_args(
    script_path: Path,
    input_path: Path,
    output_path: Path,
    fonts_dir: Path,
    duration_seconds: str,
    fps: str,
) -> List[str]:
    """Build common CLI arguments for render_text_video.py."""
    return [
        str(script_path),
        "--input-text-file",
        str(input_path),
        "--output-video-file",
        str(output_path),
        "--width",
        "64",
        "--height",
        "64",
        "--duration-seconds",
        duration_seconds,
        "--fps",
        fps,
        "--background",
        "transparent",
        "--fonts-dir",
        str(fonts_dir),
    ]


def test_srt_window_too_small(tmp_path: Path) -> None:
    """Fail when an SRT window is too short for its words."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    srt_content = """1\n00:00:00,000 --> 00:00:00,200\nalpha beta gamma delta\n"""
    srt_path = tmp_path / "short.srt"
    write_srt_file(srt_path, srt_content)

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=srt_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="1.0",
        fps="10",
    )

    result = run_render_text_video(args, repo_root)

    assert result.returncode != 0
    assert "render_text_video.input.invalid_window" in result.stderr


def test_srt_success(tmp_path: Path) -> None:
    """Render a short SRT successfully."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    srt_content = """1\n00:00:00,000 --> 00:00:01,000\nhello world\n\n2\n00:00:01,000 --> 00:00:02,000\nsecond line\n"""
    srt_path = tmp_path / "ok.srt"
    write_srt_file(srt_path, srt_content)

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=srt_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="2.0",
        fps="10",
    )

    result = run_render_text_video(args, repo_root)

    assert result.returncode == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0
