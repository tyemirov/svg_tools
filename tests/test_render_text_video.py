"""Integration tests for render_text_video CLI."""

from __future__ import annotations

import json
import struct
import subprocess
import zlib
from pathlib import Path
from typing import List

LETTER_TRACKING_RATIO = 0.15
MIN_TRACKING_PIXELS = 2


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


def write_png(
    target_path: Path, width: int, height: int, color: tuple[int, int, int, int]
) -> None:
    """Write a solid-color RGBA PNG using the standard library."""

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    row = bytes([0]) + bytes(color) * width
    raw = row * height
    compressed = zlib.compress(raw)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png_bytes = (
        signature
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", compressed)
        + png_chunk(b"IEND", b"")
    )
    target_path.write_bytes(png_bytes)


def build_common_args(
    script_path: Path,
    input_path: Path,
    output_path: Path,
    fonts_dir: Path,
    duration_seconds: str,
    fps: str,
    include_dimensions: bool = True,
) -> List[str]:
    """Build common CLI arguments for render_text_video.py."""
    args = [
        str(script_path),
        "--input-text-file",
        str(input_path),
        "--output-video-file",
        str(output_path),
        "--duration-seconds",
        duration_seconds,
        "--fps",
        fps,
        "--background",
        "transparent",
        "--fonts-dir",
        str(fonts_dir),
    ]
    if include_dimensions:
        args.extend(["--width", "64", "--height", "64"])
    return args


def expected_font_size_range(width: int, height: int) -> tuple[int, int]:
    """Compute the expected font size range for a frame size."""
    min_dimension = min(width, height)
    base_size = max(24, min_dimension // 8)
    min_size = max(32, base_size + 4, min_dimension // 5)
    max_size = max(min_size * 2, int(min_dimension * 2.0))
    return min_size, max_size


def expected_letter_bands(letter_band_sizes: list[int]) -> list[int]:
    """Compute expected band offsets for letter placement."""
    if not letter_band_sizes:
        return []
    tracking_sizes = [
        max(MIN_TRACKING_PIXELS, int(round(size * LETTER_TRACKING_RATIO)))
        for size in letter_band_sizes
    ]
    total_span = sum(letter_band_sizes) + sum(tracking_sizes[:-1])
    cursor = -total_span / 2.0
    positions: list[int] = []
    for index_value, size in enumerate(letter_band_sizes):
        positions.append(int(round(cursor + size / 2.0)))
        cursor += size
        if index_value < len(letter_band_sizes) - 1:
            cursor += tracking_sizes[index_value]
    return positions


def assert_non_overlapping_bands(bands: list[int], sizes: list[int]) -> None:
    """Assert bands do not overlap given their sizes."""
    intervals = sorted(
        (
            band_center - band_size / 2.0,
            band_center + band_size / 2.0,
        )
        for band_center, band_size in zip(bands, sizes)
    )
    for (previous_left, previous_right), (current_left, current_right) in zip(
        intervals, intervals[1:]
    ):
        assert previous_left <= previous_right
        assert current_left <= current_right
        assert current_left >= previous_right


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


def test_direction_seed_is_deterministic(tmp_path: Path) -> None:
    """Use a seed to make direction selection deterministic."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    input_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    base_args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="3.0",
        fps="10",
    )

    args_seeded = base_args + ["--emit-directions", "--direction-seed", "7"]
    first = run_render_text_video(args_seeded, repo_root)
    second = run_render_text_video(args_seeded, repo_root)

    assert first.returncode == 0
    assert second.returncode == 0

    first_payload = json.loads(first.stdout or "{}")
    second_payload = json.loads(second.stdout or "{}")

    assert first_payload["directions"] == second_payload["directions"]
    assert first_payload["font_sizes"] == second_payload["font_sizes"]
    assert first_payload["words"] == second_payload["words"]
    assert first_payload["letter_offsets"] == second_payload["letter_offsets"]
    assert first_payload["letter_bands"] == second_payload["letter_bands"]
    assert first_payload["letter_band_sizes"] == second_payload["letter_band_sizes"]

    expected_min_size, expected_max_size = expected_font_size_range(64, 64)
    for font_size in first_payload["font_sizes"]:
        assert expected_min_size <= font_size <= expected_max_size

    first_offsets = first_payload["letter_offsets"][0]
    assert any(offset != 0 for offset in first_offsets)
    second_offsets = first_payload["letter_offsets"][1]
    assert all(offset == 0 for offset in second_offsets)

    for word, direction, bands, band_sizes in zip(
        first_payload["words"],
        first_payload["directions"],
        first_payload["letter_bands"],
        first_payload["letter_band_sizes"],
    ):
        assert len(band_sizes) == len(word)
        assert all(size >= 1 for size in band_sizes)
        assert bands == expected_letter_bands(band_sizes)

    assert any(
        direction in ("T2B", "B2T") for direction in first_payload["directions"]
    )
    for direction, bands, band_sizes in zip(
        first_payload["directions"],
        first_payload["letter_bands"],
        first_payload["letter_band_sizes"],
    ):
        if direction in ("T2B", "B2T"):
            assert_non_overlapping_bands(bands, band_sizes)

    args_other_seed = base_args + ["--emit-directions", "--direction-seed", "8"]
    other = run_render_text_video(args_other_seed, repo_root)
    assert other.returncode == 0

    other_payload = json.loads(other.stdout or "{}")
    assert other_payload["directions"] != first_payload["directions"]
    assert other_payload["font_sizes"] != first_payload["font_sizes"]
    assert other_payload["letter_offsets"] != first_payload["letter_offsets"]


def test_remove_punctuation(tmp_path: Path) -> None:
    """Strip punctuation when requested."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    input_text = "Hello, world! (Testing) punctuation... OK?"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    base_args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="2.0",
        fps="10",
    )

    args_strip = base_args + ["--emit-directions", "--remove-punctuation"]
    stripped = run_render_text_video(args_strip, repo_root)
    assert stripped.returncode == 0
    stripped_payload = json.loads(stripped.stdout or "{}")
    assert stripped_payload["words"] == ["Hello", "world", "Testing", "punctuation", "OK"]

    args_keep = base_args + ["--emit-directions"]
    kept = run_render_text_video(args_keep, repo_root)
    assert kept.returncode == 0
    kept_payload = json.loads(kept.stdout or "{}")
    assert kept_payload["words"] == [
        "Hello,",
        "world!",
        "(Testing)",
        "punctuation...",
        "OK?",
    ]


def test_background_image_derives_dimensions(tmp_path: Path) -> None:
    """Use background image to derive frame dimensions."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    input_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    background_path = tmp_path / "background.png"
    write_png(background_path, 10, 12, (20, 40, 60, 255))

    output_path = tmp_path / "out.mov"
    base_args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="3.0",
        fps="10",
        include_dimensions=False,
    )

    args = base_args + [
        "--background-image",
        str(background_path),
        "--emit-directions",
        "--direction-seed",
        "7",
    ]
    result = run_render_text_video(args, repo_root)
    assert result.returncode == 0

    payload = json.loads(result.stdout or "{}")
    _, expected_max_size = expected_font_size_range(10, 12)
    assert max(payload["font_sizes"]) <= expected_max_size


def test_background_image_conflicts_with_dimensions(tmp_path: Path) -> None:
    """Fail when background image and dimensions are both provided."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    input_text = "alpha beta"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    background_path = tmp_path / "background.png"
    write_png(background_path, 8, 8, (0, 0, 0, 255))

    output_path = tmp_path / "out.mov"
    base_args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="1.0",
        fps="10",
    )
    args = base_args + ["--background-image", str(background_path), "--emit-directions"]
    result = run_render_text_video(args, repo_root)
    assert result.returncode != 0
    assert "render_text_video.input.invalid_config" in result.stderr


def test_requires_dimensions_without_background(tmp_path: Path) -> None:
    """Fail when no dimensions or background image are provided."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    input_text = "alpha beta"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    args = [
        str(script_path),
        "--input-text-file",
        str(input_path),
        "--output-video-file",
        str(output_path),
        "--duration-seconds",
        "1.0",
        "--fps",
        "10",
        "--background",
        "transparent",
        "--fonts-dir",
        str(fonts_dir),
        "--emit-directions",
    ]

    result = run_render_text_video(args, repo_root)
    assert result.returncode != 0
    assert "render_text_video.input.invalid_config" in result.stderr
