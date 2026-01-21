"""Integration tests for render_text_video CLI."""

from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import List

BYTES_PER_PIXEL = 4
ALPHA_THRESHOLD = 10
TEMPLATE_MATCH_RATIO = 0.98


def run_render_text_video(args: List[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    """Run render_text_video.py with the provided arguments."""
    return subprocess.run(
        args,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def extract_raw_frame(video_path: Path, time_seconds: float) -> bytes:
    """Extract a raw RGBA frame from a video at the requested time."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{time_seconds:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


def alpha_bbox(frame_bytes: bytes, width: int, height: int) -> tuple[int, int, int, int]:
    """Return the bounding box of non-transparent pixels."""
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y_value in range(height):
        row_offset = y_value * width * BYTES_PER_PIXEL
        for x_value in range(width):
            alpha_value = frame_bytes[row_offset + x_value * BYTES_PER_PIXEL + 3]
            if alpha_value == 0:
                continue
            min_x = min(min_x, x_value)
            min_y = min(min_y, y_value)
            max_x = max(max_x, x_value)
            max_y = max(max_y, y_value)
    if max_x < 0:
        raise AssertionError("no non-transparent pixels found")
    return min_x, min_y, max_x, max_y


def crop_rgba(
    frame_bytes: bytes,
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
) -> tuple[bytes, int, int]:
    """Crop an RGBA frame to a bounding box."""
    left, top, right, bottom = bbox
    crop_width = right - left + 1
    crop_height = bottom - top + 1
    rows: list[bytes] = []
    for y_value in range(top, bottom + 1):
        start = (y_value * width + left) * BYTES_PER_PIXEL
        end = start + crop_width * BYTES_PER_PIXEL
        rows.append(frame_bytes[start:end])
    return b"".join(rows), crop_width, crop_height


def alpha_mask_rows(frame_bytes: bytes, width: int, height: int) -> list[bytes]:
    """Extract alpha mask rows from an RGBA frame."""
    rows: list[bytes] = []
    for y_value in range(height):
        row_start = y_value * width * BYTES_PER_PIXEL + 3
        row = bytes(
            1
            if frame_bytes[row_start + x_value * BYTES_PER_PIXEL] >= ALPHA_THRESHOLD
            else 0
            for x_value in range(width)
        )
        rows.append(row)
    return rows


def find_template_leftmost_x(
    frame_bytes: bytes,
    width: int,
    height: int,
    template_bytes: bytes,
    template_width: int,
    template_height: int,
    search_bbox: tuple[int, int, int, int],
) -> int:
    """Find the leftmost x position where the template alpha matches."""
    frame_alpha = alpha_mask_rows(frame_bytes, width, height)
    template_alpha = alpha_mask_rows(template_bytes, template_width, template_height)
    template_on_pixels = sum(sum(row) for row in template_alpha)
    if template_on_pixels == 0:
        raise AssertionError("template has no visible pixels")
    left, top, right, bottom = search_bbox
    max_x = right - template_width + 1
    max_y = bottom - template_height + 1
    best_x = None
    for y_value in range(top, max_y + 1):
        for x_value in range(left, max_x + 1):
            matched_on = 0
            for row_index, template_row in enumerate(template_alpha):
                frame_row = frame_alpha[y_value + row_index]
                for col_index, template_pixel in enumerate(template_row):
                    if template_pixel and frame_row[x_value + col_index]:
                        matched_on += 1
            if matched_on / template_on_pixels >= TEMPLATE_MATCH_RATIO:
                if best_x is None or x_value < best_x:
                    best_x = x_value
    if best_x is None:
        raise AssertionError("template not found in frame")
    return best_x


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
    width: int = 64,
    height: int = 64,
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
        args.extend(["--width", str(width), "--height", str(height)])
    return args


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

    input_text = "alpha"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    duration_seconds = "1.2"
    fps = "8"

    def render(seed: str, output_name: str) -> Path:
        output_path = tmp_path / output_name
        args = build_common_args(
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            fonts_dir=fonts_dir,
            duration_seconds=duration_seconds,
            fps=fps,
        )
        args.extend(["--direction-seed", seed])
        result = run_render_text_video(args, repo_root)
        assert result.returncode == 0
        assert output_path.exists()
        return output_path

    first_path = render("2", "out-1.mov")
    second_path = render("2", "out-2.mov")
    other_path = render("5", "out-3.mov")

    time_seconds = float(duration_seconds) / 2.0
    first_frame = extract_raw_frame(first_path, time_seconds)
    second_frame = extract_raw_frame(second_path, time_seconds)
    other_frame = extract_raw_frame(other_path, time_seconds)

    assert first_frame == second_frame
    assert first_frame != other_frame


def test_l2r_cyrillic_word_is_reversed(tmp_path: Path) -> None:
    """Ensure L2R words lead with the first letter."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    source_font = repo_root / "assets" / "fonts" / "NotoSans-Bold.ttf"
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(source_font, fonts_dir / source_font.name)

    width = 300
    height = 200
    duration_seconds = "1.5"
    fps = "10"
    seed = "42"

    def render_word(text_value: str, output_name: str) -> Path:
        input_path = tmp_path / f"{output_name}.txt"
        input_path.write_text(text_value, encoding="utf-8")
        output_path = tmp_path / f"{output_name}.mov"
        args = build_common_args(
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            fonts_dir=fonts_dir,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
        )
        args.extend(["--direction-seed", seed])
        result = run_render_text_video(args, repo_root)
        assert result.returncode == 0
        return output_path

    word_path = render_word("писать", "word")
    pe_path = render_word("п", "pe")
    soft_path = render_word("ь", "soft-sign")

    time_seconds = float(duration_seconds) / 2.0
    word_frame = extract_raw_frame(word_path, time_seconds)
    word_bbox = alpha_bbox(word_frame, width, height)

    pe_frame = extract_raw_frame(pe_path, time_seconds)
    pe_template, pe_width, pe_height = crop_rgba(
        pe_frame, width, height, alpha_bbox(pe_frame, width, height)
    )
    soft_frame = extract_raw_frame(soft_path, time_seconds)
    soft_template, soft_width, soft_height = crop_rgba(
        soft_frame, width, height, alpha_bbox(soft_frame, width, height)
    )

    pe_x = find_template_leftmost_x(
        word_frame, width, height, pe_template, pe_width, pe_height, word_bbox
    )
    soft_x = find_template_leftmost_x(
        word_frame, width, height, soft_template, soft_width, soft_height, word_bbox
    )

    assert pe_x > soft_x


def test_remove_punctuation(tmp_path: Path) -> None:
    """Strip punctuation when requested."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = repo_root / "assets" / "fonts"

    input_text = "Hello, world! (Testing) punctuation... OK?"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    duration_seconds = "2.0"
    fps = "10"
    seed = "11"

    output_path = tmp_path / "out-strip.mov"
    base_args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds=duration_seconds,
        fps=fps,
    )

    args_strip = base_args + ["--remove-punctuation", "--direction-seed", seed]
    stripped = run_render_text_video(args_strip, repo_root)
    assert stripped.returncode == 0

    output_path_keep = tmp_path / "out-keep.mov"
    args_keep = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path_keep,
        fonts_dir=fonts_dir,
        duration_seconds=duration_seconds,
        fps=fps,
    )
    args_keep.extend(["--direction-seed", seed])
    kept = run_render_text_video(args_keep, repo_root)
    assert kept.returncode == 0

    time_seconds = float(duration_seconds) / 2.0
    stripped_frame = extract_raw_frame(output_path, time_seconds)
    kept_frame = extract_raw_frame(output_path_keep, time_seconds)
    assert stripped_frame != kept_frame


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

    duration_seconds = "1.2"
    fps = "10"
    output_path = tmp_path / "out.mov"
    base_args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds=duration_seconds,
        fps=fps,
        include_dimensions=False,
    )

    args = base_args + [
        "--background-image",
        str(background_path),
        "--direction-seed",
        "7",
    ]
    result = run_render_text_video(args, repo_root)
    assert result.returncode == 0
    frame_bytes = extract_raw_frame(output_path, float(duration_seconds) / 2.0)
    assert len(frame_bytes) == 10 * 12 * BYTES_PER_PIXEL


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
    args = base_args + ["--background-image", str(background_path)]
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
    ]

    result = run_render_text_video(args, repo_root)
    assert result.returncode != 0
    assert "render_text_video.input.invalid_config" in result.stderr
