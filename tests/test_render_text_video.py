"""Integration tests for render_text_video CLI."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import List

BYTES_PER_PIXEL = 4
ALPHA_THRESHOLD = 10
TEMPLATE_MATCH_RATIO = 0.98
FIRST_LETTER_MATCH_RATIO = 0.3
ORDER_MATCH_RATIO = 0.3
RSVP_MATCH_RATIO = 0.3
DIRECTION_SEEDS = {
    "L2R": 2,
    "R2L": 1,
    "T2B": 5,
    "B2T": 0,
}
ENTRY_FIRST_SEEDS = {
    "L2R": DIRECTION_SEEDS["L2R"],
    "R2L": DIRECTION_SEEDS["R2L"],
    "T2B": DIRECTION_SEEDS["T2B"],
}


def run_render_text_video(args: List[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    """Run render_text_video.py with the provided arguments."""
    return subprocess.run(
        args,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def get_test_fonts_dir(repo_root: Path) -> Path:
    """Return the fonts directory for tests."""
    return repo_root / "tests" / "fixtures" / "fonts"


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


def probe_video_stream(video_path: Path) -> dict[str, str]:
    """Return codec metadata for the first video stream."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    assert streams, "no video streams found"
    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name", ""),
        "pix_fmt": stream.get("pix_fmt", ""),
    }


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


def frame_has_alpha(frame_bytes: bytes, width: int, height: int) -> bool:
    """Return True when any alpha pixel is visible."""
    if len(frame_bytes) != width * height * BYTES_PER_PIXEL:
        return False
    for y_value in range(height):
        row_offset = y_value * width * BYTES_PER_PIXEL
        for x_value in range(width):
            alpha_value = frame_bytes[row_offset + x_value * BYTES_PER_PIXEL + 3]
            if alpha_value >= ALPHA_THRESHOLD:
                return True
    return False


def frame_has_expected_size(frame_bytes: bytes, width: int, height: int) -> bool:
    """Return True when the frame has the expected RGBA size."""
    return len(frame_bytes) == width * height * BYTES_PER_PIXEL


def alpha_pixel_count(frame_bytes: bytes, width: int, height: int) -> int:
    """Return the count of non-transparent pixels in a frame."""
    if len(frame_bytes) != width * height * BYTES_PER_PIXEL:
        return 0
    count = 0
    for y_value in range(height):
        row_offset = y_value * width * BYTES_PER_PIXEL
        for x_value in range(width):
            alpha_value = frame_bytes[row_offset + x_value * BYTES_PER_PIXEL + 3]
            if alpha_value >= ALPHA_THRESHOLD:
                count += 1
    return count


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


def template_from_best_frame(
    video_path: Path,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
) -> tuple[bytes, int, int]:
    """Extract a template from the frame with the most visible alpha."""
    best_frame = None
    best_count = -1
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(video_path, frame_index / fps)
        if not frame_has_expected_size(frame_bytes, width, height):
            continue
        visible_pixels = alpha_pixel_count(frame_bytes, width, height)
        if visible_pixels > best_count:
            best_count = visible_pixels
            best_frame = frame_bytes
    assert best_frame is not None
    assert best_count > 0
    return crop_rgba(
        best_frame, width, height, alpha_bbox(best_frame, width, height)
    )


def template_from_first_visible_frame(
    video_path: Path,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
) -> tuple[bytes, int, int]:
    """Extract a template from the first frame that shows alpha."""
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(video_path, frame_index / fps)
        if not frame_has_expected_size(frame_bytes, width, height):
            continue
        if not frame_has_alpha(frame_bytes, width, height):
            continue
        return crop_rgba(
            frame_bytes, width, height, alpha_bbox(frame_bytes, width, height)
        )
    raise AssertionError("no visible frame found")


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


def best_template_match_ratio(
    frame_bytes: bytes,
    width: int,
    height: int,
    template_bytes: bytes,
    template_width: int,
    template_height: int,
    search_bbox: tuple[int, int, int, int],
) -> float:
    """Return the best overlap ratio for a template alpha mask."""
    frame_alpha = alpha_mask_rows(frame_bytes, width, height)
    template_alpha = alpha_mask_rows(template_bytes, template_width, template_height)
    template_on_pixels = sum(sum(row) for row in template_alpha)
    if template_on_pixels == 0:
        raise AssertionError("template has no visible pixels")
    left, top, right, bottom = search_bbox
    max_x = right - template_width + 1
    max_y = bottom - template_height + 1
    best_ratio = 0.0
    for y_value in range(top, max_y + 1):
        for x_value in range(left, max_x + 1):
            matched_on = 0
            for row_index, template_row in enumerate(template_alpha):
                frame_row = frame_alpha[y_value + row_index]
                for col_index, template_pixel in enumerate(template_row):
                    if template_pixel and frame_row[x_value + col_index]:
                        matched_on += 1
            ratio = matched_on / template_on_pixels
            if ratio > best_ratio:
                best_ratio = ratio
    return best_ratio


def best_template_match_location(
    frame_bytes: bytes,
    width: int,
    height: int,
    template_bytes: bytes,
    template_width: int,
    template_height: int,
    search_bbox: tuple[int, int, int, int],
) -> tuple[float, int, int]:
    """Return the best overlap ratio and its location for a template."""
    frame_alpha = alpha_mask_rows(frame_bytes, width, height)
    template_alpha = alpha_mask_rows(template_bytes, template_width, template_height)
    template_on_pixels = sum(sum(row) for row in template_alpha)
    if template_on_pixels == 0:
        raise AssertionError("template has no visible pixels")
    left, top, right, bottom = search_bbox
    max_x = right - template_width + 1
    max_y = bottom - template_height + 1
    best_ratio = 0.0
    best_x = left
    best_y = top
    for y_value in range(top, max_y + 1):
        for x_value in range(left, max_x + 1):
            matched_on = 0
            for row_index, template_row in enumerate(template_alpha):
                frame_row = frame_alpha[y_value + row_index]
                for col_index, template_pixel in enumerate(template_row):
                    if template_pixel and frame_row[x_value + col_index]:
                        matched_on += 1
            ratio = matched_on / template_on_pixels
            if ratio > best_ratio:
                best_ratio = ratio
                best_x = x_value
                best_y = y_value
    return best_ratio, best_x, best_y


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
    fonts_dir = get_test_fonts_dir(repo_root)

    srt_content = """1\n00:00:00,000 --> 00:00:00,200\nalpha beta gamma delta\n"""
    srt_path = tmp_path / "short.srt"
    write_srt_file(srt_path, srt_content)

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=srt_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="0.6",
        fps="6",
    )

    result = run_render_text_video(args, repo_root)

    assert result.returncode != 0
    assert "render_text_video.input.invalid_window" in result.stderr


def test_srt_success(tmp_path: Path) -> None:
    """Render a short SRT successfully."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    srt_content = """1\n00:00:00,000 --> 00:00:00,400\nhello world\n\n2\n00:00:00,400 --> 00:00:00,800\nsecond line\n"""
    srt_path = tmp_path / "ok.srt"
    write_srt_file(srt_path, srt_content)

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=srt_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="0.8",
        fps="6",
    )

    result = run_render_text_video(args, repo_root)

    assert result.returncode == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_direction_seed_is_deterministic(tmp_path: Path) -> None:
    """Use a seed to make direction selection deterministic."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "alpha"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    duration_seconds = "0.8"
    fps = "6"

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
    source_font = get_test_fonts_dir(repo_root) / "NotoSans-Bold.ttf"
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(source_font, fonts_dir / source_font.name)

    width = 300
    height = 200
    duration_seconds = "1.0"
    fps = "6"
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

    total_frames = int(round(float(duration_seconds) * float(fps)))
    fps_value = float(fps)

    pe_template, pe_width, pe_height = template_from_best_frame(
        pe_path, total_frames, fps_value, width, height
    )
    soft_template, soft_width, soft_height = template_from_best_frame(
        soft_path, total_frames, fps_value, width, height
    )

    first_letter = None
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(word_path, frame_index / fps_value)
        if not frame_has_expected_size(frame_bytes, width, height):
            continue
        if not frame_has_alpha(frame_bytes, width, height):
            continue
        word_bbox = alpha_bbox(frame_bytes, width, height)
        ratios = {
            "п": best_template_match_ratio(
                frame_bytes,
                width,
                height,
                pe_template,
                pe_width,
                pe_height,
                word_bbox,
            ),
            "ь": best_template_match_ratio(
                frame_bytes,
                width,
                height,
                soft_template,
                soft_width,
                soft_height,
                word_bbox,
            ),
        }
        best_letter = max(ratios, key=ratios.get)
        if ratios[best_letter] < FIRST_LETTER_MATCH_RATIO:
            continue
        first_letter = best_letter
        break

    assert first_letter is not None
    assert first_letter == "п"


def test_hard_first_letter_appears_first_entry_directions(tmp_path: Path) -> None:
    """Ensure HARD leads with H as the first visible letter where entry is leading."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    source_font = get_test_fonts_dir(repo_root) / "NotoSans-Bold.ttf"
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(source_font, fonts_dir / source_font.name)

    width = 240
    height = 180
    duration_seconds = "0.9"
    fps = "6"
    word_text = "HARD"

    def render_word(text_value: str, seed: str, output_name: str) -> Path:
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

    total_frames = int(round(float(duration_seconds) * float(fps)))
    fps_value = float(fps)
    for direction, seed in ENTRY_FIRST_SEEDS.items():
        word_path = render_word(word_text, str(seed), f"word-{direction}")
        templates = {}
        for letter in word_text:
            letter_path = render_word(letter, str(seed), f"letter-{letter}-{direction}")
            templates[letter] = template_from_first_visible_frame(
                letter_path,
                total_frames,
                fps_value,
                width,
                height,
            )

        word_frames: list[bytes | None] = []
        frame_bboxes: list[tuple[int, int, int, int] | None] = []
        for frame_index in range(total_frames):
            frame_bytes = extract_raw_frame(word_path, frame_index / fps_value)
            if not frame_has_expected_size(frame_bytes, width, height):
                word_frames.append(None)
                frame_bboxes.append(None)
                continue
            word_frames.append(frame_bytes)
            frame_bboxes.append(
                alpha_bbox(frame_bytes, width, height)
                if frame_has_alpha(frame_bytes, width, height)
                else None
            )

        earliest_frames: dict[str, int] = {}
        for letter, (template_bytes, template_width, template_height) in templates.items():
            for frame_index, frame_bytes in enumerate(word_frames):
                if frame_bytes is None:
                    continue
                search_bbox = frame_bboxes[frame_index]
                if search_bbox is None:
                    continue
                ratio = best_template_match_ratio(
                    frame_bytes,
                    width,
                    height,
                    template_bytes,
                    template_width,
                    template_height,
                    search_bbox,
                )
                if ratio >= FIRST_LETTER_MATCH_RATIO:
                    earliest_frames[letter] = frame_index
                    break

        assert "H" in earliest_frames
        for letter, earliest_frame in earliest_frames.items():
            if letter == "H":
                continue
            assert earliest_frame >= earliest_frames["H"], (
                f"{direction} expected H first, got {earliest_frames}"
            )


def test_b2t_word_is_natural_and_complete(tmp_path: Path) -> None:
    """Ensure B2T words are ordered naturally with all letters visible."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    source_font = get_test_fonts_dir(repo_root) / "NotoSans-Bold.ttf"
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(source_font, fonts_dir / source_font.name)

    width = 240
    height = 180
    duration_seconds = "0.9"
    fps = "6"
    word_text = "HARD"
    seed = str(DIRECTION_SEEDS["B2T"])

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

    total_frames = int(round(float(duration_seconds) * float(fps)))
    fps_value = float(fps)
    word_path = render_word(word_text, "word-b2t")
    templates = {}
    for letter in word_text:
        letter_path = render_word(letter, f"letter-{letter}-b2t")
        templates[letter] = template_from_first_visible_frame(
            letter_path,
            total_frames,
            fps_value,
            width,
            height,
        )

    word_frames: list[bytes | None] = []
    frame_bboxes: list[tuple[int, int, int, int] | None] = []
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(word_path, frame_index / fps_value)
        if not frame_has_expected_size(frame_bytes, width, height):
            word_frames.append(None)
            frame_bboxes.append(None)
            continue
        word_frames.append(frame_bytes)
        frame_bboxes.append(
            alpha_bbox(frame_bytes, width, height)
            if frame_has_alpha(frame_bytes, width, height)
            else None
        )

    best_positions = {}
    best_ratios = {}
    for letter in word_text:
        best_positions[letter] = None
        best_ratios[letter] = 0.0

    for frame_index, frame_bytes in enumerate(word_frames):
        if frame_bytes is None:
            continue
        search_bbox = frame_bboxes[frame_index]
        if search_bbox is None:
            continue
        for letter, (template_bytes, template_width, template_height) in templates.items():
            ratio, _, match_y = best_template_match_location(
                frame_bytes,
                width,
                height,
                template_bytes,
                template_width,
                template_height,
                search_bbox,
            )
            if ratio > best_ratios[letter]:
                best_ratios[letter] = ratio
                best_positions[letter] = (match_y, template_height)

    assert all(ratio >= ORDER_MATCH_RATIO for ratio in best_ratios.values())

    y_positions = [best_positions[letter][0] for letter in word_text]
    assert y_positions == sorted(y_positions)


def test_rsvp_orp_anchor_is_stable(tmp_path: Path) -> None:
    """Ensure RSVP ORP anchor stays stable across words."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    source_font = get_test_fonts_dir(repo_root) / "NotoSans-Bold.ttf"
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(source_font, fonts_dir / source_font.name)

    width = 320
    height = 240
    sentence_duration_seconds = "1.2"
    template_duration_seconds = "0.6"
    fps = "10"
    words = ["READING", "HARD"]
    sentence_srt_end = "00:00:01,200"
    template_srt_end = "00:00:00,600"

    def render_words(word_text: str, output_name: str, srt_end: str, duration: str) -> Path:
        srt_content = (
            f"1\n00:00:00,000 --> {srt_end}\n"
            f"{word_text}\n"
        )
        input_path = tmp_path / f"{output_name}.srt"
        input_path.write_text(srt_content, encoding="utf-8")
        output_path = tmp_path / f"{output_name}.mov"
        args = build_common_args(
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            fonts_dir=fonts_dir,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
        )
        args.extend(["--subtitle-renderer", "rsvp_orp", "--keep-punctuation"])
        result = run_render_text_video(args, repo_root)
        assert result.returncode == 0
        return output_path

    sentence_path = render_words(
        " ".join(words), "rsvp-words", sentence_srt_end, sentence_duration_seconds
    )
    total_frames = int(round(float(sentence_duration_seconds) * float(fps)))
    fps_value = float(fps)

    templates = {}
    for word in words:
        word_path = render_words(
            word, f"rsvp-{word}", template_srt_end, template_duration_seconds
        )
        templates[word] = template_from_first_visible_frame(
            word_path,
            int(round(float(template_duration_seconds) * float(fps))),
            fps_value,
            width,
            height,
        )

    def orp_char(word_text: str) -> str:
        core = word_text.rstrip(".,!?:;")
        if not core:
            core = word_text
        if len(core) <= 1:
            return core[0]
        index = max(0, int(math.floor(len(core) * 0.35) - 1))
        return core[index]

    orp_templates = {}
    for word in words:
        letter = orp_char(word)
        letter_path = render_words(
            letter,
            f"rsvp-orp-{word}",
            template_srt_end,
            template_duration_seconds,
        )
        orp_templates[word] = template_from_first_visible_frame(
            letter_path,
            int(round(float(template_duration_seconds) * float(fps))),
            fps_value,
            width,
            height,
        )

    centers_by_word: dict[str, list[float]] = {word: [] for word in words}
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(sentence_path, frame_index / fps_value)
        if not frame_has_expected_size(frame_bytes, width, height):
            continue
        if not frame_has_alpha(frame_bytes, width, height):
            continue
        search_bbox = alpha_bbox(frame_bytes, width, height)
        ratios = {}
        for word, (template_bytes, template_width, template_height) in templates.items():
            ratios[word] = best_template_match_ratio(
                frame_bytes,
                width,
                height,
                template_bytes,
                template_width,
                template_height,
                search_bbox,
            )
        best_word = max(ratios, key=ratios.get)
        if ratios[best_word] < RSVP_MATCH_RATIO:
            continue
        template_bytes, template_width, template_height = orp_templates[best_word]
        ratio, match_x, _ = best_template_match_location(
            frame_bytes,
            width,
            height,
            template_bytes,
            template_width,
            template_height,
            search_bbox,
        )
        if ratio < RSVP_MATCH_RATIO:
            continue
        center_x = match_x + (template_width / 2.0)
        centers_by_word[best_word].append(center_x)

    for word in words:
        assert centers_by_word[word]

    avg_centers = {
        word: sum(centers) / len(centers) for word, centers in centers_by_word.items()
    }
    assert abs(avg_centers[words[0]] - avg_centers[words[1]]) <= 2


def test_rsvp_orp_punctuation_pause_extends_word(tmp_path: Path) -> None:
    """Ensure RSVP punctuation pauses extend the punctuated word."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    source_font = get_test_fonts_dir(repo_root) / "NotoSans-Bold.ttf"
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    shutil.copy(source_font, fonts_dir / source_font.name)

    width = 320
    height = 240
    sentence_duration_seconds = "1.0"
    template_duration_seconds = "0.6"
    fps = "10"
    words = ["HELLO,", "WORLD"]
    sentence_srt_end = "00:00:01,000"
    template_srt_end = "00:00:00,600"

    def render_words(word_text: str, output_name: str, srt_end: str, duration: str) -> Path:
        srt_content = (
            f"1\n00:00:00,000 --> {srt_end}\n"
            f"{word_text}\n"
        )
        input_path = tmp_path / f"{output_name}.srt"
        input_path.write_text(srt_content, encoding="utf-8")
        output_path = tmp_path / f"{output_name}.mov"
        args = build_common_args(
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            fonts_dir=fonts_dir,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
        )
        args.extend(["--subtitle-renderer", "rsvp_orp", "--keep-punctuation"])
        result = run_render_text_video(args, repo_root)
        assert result.returncode == 0
        return output_path

    sentence_path = render_words(
        " ".join(words),
        "rsvp-punct",
        sentence_srt_end,
        sentence_duration_seconds,
    )
    total_frames = int(round(float(sentence_duration_seconds) * float(fps)))
    fps_value = float(fps)

    templates = {}
    for index, word in enumerate(words):
        word_path = render_words(
            word,
            f"rsvp-punct-word-{index}",
            template_srt_end,
            template_duration_seconds,
        )
        templates[word] = template_from_first_visible_frame(
            word_path,
            int(round(float(template_duration_seconds) * float(fps))),
            fps_value,
            width,
            height,
        )

    counts = {word: 0 for word in words}
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(sentence_path, frame_index / fps_value)
        if not frame_has_expected_size(frame_bytes, width, height):
            continue
        if not frame_has_alpha(frame_bytes, width, height):
            continue
        search_bbox = alpha_bbox(frame_bytes, width, height)
        ratios = {}
        for word, (template_bytes, template_width, template_height) in templates.items():
            ratios[word] = best_template_match_ratio(
                frame_bytes,
                width,
                height,
                template_bytes,
                template_width,
                template_height,
                search_bbox,
            )
        best_word = max(ratios, key=ratios.get)
        if ratios[best_word] < RSVP_MATCH_RATIO:
            continue
        counts[best_word] += 1

    pause_frames = int(math.ceil(0.160 * float(fps)))
    base_frames = (total_frames - pause_frames) // 2
    assert counts[words[0]] == base_frames + pause_frames
    assert counts[words[1]] == base_frames
    assert counts[words[0]] + counts[words[1]] == total_frames


def test_rsvp_orp_allows_long_window(tmp_path: Path) -> None:
    """Allow RSVP windows longer than max per-word timing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    width = 240
    height = 180
    duration_seconds = "3.0"
    fps = "10"
    input_text = "alpha beta"

    srt_content = "1\n00:00:00,000 --> 00:00:03,000\nalpha beta\n"
    input_path = tmp_path / "long.srt"
    input_path.write_text(srt_content, encoding="utf-8")

    output_path = tmp_path / "out.mov"
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
    args.extend(["--subtitle-renderer", "rsvp_orp"])
    result = run_render_text_video(args, repo_root)
    assert result.returncode == 0

    total_frames = int(round(float(duration_seconds) * float(fps)))
    alpha_frames = 0
    for frame_index in range(total_frames):
        frame_bytes = extract_raw_frame(output_path, frame_index / float(fps))
        if not frame_has_expected_size(frame_bytes, width, height):
            continue
        if frame_has_alpha(frame_bytes, width, height):
            alpha_frames += 1

    max_frames = int(math.floor(0.700 * float(fps)))
    expected_frames = max_frames * len(input_text.split())
    assert alpha_frames == expected_frames


def test_font_bounds_require_criss_cross_renderer(tmp_path: Path) -> None:
    """Reject font bounds unless criss_cross renderer is selected."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "alpha beta gamma"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="0.9",
        fps="6",
    )
    args.extend(["--font-min", "40"])
    result = run_render_text_video(args, repo_root)
    assert result.returncode != 0
    assert "render_text_video.input.invalid_config" in result.stderr


def test_font_bounds_apply_for_criss_cross(tmp_path: Path) -> None:
    """Apply font bounds when criss_cross renderer is used."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "alpha beta gamma"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="0.9",
        fps="6",
    )
    args.extend(
        [
            "--subtitle-renderer",
            "criss_cross",
            "--font-min",
            "40",
            "--font-max",
            "50",
            "--emit-directions",
        ]
    )
    result = run_render_text_video(args, repo_root)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    sizes = payload["font_sizes"]
    assert sizes
    assert all(40 <= size <= 50 for size in sizes)


def test_font_max_clamps_default_min(tmp_path: Path) -> None:
    """Clamp the default minimum when only font-max is provided."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "alpha beta gamma"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="0.9",
        fps="6",
    )
    args.extend(
        [
            "--subtitle-renderer",
            "criss_cross",
            "--font-max",
            "20",
            "--emit-directions",
        ]
    )
    result = run_render_text_video(args, repo_root)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    sizes = payload["font_sizes"]
    assert sizes
    assert all(size == 20 for size in sizes)


def test_font_bounds_invalid_range_fails(tmp_path: Path) -> None:
    """Reject invalid font bounds where min exceeds max."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "alpha beta"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    output_path = tmp_path / "out.mov"
    args = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path,
        fonts_dir=fonts_dir,
        duration_seconds="0.9",
        fps="6",
    )
    args.extend(
        [
            "--subtitle-renderer",
            "criss_cross",
            "--font-min",
            "60",
            "--font-max",
            "20",
        ]
    )
    result = run_render_text_video(args, repo_root)
    assert result.returncode != 0
    assert "render_text_video.input.invalid_config" in result.stderr


def test_remove_punctuation(tmp_path: Path) -> None:
    """Strip punctuation when requested."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "Hello, world! (Testing) punctuation... OK?"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    duration_seconds = "1.0"
    fps = "6"
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

    args_default = base_args + ["--direction-seed", seed]
    default_result = run_render_text_video(args_default, repo_root)
    assert default_result.returncode == 0

    output_path_keep = tmp_path / "out-keep.mov"
    args_keep = build_common_args(
        script_path=script_path,
        input_path=input_path,
        output_path=output_path_keep,
        fonts_dir=fonts_dir,
        duration_seconds=duration_seconds,
        fps=fps,
    )
    args_keep.extend(["--direction-seed", seed, "--keep-punctuation"])
    kept = run_render_text_video(args_keep, repo_root)
    assert kept.returncode == 0

    time_seconds = float(duration_seconds) / 2.0
    default_frame = extract_raw_frame(output_path, time_seconds)
    kept_frame = extract_raw_frame(output_path_keep, time_seconds)
    assert default_frame != kept_frame


def test_background_image_derives_dimensions(tmp_path: Path) -> None:
    """Use background image to derive frame dimensions."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

    input_text = "alpha beta gamma delta"
    input_path = tmp_path / "words.txt"
    input_path.write_text(input_text, encoding="utf-8")

    background_path = tmp_path / "background.png"
    write_png(background_path, 10, 12, (20, 40, 60, 255))

    duration_seconds = "0.8"
    fps = "6"
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
    stream_info = probe_video_stream(output_path)
    assert stream_info["codec_name"] == "h264"
    assert stream_info["pix_fmt"] == "yuv420p"


def test_background_image_conflicts_with_dimensions(tmp_path: Path) -> None:
    """Fail when background image and dimensions are both provided."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "render_text_video.py"
    fonts_dir = get_test_fonts_dir(repo_root)

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
    fonts_dir = get_test_fonts_dir(repo_root)

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
