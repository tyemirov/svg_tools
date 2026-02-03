"""Integration tests for audio_to_text CLI."""

from __future__ import annotations

import os
import platform
import subprocess
import wave
import json
import math
import sys
from pathlib import Path
from typing import List

import pytest
from reel import audio_to_text

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
    ]
    for args, expected in cases:
        result = run_audio_to_text(args, repo_root)
        assert result.returncode != 0
        assert expected in result.stderr


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
