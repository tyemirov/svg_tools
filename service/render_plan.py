"""Render plan construction for render_text_video."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from domain.text_video import (
    EMPTY_TEXT_CODE,
    INVALID_CONFIG_CODE,
    INVALID_WINDOW_CODE,
    RenderValidationError,
    SubtitleWindow,
    split_trailing_punctuation,
)

RSVP_MIN_MS = 80
RSVP_MAX_MS = 700
RSVP_PUNCT_PAUSE_MS = 160


@dataclass(frozen=True)
class ScheduledWord:
    """A word scheduled over a contiguous frame range."""

    token_index: int
    start_frame: int
    frame_count: int

    def __post_init__(self) -> None:
        if self.token_index < 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "token_index must be non-negative"
            )
        if self.start_frame < 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "start_frame must be non-negative"
            )
        if self.frame_count <= 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "frame_count must be positive"
            )


@dataclass(frozen=True)
class RenderPlan:
    """Plan describing word ordering and timing across frames."""

    total_frames: int
    words: Tuple[str, ...]
    scheduled_words: Tuple[ScheduledWord, ...]

    def __post_init__(self) -> None:
        if self.total_frames <= 0:
            raise RenderValidationError(
                INVALID_CONFIG_CODE, "total_frames must be positive"
            )
        if not self.words:
            raise RenderValidationError(EMPTY_TEXT_CODE, "no words to render")

        last_end_frame = 0
        for scheduled_word in self.scheduled_words:
            if scheduled_word.token_index >= len(self.words):
                raise RenderValidationError(
                    INVALID_CONFIG_CODE, "token_index out of bounds"
                )
            end_frame = scheduled_word.start_frame + scheduled_word.frame_count
            if end_frame > self.total_frames:
                raise RenderValidationError(
                    INVALID_CONFIG_CODE, "scheduled word exceeds total frames"
                )
            if scheduled_word.start_frame < last_end_frame:
                raise RenderValidationError(
                    INVALID_CONFIG_CODE, "scheduled words overlap"
                )
            last_end_frame = end_frame


def build_render_plan(
    windows: Sequence[SubtitleWindow], fps: int, duration_seconds: float
) -> RenderPlan:
    """Build a render plan from subtitle windows and timing."""
    if not windows:
        raise RenderValidationError(EMPTY_TEXT_CODE, "no subtitle windows")

    total_frames = int(round(duration_seconds * fps))
    if total_frames <= 0:
        raise RenderValidationError(
            INVALID_CONFIG_CODE, "duration and fps produce zero frames"
        )

    scheduled_words: list[ScheduledWord] = []
    words: list[str] = []
    current_end_frame = 0
    token_offset = 0

    for window in windows:
        window_start_frame, window_end_frame = window_to_frames(
            window.start_seconds, window.end_seconds, fps
        )
        if window_start_frame < current_end_frame:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle windows overlap"
            )
        if window_end_frame > total_frames:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle window exceeds duration"
            )

        window_frames = window_end_frame - window_start_frame
        word_count = len(window.words)
        if window_frames < word_count:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle window too short for words"
            )

        frames_per_word = window_frames // word_count
        remainder = window_frames % word_count
        cursor_frame = window_start_frame

        for index, word in enumerate(window.words):
            frame_count = frames_per_word + (1 if index < remainder else 0)
            scheduled_words.append(
                ScheduledWord(
                    token_index=token_offset + index,
                    start_frame=cursor_frame,
                    frame_count=frame_count,
                )
            )
            cursor_frame += frame_count
            words.append(word)

        token_offset += word_count
        current_end_frame = window_end_frame

    return RenderPlan(
        total_frames=total_frames,
        words=tuple(words),
        scheduled_words=tuple(scheduled_words),
    )


def window_to_frames(
    start_seconds: float, end_seconds: float, fps: int
) -> Tuple[int, int]:
    """Convert subtitle window seconds to a non-empty frame range."""
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    return start_frame, end_frame


def ms_to_frames_ceil(milliseconds: float, fps: int) -> int:
    """Convert milliseconds to frames, rounding up."""
    return max(1, int(math.ceil(milliseconds * fps / 1000.0)))


def ms_to_frames_floor(milliseconds: float, fps: int) -> int:
    """Convert milliseconds to frames, rounding down."""
    return max(1, int(math.floor(milliseconds * fps / 1000.0)))


def allocate_weighted_frames(
    weights: Sequence[int],
    total_frames: int,
    min_frames: int,
    max_frames: int,
) -> Tuple[int, ...]:
    """Allocate frames to weights while respecting min/max bounds."""
    if not weights:
        raise RenderValidationError(INVALID_WINDOW_CODE, "subtitle window has no words")

    word_count = len(weights)
    min_total = min_frames * word_count
    max_total = max_frames * word_count
    if total_frames < min_total:
        raise RenderValidationError(
            INVALID_WINDOW_CODE, "subtitle window too short for RSVP timing"
        )
    if total_frames > max_total:
        raise RenderValidationError(
            INVALID_WINDOW_CODE, "subtitle window too long for RSVP timing"
        )

    allocations = [min_frames for _ in range(word_count)]
    remaining = total_frames - min_total
    if remaining <= 0:
        return tuple(allocations)

    active = {index for index in range(word_count) if allocations[index] < max_frames}
    while remaining > 0 and active:
        total_weight = sum(weights[index] for index in active)
        if total_weight <= 0:
            break

        remainders: list[tuple[float, int]] = []
        used = 0
        for index in active:
            share = remaining * weights[index] / total_weight
            capacity = max_frames - allocations[index]
            add = min(capacity, int(math.floor(share)))
            if add:
                allocations[index] += add
                used += add
            remainders.append((share - add, index))

        remaining -= used
        if remaining <= 0:
            break

        remainders.sort(key=lambda pair: pair[0], reverse=True)
        for _, index in remainders:
            if remaining <= 0:
                break
            if allocations[index] < max_frames:
                allocations[index] += 1
                remaining -= 1

        active = {index for index in active if allocations[index] < max_frames}

    if remaining > 0:
        raise RenderValidationError(
            INVALID_WINDOW_CODE, "subtitle window cannot allocate RSVP frames"
        )

    return tuple(allocations)


def compute_rsvp_start_frame(
    window_start_frame: int,
    window_frames: int,
    sequence_frames: int,
    current_end_frame: int,
    total_frames: int,
) -> int:
    """Compute a best-effort start frame for RSVP scheduling."""
    if sequence_frames > total_frames:
        raise RenderValidationError(
            INVALID_WINDOW_CODE, "subtitle window exceeds duration"
        )

    start_frame = window_start_frame
    if sequence_frames > window_frames:
        overflow = sequence_frames - window_frames
        start_frame = max(0, window_start_frame - overflow // 2)

    start_frame = max(start_frame, current_end_frame)
    if start_frame + sequence_frames > total_frames:
        start_frame = max(current_end_frame, total_frames - sequence_frames)

    return start_frame


def build_rsvp_render_plan(
    windows: Sequence[SubtitleWindow], fps: int, duration_seconds: float
) -> RenderPlan:
    """Build a render plan for RSVP-style word timing."""
    if not windows:
        raise RenderValidationError(EMPTY_TEXT_CODE, "no subtitle windows")

    total_frames = int(round(duration_seconds * fps))
    if total_frames <= 0:
        raise RenderValidationError(
            INVALID_CONFIG_CODE, "duration and fps produce zero frames"
        )

    min_frames = ms_to_frames_ceil(RSVP_MIN_MS, fps)
    max_frames = ms_to_frames_floor(RSVP_MAX_MS, fps)
    max_frames = max(min_frames, max_frames)
    pause_frames = ms_to_frames_ceil(RSVP_PUNCT_PAUSE_MS, fps)

    scheduled_words: list[ScheduledWord] = []
    words: list[str] = []
    current_end_frame = 0
    token_offset = 0

    for window in windows:
        window_start_frame, window_end_frame = window_to_frames(
            window.start_seconds, window.end_seconds, fps
        )
        if window_end_frame > total_frames:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle window exceeds duration"
            )

        window_frames = window_end_frame - window_start_frame
        word_count = len(window.words)
        if word_count <= 0:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle window has no words"
            )

        punctuation_flags = []
        weights: list[int] = []
        for word in window.words:
            core, trailing = split_trailing_punctuation(word)
            punctuation_flags.append(bool(trailing))
            weights.append(max(1, len(core)))

        pause_frames_used = pause_frames
        pause_total = pause_frames_used * sum(1 for flag in punctuation_flags if flag)
        available_frames = window_frames - pause_total
        if available_frames <= 0:
            pause_frames_used = 0
            pause_total = 0
            available_frames = window_frames

        min_total = min_frames * word_count
        max_total = max_frames * word_count
        target_frames = min(available_frames, max_total)
        min_frames_override = min_frames
        if available_frames < min_total:
            pause_frames_used = 0
            pause_total = 0
            available_frames = window_frames
            min_frames_override = max(1, available_frames // word_count)
            min_total_override = min_frames_override * word_count
            if available_frames < min_total_override:
                target_frames = min_total_override
            else:
                target_frames = available_frames

        max_frames_override = max(
            min_frames_override, min(max_frames, target_frames)
        )
        base_frames = allocate_weighted_frames(
            weights, target_frames, min_frames_override, max_frames_override
        )
        per_word_frames = [
            base + (pause_frames_used if flagged else 0)
            for base, flagged in zip(base_frames, punctuation_flags)
        ]

        sequence_frames = sum(per_word_frames)
        cursor_frame = compute_rsvp_start_frame(
            window_start_frame,
            window_frames,
            sequence_frames,
            current_end_frame,
            total_frames,
        )
        for index, word in enumerate(window.words):
            frame_count = per_word_frames[index]
            scheduled_words.append(
                ScheduledWord(
                    token_index=token_offset + index,
                    start_frame=cursor_frame,
                    frame_count=frame_count,
                )
            )
            cursor_frame += frame_count
            words.append(word)

        token_offset += word_count
        current_end_frame = cursor_frame

    return RenderPlan(
        total_frames=total_frames,
        words=tuple(words),
        scheduled_words=tuple(scheduled_words),
    )
