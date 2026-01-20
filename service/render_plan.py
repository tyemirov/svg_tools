"""Render plan construction for render_text_video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from domain.text_video import (
    EMPTY_TEXT_CODE,
    INVALID_CONFIG_CODE,
    INVALID_WINDOW_CODE,
    RenderValidationError,
    SubtitleWindow,
)


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
        window_start_frame = int(round(window.start_seconds * fps))
        window_end_frame = int(round(window.end_seconds * fps))
        if window_end_frame <= window_start_frame:
            raise RenderValidationError(
                INVALID_WINDOW_CODE, "subtitle window has no frames"
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
