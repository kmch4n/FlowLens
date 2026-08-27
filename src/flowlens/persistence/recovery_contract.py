"""Shared deterministic contracts for recovered session timelines."""

from collections.abc import Sequence
from dataclasses import dataclass

from flowlens.domain.enums import EventType
from flowlens.domain.messages import EventRecord
from flowlens.domain.session import PauseInterval


class RecoveryPauseContractError(ValueError):
    """Raised when retained pause events cannot form a recovered timeline."""


@dataclass(frozen=True, slots=True)
class RecoveredTimeline:
    """One authoritative recovered wall timeline derived from persisted evidence."""

    pause_intervals: tuple[PauseInterval, ...]
    completed_pause_duration_ms: int
    terminal_time_ms: int
    closed_open_pause_at_ms: int | None


def reconstruct_recovered_timeline(
    events: Sequence[EventRecord],
    active_duration_ms: int,
) -> RecoveredTimeline:
    """Reconstruct pause intervals and the wall-time recovery boundary once."""

    if (
        not isinstance(active_duration_ms, int)
        or isinstance(active_duration_ms, bool)
        or active_duration_ms < 0
    ):
        raise ValueError("active_duration_ms must be a non-negative integer")
    intervals: list[PauseInterval] = []
    completed_pause_duration_ms = 0
    open_pause: int | None = None
    last_pause_event_ms: int | None = None
    for event in events:
        if not isinstance(event, EventRecord):
            raise TypeError("events must contain EventRecord values")
        if event.event_type not in {EventType.PAUSE_START, EventType.PAUSE_END}:
            continue
        if (
            last_pause_event_ms is not None
            and event.session_time_ms < last_pause_event_ms
        ):
            raise RecoveryPauseContractError("pause events must be chronological")
        last_pause_event_ms = event.session_time_ms
        if event.event_type is EventType.PAUSE_START:
            if open_pause is not None:
                raise RecoveryPauseContractError(
                    "PAUSE_START occurred while already paused"
                )
            open_pause = event.session_time_ms
        elif event.event_type is EventType.PAUSE_END:
            if open_pause is None:
                raise RecoveryPauseContractError(
                    "PAUSE_END occurred without PAUSE_START"
                )
            if event.session_time_ms < open_pause:
                raise RecoveryPauseContractError(
                    "PAUSE_END must not precede PAUSE_START"
                )
            intervals.append(PauseInterval(open_pause, event.session_time_ms))
            completed_pause_duration_ms += event.session_time_ms - open_pause
            open_pause = None
    last_event_time_ms = events[-1].session_time_ms if events else 0
    terminal_time_ms = max(
        active_duration_ms + completed_pause_duration_ms,
        last_event_time_ms,
    )
    if open_pause is not None:
        if terminal_time_ms < open_pause:
            raise RecoveryPauseContractError(
                "recovery boundary must not precede open PAUSE_START"
            )
        intervals.append(PauseInterval(open_pause, terminal_time_ms))
    return RecoveredTimeline(
        pause_intervals=tuple(intervals),
        completed_pause_duration_ms=completed_pause_duration_ms,
        terminal_time_ms=terminal_time_ms,
        closed_open_pause_at_ms=open_pause,
    )
