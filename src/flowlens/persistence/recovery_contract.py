"""Shared deterministic contracts for recovered session timelines."""

from collections.abc import Sequence

from flowlens.domain.enums import EventType
from flowlens.domain.messages import EventRecord
from flowlens.domain.session import PauseInterval


class RecoveryPauseContractError(ValueError):
    """Raised when retained pause events cannot form a recovered timeline."""


def recovered_terminal_time_ms(
    events: Sequence[EventRecord],
    active_duration_ms: int,
) -> int:
    """Return the persisted recovery boundary for retained base events."""

    if (
        not isinstance(active_duration_ms, int)
        or isinstance(active_duration_ms, bool)
        or active_duration_ms < 0
    ):
        raise ValueError("active_duration_ms must be a non-negative integer")
    if not all(isinstance(event, EventRecord) for event in events):
        raise TypeError("events must contain EventRecord values")
    last_event_time_ms = events[-1].session_time_ms if events else 0
    return max(active_duration_ms, last_event_time_ms)


def reconstruct_recovered_pause_intervals(
    events: Sequence[EventRecord],
    active_duration_ms: int,
) -> tuple[tuple[PauseInterval, ...], int | None]:
    """Reconstruct pauses, closing one final open pause at the WAV boundary."""

    if (
        not isinstance(active_duration_ms, int)
        or isinstance(active_duration_ms, bool)
        or active_duration_ms < 0
    ):
        raise ValueError("active_duration_ms must be a non-negative integer")
    intervals: list[PauseInterval] = []
    open_pause: int | None = None
    for event in events:
        if not isinstance(event, EventRecord):
            raise TypeError("events must contain EventRecord values")
        if event.event_type is EventType.PAUSE_START:
            if open_pause is not None:
                raise RecoveryPauseContractError(
                    "PAUSE_START occurred while already paused"
                )
            if event.session_time_ms > active_duration_ms:
                raise RecoveryPauseContractError(
                    "PAUSE_START exceeds recovered duration"
                )
            open_pause = event.session_time_ms
        elif event.event_type is EventType.PAUSE_END:
            if open_pause is None:
                raise RecoveryPauseContractError(
                    "PAUSE_END occurred without PAUSE_START"
                )
            if event.session_time_ms > active_duration_ms:
                raise RecoveryPauseContractError("PAUSE_END exceeds recovered duration")
            intervals.append(PauseInterval(open_pause, event.session_time_ms))
            open_pause = None
    if open_pause is not None:
        intervals.append(PauseInterval(open_pause, active_duration_ms))
    return tuple(intervals), open_pause
