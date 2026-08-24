"""Spawn-safe synchronous arbitration for Writer terminal outcomes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Lock
from typing import Any, Self

from flowlens.domain._validation import json_dumps
from flowlens.domain.messages import (
    EventRecord,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterForceCloseResult,
)

_REQUEST_CAPACITY = 8_192


class _GateState(IntEnum):
    OPEN = 0
    FORCE_REQUESTED = 1
    FINALIZE_COMMITTING = 2
    FORCE_COMMITTING = 3
    COMPLETED = 4
    INCOMPLETE = 5


@dataclass(frozen=True, slots=True)
class WriterTerminalClaim:
    """One Writer-owned terminal candidate selected under the gate lock."""

    outcome: WriterForceCloseOutcome
    event: EventRecord
    force_was_requested: bool


class WriterFinalizationGate:
    """Hold force payload and terminal result in synchronous shared memory."""

    def __init__(
        self,
        lock: Lock,
        state: Any,
        request_length: Any,
        request_buffer: Any,
        result_timestamp_us: Any,
    ) -> None:
        self._lock = lock
        self._state = state
        self._request_length = request_length
        self._request_buffer = request_buffer
        self._result_timestamp_us = result_timestamp_us

    @classmethod
    def create(cls, context: BaseContext) -> Self:
        """Create one gate using primitives from the runtime process context."""

        return cls(
            context.Lock(),
            context.RawValue("b", int(_GateState.OPEN)),
            context.RawValue("I", 0),
            context.RawArray("B", _REQUEST_CAPACITY),
            context.RawValue("q", -1),
        )

    def request_force_close(
        self,
        request: WriterForceCloseRequest,
        *,
        timeout_seconds: float,
    ) -> WriterForceCloseResult | None:
        """Publish force intent with bounded lock acquisition."""

        if type(request) is not WriterForceCloseRequest:
            raise TypeError("request must be an exact WriterForceCloseRequest")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be a non-negative finite number")
        if not self._lock.acquire(timeout=float(timeout_seconds)):
            raise TimeoutError("Writer finalization gate request timed out")
        try:
            state = self._read_state()
            if state is _GateState.OPEN:
                self._write_request(request)
                self._state.value = int(_GateState.FORCE_REQUESTED)
                return None
            if state is _GateState.FORCE_REQUESTED:
                if self._read_request() != request:
                    raise RuntimeError("a different force-close request is pending")
                return None
            if state in {_GateState.FORCE_COMMITTING, _GateState.INCOMPLETE}:
                return self._result_for_state(state)
            if state in {_GateState.FINALIZE_COMMITTING, _GateState.COMPLETED}:
                return self._result_for_state(state)
            raise RuntimeError("unknown Writer finalization gate state")
        finally:
            self._lock.release()

    def claim_force_if_requested(self) -> WriterTerminalClaim | None:
        """Let Writer claim an early force request without a finalize command."""

        with self._lock:
            if self._read_state() is not _GateState.FORCE_REQUESTED:
                return None
            request = self._read_request()
            self._state.value = int(_GateState.FORCE_COMMITTING)
            return WriterTerminalClaim(
                WriterForceCloseOutcome.INCOMPLETE,
                request.event,
                True,
            )

    def claim_terminal(self, completion_event: EventRecord) -> WriterTerminalClaim:
        """Select exactly one terminal event at the short commit boundary."""

        if type(completion_event) is not EventRecord:
            raise TypeError("completion_event must be an exact EventRecord")
        with self._lock:
            state = self._read_state()
            if state is _GateState.OPEN:
                self._state.value = int(_GateState.FINALIZE_COMMITTING)
                return WriterTerminalClaim(
                    WriterForceCloseOutcome.COMPLETED,
                    completion_event,
                    False,
                )
            if state is _GateState.FORCE_REQUESTED:
                request = self._read_request()
                if (
                    request.event.session_id != completion_event.session_id
                    or request.event.sequence != completion_event.sequence
                ):
                    raise RuntimeError(
                        "force and completion events must share one terminal candidate"
                    )
                self._state.value = int(_GateState.FORCE_COMMITTING)
                return WriterTerminalClaim(
                    WriterForceCloseOutcome.INCOMPLETE,
                    request.event,
                    True,
                )
            raise RuntimeError("Writer terminal outcome was already claimed")

    def publish_result(
        self,
        outcome: WriterForceCloseOutcome,
        saved_at: datetime,
    ) -> WriterForceCloseResult:
        """Publish a durable terminal result after the selected commit succeeds."""

        result = WriterForceCloseResult(outcome, saved_at)
        timestamp_us = int(
            result.latest_successful_save_at.astimezone(UTC).timestamp() * 1_000_000
        )
        with self._lock:
            state = self._read_state()
            expected = (
                _GateState.FORCE_COMMITTING
                if outcome is WriterForceCloseOutcome.INCOMPLETE
                else _GateState.FINALIZE_COMMITTING
            )
            if state is not expected:
                raise RuntimeError("terminal result does not match the claimed outcome")
            self._result_timestamp_us.value = timestamp_us
            self._state.value = int(
                _GateState.INCOMPLETE
                if outcome is WriterForceCloseOutcome.INCOMPLETE
                else _GateState.COMPLETED
            )
        return result

    def result(self) -> WriterForceCloseResult | None:
        """Inspect a final outcome without waiting on a possibly orphaned lock."""

        return self._result_for_state(self._read_state())

    def _result_for_state(
        self,
        state: _GateState,
    ) -> WriterForceCloseResult | None:
        if state not in {_GateState.COMPLETED, _GateState.INCOMPLETE}:
            return None
        timestamp_us = int(self._result_timestamp_us.value)
        if timestamp_us < 0:
            return None
        saved_at = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC)
        outcome = (
            WriterForceCloseOutcome.COMPLETED
            if state is _GateState.COMPLETED
            else WriterForceCloseOutcome.INCOMPLETE
        )
        return WriterForceCloseResult(outcome, saved_at)

    def _write_request(self, request: WriterForceCloseRequest) -> None:
        payload = json_dumps(request.event.to_dict()).encode("utf-8")
        if len(payload) > _REQUEST_CAPACITY:
            raise ValueError("force-close request exceeds shared gate capacity")
        self._request_buffer[: len(payload)] = payload
        self._request_length.value = len(payload)

    def _read_request(self) -> WriterForceCloseRequest:
        length = int(self._request_length.value)
        if length <= 0 or length > _REQUEST_CAPACITY:
            raise RuntimeError("shared force-close request is unavailable")
        payload = bytes(self._request_buffer[:length]).decode("utf-8")
        return WriterForceCloseRequest(EventRecord.from_dict(json.loads(payload)))

    def _read_state(self) -> _GateState:
        try:
            return _GateState(int(self._state.value))
        except ValueError as error:
            raise RuntimeError("shared Writer finalization state is invalid") from error
