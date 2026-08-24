"""Pure discussion state machine and bounded multiprocessing queue loop."""

from __future__ import annotations

import multiprocessing
import queue
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from flowlens.discussion.context import select_recent_records
from flowlens.discussion.contracts import (
    DiscussionBackend,
    DiscussionContextError,
    DiscussionGenerationError,
    DiscussionOutputError,
    DiscussionRequest,
    DiscussionStatusPayload,
    DiscussionStoppedPayload,
)
from flowlens.discussion.llama_cpp_adapter import (
    DiscussionModelConfig,
    load_llama_cpp_backend,
)
from flowlens.discussion.prompt import build_messages
from flowlens.discussion.scheduler import DiscussionScheduler
from flowlens.discussion.schema import discussion_state_schema, parse_discussion_state
from flowlens.domain._validation import ContractValidationError
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    DiscussionStateReplaced,
    MessageEnvelope,
    MessageSequenceError,
    TranscriptCommitted,
)

BackendLoader = Callable[[DiscussionModelConfig], DiscussionBackend]
MonotonicClock = Callable[[], int]
WallClock = Callable[[], datetime]
ParentProbe = Callable[[], bool]
_SESSION_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
_DEFAULT_POLL_TIMEOUT_SECONDS = 0.05


class _QueueIn(Protocol):
    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> object: ...


class _QueueOut(Protocol):
    def put(self, item: MessageEnvelope[object]) -> None: ...

    def close(self) -> None: ...

    def join_thread(self) -> None: ...


class DiscussionWorkerProtocolError(ValueError):
    """Raised when a control envelope violates the Discussion protocol."""


@dataclass(frozen=True, slots=True)
class DiscussionWorkerConfig:
    """Spawn-picklable configuration for one local Discussion worker."""

    session_id: str
    model: DiscussionModelConfig
    initial_state: DiscussionState
    coalesce_ms: int = 500

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or _SESSION_ID_PATTERN.fullmatch(self.session_id) is None
        ):
            raise ContractValidationError(
                "session_id must contain 26 uppercase Crockford characters"
            )
        if not isinstance(self.model, DiscussionModelConfig):
            raise ContractValidationError("model must be a DiscussionModelConfig")
        if not isinstance(self.initial_state, DiscussionState):
            raise ContractValidationError("initial_state must be a DiscussionState")
        if (
            not isinstance(self.coalesce_ms, int)
            or isinstance(self.coalesce_ms, bool)
            or self.coalesce_ms <= 0
        ):
            raise ContractValidationError("coalesce_ms must be a positive integer")


class DiscussionWorkerCore:
    """Deterministic Discussion lifecycle independent of queues and processes."""

    def __init__(
        self,
        config: DiscussionWorkerConfig,
        *,
        backend_loader: BackendLoader,
        monotonic_ms: MonotonicClock,
        wall_clock: WallClock,
    ) -> None:
        if not isinstance(config, DiscussionWorkerConfig):
            raise TypeError("config must be a DiscussionWorkerConfig")
        self._config = config
        self._backend_loader = backend_loader
        self._monotonic_ms = monotonic_ms
        self._wall_clock = wall_clock
        self._scheduler = DiscussionScheduler(
            config.initial_state,
            coalesce_ms=config.coalesce_ms,
        )
        self._backend: DiscussionBackend | None = None
        self._lifecycle = "WAITING_START"
        self._expected_sequence = 1
        self._outgoing_sequence = 0
        self._last_failed_pending_count = 0

    @property
    def state(self) -> DiscussionState:
        """Return the latest valid immutable discussion state."""

        return self._scheduler.current_state

    @property
    def stopped(self) -> bool:
        """Return whether the exact final drain acknowledgement was emitted."""

        return self._lifecycle == "STOPPED"

    @property
    def failed(self) -> bool:
        """Return whether a terminal worker error was emitted."""

        return self._lifecycle == "FAILED"

    def handle(
        self,
        envelope: MessageEnvelope[object],
    ) -> tuple[MessageEnvelope[object], ...]:
        """Validate and apply one controller envelope transactionally."""

        self._validate_envelope(envelope)
        if envelope.sequence != self._expected_sequence:
            raise MessageSequenceError(
                f"expected control sequence {self._expected_sequence}, "
                f"got {envelope.sequence}"
            )

        outgoing = self._dispatch(envelope)
        self._expected_sequence += 1
        return outgoing

    def tick(
        self,
        now_ms: int,
        now: datetime,
    ) -> tuple[MessageEnvelope[object], ...]:
        """Launch one due analysis request while the worker is running."""

        if self._lifecycle != "RUNNING":
            return ()
        request = self._scheduler.next_request(now_ms, now)
        if request is None:
            return ()
        return self._generate(request)

    def fatal(self, code: str) -> MessageEnvelope[object]:
        """Build one metadata-only terminal worker error."""

        self._lifecycle = "FAILED"
        return self._emit(
            MessageType.WORKER_ERROR,
            {"worker": "DISCUSSION", "code": code},
        )

    def _validate_envelope(self, envelope: object) -> None:
        if not isinstance(envelope, MessageEnvelope):
            raise DiscussionWorkerProtocolError(
                "control queue item must be a MessageEnvelope"
            )
        try:
            envelope.validate_schema()
        except ValueError as error:
            raise DiscussionWorkerProtocolError(
                "control envelope schema is unsupported"
            ) from error
        if envelope.session_id != self._config.session_id:
            raise DiscussionWorkerProtocolError(
                "control envelope targets a different session"
            )
        if envelope.source is not ProcessSource.GUI:
            raise DiscussionWorkerProtocolError("control source must be GUI")
        self._validate_payload(envelope)

    @staticmethod
    def _validate_payload(envelope: MessageEnvelope[object]) -> None:
        lifecycle_keys: dict[MessageType, frozenset[str]] = {
            MessageType.WORKER_START: frozenset({"worker"}),
            MessageType.WORKER_PAUSE: frozenset({"worker"}),
            MessageType.WORKER_RESUME: frozenset({"worker"}),
            MessageType.WORKER_STOP: frozenset({"worker", "finalize"}),
        }
        expected_keys = lifecycle_keys.get(envelope.message_type)
        if expected_keys is not None:
            payload = envelope.payload
            if (
                type(payload) is not dict
                or frozenset(payload) != expected_keys
                or type(payload["worker"]) is not str
                or payload["worker"] != "DISCUSSION"
                or (
                    envelope.message_type is MessageType.WORKER_STOP
                    and payload["finalize"] is not True
                )
            ):
                raise DiscussionWorkerProtocolError(
                    "lifecycle payload does not target Discussion exactly"
                )
            return
        if envelope.message_type is MessageType.TRANSCRIPT_COMMITTED:
            if not isinstance(envelope.payload, TranscriptCommitted):
                raise DiscussionWorkerProtocolError(
                    "committed payload must be TranscriptCommitted"
                )
            return
        raise DiscussionWorkerProtocolError(
            "message type does not target the Discussion worker"
        )

    def _dispatch(
        self,
        envelope: MessageEnvelope[object],
    ) -> tuple[MessageEnvelope[object], ...]:
        message_type = envelope.message_type
        if message_type is MessageType.WORKER_START:
            return self._start()
        if message_type is MessageType.TRANSCRIPT_COMMITTED:
            return self._commit(envelope)
        if message_type is MessageType.WORKER_PAUSE:
            return self._pause()
        if message_type is MessageType.WORKER_RESUME:
            return self._resume()
        if message_type is MessageType.WORKER_STOP:
            return self._stop()
        raise DiscussionWorkerProtocolError("unsupported Discussion command")

    def _start(self) -> tuple[MessageEnvelope[object], ...]:
        self._require_state("WAITING_START")
        try:
            self._backend = self._backend_loader(self._config.model)
        except Exception as error:
            code = _load_error_code(error)
            return (self.fatal(code),)
        self._lifecycle = "RUNNING"
        return (self._emit(MessageType.WORKER_READY, {"worker": "DISCUSSION"}),)

    def _commit(
        self,
        envelope: MessageEnvelope[object],
    ) -> tuple[MessageEnvelope[object], ...]:
        if self._lifecycle not in {"RUNNING", "PAUSED"}:
            raise DiscussionWorkerProtocolError(
                f"TRANSCRIPT_COMMITTED is invalid in state {self._lifecycle}"
            )
        payload = envelope.payload
        if not isinstance(payload, TranscriptCommitted):
            raise DiscussionWorkerProtocolError(
                "committed payload must be TranscriptCommitted"
            )
        self._scheduler.add(payload.record, envelope.created_monotonic_ms)
        return ()

    def _pause(self) -> tuple[MessageEnvelope[object], ...]:
        self._require_state("RUNNING")
        self._scheduler.set_paused(True)
        self._lifecycle = "PAUSED"
        return ()

    def _resume(self) -> tuple[MessageEnvelope[object], ...]:
        self._require_state("PAUSED")
        self._scheduler.set_paused(False)
        self._lifecycle = "RUNNING"
        return ()

    def _stop(self) -> tuple[MessageEnvelope[object], ...]:
        if self._lifecycle == "STOPPED":
            return ()
        if self._lifecycle not in {"RUNNING", "PAUSED"}:
            raise DiscussionWorkerProtocolError(
                f"WORKER_STOP is invalid in state {self._lifecycle}"
            )
        if self._lifecycle == "PAUSED":
            self._scheduler.set_paused(False)
        request = self._scheduler.final_request(self._wall_clock())
        outgoing = () if request is None else self._generate(request)
        self._lifecycle = "STOPPED"
        stopped = DiscussionStoppedPayload(
            worker="DISCUSSION",
            drained=True,
            final_revision=self.state.revision,
            pending_count=self._pending_count_after_final(outgoing),
        )
        return (*outgoing, self._emit(MessageType.WORKER_STOPPED, stopped))

    def _pending_count_after_final(
        self,
        outgoing: tuple[MessageEnvelope[object], ...],
    ) -> int:
        if any(
            item.message_type is MessageType.DISCUSSION_STATE_REPLACED
            for item in outgoing
        ):
            return 0
        return self._last_failed_pending_count if self._scheduler.has_pending else 0

    def _generate(
        self,
        scheduler_request: DiscussionRequest,
    ) -> tuple[MessageEnvelope[object], ...]:
        backend = self._backend
        if backend is None:
            raise DiscussionWorkerProtocolError("Discussion backend is not loaded")
        try:
            selected = select_recent_records(
                scheduler_request.records,
                backend.count_tokens,
            )
            generation_request = DiscussionRequest(
                current_state=scheduler_request.current_state,
                records=selected,
                requested_revision=scheduler_request.requested_revision,
                updated_at=scheduler_request.updated_at,
            )
            raw = backend.generate(
                build_messages(generation_request),
                discussion_state_schema(generation_request),
            )
            new_state = parse_discussion_state(raw, generation_request)
            self._scheduler.succeed(scheduler_request, new_state)
        except Exception as error:
            self._scheduler.fail(scheduler_request)
            self._last_failed_pending_count = len(scheduler_request.records)
            return (
                self._emit(
                    MessageType.DISCUSSION_STATUS,
                    DiscussionStatusPayload(
                        state="FAILED",
                        revision=self.state.revision,
                        pending_count=len(scheduler_request.records),
                        error_code=_generation_error_code(error),
                    ),
                ),
            )
        self._last_failed_pending_count = 0
        return (
            self._emit(
                MessageType.DISCUSSION_STATE_REPLACED,
                DiscussionStateReplaced(
                    previous_revision=scheduler_request.current_state.revision,
                    state=new_state,
                ),
            ),
            self._emit(
                MessageType.DISCUSSION_STATUS,
                DiscussionStatusPayload(
                    state="UPDATED",
                    revision=new_state.revision,
                    pending_count=0,
                    error_code=None,
                ),
            ),
        )

    def _require_state(self, expected: str) -> None:
        if self._lifecycle != expected:
            raise DiscussionWorkerProtocolError(
                f"command is invalid in state {self._lifecycle}"
            )

    def _emit(
        self,
        message_type: MessageType,
        payload: object,
    ) -> MessageEnvelope[object]:
        self._outgoing_sequence += 1
        return MessageEnvelope(
            schema_version=1,
            session_id=self._config.session_id,
            message_type=message_type,
            sequence=self._outgoing_sequence,
            source=ProcessSource.DISCUSSION,
            created_monotonic_ms=self._monotonic_ms(),
            payload=payload,
        )


def _generation_error_code(error: Exception) -> str:
    if isinstance(error, DiscussionOutputError):
        return "INVALID_OUTPUT"
    if isinstance(error, DiscussionContextError):
        return "CONTEXT_LIMIT"
    if isinstance(error, DiscussionGenerationError):
        return "GENERATION_FAILED"
    return "GENERATION_FAILED"


def _load_error_code(error: Exception) -> str:
    if isinstance(error, MemoryError):
        return "GPU_OOM"
    error_name = type(error).__name__.lower()
    try:
        error_text = str(error).lower()
    except Exception:
        error_text = ""
    if "outofmemory" in error_name or "out of memory" in error_text:
        return "GPU_OOM"
    return "MODEL_LOAD_FAILED"


def _parent_alive() -> bool:
    parent = multiprocessing.parent_process()
    return parent is None or parent.is_alive()


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1_000)


def _wall_clock() -> datetime:
    return datetime.now(UTC)


def _close_output(output: _QueueOut) -> None:
    for operation in (output.close, output.join_thread):
        try:
            operation()
        except BaseException:
            continue


def _discussion_worker_loop(
    config: DiscussionWorkerConfig,
    control_in: _QueueIn,
    control_out: _QueueOut,
    *,
    backend_loader: BackendLoader,
    monotonic_ms: MonotonicClock,
    wall_clock: WallClock,
    parent_alive: ParentProbe,
    poll_timeout_seconds: float = _DEFAULT_POLL_TIMEOUT_SECONDS,
) -> int:
    """Run the fakeable bounded Discussion queue loop."""

    if (
        not isinstance(poll_timeout_seconds, float)
        or poll_timeout_seconds <= 0.0
        or poll_timeout_seconds > _DEFAULT_POLL_TIMEOUT_SECONDS
    ):
        raise ValueError("poll_timeout_seconds must be in (0.0, 0.05]")
    core = DiscussionWorkerCore(
        config,
        backend_loader=backend_loader,
        monotonic_ms=monotonic_ms,
        wall_clock=wall_clock,
    )
    try:
        while True:
            if not parent_alive():
                return 0
            try:
                value = control_in.get(timeout=poll_timeout_seconds)
            except queue.Empty:
                if not parent_alive():
                    return 0
                for tick_item in core.tick(monotonic_ms(), wall_clock()):
                    control_out.put(tick_item)
                continue
            except (EOFError, OSError, ValueError):
                return 0

            if not parent_alive():
                return 0
            try:
                outgoing_envelopes = core.handle(value)  # type: ignore[arg-type]
            except (DiscussionWorkerProtocolError, MessageSequenceError, ValueError):
                control_out.put(core.fatal("PROTOCOL_ERROR"))
                return 1
            for item in outgoing_envelopes:
                control_out.put(item)
            if core.failed:
                return 1
            if core.stopped:
                return 0
    except Exception:
        try:
            control_out.put(core.fatal("WORKER_FAILED"))
        except Exception:
            pass
        return 1
    finally:
        _close_output(control_out)


def run_discussion_worker(
    config: DiscussionWorkerConfig,
    control_in: multiprocessing.Queue[MessageEnvelope[object]],
    control_out: multiprocessing.Queue[MessageEnvelope[object]],
) -> None:
    """Run the spawn-safe production Discussion worker entry point."""

    exit_code = _discussion_worker_loop(
        config,
        control_in,
        control_out,
        backend_loader=load_llama_cpp_backend,
        monotonic_ms=_monotonic_ms,
        wall_clock=_wall_clock,
        parent_alive=_parent_alive,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)
