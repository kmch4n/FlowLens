"""Acknowledgement-driven ordered session finalization."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from flowlens.discussion.contracts import DiscussionStoppedPayload
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    EventRecord,
    MessageEnvelope,
    WriterAck,
    WriterFinalize,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterForceCloseResult,
)

SLOW_FINALIZATION_MS = 30_000
FORCE_REQUEST_TIMEOUT_SECONDS = 0.25
FORCE_RESULT_TIMEOUT_MS = 5_000

SendCommand = Callable[[ProcessSource, MessageType, object], MessageEnvelope[object]]
EnvelopeCommand = tuple[ProcessSource, MessageEnvelope[object]]


class FinalizationStep(str, Enum):
    """Exact ordered finalization states."""

    STOP_AUDIO = "STOP_AUDIO"
    DRAIN_AUDIO = "DRAIN_AUDIO"
    FINALIZE_ASR = "FINALIZE_ASR"
    FINAL_ANALYSIS = "FINAL_ANALYSIS"
    FINALIZE_WRITER = "FINALIZE_WRITER"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class FinalizationSnapshot:
    """Immutable UI-facing finalization state."""

    step: FinalizationStep
    started_ms: int | None
    slow_visible: bool
    incomplete: bool
    completed: bool
    force_requested: bool
    force_deadline_ms: int | None


@dataclass(frozen=True, slots=True)
class FinalizationTick:
    """One clock observation without implicit control commands."""

    show_slow_message: bool
    commands: tuple[EnvelopeCommand, ...] = ()
    force_result_timed_out: bool = False


class FinalizationCoordinator:
    """Advance finalization only after the exact current acknowledgement."""

    def __init__(
        self,
        *,
        session_id: str,
        send: SendCommand,
        writer_finalize_factory: Callable[[], WriterFinalize],
        force_close_event_factory: Callable[[], EventRecord],
        acknowledge_expected_stop: Callable[[ProcessSource], None],
        request_writer_force_close: Callable[
            [WriterForceCloseRequest, float], WriterForceCloseResult | None
        ],
        shutdown: Callable[[], None],
    ) -> None:
        if type(session_id) is not str or not session_id:
            raise ValueError("session_id must be a non-empty exact string")
        self._session_id = session_id
        self._send = send
        self._writer_finalize_factory = writer_finalize_factory
        self._force_close_event_factory = force_close_event_factory
        self._acknowledge_expected_stop = acknowledge_expected_stop
        self._request_writer_force_close = request_writer_force_close
        self._shutdown = shutdown
        self._step = FinalizationStep.STOP_AUDIO
        self._started_ms: int | None = None
        self._last_tick_ms: int | None = None
        self._next_slow_offer_ms: int | None = None
        self._slow_visible = False
        self._incomplete = False
        self._force_requested = False
        self._force_deadline_ms: int | None = None
        self._finalize_sequence: int | None = None

    @property
    def step(self) -> FinalizationStep:
        """Return the current immutable state-machine step."""

        return self._step

    @property
    def completed(self) -> bool:
        """Return whether matching Writer acknowledgement completed the session."""

        return self._step is FinalizationStep.COMPLETE

    def snapshot(self) -> FinalizationSnapshot:
        """Return a pure finalization snapshot for later UI tasks."""

        return FinalizationSnapshot(
            self._step,
            self._started_ms,
            self._slow_visible,
            self._incomplete,
            self.completed,
            self._force_requested,
            self._force_deadline_ms,
        )

    def begin(self, now_ms: int) -> tuple[EnvelopeCommand, ...]:
        """Stop Audio once and establish the non-refreshing hard deadline."""

        self._observe_time(now_ms)
        if self._started_ms is not None:
            return ()
        command = self._send_one(
            ProcessSource.AUDIO,
            MessageType.WORKER_STOP,
            {"worker": "AUDIO"},
        )
        self._started_ms = now_ms
        self._next_slow_offer_ms = now_ms + SLOW_FINALIZATION_MS
        self._step = FinalizationStep.DRAIN_AUDIO
        return (command,)

    def acknowledge(
        self,
        envelope: MessageEnvelope[object],
    ) -> tuple[EnvelopeCommand, ...]:
        """Advance only for an exact current-step acknowledgement."""

        if not self.accepts(envelope):
            return ()
        if envelope.message_type is MessageType.WRITER_FORCE_CLOSE_RESULT:
            result = envelope.payload
            if type(result) is not WriterForceCloseResult:
                raise RuntimeError("validated force-close result changed type")
            self.resolve_force_result(result)
            return ()
        if self._step is FinalizationStep.DRAIN_AUDIO:
            return self._advance_worker(
                ProcessSource.AUDIO,
                FinalizationStep.FINALIZE_ASR,
                ProcessSource.ASR,
                {"worker": "ASR", "finalize": True},
            )
        if self._step is FinalizationStep.FINALIZE_ASR:
            return self._advance_worker(
                ProcessSource.ASR,
                FinalizationStep.FINAL_ANALYSIS,
                ProcessSource.DISCUSSION,
                {"worker": "DISCUSSION", "finalize": True},
            )
        if self._step is FinalizationStep.FINAL_ANALYSIS:
            self._acknowledge_expected_stop(ProcessSource.DISCUSSION)
            payload = self._writer_finalize_factory()
            if type(payload) is not WriterFinalize:
                raise TypeError("writer finalize factory must return WriterFinalize")
            command = self._send_one(
                ProcessSource.WRITER,
                MessageType.WRITER_FINALIZE,
                payload,
            )
            self._finalize_sequence = command[1].sequence
            self._step = FinalizationStep.FINALIZE_WRITER
            return (command,)
        if self._step is FinalizationStep.FINALIZE_WRITER:
            self._step = FinalizationStep.COMPLETE
            self._slow_visible = False
            self._shutdown()
        return ()

    def accepts(self, envelope: object) -> bool:
        """Validate an acknowledgement without mutating coordinator state."""

        if type(envelope) is not MessageEnvelope:
            return False
        if (
            type(envelope.schema_version) is not int
            or envelope.schema_version != 1
            or type(envelope.session_id) is not str
            or envelope.session_id != self._session_id
            or type(envelope.sequence) is not int
            or envelope.sequence <= 0
            or type(envelope.source) is not ProcessSource
        ):
            return False
        if self._force_requested:
            if envelope.message_type is MessageType.WRITER_FORCE_CLOSE_RESULT:
                return self._writer_force_close_result(envelope)
            if self._step is FinalizationStep.FINALIZE_WRITER:
                return self._writer_ack(envelope)
            return False
        if self._step is FinalizationStep.DRAIN_AUDIO:
            return self._audio_ack(envelope)
        if self._step is FinalizationStep.FINALIZE_ASR:
            return self._asr_ack(envelope)
        if self._step is FinalizationStep.FINAL_ANALYSIS:
            return self._discussion_ack(envelope)
        if self._step is FinalizationStep.FINALIZE_WRITER:
            return self._writer_ack(envelope)
        return False

    def tick(self, now_ms: int) -> FinalizationTick:
        """Offer the slow choice once at the exact hard threshold."""

        self._observe_time(now_ms)
        if self._force_requested:
            deadline = self._force_deadline_ms
            return FinalizationTick(
                False,
                force_result_timed_out=deadline is not None and now_ms >= deadline,
            )
        if (
            self._started_ms is None
            or self.completed
            or self._incomplete
            or self._slow_visible
            or self._next_slow_offer_ms is None
            or now_ms < self._next_slow_offer_ms
        ):
            return FinalizationTick(False)
        while self._next_slow_offer_ms <= now_ms:
            self._next_slow_offer_ms += SLOW_FINALIZATION_MS
        self._slow_visible = True
        return FinalizationTick(True)

    def keep_waiting(self) -> None:
        """Hide an offered slow prompt without changing the hard start time."""

        self._slow_visible = False

    def force_close(self, now_ms: int) -> WriterForceCloseRequest | None:
        """Request Writer-owned out-of-band force-close linearization once."""

        self._observe_time(now_ms)
        if self._force_requested or self._incomplete:
            return None
        if self._started_ms is None or self.completed:
            raise RuntimeError("force close requires active finalization")
        event = self._force_close_event_factory()
        if type(event) is not EventRecord:
            raise TypeError("force close event factory must return EventRecord")
        request = WriterForceCloseRequest(event)
        immediate = self._request_writer_force_close(
            request,
            FORCE_REQUEST_TIMEOUT_SECONDS,
        )
        self._force_requested = True
        self._force_deadline_ms = now_ms + FORCE_RESULT_TIMEOUT_MS
        self._slow_visible = False
        if immediate is not None:
            self.resolve_force_result(immediate)
        return request

    def resolve_force_result(self, result: WriterForceCloseResult) -> None:
        """Resolve a queue-delivered or shared-state terminal result once."""

        if type(result) is not WriterForceCloseResult:
            raise TypeError("result must be an exact WriterForceCloseResult")
        if self._incomplete or self.completed:
            return
        if not self._force_requested:
            raise RuntimeError("force-close result requires a pending request")
        if result.outcome is WriterForceCloseOutcome.INCOMPLETE:
            self._incomplete = True
        else:
            self._step = FinalizationStep.COMPLETE
        self._force_deadline_ms = None
        self._slow_visible = False
        self._shutdown()

    def resolve_completed_result(self, result: WriterForceCloseResult) -> None:
        """Accept Writer's durable completed result if its queue ACK was lost."""

        if type(result) is not WriterForceCloseResult:
            raise TypeError("result must be an exact WriterForceCloseResult")
        if self._step is not FinalizationStep.FINALIZE_WRITER:
            raise RuntimeError("completed result requires Writer finalization")
        if self._force_requested or self._incomplete:
            raise RuntimeError("completed result conflicts with force close")
        if result.outcome is not WriterForceCloseOutcome.COMPLETED:
            raise RuntimeError("normal finalization requires a completed result")
        self._step = FinalizationStep.COMPLETE
        self._slow_visible = False
        self._shutdown()

    def _advance_worker(
        self,
        worker: ProcessSource,
        next_step: FinalizationStep,
        target: ProcessSource,
        payload: dict[str, object],
    ) -> tuple[EnvelopeCommand, ...]:
        self._acknowledge_expected_stop(worker)
        command = self._send_one(target, MessageType.WORKER_STOP, payload)
        self._step = next_step
        return (command,)

    def _send_one(
        self,
        target: ProcessSource,
        message_type: MessageType,
        payload: object,
    ) -> EnvelopeCommand:
        envelope = self._send(target, message_type, payload)
        if (
            type(envelope) is not MessageEnvelope
            or envelope.session_id != self._session_id
            or envelope.message_type is not message_type
            or envelope.source is not ProcessSource.GUI
            or envelope.payload != payload
            or type(envelope.sequence) is not int
            or envelope.sequence <= 0
        ):
            raise RuntimeError("send returned an invalid command envelope")
        return target, envelope

    def _observe_time(self, now_ms: int) -> None:
        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("now_ms must be a non-negative exact integer")
        if self._last_tick_ms is not None and now_ms < self._last_tick_ms:
            raise RuntimeError("finalization clock rollback")
        self._last_tick_ms = now_ms

    @staticmethod
    def _audio_ack(envelope: MessageEnvelope[object]) -> bool:
        payload = envelope.payload
        return (
            envelope.source is ProcessSource.AUDIO
            and envelope.message_type is MessageType.WORKER_STOPPED
            and type(payload) is dict
            and frozenset(payload)
            == {"worker", "drained", "writer_frames", "asr_frames"}
            and type(payload.get("worker")) is str
            and payload["worker"] == "AUDIO"
            and type(payload.get("drained")) is bool
            and payload["drained"] is True
            and type(payload.get("writer_frames")) is int
            and payload["writer_frames"] >= 0
            and type(payload.get("asr_frames")) is int
            and payload["asr_frames"] >= 0
        )

    @staticmethod
    def _asr_ack(envelope: MessageEnvelope[object]) -> bool:
        payload = envelope.payload
        return (
            envelope.source is ProcessSource.ASR
            and envelope.message_type is MessageType.WORKER_STOPPED
            and type(payload) is dict
            and frozenset(payload) == {"worker", "drained", "committed_count"}
            and type(payload.get("worker")) is str
            and payload["worker"] == "ASR"
            and type(payload.get("drained")) is bool
            and payload["drained"] is True
            and type(payload.get("committed_count")) is int
            and payload["committed_count"] >= 0
        )

    @staticmethod
    def _discussion_ack(envelope: MessageEnvelope[object]) -> bool:
        payload = envelope.payload
        return (
            envelope.source is ProcessSource.DISCUSSION
            and envelope.message_type is MessageType.WORKER_STOPPED
            and type(payload) is DiscussionStoppedPayload
            and type(payload.worker) is str
            and payload.worker == "DISCUSSION"
            and type(payload.drained) is bool
            and payload.drained is True
            and type(payload.final_revision) is int
            and payload.final_revision >= 0
            and type(payload.pending_count) is int
            and payload.pending_count == 0
        )

    def _writer_ack(self, envelope: MessageEnvelope[object]) -> bool:
        payload = envelope.payload
        return (
            envelope.source is ProcessSource.WRITER
            and envelope.message_type is MessageType.WRITER_ACK
            and type(payload) is WriterAck
            and type(payload.acknowledged_sequence) is int
            and payload.acknowledged_sequence == self._finalize_sequence
            and type(payload.latest_successful_save_at) is datetime
            and payload.latest_successful_save_at.utcoffset() is not None
        )

    @staticmethod
    def _writer_force_close_result(envelope: MessageEnvelope[object]) -> bool:
        payload = envelope.payload
        return (
            envelope.source is ProcessSource.WRITER
            and envelope.message_type is MessageType.WRITER_FORCE_CLOSE_RESULT
            and type(payload) is WriterForceCloseResult
            and type(payload.outcome) is WriterForceCloseOutcome
            and type(payload.latest_successful_save_at) is datetime
            and payload.latest_successful_save_at.utcoffset() is not None
        )
