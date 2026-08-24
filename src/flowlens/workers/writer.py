"""Single-owner Writer Worker queue dispatch and fatal reporting."""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from multiprocessing import parent_process
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from queue import Empty

from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    DiscussionStateReplaced,
    MessageEnvelope,
    MessageSequenceError,
    TranscriptCommitted,
    WriterAck,
    WriterAppendEvent,
    WriterFatal,
    WriterFinalize,
    WriterFlush,
    WriterForceCloseOutcome,
    WriterForceCloseResult,
    WriterOpenSession,
    WriterShutdown,
)
from flowlens.persistence.session_writer import PreparedFinalization, SessionWriter
from flowlens.workers.finalization_gate import (
    WriterFinalizationGate,
    WriterTerminalClaim,
)

_AUDIO_BATCH_LIMIT = 64
_QUEUE_TIMEOUT_SECONDS = 0.1
_TRANSCRIPT_AUDIO_TIMEOUT_SECONDS = 5.0
_UNKNOWN_SESSION_ID = "00000000000000000000000000"
_CONTROL_PAYLOAD_TYPES: dict[MessageType, type[object]] = {
    MessageType.WRITER_OPEN_SESSION: WriterOpenSession,
    MessageType.TRANSCRIPT_COMMITTED: TranscriptCommitted,
    MessageType.DISCUSSION_STATE_REPLACED: DiscussionStateReplaced,
    MessageType.EVENT_APPENDED: WriterAppendEvent,
    MessageType.WRITER_FLUSH: WriterFlush,
    MessageType.WRITER_FINALIZE: WriterFinalize,
    MessageType.WRITER_SHUTDOWN: WriterShutdown,
}


class WriterWorkerProtocolError(ValueError):
    """Raised when a queue item does not target the Writer contract."""


def _open_session(command: WriterOpenSession) -> SessionWriter:
    """Open the production persistence owner behind an injectable seam."""

    return SessionWriter.open(
        command.session_dir,
        command.manifest,
        command.initial_state,
    )


def _monotonic() -> float:
    """Return the process monotonic clock behind an injectable seam."""

    return time.monotonic()


def _wall_clock() -> datetime:
    """Return an aware wall-clock timestamp behind an injectable seam."""

    return datetime.now(UTC)


def _parent_is_dead() -> bool:
    """Return whether a spawned worker has lost its parent process."""

    parent = parent_process()
    return parent is not None and not parent.is_alive()


def _validate_controller_source(envelope: MessageEnvelope[object]) -> None:
    """Require a controller-rewrapped envelope for the Writer queue."""

    if envelope.source is not ProcessSource.GUI:
        raise WriterWorkerProtocolError("Writer control source must be GUI")


class _WriterWorker:
    """Stateful bounded-fair dispatcher for one Writer process."""

    def __init__(
        self,
        control_queue: Queue[object],
        audio_queue: Queue[object],
        response_queue: Queue[object],
        stop_event: Event,
        finalization_gate: WriterFinalizationGate | None,
        *,
        open_session: Callable[[WriterOpenSession], SessionWriter],
    ) -> None:
        self._control_queue = control_queue
        self._audio_queue = audio_queue
        self._response_queue = response_queue
        self._stop_event = stop_event
        self._finalization_gate = finalization_gate
        self._open_session = open_session
        self._expected_control_sequence = 1
        self._writer: SessionWriter | None = None
        self._session_id = _UNKNOWN_SESSION_ID
        self._response_sequence = 0
        self._failed_sequence = 0
        self._latest_successful_save_at: datetime | None = None
        self._pending_control: object | None = None
        self._pending_transcript_deadline: float | None = None
        self._persisted_audio_cursors = {
            AudioSource.ME: 0,
            AudioSource.OTHERS: 0,
        }
        self._audio_drain_fence_seen = False
        self._cleanup_attempted = False
        self._finalized = False
        self._exiting = False
        self._force_close_processed = False

    def run(self) -> None:
        """Run until shutdown, lifecycle stop, or a fail-closed exception."""

        try:
            self._wait_for_open()
            if self._exiting:
                return
            self._dispatch_open_session()
        except BaseException as error:
            self._report_fatal(error)
            raise

    def _wait_for_open(self) -> None:
        while self._writer is None:
            if self._lifecycle_exit_requested():
                self._close_incomplete_once()
                self._exiting = True
                return
            try:
                item = self._control_queue.get(timeout=_QUEUE_TIMEOUT_SECONDS)
            except Empty:
                continue
            self._handle_control(item, opening=True)

    def _dispatch_open_session(self) -> None:
        while not self._exiting:
            if self._lifecycle_exit_requested():
                self._close_incomplete_once()
                return
            if self._process_force_close_if_requested():
                return

            self._raise_if_pending_transcript_timed_out()
            wait_for_audio = self._should_wait_for_audio()
            audio_processed = self._drain_audio_batch(wait_for_first=wait_for_audio)
            if self._lifecycle_exit_requested():
                self._close_incomplete_once()
                return

            if self._pending_control is not None:
                if self._pending_transcript_waits_for_audio():
                    self._raise_if_pending_transcript_timed_out()
                else:
                    item = self._pending_control
                    self._pending_control = None
                    self._pending_transcript_deadline = None
                    self._process_or_defer_control(item)
            else:
                try:
                    item = self._control_queue.get_nowait()
                except Empty:
                    if audio_processed == 0:
                        self._wait_for_control()
                else:
                    self._process_or_defer_control(item)

            self._failed_sequence = 0
            self._sync_if_due()

    def _wait_for_control(self) -> None:
        try:
            self._pending_control = self._control_queue.get(
                timeout=_QUEUE_TIMEOUT_SECONDS
            )
        except Empty:
            self._pending_control = None

    def _process_or_defer_control(
        self,
        item: object,
    ) -> None:
        if self._is_terminal_control(item) and not self._audio_drain_fence_seen:
            envelope = self._validate_control_metadata(item, opening=False)
            if envelope.sequence < self._expected_control_sequence:
                self._emit_ack(envelope.sequence, mutation_succeeded=False)
                self._failed_sequence = 0
                return
            if envelope.sequence > self._expected_control_sequence:
                self._raise_control_gap(envelope.sequence)
            self._pending_control = envelope
            return
        if self._is_transcript_control(item):
            envelope = self._validate_control_metadata(item, opening=False)
            if envelope.sequence < self._expected_control_sequence:
                self._emit_ack(envelope.sequence, mutation_succeeded=False)
                self._failed_sequence = 0
                return
            if envelope.sequence > self._expected_control_sequence:
                self._raise_control_gap(envelope.sequence)
            if self._transcript_requires_audio(envelope):
                if self._audio_drain_fence_seen:
                    self._handle_control(envelope, opening=False)
                    return
                self._pending_control = envelope
                self._pending_transcript_deadline = (
                    _monotonic() + _TRANSCRIPT_AUDIO_TIMEOUT_SECONDS
                )
                return
        self._handle_control(item, opening=False)

    @staticmethod
    def _is_terminal_control(item: object) -> bool:
        return isinstance(item, MessageEnvelope) and item.message_type in {
            MessageType.WRITER_FINALIZE,
            MessageType.WRITER_SHUTDOWN,
        }

    @staticmethod
    def _is_transcript_control(item: object) -> bool:
        return (
            isinstance(item, MessageEnvelope)
            and item.message_type is MessageType.TRANSCRIPT_COMMITTED
        )

    def _should_wait_for_audio(self) -> bool:
        return self._pending_transcript_deadline is not None or (
            self._pending_control is not None
            and self._is_terminal_control(self._pending_control)
            and not self._audio_drain_fence_seen
        )

    def _pending_transcript_waits_for_audio(self) -> bool:
        if self._pending_transcript_deadline is None:
            return False
        envelope = self._pending_control
        if not isinstance(envelope, MessageEnvelope):
            raise WriterWorkerProtocolError(
                "pending transcript control must be a MessageEnvelope"
            )
        return not self._audio_drain_fence_seen and self._transcript_requires_audio(
            envelope
        )

    def _transcript_requires_audio(
        self,
        envelope: MessageEnvelope[object],
    ) -> bool:
        payload = envelope.payload
        if not isinstance(payload, TranscriptCommitted):
            raise WriterWorkerProtocolError(
                "transcript control payload must be TranscriptCommitted"
            )
        record = payload.record
        return record.source_end_sample > self._persisted_audio_cursors[record.source]

    def _raise_if_pending_transcript_timed_out(self) -> None:
        deadline = self._pending_transcript_deadline
        if deadline is None or _monotonic() < deadline:
            return
        envelope = self._pending_control
        if isinstance(envelope, MessageEnvelope):
            self._failed_sequence = envelope.sequence
        raise WriterWorkerProtocolError(
            "timed out waiting for transcript audio persistence"
        )

    def _drain_audio_batch(self, *, wait_for_first: bool = False) -> int:
        writer = self._require_open_writer()
        processed = 0
        first_read = True
        while processed < _AUDIO_BATCH_LIMIT:
            try:
                if wait_for_first and first_read:
                    item = self._audio_queue.get(timeout=_QUEUE_TIMEOUT_SECONDS)
                else:
                    item = self._audio_queue.get_nowait()
            except Empty:
                break
            first_read = False
            if isinstance(item, AudioDrainFence):
                if self._audio_drain_fence_seen:
                    raise WriterWorkerProtocolError("duplicate audio drain fence")
                self._audio_drain_fence_seen = True
                continue
            if isinstance(item, AudioWriteCommand):
                if self._audio_drain_fence_seen:
                    raise WriterWorkerProtocolError(
                        "audio command received after audio drain fence"
                    )
                writer.append_audio(item)
                self._persisted_audio_cursors[item.source] = item.source_end_sample
                processed += 1
                if (
                    self._pending_transcript_deadline is not None
                    and not self._pending_transcript_waits_for_audio()
                ):
                    break
                continue
            if self._audio_drain_fence_seen:
                raise WriterWorkerProtocolError(
                    "audio queue item received after audio drain fence"
                )
            else:
                raise WriterWorkerProtocolError(
                    "audio queue item must be an AudioWriteCommand or AudioDrainFence"
                )
        return processed

    def _handle_control(self, item: object, *, opening: bool) -> None:
        envelope = self._validate_control_metadata(item, opening=opening)
        if envelope.sequence < self._expected_control_sequence:
            self._emit_ack(envelope.sequence, mutation_succeeded=False)
            self._failed_sequence = 0
            return
        if envelope.sequence > self._expected_control_sequence:
            self._raise_control_gap(envelope.sequence)
        self._expected_control_sequence += 1

        if opening:
            self._open(envelope)
        else:
            self._mutate(envelope)
        if not self._force_close_processed:
            self._emit_ack(envelope.sequence)
        self._failed_sequence = 0

    def _validate_control_metadata(
        self,
        item: object,
        *,
        opening: bool,
    ) -> MessageEnvelope[object]:
        if not isinstance(item, MessageEnvelope):
            self._failed_sequence = 0
            raise WriterWorkerProtocolError(
                "control queue item must be a MessageEnvelope"
            )

        self._failed_sequence = item.sequence
        if self._writer is None:
            self._session_id = item.session_id
        item.validate_schema()
        self._validate_control_target(item, opening=opening)
        return item

    def _raise_control_gap(self, actual_sequence: int) -> None:
        raise MessageSequenceError(
            f"expected control sequence {self._expected_control_sequence}, "
            f"got {actual_sequence}"
        )

    def _validate_control_target(
        self,
        envelope: MessageEnvelope[object],
        *,
        opening: bool,
    ) -> None:
        expected_payload_type = _CONTROL_PAYLOAD_TYPES.get(envelope.message_type)
        if expected_payload_type is None:
            raise WriterWorkerProtocolError(
                "control envelope does not target a Writer mutation"
            )
        if not isinstance(envelope.payload, expected_payload_type):
            raise WriterWorkerProtocolError(
                "control envelope payload does not match its message type"
            )
        _validate_controller_source(envelope)

        if opening:
            if envelope.message_type is not MessageType.WRITER_OPEN_SESSION:
                raise WriterWorkerProtocolError(
                    "the first control must open the Writer session"
                )
            if envelope.sequence != 1:
                raise MessageSequenceError("the first control sequence must be 1")
            payload = envelope.payload
            if not isinstance(payload, WriterOpenSession):
                raise WriterWorkerProtocolError(
                    "the first control payload must be WriterOpenSession"
                )
            if payload.manifest.session_id != envelope.session_id:
                raise WriterWorkerProtocolError(
                    "open control session does not match its manifest"
                )
            return

        if envelope.session_id != self._session_id:
            raise WriterWorkerProtocolError(
                "control envelope does not target the open session"
            )

    def _open(self, envelope: MessageEnvelope[object]) -> None:
        payload = envelope.payload
        if not isinstance(payload, WriterOpenSession):
            raise WriterWorkerProtocolError("open payload must be WriterOpenSession")
        self._writer = self._open_session(payload)
        self._session_id = envelope.session_id

    def _mutate(self, envelope: MessageEnvelope[object]) -> None:
        writer = self._require_open_writer()
        payload = envelope.payload

        if self._finalized and envelope.message_type is not MessageType.WRITER_SHUTDOWN:
            raise WriterWorkerProtocolError(
                "only Writer shutdown is valid after finalization"
            )
        if isinstance(payload, TranscriptCommitted):
            writer.append_transcript(payload.record)
        elif isinstance(payload, DiscussionStateReplaced):
            writer.replace_discussion_state(payload.previous_revision, payload.state)
        elif isinstance(payload, WriterAppendEvent):
            writer.append_event(payload.record)
        elif isinstance(payload, WriterFlush):
            writer.force_sync()
        elif isinstance(payload, WriterFinalize):
            gate = self._finalization_gate
            if gate is None:
                writer.finalize(payload)
                self._finalized = True
            else:
                prepared = writer.prepare_finalize(payload)
                claim = gate.claim_terminal(payload.completion_event)
                self._commit_terminal_claim(claim, prepared=prepared)
        elif isinstance(payload, WriterShutdown):
            self._close_incomplete_once()
            self._exiting = True
        elif isinstance(payload, WriterOpenSession):
            raise WriterWorkerProtocolError("Writer session is already open")
        else:
            raise WriterWorkerProtocolError("unsupported Writer mutation payload")

    def _sync_if_due(self) -> None:
        if self._writer is None or self._finalized or self._cleanup_attempted:
            return
        self._writer.sync_if_due(_monotonic())

    def _emit_ack(
        self,
        acknowledged_sequence: int,
        *,
        mutation_succeeded: bool = True,
    ) -> None:
        latest_save = self._latest_successful_save_at
        if mutation_succeeded:
            latest_save = _wall_clock()
            self._latest_successful_save_at = latest_save
        elif latest_save is None:
            raise WriterWorkerProtocolError(
                "duplicate control has no prior successful save"
            )
        response = self._response_envelope(
            MessageType.WRITER_ACK,
            WriterAck(acknowledged_sequence, latest_save),
        )
        self._response_queue.put(response)

    def _response_envelope(
        self,
        message_type: MessageType,
        payload: WriterAck | WriterFatal | WriterForceCloseResult,
    ) -> MessageEnvelope[WriterAck | WriterFatal | WriterForceCloseResult]:
        self._response_sequence += 1
        return MessageEnvelope(
            schema_version=1,
            session_id=self._session_id,
            message_type=message_type,
            sequence=self._response_sequence,
            source=ProcessSource.WRITER,
            created_monotonic_ms=int(_monotonic() * 1_000),
            payload=payload,
        )

    def _process_force_close_if_requested(self) -> bool:
        gate = self._finalization_gate
        if gate is None:
            return False
        claim = gate.claim_force_if_requested()
        if claim is None:
            return False
        self._commit_terminal_claim(claim)
        return True

    def _commit_terminal_claim(
        self,
        claim: WriterTerminalClaim,
        *,
        prepared: PreparedFinalization | None = None,
    ) -> None:
        gate = self._finalization_gate
        if gate is None:
            raise RuntimeError("terminal claim requires a finalization gate")
        writer = self._require_open_writer()
        if claim.outcome is WriterForceCloseOutcome.INCOMPLETE:
            writer.commit_force_close(claim.event)
            self._cleanup_attempted = True
            self._exiting = True
            self._force_close_processed = True
        else:
            if type(prepared) is not PreparedFinalization:
                raise RuntimeError("completion claim requires prepared finalization")
            writer.commit_finalize(prepared)
            self._finalized = True
        latest_save = _wall_clock()
        self._latest_successful_save_at = latest_save
        result = gate.publish_result(claim.outcome, latest_save)
        if claim.force_was_requested:
            self._response_queue.put(
                self._response_envelope(
                    MessageType.WRITER_FORCE_CLOSE_RESULT,
                    result,
                )
            )

    def _report_fatal(self, error: BaseException) -> None:
        self._close_incomplete_once(error)
        try:
            error_type = type(error).__name__.strip() or "Exception"
            message = str(error).strip() or error_type or "Writer worker failed"
            fatal = WriterFatal(
                failed_sequence=self._failed_sequence,
                error_type=error_type,
                message=message,
            )
            self._response_queue.put(
                self._response_envelope(MessageType.WRITER_FATAL, fatal)
            )
        except BaseException as delivery_error:
            error.add_note(
                "Writer fatal response delivery failed: "
                f"{type(delivery_error).__name__}: {delivery_error}"
            )
        finally:
            self._close_response_queue(error)

    def _close_incomplete_once(
        self,
        primary_error: BaseException | None = None,
    ) -> None:
        if self._writer is None or self._cleanup_attempted:
            return
        self._cleanup_attempted = True
        try:
            self._writer.close_incomplete()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "Writer incomplete cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    def _close_response_queue(self, primary_error: BaseException) -> None:
        for operation_name, operation in (
            ("close", self._response_queue.close),
            ("join_thread", self._response_queue.join_thread),
        ):
            try:
                operation()
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"Writer response queue {operation_name} failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    def _lifecycle_exit_requested(self) -> bool:
        return self._stop_event.is_set() or _parent_is_dead()

    def _require_open_writer(self) -> SessionWriter:
        if self._writer is None:
            raise WriterWorkerProtocolError("Writer session is not open")
        return self._writer


def run_writer_worker(
    control_queue: Queue[object],
    audio_queue: Queue[object],
    response_queue: Queue[object],
    stop_event: Event,
    finalization_gate: WriterFinalizationGate | None = None,
) -> None:
    """Run the spawn-safe Writer process entry point."""

    worker = _WriterWorker(
        control_queue,
        audio_queue,
        response_queue,
        stop_event,
        finalization_gate,
        open_session=_open_session,
    )
    worker.run()
