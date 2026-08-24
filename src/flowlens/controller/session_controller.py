"""Pure transactional session lifecycle and IPC coordination."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast

from flowlens.asr.types import AsrWorkerConfig, PartialTranscript
from flowlens.audio.types import AudioWorkerConfig
from flowlens.controller.finalization import FinalizationCoordinator
from flowlens.controller.models import PreflightReport, PreflightSelection
from flowlens.controller.ports import AccessibilityAnnouncer, Clock, WorkerRuntime
from flowlens.controller.routing import (
    SequenceTracker,
    rewrap_for_gui,
    validate_worker_payload,
)
from flowlens.controller.supervision import (
    RecoveryAction,
    RecoveryDecision,
    WorkerSupervisor,
)
from flowlens.discussion.contracts import DiscussionStatusPayload
from flowlens.discussion.worker import DiscussionWorkerConfig
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import (
    AudioSource,
    EventType,
    MessageType,
    ProcessSource,
    SessionStatus,
)
from flowlens.domain.messages import (
    DiscussionStateReplaced,
    EventRecord,
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    WriterAck,
    WriterAppendEvent,
    WriterFatal,
    WriterFinalize,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterForceCloseResult,
    WriterOpenSession,
)
from flowlens.domain.session import PauseInterval, SessionManifest

READINESS_TIMEOUT_MS = 60_000
_START_WORKERS = (
    ProcessSource.AUDIO,
    ProcessSource.ASR,
    ProcessSource.DISCUSSION,
)


class SessionState(str, Enum):
    """The exact eight controller lifecycle states."""

    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    STARTING = "STARTING"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class _AnalysisPauseReason(str, Enum):
    """Independent reasons that keep Discussion generation paused."""

    SESSION = "SESSION"
    ASR_LAG = "ASR_LAG"
    DISABLED = "DISABLED"


class InvalidTransition(RuntimeError):
    """Raised before any side effect for an illegal lifecycle call."""


class StartBlocked(RuntimeError):
    """Raised when exact preflight readiness does not permit start."""


class _Preflight(Protocol):
    def evaluate(self, selection: PreflightSelection) -> PreflightReport: ...


LaunchFactory = Callable[[PreflightReport, datetime, int], "SessionLaunch"]


@dataclass(frozen=True, slots=True)
class SessionLaunch:
    """Complete immutable spawn-picklable five-process launch value."""

    session_id: str
    session_dir: Path
    manifest: SessionManifest
    initial_state: DiscussionState
    audio_config: AudioWorkerConfig
    asr_config: AsrWorkerConfig
    discussion_config: DiscussionWorkerConfig

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise ValueError("session_id must be a non-empty exact string")
        if (
            not isinstance(self.session_dir, Path)
            or not self.session_dir.is_absolute()
            or self.session_dir.resolve(strict=False) != self.session_dir
        ):
            raise ValueError("session_dir must be an absolute normalized Path")
        if not isinstance(self.manifest, SessionManifest):
            raise TypeError("manifest must be a SessionManifest")
        if self.manifest.status is not SessionStatus.INCOMPLETE:
            raise ValueError("launch manifest must be incomplete")
        if not isinstance(self.initial_state, DiscussionState):
            raise TypeError("initial_state must be a DiscussionState")
        if not isinstance(self.audio_config, AudioWorkerConfig):
            raise TypeError("audio_config must be an AudioWorkerConfig")
        if not isinstance(self.asr_config, AsrWorkerConfig):
            raise TypeError("asr_config must be an AsrWorkerConfig")
        if not isinstance(self.discussion_config, DiscussionWorkerConfig):
            raise TypeError("discussion_config must be a DiscussionWorkerConfig")
        session_ids = {
            self.manifest.session_id,
            self.audio_config.session_id,
            self.asr_config.session_id,
            self.discussion_config.session_id,
        }
        if session_ids != {self.session_id}:
            raise ValueError("all launch values must target one session")
        if self.discussion_config.initial_state != self.initial_state:
            raise ValueError("discussion initial state must match the launch")
        if self.manifest.mode is not self.initial_state.mode:
            raise ValueError("manifest and initial state modes must match")


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    """Immutable UI-facing controller value updated synchronously."""

    state: SessionState
    preflight: PreflightReport | None
    issue: str | None
    recording_status: str
    transcript: tuple[TranscriptRecord, ...]
    partials: tuple[PartialTranscript, ...]
    discussion_state: DiscussionState | None
    microphone_level: float
    loopback_level: float
    asr_status: str
    asr_backlog_ms: int
    maximum_asr_backlog_ms: int
    analysis_status: str
    latest_successful_save_at: datetime | None
    fatal_error: str | None
    stop_confirmation_visible: bool
    slow_finalization_visible: bool


class SessionController:
    """Coordinate one local session without owning processes or persistence."""

    def __init__(
        self,
        *,
        preflight: _Preflight,
        runtime: WorkerRuntime,
        clock: Clock,
        announcer: AccessibilityAnnouncer,
        launch_factory: LaunchFactory,
    ) -> None:
        self._preflight_service = preflight
        self._runtime = runtime
        self._clock = clock
        self._announcer = announcer
        self._launch_factory = launch_factory
        self._state = SessionState.IDLE
        self._snapshot = ControllerSnapshot(
            state=self._state,
            preflight=None,
            issue=None,
            recording_status="Idle",
            transcript=(),
            partials=(),
            discussion_state=None,
            microphone_level=0.0,
            loopback_level=0.0,
            asr_status="Idle",
            asr_backlog_ms=0,
            maximum_asr_backlog_ms=0,
            analysis_status="Idle",
            latest_successful_save_at=None,
            fatal_error=None,
            stop_confirmation_visible=False,
            slow_finalization_visible=False,
        )
        self._launch: SessionLaunch | None = None
        self._sequence_tracker: SequenceTracker | None = None
        self._supervisor: WorkerSupervisor | None = None
        self._outgoing_sequences: dict[ProcessSource, int] = {}
        self._writer_open_sequence: int | None = None
        self._started_ms: int | None = None
        self._last_clock_ms: int | None = None
        self._workers_started = False
        self._ready: set[ProcessSource] = set()
        self._early_ready: set[ProcessSource] = set()
        self._event_sequence = 0
        self._terminal_event_consumed = False
        self._analysis_paused_for_lag = False
        self._analysis_disabled = False
        self._announced: set[tuple[str, str]] = set()
        self._protocol_event_in_progress = False
        self._runtime_shutdown = False
        self._drained_workers: set[ProcessSource] = set()
        self._source_connected = {source: True for source in AudioSource}
        self._source_transition_generation = {source: 0 for source in AudioSource}
        self._finalization: FinalizationCoordinator | None = None
        self._pause_started_ms: int | None = None
        self._pause_intervals: list[PauseInterval] = []
        self._stop_confirmed_ms: int | None = None
        self._stop_confirmed_at: datetime | None = None

    @property
    def state(self) -> SessionState:
        """Return the exact lifecycle state."""

        return self._state

    def snapshot(self) -> ControllerSnapshot:
        """Return the latest immutable UI snapshot."""

        return self._snapshot

    def enter_preflight(self) -> None:
        """Enter preflight from an idle or terminal state."""

        self._require_state(
            "PREFLIGHT",
            {SessionState.IDLE, SessionState.COMPLETED, SessionState.ERROR},
        )
        self._set_state(SessionState.PREFLIGHT, recording_status="Ready")

    def refresh_preflight(self, selection: PreflightSelection) -> PreflightReport:
        """Evaluate current selections while remaining in preflight."""

        self._require_state("PREFLIGHT", {SessionState.PREFLIGHT})
        if not isinstance(selection, PreflightSelection):
            raise TypeError("selection must be a PreflightSelection")
        report = self._preflight_service.evaluate(selection)
        if not isinstance(report, PreflightReport):
            raise TypeError("preflight must return a PreflightReport")
        self._snapshot = replace(
            self._snapshot,
            preflight=report,
            issue=report.issues[0].message if report.issues else None,
            microphone_level=report.mic_level,
            loopback_level=report.loopback_level,
        )
        return report

    def start(self, selection: PreflightSelection) -> None:
        """Build one complete launch and open Writer before worker start."""

        self._require_state("STARTING", {SessionState.PREFLIGHT})
        checked = self._preflight_service.evaluate(selection)
        if not isinstance(checked, PreflightReport):
            raise TypeError("preflight must return a PreflightReport")
        if not checked.can_start:
            self._snapshot = replace(
                self._snapshot,
                preflight=checked,
                issue=checked.issues[0].message if checked.issues else "Start blocked.",
            )
            raise StartBlocked(self._snapshot.issue)
        started_ms = self._read_clock()
        now = self._clock.now()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("clock now must be timezone-aware")
        launch = self._launch_factory(checked, now, started_ms)
        if not isinstance(launch, SessionLaunch):
            raise TypeError("launch_factory must return a SessionLaunch")

        self._outgoing_sequences.clear()
        self._announced.clear()
        self._runtime_shutdown = False
        try:
            self._runtime.start_all(launch)
            open_payload = WriterOpenSession(
                launch.session_dir,
                launch.manifest,
                launch.initial_state,
            )
            open_envelope = self._send(
                ProcessSource.WRITER,
                MessageType.WRITER_OPEN_SESSION,
                open_payload,
                launch=launch,
            )
        except Exception:
            self._bounded_shutdown()
            self._snapshot = replace(
                self._snapshot,
                preflight=checked,
                issue="Worker runtime could not start safely.",
            )
            return

        self._launch = launch
        self._sequence_tracker = SequenceTracker(launch.session_id)
        self._supervisor = WorkerSupervisor(launch.session_id)
        self._writer_open_sequence = open_envelope.sequence
        self._started_ms = started_ms
        self._workers_started = False
        self._ready = set()
        self._early_ready = set()
        self._event_sequence = 0
        self._terminal_event_consumed = False
        self._analysis_paused_for_lag = False
        self._analysis_disabled = False
        self._drained_workers.clear()
        self._source_connected = {source: True for source in AudioSource}
        self._source_transition_generation = {source: 0 for source in AudioSource}
        self._finalization = None
        self._pause_started_ms = None
        self._pause_intervals = []
        self._stop_confirmed_ms = None
        self._stop_confirmed_at = None
        self._set_state(
            SessionState.STARTING,
            preflight=checked,
            issue=None,
            recording_status="Starting",
            discussion_state=launch.initial_state,
            transcript=(),
            partials=(),
            asr_status="Starting",
            analysis_status="Starting",
            fatal_error=None,
        )

    def pause(self) -> None:
        """Synchronously show paused and stop all live producers."""

        self._require_state("PAUSED", {SessionState.RECORDING})
        paused_ms = self._read_clock()
        before = self._analysis_pause_reasons()
        after = self._analysis_pause_reasons(state=SessionState.PAUSED)
        targets = [ProcessSource.AUDIO, ProcessSource.ASR]
        if not before and after:
            targets.append(ProcessSource.DISCUSSION)
        if not self._broadcast(MessageType.WORKER_PAUSE, tuple(targets)):
            return
        self._set_state(
            SessionState.PAUSED,
            recording_status="Paused",
            analysis_status=self._analysis_status(after),
        )
        self._pause_started_ms = paused_ms
        self._persist_operational(EventType.PAUSE_START, ProcessSource.GUI, {})

    def resume(self) -> None:
        """Synchronously show recording and resume all live producers."""

        self._require_state("RECORDING", {SessionState.PAUSED})
        resumed_ms = self._read_clock()
        before = self._analysis_pause_reasons()
        after = self._analysis_pause_reasons(state=SessionState.RECORDING)
        targets = [ProcessSource.AUDIO, ProcessSource.ASR]
        if before and not after:
            targets.append(ProcessSource.DISCUSSION)
        if not self._broadcast(MessageType.WORKER_RESUME, tuple(targets)):
            return
        self._set_state(
            SessionState.RECORDING,
            recording_status="Recording",
            analysis_status=self._analysis_status(after),
        )
        self._close_pause_interval(resumed_ms)
        self._persist_operational(EventType.PAUSE_END, ProcessSource.GUI, {})

    def request_stop(self) -> None:
        """Expose Task 7's confirmation seam without stopping capture."""

        self._require_state(
            "STOPPING",
            {SessionState.RECORDING, SessionState.PAUSED},
        )
        self._snapshot = replace(self._snapshot, stop_confirmation_visible=True)

    def cancel_stop(self) -> None:
        """Hide the stop confirmation without changing capture."""

        self._require_state(
            self._state.value,
            {SessionState.RECORDING, SessionState.PAUSED},
        )
        if not self._snapshot.stop_confirmation_visible:
            raise InvalidTransition(f"{self._state.value} -> CANCEL_STOP")
        self._snapshot = replace(self._snapshot, stop_confirmation_visible=False)

    def confirm_stop(self) -> None:
        """Enter the Task 7 finalization seam."""

        if self._state is SessionState.STOPPING and self._finalization is not None:
            return
        self._require_state(
            "STOPPING",
            {SessionState.RECORDING, SessionState.PAUSED},
        )
        if not self._snapshot.stop_confirmation_visible:
            raise InvalidTransition(f"{self._state.value} -> STOPPING")
        if self._finalization is not None:
            return
        try:
            now_ms = self._read_clock()
            now = self._clock.now()
            if not isinstance(now, datetime) or now.utcoffset() is None:
                raise ValueError("clock now must be timezone-aware")
            coordinator = FinalizationCoordinator(
                session_id=self._require_launch().session_id,
                send=self._send,
                writer_finalize_factory=self._build_writer_finalize,
                force_close_event_factory=self._build_force_close_event,
                acknowledge_expected_stop=self.acknowledge_expected_stop,
                request_writer_force_close=self._request_writer_force_close,
                shutdown=self._bounded_shutdown,
            )
            coordinator.begin(now_ms)
        except Exception:
            self._fatal_runtime("Worker runtime became unavailable.")
            return
        self._finalization = coordinator
        self._stop_confirmed_ms = now_ms
        self._stop_confirmed_at = now
        self._close_pause_interval(now_ms)
        self._set_state(
            SessionState.STOPPING,
            recording_status="Finalizing",
            stop_confirmation_visible=False,
        )

    def keep_waiting(self) -> None:
        """Hide Task 7's slow-finalization message."""

        self._require_state("STOPPING", {SessionState.STOPPING})
        if self._finalization is None:
            raise InvalidTransition("STOPPING -> KEEP_WAITING")
        self._finalization.keep_waiting()
        self._snapshot = replace(self._snapshot, slow_finalization_visible=False)

    def acknowledge_expected_stop(self, worker: ProcessSource) -> None:
        """Mask health loss after Task 7 accepts the current drain-step ACK."""

        self._require_state("ACKNOWLEDGE_STOP", {SessionState.STOPPING})
        if not isinstance(worker, ProcessSource) or worker not in _START_WORKERS:
            raise ValueError("worker must be a stoppable process source")
        self._drained_workers.add(worker)

    def force_close(self) -> None:
        """Provide Task 7's explicit incomplete-shutdown seam."""

        self._require_state("ERROR", {SessionState.STOPPING})
        if self._finalization is None:
            raise InvalidTransition("STOPPING -> FORCE_CLOSE")
        try:
            self._finalization.force_close(self._read_clock())
        except Exception:
            self._fatal_runtime("Worker runtime became unavailable.")
            return
        finalization_snapshot = self._finalization.snapshot()
        if finalization_snapshot.incomplete or finalization_snapshot.completed:
            result = self._runtime.writer_force_close_result()
            if result is None:
                self._fatal_runtime("Writer force-close outcome could not be resolved.")
                return
            self._apply_force_close_result(result)
            return
        self._snapshot = replace(
            self._snapshot,
            recording_status="Resolving force close",
            slow_finalization_visible=False,
        )

    def persist_event(self, record: EventRecord) -> None:
        """Persist one already validated operational record in Writer order."""

        if not isinstance(record, EventRecord):
            raise TypeError("record must be an EventRecord")
        launch = self._require_launch()
        if record.session_id != launch.session_id:
            raise ValueError("event does not target the active session")
        self._event_sequence = max(self._event_sequence, record.sequence)
        self._send(
            ProcessSource.WRITER,
            MessageType.EVENT_APPENDED,
            WriterAppendEvent(record),
        )

    def handle_message(self, envelope: MessageEnvelope[object]) -> None:
        """Validate one worker payload before any state or sequence mutation."""

        if self._launch is None or self._sequence_tracker is None:
            return
        if self._is_finalization_ack(envelope):
            finalization = self._finalization
            tracker = self._sequence_tracker
            if (
                finalization is None
                or not finalization.accepts(envelope)
                or envelope.sequence != tracker.expected(envelope.source)
            ):
                return
        try:
            payload = validate_worker_payload(envelope)
        except Exception:
            self._protocol_event(envelope, "invalid_payload_or_schema")
            return
        if not self._contextual_payload_is_valid(envelope, payload):
            self._protocol_event_metadata(envelope.source, "inconsistent_status")
            return
        try:
            result = self._sequence_tracker.accept(envelope)
        except Exception:
            self._protocol_event(envelope, "invalid_sequence_or_session")
            return
        if result.duplicate:
            return
        if result.gap is not None:
            if not self._persist_operational(
                EventType.WORKER_EXITED,
                envelope.source,
                {
                    "kind": "message_gap",
                    "missing_start": result.gap[0],
                    "missing_end": result.gap[1],
                },
            ):
                return
        try:
            self._route(envelope, payload)
        except Exception:
            self._fatal_runtime("Worker runtime became unavailable.")

    def handle_worker_exit(self, worker: ProcessSource) -> None:
        """Apply one supervisor decision and announce it once."""

        if self._state is SessionState.STOPPING:
            if worker in self._drained_workers:
                return
            self._fatal_runtime("Worker exited during finalization.")
            return
        supervisor = self._require_supervisor()
        self._apply_recovery(supervisor.on_exit(worker))

    def _contextual_payload_is_valid(
        self,
        envelope: MessageEnvelope[object],
        payload: object,
    ) -> bool:
        if envelope.message_type is not MessageType.ASR_STATUS:
            return True
        if not isinstance(payload, dict):
            return False
        state = cast(str, payload["state"])
        backlog = cast(int, payload["backlog_ms"])
        maximum = cast(int, payload["maximum_backlog_ms"])
        analysis_paused = cast(bool, payload["analysis_paused"])
        if maximum < self._snapshot.maximum_asr_backlog_ms:
            return False
        if state == "READY":
            return (
                self._state is SessionState.STARTING
                and backlog == 0
                and not analysis_paused
            )
        if state == "STOPPED":
            return (
                self._state is SessionState.STOPPING
                and backlog == 0
                and not analysis_paused
            )
        if backlog > 2_000:
            expected_state = "DELAYED"
        elif backlog < 2_000:
            expected_state = "RUNNING"
        else:
            expected_state = (
                "DELAYED" if self._snapshot.asr_status == "Delayed" else "RUNNING"
            )
        if backlog > 5_000:
            expected_paused = True
        elif backlog < 2_000:
            expected_paused = False
        else:
            expected_paused = self._analysis_paused_for_lag
        return state == expected_state and analysis_paused is expected_paused

    def tick(self) -> None:
        """Poll readiness, control messages, and health with bounded failures."""

        if self._state not in {
            SessionState.STARTING,
            SessionState.RECORDING,
            SessionState.PAUSED,
            SessionState.STOPPING,
        }:
            return
        if (
            self._state is SessionState.STOPPING
            and self._finalization is not None
            and self._finalization.snapshot().incomplete
        ):
            return
        try:
            now_ms = self._read_clock()
        except Exception:
            self._fatal_runtime("Controller clock became unsafe.")
            return
        if (
            self._state is SessionState.STARTING
            and self._started_ms is not None
            and now_ms - self._started_ms >= READINESS_TIMEOUT_MS
        ):
            self._startup_timeout()
            return
        try:
            incoming = self._runtime.poll()
            if not isinstance(incoming, tuple) or not all(
                isinstance(item, MessageEnvelope) for item in incoming
            ):
                raise ValueError("runtime poll contract")
            for envelope in incoming:
                self.handle_message(envelope)
            if self._state in {SessionState.COMPLETED, SessionState.ERROR}:
                return
            if self._state is SessionState.STOPPING and self._finalization is not None:
                if self._finalization.snapshot().incomplete:
                    return
                finalization_tick = self._finalization.tick(now_ms)
                if finalization_tick.show_slow_message:
                    self._snapshot = replace(
                        self._snapshot,
                        slow_finalization_visible=True,
                    )
                if self._finalization.snapshot().force_requested:
                    result = self._runtime.writer_force_close_result()
                    if result is not None:
                        self._apply_force_close_result(result)
                        return
                    health = self._runtime.health()
                    if not isinstance(health, Mapping):
                        raise ValueError("runtime health contract")
                    writer_alive = health.get(ProcessSource.WRITER)
                    if (
                        writer_alive is not True
                        or finalization_tick.force_result_timed_out
                    ):
                        result = self._runtime.writer_force_close_result()
                        if result is not None:
                            self._apply_force_close_result(result)
                        else:
                            self._fatal_runtime(
                                "Writer force-close outcome could not be resolved."
                            )
                    return
            supervisor = self._require_supervisor()
            if supervisor.health_poll_due(now_ms):
                health = self._runtime.health()
                if not isinstance(health, Mapping):
                    raise ValueError("runtime health contract")
                if self._state is SessionState.STOPPING and any(
                    health.get(worker) is False
                    for worker in (
                        ProcessSource.AUDIO,
                        ProcessSource.ASR,
                        ProcessSource.DISCUSSION,
                        ProcessSource.WRITER,
                    )
                    if worker not in self._drained_workers
                ):
                    self._fatal_runtime("Worker exited during finalization.")
                    return
                effective_health = dict(health)
                if self._state is SessionState.STOPPING:
                    for worker in self._drained_workers:
                        effective_health[worker] = True
                for decision in supervisor.observe_health(effective_health):
                    self._apply_recovery(decision)
                    if self._state in {SessionState.ERROR, SessionState.STOPPING}:
                        break
        except Exception:
            self._fatal_runtime("Worker runtime became unavailable.")

    def _route(self, envelope: MessageEnvelope[object], payload: object) -> None:
        message_type = envelope.message_type
        if self._is_finalization_ack(envelope):
            finalization = self._finalization
            if finalization is None:
                return
            was_terminal = finalization.completed or finalization.snapshot().incomplete
            finalization.acknowledge(envelope)
            finalization_snapshot = finalization.snapshot()
            if not was_terminal and (
                finalization_snapshot.incomplete or finalization_snapshot.completed
            ):
                self._consume_terminal_event_sequence()
            if finalization_snapshot.incomplete:
                if isinstance(payload, WriterForceCloseResult):
                    self._snapshot = replace(
                        self._snapshot,
                        latest_successful_save_at=payload.latest_successful_save_at,
                        recording_status="Incomplete",
                        slow_finalization_visible=False,
                    )
                return
            if finalization.completed:
                if isinstance(payload, WriterAck | WriterForceCloseResult):
                    self._snapshot = replace(
                        self._snapshot,
                        latest_successful_save_at=payload.latest_successful_save_at,
                    )
                self._set_state(
                    SessionState.COMPLETED,
                    recording_status="Completed",
                    slow_finalization_visible=False,
                )
            return
        if message_type is MessageType.WRITER_ACK and isinstance(payload, WriterAck):
            self._writer_ack(payload)
            return
        if message_type is MessageType.WRITER_FATAL and isinstance(
            payload, WriterFatal
        ):
            self._fatal_storage()
            return
        if message_type is MessageType.WORKER_READY:
            self._worker_ready(envelope.source)
            return
        if message_type is MessageType.TRANSCRIPT_COMMITTED and isinstance(
            payload, TranscriptCommitted
        ):
            self._route_transcript(envelope, payload)
            return
        if message_type is MessageType.TRANSCRIPT_PARTIAL and isinstance(
            payload, PartialTranscript
        ):
            self._replace_partial(payload)
            return
        if message_type is MessageType.DISCUSSION_STATE_REPLACED and isinstance(
            payload, DiscussionStateReplaced
        ):
            self._send_rewrapped(ProcessSource.WRITER, envelope, payload)
            self._snapshot = replace(self._snapshot, discussion_state=payload.state)
            return
        if message_type is MessageType.ASR_STATUS and isinstance(payload, dict):
            self._asr_status(payload)
            return
        if message_type is MessageType.DISCUSSION_STATUS and isinstance(
            payload, DiscussionStatusPayload
        ):
            self._discussion_status(payload)
            return
        if message_type is MessageType.AUDIO_LEVEL and isinstance(payload, dict):
            level = _dbfs_level(payload["peak_dbfs"])
            field = (
                "microphone_level"
                if payload["source"] == AudioSource.ME.value
                else "loopback_level"
            )
            if field == "microphone_level":
                self._snapshot = replace(
                    self._snapshot,
                    microphone_level=level,
                )
            else:
                self._snapshot = replace(
                    self._snapshot,
                    loopback_level=level,
                )
            return
        if message_type in {
            MessageType.SOURCE_DISCONNECTED,
            MessageType.SOURCE_RECONNECTED,
        } and isinstance(payload, dict):
            self._source_status(message_type, payload)
            return
        if message_type is MessageType.WORKER_ERROR and isinstance(payload, dict):
            self._worker_error(envelope.source, payload)
            return
        if message_type is MessageType.WORKER_STOPPED:
            return

    def _writer_ack(self, payload: WriterAck) -> None:
        if self._state is not SessionState.STARTING or self._workers_started:
            self._snapshot = replace(
                self._snapshot,
                latest_successful_save_at=payload.latest_successful_save_at,
            )
            return
        if payload.acknowledged_sequence != self._writer_open_sequence:
            self._protocol_event_metadata(ProcessSource.WRITER, "unexpected_writer_ack")
            return
        self._snapshot = replace(
            self._snapshot,
            latest_successful_save_at=payload.latest_successful_save_at,
        )
        for worker in _START_WORKERS:
            self._send(
                worker,
                MessageType.WORKER_START,
                {"worker": worker.value},
            )
        self._workers_started = True
        self._ready.add(ProcessSource.WRITER)
        self._ready.update(self._early_ready)
        self._early_ready.clear()
        self._enter_recording_if_ready()

    def _worker_ready(self, worker: ProcessSource) -> None:
        if self._state is not SessionState.STARTING:
            return
        if worker not in _START_WORKERS:
            return
        if not self._workers_started:
            self._early_ready.add(worker)
            return
        self._ready.add(worker)
        self._enter_recording_if_ready()

    def _enter_recording_if_ready(self) -> None:
        if self._ready == {
            ProcessSource.WRITER,
            ProcessSource.AUDIO,
            ProcessSource.ASR,
            ProcessSource.DISCUSSION,
        }:
            self._set_state(
                SessionState.RECORDING,
                recording_status="Recording",
                asr_status="Running",
                analysis_status="Running",
            )

    def _route_transcript(
        self,
        envelope: MessageEnvelope[object],
        payload: TranscriptCommitted,
    ) -> None:
        self._send_rewrapped(ProcessSource.WRITER, envelope, payload)
        self._send_rewrapped(ProcessSource.DISCUSSION, envelope, payload)
        self._snapshot = replace(
            self._snapshot,
            transcript=(*self._snapshot.transcript, payload.record),
            partials=tuple(
                item
                for item in self._snapshot.partials
                if item.source is not payload.record.source
            ),
        )

    def _replace_partial(self, payload: PartialTranscript) -> None:
        retained = tuple(
            item
            for item in self._snapshot.partials
            if item.source is not payload.source
        )
        self._snapshot = replace(
            self._snapshot,
            partials=retained if not payload.text else (*retained, payload),
        )

    def _asr_status(self, payload: dict[str, object]) -> None:
        state = cast(str, payload["state"])
        backlog = cast(int, payload["backlog_ms"])
        maximum = cast(int, payload["maximum_backlog_ms"])
        analysis_paused = cast(bool, payload["analysis_paused"])
        was_delayed = self._snapshot.asr_status == "Delayed"
        delayed = state == "DELAYED"
        status_label = state.title()
        self._snapshot = replace(
            self._snapshot,
            asr_status=status_label,
            asr_backlog_ms=backlog,
            maximum_asr_backlog_ms=maximum,
        )
        if self._state is not SessionState.RECORDING or state not in {
            "RUNNING",
            "DELAYED",
        }:
            self._analysis_paused_for_lag = analysis_paused
            if self._state is SessionState.PAUSED:
                self._snapshot = replace(
                    self._snapshot,
                    analysis_status=self._analysis_status(
                        self._analysis_pause_reasons()
                    ),
                )
            return
        if delayed and not was_delayed:
            if not self._persist_operational(
                EventType.ASR_LAG_STARTED,
                ProcessSource.ASR,
                {"backlog_ms": backlog},
            ):
                return
        elif not delayed and was_delayed:
            if not self._persist_operational(
                EventType.ASR_LAG_ENDED,
                ProcessSource.ASR,
                {"backlog_ms": backlog},
            ):
                return
        was_paused = self._analysis_paused_for_lag
        if analysis_paused and not was_paused:
            if not self._analysis_disabled:
                if not self._persist_operational(
                    EventType.ANALYSIS_PAUSED,
                    ProcessSource.ASR,
                    {"backlog_ms": backlog},
                ):
                    return
                self._send(
                    ProcessSource.DISCUSSION,
                    MessageType.WORKER_PAUSE,
                    {"worker": "DISCUSSION"},
                )
                self._snapshot = replace(
                    self._snapshot,
                    analysis_status="Paused for ASR delay",
                )
        elif not analysis_paused and was_paused:
            if not self._analysis_disabled:
                if not self._persist_operational(
                    EventType.ANALYSIS_RESUMED,
                    ProcessSource.ASR,
                    {"backlog_ms": backlog},
                ):
                    return
                self._send(
                    ProcessSource.DISCUSSION,
                    MessageType.WORKER_RESUME,
                    {"worker": "DISCUSSION"},
                )
                self._snapshot = replace(
                    self._snapshot,
                    analysis_status="Running",
                )
        self._analysis_paused_for_lag = analysis_paused

    def _discussion_status(self, payload: DiscussionStatusPayload) -> None:
        if payload.error_code == "GPU_OOM":
            self._apply_recovery(self._require_supervisor().on_gpu_oom())
            return
        if payload.error_code is not None:
            self._persist_operational(
                EventType.ANALYSIS_FAILED,
                ProcessSource.DISCUSSION,
                {"error_code": payload.error_code},
            )

    def _source_status(
        self,
        message_type: MessageType,
        payload: dict[str, object],
    ) -> None:
        disconnected = message_type is MessageType.SOURCE_DISCONNECTED
        event_type = (
            EventType.SOURCE_DISCONNECTED
            if disconnected
            else EventType.SOURCE_RECONNECTED
        )
        source = AudioSource(cast(str, payload["source"]))
        next_connected = not disconnected
        if self._source_connected[source] is next_connected:
            return
        if not self._persist_operational(
            event_type,
            ProcessSource.AUDIO,
            {"source": source.value, "device_id": cast(str, payload["device_id"])},
        ):
            return
        self._source_connected[source] = next_connected
        self._source_transition_generation[source] += 1
        generation = self._source_transition_generation[source]
        self._announce_once(
            (event_type.value, f"{source.value}:{generation}"),
            f"{source.value} source "
            f"{'disconnected' if disconnected else 'reconnected'}.",
            disconnected,
        )

    def _worker_error(
        self,
        worker: ProcessSource,
        payload: dict[str, object],
    ) -> None:
        code = str(payload["code"])
        if worker is ProcessSource.DISCUSSION and code == "GPU_OOM":
            self._apply_recovery(self._require_supervisor().on_gpu_oom())
            return
        if worker is ProcessSource.AUDIO:
            self._safe_stop("Audio worker failed.")
            return
        self.handle_worker_exit(worker)

    def _apply_recovery(self, decision: RecoveryDecision) -> None:
        worker = decision.worker
        if decision.action is RecoveryAction.FATAL:
            self._fatal_storage()
            return
        if decision.action is RecoveryAction.SAFE_STOP:
            if not self._persist_operational(
                EventType.WORKER_EXITED,
                worker,
                {"worker": worker.value},
            ):
                return
            self._safe_stop(f"{worker.value} worker exited.")
            return
        if decision.action is RecoveryAction.RESTART:
            if not self._persist_operational(
                EventType.WORKER_EXITED,
                worker,
                {"worker": worker.value},
            ):
                return
            try:
                self._runtime.restart(worker)
            except Exception:
                self._safe_stop(f"{worker.value} worker restart failed.")
                return
            if not self._persist_operational(
                EventType.WORKER_RESTARTED,
                worker,
                {"worker": worker.value},
            ):
                return
            self._announce_once(
                ("restart", worker.value),
                f"{worker.value} worker restarted.",
                True,
            )
            return
        if not self._persist_operational(
            EventType.ANALYSIS_FAILED,
            ProcessSource.DISCUSSION,
            {"error_code": decision.reason.value},
        ):
            return
        before = self._analysis_pause_reasons()
        self._analysis_disabled = True
        after = self._analysis_pause_reasons()
        if not before and after:
            try:
                self._send(
                    ProcessSource.DISCUSSION,
                    MessageType.WORKER_PAUSE,
                    {"worker": "DISCUSSION"},
                )
            except Exception:
                self._fatal_runtime("Discussion control queue became unavailable.")
                return
        self._snapshot = replace(self._snapshot, analysis_status="Unavailable")
        self._announce_once(
            ("analysis", decision.reason.value),
            "Discussion analysis is unavailable.",
            True,
        )

    def _startup_timeout(self) -> None:
        missing = next(
            worker
            for worker in (
                ProcessSource.WRITER,
                ProcessSource.AUDIO,
                ProcessSource.ASR,
                ProcessSource.DISCUSSION,
            )
            if worker not in self._ready
        )
        self._bounded_shutdown()
        issue = f"{missing.value} worker did not become ready within 60 seconds."
        self._set_state(
            SessionState.PREFLIGHT,
            issue=issue,
            recording_status="Ready",
            asr_status="Idle",
            analysis_status="Idle",
        )
        self._clear_runtime_session()

    def _safe_stop(self, issue: str) -> None:
        if self._state not in {SessionState.ERROR, SessionState.STOPPING}:
            self._set_state(
                SessionState.STOPPING,
                issue=issue,
                recording_status="Finalizing",
            )
        self._bounded_shutdown()

    def _is_finalization_ack(self, envelope: MessageEnvelope[object]) -> bool:
        return (
            self._state is SessionState.STOPPING
            and self._finalization is not None
            and envelope.message_type
            in {
                MessageType.WORKER_STOPPED,
                MessageType.WRITER_ACK,
                MessageType.WRITER_FORCE_CLOSE_RESULT,
            }
        )

    def _request_writer_force_close(
        self,
        request: WriterForceCloseRequest,
        timeout_seconds: float,
    ) -> WriterForceCloseResult | None:
        if type(request) is not WriterForceCloseRequest:
            raise TypeError("request must be an exact WriterForceCloseRequest")
        return self._runtime.request_writer_force_close(request, timeout_seconds)

    def _apply_force_close_result(self, result: WriterForceCloseResult) -> None:
        finalization = self._finalization
        if finalization is None:
            raise RuntimeError("finalization is unavailable")
        was_terminal = finalization.completed or finalization.snapshot().incomplete
        finalization.resolve_force_result(result)
        if not was_terminal:
            self._consume_terminal_event_sequence()
        self._snapshot = replace(
            self._snapshot,
            latest_successful_save_at=result.latest_successful_save_at,
            recording_status=(
                "Incomplete"
                if result.outcome is WriterForceCloseOutcome.INCOMPLETE
                else "Completed"
            ),
            slow_finalization_visible=False,
        )
        if result.outcome is WriterForceCloseOutcome.COMPLETED:
            self._set_state(SessionState.COMPLETED)

    def _consume_terminal_event_sequence(self) -> None:
        if self._terminal_event_consumed:
            return
        self._event_sequence += 1
        self._terminal_event_consumed = True

    def _build_writer_finalize(self) -> WriterFinalize:
        launch = self._require_launch()
        now_ms = self._stop_confirmed_ms
        ended_at = self._stop_confirmed_at
        if now_ms is None or ended_at is None:
            raise RuntimeError("stop confirmation time is unavailable")
        started_ms = self._started_ms
        if started_ms is None:
            raise RuntimeError("session start time is unavailable")
        elapsed_ms = now_ms - started_ms
        paused_ms = sum(
            interval.ended_ms - interval.started_ms
            for interval in self._pause_intervals
        )
        final_state = self._snapshot.discussion_state
        if final_state is None:
            raise RuntimeError("final discussion state is unavailable")
        completion_event = EventRecord(
            schema_version=1,
            session_id=launch.session_id,
            sequence=self._event_sequence + 1,
            event_type=EventType.SESSION_COMPLETED,
            source=ProcessSource.GUI,
            session_time_ms=elapsed_ms,
            created_at=ended_at,
            details={},
        )
        return WriterFinalize(
            ended_at=ended_at,
            active_duration_ms=max(0, elapsed_ms - paused_ms),
            pause_intervals=tuple(self._pause_intervals),
            final_state=final_state,
            completion_event=completion_event,
        )

    def _build_force_close_event(self) -> EventRecord:
        launch = self._require_launch()
        now_ms = self._read_clock()
        now = self._clock.now()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("clock now must be timezone-aware")
        started_ms = self._started_ms
        if started_ms is None:
            raise RuntimeError("session start time is unavailable")
        finalization = self._finalization
        if finalization is None:
            raise RuntimeError("finalization is unavailable")
        return EventRecord(
            schema_version=1,
            session_id=launch.session_id,
            sequence=self._event_sequence + 1,
            event_type=EventType.FORCE_CLOSE_REQUESTED,
            source=ProcessSource.GUI,
            session_time_ms=now_ms - started_ms,
            created_at=now,
            details={"finalization_step": finalization.step.value},
        )

    def _close_pause_interval(self, now_ms: int) -> None:
        pause_started = self._pause_started_ms
        if pause_started is None:
            return
        started_ms = self._started_ms
        if started_ms is None:
            raise RuntimeError("session start time is unavailable")
        self._pause_intervals.append(
            PauseInterval(pause_started - started_ms, now_ms - started_ms)
        )
        self._pause_started_ms = None

    def _fatal_storage(self) -> None:
        self._snapshot = replace(
            self._snapshot,
            fatal_error="Session storage is unsafe; recording stopped.",
            issue="Session storage is unsafe; recording stopped.",
        )
        self._bounded_shutdown()
        self._set_state(SessionState.ERROR, recording_status="Error")
        self._announce_once(
            ("fatal", "storage"),
            "Session storage is unsafe; recording stopped.",
            True,
        )

    def _fatal_runtime(self, message: str) -> None:
        self._bounded_shutdown()
        self._set_state(
            SessionState.ERROR,
            issue=message,
            fatal_error=message,
            recording_status="Error",
        )
        self._announce_once(("fatal", "runtime"), message, True)

    def _broadcast(
        self,
        message_type: MessageType,
        workers: tuple[ProcessSource, ...] = _START_WORKERS,
    ) -> bool:
        for worker in workers:
            try:
                self._send(
                    worker,
                    message_type,
                    {"worker": worker.value},
                )
            except Exception:
                persisted = self._persist_operational(
                    EventType.WORKER_EXITED,
                    ProcessSource.GUI,
                    {
                        "kind": "lifecycle_broadcast_failed",
                        "target": worker.value,
                        "message_type": message_type.value,
                    },
                )
                if persisted:
                    self._fatal_runtime("Worker control queue became unavailable.")
                return False
        return True

    def _analysis_pause_reasons(
        self,
        *,
        state: SessionState | None = None,
    ) -> frozenset[_AnalysisPauseReason]:
        active_state = self._state if state is None else state
        reasons: set[_AnalysisPauseReason] = set()
        if active_state is SessionState.PAUSED:
            reasons.add(_AnalysisPauseReason.SESSION)
        if self._analysis_paused_for_lag:
            reasons.add(_AnalysisPauseReason.ASR_LAG)
        if self._analysis_disabled:
            reasons.add(_AnalysisPauseReason.DISABLED)
        return frozenset(reasons)

    @staticmethod
    def _analysis_status(reasons: frozenset[_AnalysisPauseReason]) -> str:
        if _AnalysisPauseReason.DISABLED in reasons:
            return "Unavailable"
        if _AnalysisPauseReason.ASR_LAG in reasons:
            return "Paused for ASR delay"
        if _AnalysisPauseReason.SESSION in reasons:
            return "Paused"
        return "Running"

    def _send(
        self,
        target: ProcessSource,
        message_type: MessageType,
        payload: object,
        *,
        launch: SessionLaunch | None = None,
    ) -> MessageEnvelope[object]:
        active = self._launch if launch is None else launch
        if active is None:
            raise RuntimeError("session launch is not active")
        sequence = self._outgoing_sequences.get(target, 0) + 1
        envelope = MessageEnvelope(
            schema_version=1,
            session_id=active.session_id,
            message_type=message_type,
            sequence=sequence,
            source=ProcessSource.GUI,
            created_monotonic_ms=self._last_clock_ms
            or active.audio_config.session_started_monotonic_ms,
            payload=payload,
        )
        self._runtime.send(target, envelope)
        self._outgoing_sequences[target] = sequence
        return envelope

    def _send_rewrapped(
        self,
        target: ProcessSource,
        incoming: MessageEnvelope[object],
        payload: object,
    ) -> MessageEnvelope[object]:
        sequence = self._outgoing_sequences.get(target, 0) + 1
        outgoing = rewrap_for_gui(incoming, sequence=sequence, payload=payload)
        self._runtime.send(target, outgoing)
        self._outgoing_sequences[target] = sequence
        return outgoing

    def _persist_operational(
        self,
        event_type: EventType,
        source: ProcessSource,
        details: dict[str, object],
    ) -> bool:
        launch = self._launch
        if launch is None or self._state is SessionState.ERROR:
            return False
        self._event_sequence += 1
        now_ms = self._last_clock_ms
        if now_ms is None:
            now_ms = launch.audio_config.session_started_monotonic_ms
        started = self._started_ms or now_ms
        try:
            record = EventRecord(
                schema_version=1,
                session_id=launch.session_id,
                sequence=self._event_sequence,
                event_type=event_type,
                source=source,
                session_time_ms=max(0, now_ms - started),
                created_at=self._clock.now(),
                details=details,  # type: ignore[arg-type]
            )
            self._send(
                ProcessSource.WRITER,
                MessageType.EVENT_APPENDED,
                WriterAppendEvent(record),
            )
            return True
        except Exception:
            if event_type is not EventType.STORAGE_FAILED:
                self._fatal_storage()
            return False

    def _protocol_event(self, envelope: object, kind: str) -> None:
        if self._protocol_event_in_progress:
            return
        self._protocol_event_in_progress = True
        try:
            source = (
                envelope.source
                if isinstance(envelope, MessageEnvelope)
                and isinstance(envelope.source, ProcessSource)
                else ProcessSource.GUI
            )
            self._protocol_event_metadata(source, kind)
        finally:
            self._protocol_event_in_progress = False

    def _protocol_event_metadata(self, source: ProcessSource, kind: str) -> None:
        self._persist_operational(
            EventType.WORKER_EXITED,
            source,
            {"kind": kind, "source": source.value},
        )

    def _announce_once(
        self,
        key: tuple[str, str],
        message: str,
        assertive: bool,
    ) -> None:
        if key in self._announced:
            return
        self._announced.add(key)
        try:
            self._announcer.announce(self, message, assertive)
        except Exception:
            pass

    def _read_clock(self) -> int:
        value = self._clock.monotonic_ms()
        if type(value) is not int or value < 0:
            raise ValueError("monotonic clock must return a non-negative exact integer")
        if self._last_clock_ms is not None and value < self._last_clock_ms:
            raise RuntimeError("monotonic clock rollback")
        self._last_clock_ms = value
        return value

    def _bounded_shutdown(self) -> None:
        if self._runtime_shutdown:
            return
        self._runtime_shutdown = True
        try:
            self._runtime.shutdown()
        except Exception:
            pass

    def _require_state(
        self,
        destination: str,
        allowed: set[SessionState],
    ) -> None:
        if self._state not in allowed:
            raise InvalidTransition(f"{self._state.value} -> {destination}")

    def _set_state(self, state: SessionState, **changes: object) -> None:
        self._state = state
        self._snapshot = replace(
            self._snapshot,
            state=state,
            **cast(Any, changes),
        )

    def _require_launch(self) -> SessionLaunch:
        if self._launch is None:
            raise RuntimeError("session launch is not active")
        return self._launch

    def _require_supervisor(self) -> WorkerSupervisor:
        if self._supervisor is None:
            raise RuntimeError("worker supervisor is not active")
        return self._supervisor

    def _clear_runtime_session(self) -> None:
        self._launch = None
        self._sequence_tracker = None
        self._supervisor = None
        self._outgoing_sequences.clear()
        self._writer_open_sequence = None
        self._started_ms = None
        self._workers_started = False
        self._ready.clear()
        self._early_ready.clear()


def _dbfs_level(value: object) -> float:
    if type(value) is not float:
        return 0.0
    return min(1.0, max(0.0, (value + 96.0) / 96.0))
