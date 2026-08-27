"""Transactional session lifecycle and routing tests."""

import pickle
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from datetime import datetime
from pathlib import Path

import pytest

import flowlens.controller.models as controller_models
from flowlens.asr.types import AsrWorkerConfig
from flowlens.audio.types import AudioWorkerConfig
from flowlens.controller.models import (
    DeviceOption,
    ModelCheck,
    PreflightReport,
    PreflightSelection,
    StorageCheck,
)
from flowlens.controller.session_controller import (
    InvalidTransition,
    SessionController,
    SessionLaunch,
    SessionState,
)
from flowlens.discussion.contracts import (
    DiscussionStatusPayload,
    DiscussionStoppedPayload,
)
from flowlens.discussion.llama_cpp_adapter import DiscussionModelConfig
from flowlens.discussion.worker import DiscussionWorkerConfig
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import (
    EventType,
    MessageType,
    ProcessSource,
    SessionMode,
)
from flowlens.domain.messages import (
    DiscussionStateReplaced,
    EventRecord,
    MessageEnvelope,
    TranscriptCommitted,
    WriterAck,
    WriterAppendEvent,
    WriterForceCloseRequest,
    WriterForceCloseResult,
    WriterOpenSession,
)
from tests.factories import make_discussion_state, make_manifest, make_transcript_record

SESSION_ID = "01J00000000000000000000000"
NOW = datetime.fromisoformat("2026-08-19T12:00:00+09:00")


class FakeClock:
    def __init__(self) -> None:
        self.ms = 1_000

    def monotonic_ms(self) -> int:
        return self.ms

    def now(self) -> datetime:
        return NOW


class FakePreflight:
    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        self.calls: list[PreflightSelection] = []

    def evaluate(self, selection: PreflightSelection) -> PreflightReport:
        self.calls.append(selection)
        return self.report


class FakeRuntime:
    def __init__(self) -> None:
        self.launches: list[object] = []
        self.sent: list[tuple[ProcessSource, MessageEnvelope[object]]] = []
        self.polled: tuple[MessageEnvelope[object], ...] = ()
        self.restarted: list[ProcessSource] = []
        self.restart_launches: list[object | None] = []
        self.shutdown_count = 0
        self.safe_stop_audio_fences: list[bool] = []
        self.force_close_requests: list[WriterForceCloseRequest] = []
        self.force_close_result_value: WriterForceCloseResult | None = None
        self.health_values: dict[ProcessSource, bool] = {
            source: True for source in ProcessSource if source is not ProcessSource.GUI
        }

    def start_all(self, launch: object) -> None:
        self.launches.append(launch)

    def send(
        self,
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        self.sent.append((target, envelope))

    def poll(self) -> tuple[MessageEnvelope[object], ...]:
        result = self.polled
        self.polled = ()
        return result

    def restart(self, target: ProcessSource, launch: object | None = None) -> None:
        self.restarted.append(target)
        self.restart_launches.append(launch)

    def health(self) -> Mapping[ProcessSource, bool]:
        return self.health_values.copy()

    def shutdown(self) -> object:
        self.shutdown_count += 1
        return None

    def safe_stop(self, *, audio_fence_required: bool = False) -> object:
        self.safe_stop_audio_fences.append(audio_fence_required)
        self.shutdown_count += 1
        return None

    def request_writer_force_close(
        self,
        request: WriterForceCloseRequest,
        timeout_seconds: float,
    ) -> WriterForceCloseResult | None:
        assert timeout_seconds == 0.25
        self.force_close_requests.append(request)
        return self.force_close_result_value

    def writer_force_close_result(self) -> WriterForceCloseResult | None:
        return self.force_close_result_value


class FakeAnnouncer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def announce(
        self,
        widget: object,
        message: str,
        assertive: bool = False,
    ) -> None:
        del widget
        self.messages.append((message, assertive))


def selection() -> PreflightSelection:
    return PreflightSelection(SessionMode.MEETING, "mic-1", "out-1")


def report(root: Path) -> PreflightReport:
    return PreflightReport(
        selection=selection(),
        microphones=(DeviceOption("mic-1", "Microphone", False),),
        loopbacks=(DeviceOption("out-1", "Speakers", True),),
        mic_level=0.1,
        loopback_level=0.2,
        models=(
            ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
            ModelCheck("qwen3-4b-instruct-2507", None, True, None),
        ),
        storage=StorageCheck(root, 500 * 1024 * 1024, True, None),
        destination=root,
        issues=(),
        can_start=True,
    )


def discussion_config(initial_state: DiscussionState) -> DiscussionWorkerConfig:
    """Create a no-I/O model config solely for value-contract tests."""

    model = object.__new__(DiscussionModelConfig)
    object.__setattr__(model, "model_path", Path("C:/models/qwen.gguf"))
    object.__setattr__(model, "sha256", "a" * 64)
    object.__setattr__(model, "n_ctx", 8192)
    object.__setattr__(model, "n_gpu_layers", -1)
    object.__setattr__(model, "temperature", 0.0)
    object.__setattr__(model, "max_tokens", 512)
    return DiscussionWorkerConfig(SESSION_ID, model, initial_state)


def launch(root: Path) -> SessionLaunch:
    initial_state = make_discussion_state()
    return SessionLaunch(
        session_id=SESSION_ID,
        session_dir=root / SESSION_ID,
        manifest=make_manifest(session_id=SESSION_ID),
        initial_state=initial_state,
        audio_config=AudioWorkerConfig(
            SESSION_ID,
            "mic-1",
            "out-1",
            1_000,
            100,
            100,
        ),
        asr_config=AsrWorkerConfig(SESSION_ID, Path("C:/models/asr")),
        discussion_config=discussion_config(initial_state),
    )


def make_controller(
    tmp_path: Path,
    *,
    acceptance_enabled: bool = False,
) -> tuple[SessionController, FakeRuntime, FakeClock, FakeAnnouncer, SessionLaunch]:
    root = tmp_path.resolve()
    runtime = FakeRuntime()
    clock = FakeClock()
    announcer = FakeAnnouncer()
    launch_value = launch(root)
    controller = SessionController(
        preflight=FakePreflight(report(root)),
        runtime=runtime,
        clock=clock,
        announcer=announcer,
        launch_factory=lambda checked, now, now_ms: launch_value,
        acceptance_enabled=acceptance_enabled,
    )
    return controller, runtime, clock, announcer, launch_value


def worker_envelope(
    source: ProcessSource,
    message_type: MessageType,
    sequence: int,
    payload: object,
    *,
    schema_version: int = 1,
) -> MessageEnvelope[object]:
    return MessageEnvelope(
        schema_version=schema_version,
        session_id=SESSION_ID,
        message_type=message_type,
        sequence=sequence,
        source=source,
        created_monotonic_ms=1_100 + sequence,
        payload=payload,
    )


def writer_ack(sequence: int = 1) -> MessageEnvelope[object]:
    return worker_envelope(
        ProcessSource.WRITER,
        MessageType.WRITER_ACK,
        1,
        WriterAck(sequence, NOW),
    )


def ready(source: ProcessSource, sequence: int = 1) -> MessageEnvelope[object]:
    return worker_envelope(
        source,
        MessageType.WORKER_READY,
        sequence,
        {"worker": source.value},
    )


def recording_controller(
    tmp_path: Path,
) -> tuple[SessionController, FakeRuntime, FakeClock, FakeAnnouncer]:
    controller, runtime, clock, announcer, _ = make_controller(tmp_path)
    controller.enter_preflight()
    controller.start(selection())
    controller.handle_message(writer_ack())
    for source in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        controller.handle_message(ready(source))
    assert controller.snapshot().state is SessionState.RECORDING
    return controller, runtime, clock, announcer


def sent_to(
    runtime: FakeRuntime,
    target: ProcessSource,
) -> list[MessageEnvelope[object]]:
    return [message for route, message in runtime.sent if route is target]


def test_session_state_has_exact_eight_lifecycle_values() -> None:
    assert [state.value for state in SessionState] == [
        "IDLE",
        "PREFLIGHT",
        "STARTING",
        "RECORDING",
        "PAUSED",
        "STOPPING",
        "COMPLETED",
        "ERROR",
    ]


def test_recording_waits_for_writer_open_then_all_worker_readiness(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _, launch_value = make_controller(tmp_path)
    controller.enter_preflight()

    controller.start(selection())

    assert controller.state is SessionState.STARTING
    assert runtime.launches == [launch_value]
    writer_messages = sent_to(runtime, ProcessSource.WRITER)
    assert len(writer_messages) == 1
    assert writer_messages[0].message_type is MessageType.WRITER_OPEN_SESSION
    assert writer_messages[0].sequence == 1
    assert isinstance(writer_messages[0].payload, WriterOpenSession)
    assert not sent_to(runtime, ProcessSource.AUDIO)

    controller.handle_message(writer_ack())
    assert [
        sent_to(runtime, source)[0].message_type
        for source in (
            ProcessSource.AUDIO,
            ProcessSource.ASR,
            ProcessSource.DISCUSSION,
        )
    ] == [MessageType.WORKER_START] * 3
    for source in (ProcessSource.AUDIO, ProcessSource.ASR):
        controller.handle_message(ready(source))
        assert controller.state is SessionState.STARTING
    controller.handle_message(ready(ProcessSource.DISCUSSION))
    assert controller.snapshot().state is SessionState.RECORDING


def test_illegal_transition_has_no_side_effect_of_any_kind(tmp_path: Path) -> None:
    controller, runtime, clock, announcer, _ = make_controller(tmp_path)
    before = controller.snapshot()

    with pytest.raises(InvalidTransition, match="IDLE -> PAUSED"):
        controller.pause()

    assert controller.snapshot() == before
    assert runtime.sent == []
    assert runtime.launches == []
    assert runtime.shutdown_count == 0
    assert announcer.messages == []
    assert clock.ms == 1_000


def test_launch_is_complete_frozen_and_spawn_picklable(tmp_path: Path) -> None:
    launch_value = launch(tmp_path.resolve())

    restored = pickle.loads(pickle.dumps(launch_value))

    assert restored == launch_value
    with pytest.raises(FrozenInstanceError):
        launch_value.session_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(("elapsed", "timed_out"), [(59_999, False), (60_000, True)])
def test_startup_timeout_has_exact_boundary_and_returns_preflight(
    tmp_path: Path,
    elapsed: int,
    timed_out: bool,
) -> None:
    controller, runtime, clock, _, _ = make_controller(tmp_path)
    controller.enter_preflight()
    controller.start(selection())
    controller.handle_message(writer_ack())
    controller.handle_message(ready(ProcessSource.AUDIO))
    clock.ms = 1_000 + elapsed

    controller.tick()

    if timed_out:
        assert controller.state is SessionState.PREFLIGHT
        assert controller.snapshot().issue is not None
        issue = controller.snapshot().issue
        assert issue is not None
        assert "ASR worker" in issue
        assert runtime.shutdown_count == 1
    else:
        assert controller.state is SessionState.STARTING
        assert runtime.shutdown_count == 0


def test_committed_transcript_is_persisted_before_discussion_fanout(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    incoming = worker_envelope(
        ProcessSource.ASR,
        MessageType.TRANSCRIPT_COMMITTED,
        2,
        make_transcript_record().to_dict(),
    )

    controller.handle_message(incoming)

    post_ready = runtime.sent[-2:]
    assert [target for target, _ in post_ready] == [
        ProcessSource.WRITER,
        ProcessSource.DISCUSSION,
    ]
    assert all(message is not incoming for _, message in post_ready)
    assert all(
        isinstance(message.payload, TranscriptCommitted) for _, message in post_ready
    )
    assert sent_to(runtime, ProcessSource.WRITER)[-1].sequence == 2
    assert sent_to(runtime, ProcessSource.DISCUSSION)[-1].sequence == 2
    assert controller.snapshot().transcript == (make_transcript_record(),)


def test_acceptance_mode_records_commit_and_discussion_latencies(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, announcer, _ = make_controller(
        tmp_path,
        acceptance_enabled=True,
    )
    controller.enter_preflight()
    controller.start(selection())
    controller.handle_message(writer_ack())
    for source in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        controller.handle_message(ready(source))
    partial = replace(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_PARTIAL,
            2,
            {
                "source": make_transcript_record().source.value,
                "text": "今回の方針",
                "session_start_ms": 200,
                "session_end_ms": 1_000,
                "source_start_sample": 0,
                "source_end_sample": 12_800,
            },
        ),
        created_monotonic_ms=2_500,
    )
    controller.handle_message(partial)
    committed = replace(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            3,
            make_transcript_record().to_dict(),
        ),
        created_monotonic_ms=3_200,
    )
    controller.handle_message(committed)
    later_record = replace(
        make_transcript_record(2),
        committed_at=datetime.fromisoformat("2026-08-19T12:06:00+09:00"),
    )
    later = replace(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            4,
            later_record.to_dict(),
        ),
        created_monotonic_ms=4_200,
    )
    controller.handle_message(later)
    replacement = replace(
        worker_envelope(
            ProcessSource.DISCUSSION,
            MessageType.DISCUSSION_STATE_REPLACED,
            2,
            DiscussionStateReplaced(0, make_discussion_state(revision=1)),
        ),
        created_monotonic_ms=7_000,
    )
    controller.handle_message(replacement)

    snapshot = controller.snapshot()
    assert snapshot.partial_latencies_ms == (500,)
    assert snapshot.commit_latencies_ms == (400, 400)
    assert snapshot.discussion_latencies_ms == (4_200,)


def test_writer_mutations_share_one_contiguous_gui_sequence(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            2,
            make_transcript_record().to_dict(),
        )
    )
    replacement = DiscussionStateReplaced(0, make_discussion_state(revision=1))
    controller.handle_message(
        worker_envelope(
            ProcessSource.DISCUSSION,
            MessageType.DISCUSSION_STATE_REPLACED,
            2,
            replacement,
        )
    )
    event = EventRecord(
        1,
        SESSION_ID,
        1,
        EventType.WORKER_RESTARTED,
        ProcessSource.GUI,
        100,
        NOW,
        {"worker": "ASR"},
    )
    controller.persist_event(event)

    messages = sent_to(runtime, ProcessSource.WRITER)
    assert [message.sequence for message in messages] == [1, 2, 3, 4]
    assert [message.message_type for message in messages] == [
        MessageType.WRITER_OPEN_SESSION,
        MessageType.TRANSCRIPT_COMMITTED,
        MessageType.DISCUSSION_STATE_REPLACED,
        MessageType.EVENT_APPENDED,
    ]
    assert all(message.source is ProcessSource.GUI for message in messages)
    assert isinstance(messages[-1].payload, WriterAppendEvent)


def asr_status(
    sequence: int,
    backlog_ms: int,
    maximum: int,
    *,
    state: str | None = None,
    analysis_paused: bool | None = None,
) -> MessageEnvelope[object]:
    return worker_envelope(
        ProcessSource.ASR,
        MessageType.ASR_STATUS,
        sequence,
        {
            "state": state or ("DELAYED" if backlog_ms > 2_000 else "RUNNING"),
            "backlog_ms": backlog_ms,
            "maximum_backlog_ms": maximum,
            "analysis_paused": (
                backlog_ms > 5_000 if analysis_paused is None else analysis_paused
            ),
        },
    )


def test_asr_hysteresis_uses_strict_boundaries_and_emits_commands_once(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.handle_message(asr_status(2, 2_000, 2_000))
    assert controller.snapshot().asr_status == "Running"
    controller.handle_message(asr_status(3, 2_001, 2_001))
    assert controller.snapshot().asr_status == "Delayed"
    controller.handle_message(asr_status(4, 5_000, 5_000))
    assert controller.snapshot().analysis_status == "Running"
    controller.handle_message(asr_status(5, 5_001, 5_001))
    assert controller.snapshot().analysis_status == "Paused for ASR delay"
    pause_count = sum(
        message.message_type is MessageType.WORKER_PAUSE
        for message in sent_to(runtime, ProcessSource.DISCUSSION)
    )
    controller.handle_message(asr_status(6, 5_100, 5_100))
    controller.handle_message(
        asr_status(
            7,
            2_000,
            5_100,
            state="DELAYED",
            analysis_paused=True,
        )
    )
    assert (
        sum(
            message.message_type is MessageType.WORKER_PAUSE
            for message in sent_to(runtime, ProcessSource.DISCUSSION)
        )
        == pause_count
    )
    controller.handle_message(asr_status(8, 1_999, 5_100))
    assert (
        sent_to(runtime, ProcessSource.DISCUSSION)[-1].message_type
        is MessageType.WORKER_RESUME
    )
    assert controller.snapshot().maximum_asr_backlog_ms == 5_100


def test_duplicate_status_causes_no_command_or_snapshot_mutation(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 5_001, 5_001))
    before = controller.snapshot()
    sent = len(runtime.sent)

    controller.handle_message(asr_status(2, 1_000, 5_001))

    assert controller.snapshot() == before
    assert len(runtime.sent) == sent


def test_writer_exit_is_fatal_and_asr_restarts_only_once(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.handle_worker_exit(ProcessSource.ASR)
    assert runtime.restarted == [ProcessSource.ASR]
    restart_launch = runtime.restart_launches[-1]
    assert isinstance(restart_launch, SessionLaunch)
    assert restart_launch.asr_config.allow_nonzero_initial_sample is True
    assert restart_launch.asr_config.initial_transcript_sequence == 1
    controller.handle_worker_exit(ProcessSource.ASR)
    assert controller.snapshot().state is SessionState.ERROR

    other, other_runtime, _, _ = recording_controller(tmp_path)
    other.handle_worker_exit(ProcessSource.WRITER)
    assert other.state is SessionState.ERROR
    fatal_error = other.snapshot().fatal_error
    assert fatal_error is not None
    assert fatal_error.startswith("Session storage is unsafe")
    assert other_runtime.shutdown_count == 1


def test_audio_fatal_forces_an_ordered_fence_and_reaches_error_terminal(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.handle_message(
        worker_envelope(
            ProcessSource.AUDIO,
            MessageType.WORKER_ERROR,
            2,
            {"worker": "AUDIO", "code": "CAPTURE_FAILED", "detail": "device"},
        )
    )

    assert controller.state is SessionState.ERROR
    assert runtime.safe_stop_audio_fences == [True]


def test_asr_restart_resets_sequences_and_records_success_after_running(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.handle_worker_exit(ProcessSource.ASR)

    assert runtime.restarted == [ProcessSource.ASR]
    assert not any(
        isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type is EventType.WORKER_RESTARTED
        for message in sent_to(runtime, ProcessSource.WRITER)
    )

    controller.handle_message(ready(ProcessSource.ASR, sequence=1))
    assert (
        sent_to(runtime, ProcessSource.ASR)[-1].message_type is MessageType.WORKER_START
    )
    assert sent_to(runtime, ProcessSource.ASR)[-1].sequence == 1
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    assert not any(
        isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type is EventType.WORKER_RESTARTED
        for message in sent_to(runtime, ProcessSource.WRITER)
    )
    controller.handle_message(asr_status(3, 0, 0))

    assert any(
        isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type is EventType.WORKER_RESTARTED
        for message in sent_to(runtime, ProcessSource.WRITER)
    )


def test_asr_restart_preserves_paused_session_without_waiting_for_a_fence(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.pause()

    controller.handle_worker_exit(ProcessSource.ASR)

    restart_launch = runtime.restart_launches[-1]
    assert isinstance(restart_launch, SessionLaunch)
    assert restart_launch.asr_config.start_paused is True
    controller.handle_message(ready(ProcessSource.ASR, sequence=1))
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    controller.handle_message(asr_status(3, 0, 0))
    assert controller.state is SessionState.PAUSED


def test_discussion_restart_uses_latest_state_and_replays_only_pending_history(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    first = make_transcript_record()
    second = replace(
        make_transcript_record(2),
        committed_at=datetime.fromisoformat("2026-08-19T12:06:00+09:00"),
    )
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            2,
            first.to_dict(),
        )
    )
    state = replace(
        make_discussion_state(revision=1),
        updated_at=first.committed_at,
    )
    controller.handle_message(
        worker_envelope(
            ProcessSource.DISCUSSION,
            MessageType.DISCUSSION_STATE_REPLACED,
            2,
            DiscussionStateReplaced(0, state),
        )
    )
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            3,
            second.to_dict(),
        )
    )

    controller.handle_worker_exit(ProcessSource.DISCUSSION)

    restart_launch = runtime.restart_launches[-1]
    assert isinstance(restart_launch, SessionLaunch)
    assert restart_launch.initial_state == state
    assert restart_launch.discussion_config.initial_state == state
    assert sent_to(runtime, ProcessSource.DISCUSSION)[-1].sequence == 1
    assert (
        sent_to(runtime, ProcessSource.DISCUSSION)[-1].message_type
        is MessageType.WORKER_START
    )

    controller.handle_message(ready(ProcessSource.DISCUSSION, sequence=1))

    replay = sent_to(runtime, ProcessSource.DISCUSSION)[-1]
    assert replay.sequence == 2
    assert replay.message_type is MessageType.TRANSCRIPT_COMMITTED
    assert isinstance(replay.payload, TranscriptCommitted)
    assert replay.payload.record == second


def test_gpu_oom_disables_analysis_while_recording_continues(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.handle_message(
        worker_envelope(
            ProcessSource.DISCUSSION,
            MessageType.DISCUSSION_STATUS,
            2,
            DiscussionStatusPayload("FAILED", 0, 0, "GPU_OOM"),
        )
    )

    assert controller.snapshot().state is SessionState.RECORDING
    assert controller.snapshot().analysis_status == "Unavailable"
    assert (
        sent_to(runtime, ProcessSource.DISCUSSION)[-1].message_type
        is MessageType.WORKER_PAUSE
    )


def test_unknown_schema_persists_metadata_only_error_without_tracker_corruption(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    hostile = worker_envelope(
        ProcessSource.ASR,
        MessageType.WORKER_ERROR,
        2,
        {"worker": "ASR", "code": "BAD", "detail": "SECRET TRANSCRIPT"},
        schema_version=99,
    )

    controller.handle_message(hostile)
    controller.handle_message(asr_status(2, 2_001, 2_001))

    events = [
        message.payload.record
        for message in sent_to(runtime, ProcessSource.WRITER)
        if isinstance(message.payload, WriterAppendEvent)
    ]
    assert events
    assert all("SECRET" not in str(event.to_dict()) for event in events)
    assert controller.snapshot().asr_status == "Delayed"


def test_pause_resume_update_snapshot_synchronously_and_route_once(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.pause()
    assert controller.state is SessionState.PAUSED
    assert controller.snapshot().recording_status == "Paused"
    controller.resume()
    assert controller.snapshot().state is SessionState.RECORDING
    assert controller.snapshot().recording_status == "Recording"

    for target in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        types = [message.message_type for message in sent_to(runtime, target)]
        assert types[-2:] == [MessageType.WORKER_PAUSE, MessageType.WORKER_RESUME]


def test_base_exception_from_port_propagates(tmp_path: Path) -> None:
    controller, runtime, _, _, _ = make_controller(tmp_path)
    controller.enter_preflight()

    class Abort(BaseException):
        pass

    def abort(launch: object) -> None:
        del launch
        raise Abort()

    runtime.start_all = abort  # type: ignore[method-assign]

    with pytest.raises(Abort):
        controller.start(selection())


@pytest.mark.parametrize(
    "method_name",
    [
        "refresh_preflight",
        "start",
        "resume",
        "request_stop",
        "cancel_stop",
        "confirm_stop",
        "keep_waiting",
        "force_close",
    ],
)
def test_every_idle_invalid_call_is_transactional(
    tmp_path: Path,
    method_name: str,
) -> None:
    controller, runtime, _, announcer, _ = make_controller(tmp_path)
    before = controller.snapshot()
    method = getattr(controller, method_name)

    with pytest.raises(InvalidTransition):
        if method_name in {"refresh_preflight", "start"}:
            method(selection())
        else:
            method()

    assert controller.snapshot() == before
    assert runtime.launches == []
    assert runtime.sent == []
    assert runtime.shutdown_count == 0
    assert announcer.messages == []


def test_wrong_writer_ack_does_not_start_workers(tmp_path: Path) -> None:
    controller, runtime, _, _, _ = make_controller(tmp_path)
    controller.enter_preflight()
    controller.start(selection())

    controller.handle_message(writer_ack(sequence=2))

    assert controller.state is SessionState.STARTING
    assert sent_to(runtime, ProcessSource.AUDIO) == []
    assert sent_to(runtime, ProcessSource.ASR) == []
    assert sent_to(runtime, ProcessSource.DISCUSSION) == []
    assert controller.snapshot().latest_successful_save_at is None


def test_early_audio_and_asr_readiness_is_deferred_until_writer_ack(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _, _ = make_controller(tmp_path)
    controller.enter_preflight()
    controller.start(selection())
    controller.handle_message(ready(ProcessSource.AUDIO))
    controller.handle_message(ready(ProcessSource.ASR))

    assert controller.state is SessionState.STARTING
    assert sent_to(runtime, ProcessSource.AUDIO) == []
    controller.handle_message(writer_ack())
    controller.handle_message(ready(ProcessSource.DISCUSSION))

    assert controller.snapshot().state is SessionState.RECORDING


def test_cumulative_maximum_backlog_never_regresses(tmp_path: Path) -> None:
    controller, _, _, _ = recording_controller(tmp_path)

    controller.handle_message(asr_status(2, 2_001, 9_000))
    controller.handle_message(asr_status(3, 1_000, 4_000))

    assert controller.snapshot().maximum_asr_backlog_ms == 9_000


def test_discussion_exit_restarts_once_then_disables_only_analysis(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    controller.handle_worker_exit(ProcessSource.DISCUSSION)
    controller.handle_worker_exit(ProcessSource.DISCUSSION)

    assert runtime.restarted == [ProcessSource.DISCUSSION]
    assert controller.snapshot().state is SessionState.RECORDING
    assert controller.snapshot().analysis_status == "Unavailable"


def test_stop_confirmation_seam_does_not_stop_until_task7_confirmation(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    before = len(runtime.sent)

    controller.request_stop()
    assert controller.state is SessionState.RECORDING
    assert controller.snapshot().stop_confirmation_visible is True
    assert len(runtime.sent) == before
    controller.cancel_stop()
    assert controller.snapshot().stop_confirmation_visible is False
    controller.request_stop()
    controller.confirm_stop()

    assert controller.snapshot().state is SessionState.STOPPING
    assert controller.snapshot().recording_status == "Finalizing"
    assert not any(
        message.message_type is MessageType.WRITER_FINALIZE
        for _, message in runtime.sent
    )


def test_closed_runtime_during_routing_fails_closed_and_is_bounded(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    def fail_send(
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        del target, envelope
        raise RuntimeError("closed")

    runtime.send = fail_send  # type: ignore[method-assign]
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            2,
            make_transcript_record().to_dict(),
        )
    )

    assert controller.state is SessionState.ERROR
    assert runtime.shutdown_count == 1


def test_clock_rollback_and_simultaneous_writer_exit_fail_closed(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    clock.ms = 999

    controller.tick()

    assert controller.state is SessionState.ERROR
    assert runtime.shutdown_count == 1

    other, other_runtime, other_clock, _ = recording_controller(tmp_path)
    other_runtime.health_values[ProcessSource.WRITER] = False
    other_runtime.health_values[ProcessSource.ASR] = False
    other_clock.ms = 1_250
    other.tick()

    assert other.state is SessionState.ERROR
    assert other_runtime.restarted == []
    assert other_runtime.shutdown_count == 1


@pytest.mark.parametrize(
    ("worker", "prior_exits"),
    [
        (ProcessSource.ASR, 0),
        (ProcessSource.ASR, 1),
        (ProcessSource.DISCUSSION, 1),
    ],
)
def test_recovery_aborts_immediately_when_operational_event_cannot_persist(
    tmp_path: Path,
    worker: ProcessSource,
    prior_exits: int,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    for _ in range(prior_exits):
        controller.handle_worker_exit(worker)
    before_restarts = runtime.restarted.copy()
    before_discussion = len(sent_to(runtime, ProcessSource.DISCUSSION))
    original_send = runtime.send

    def fail_event(
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        if (
            target is ProcessSource.WRITER
            and envelope.message_type is MessageType.EVENT_APPENDED
        ):
            raise OSError("storage closed")
        original_send(target, envelope)

    runtime.send = fail_event  # type: ignore[method-assign]

    controller.handle_worker_exit(worker)

    assert controller.state is SessionState.ERROR
    assert runtime.shutdown_count == 1
    assert runtime.restarted == before_restarts
    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == before_discussion


@pytest.mark.parametrize("operation", ["pause", "resume"])
@pytest.mark.parametrize(
    "failed_target",
    list(
        _START_WORKERS := (
            ProcessSource.AUDIO,
            ProcessSource.ASR,
            ProcessSource.DISCUSSION,
        )
    ),
)
def test_partial_lifecycle_broadcast_failure_is_bounded_and_coherent(
    tmp_path: Path,
    operation: str,
    failed_target: ProcessSource,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    if operation == "resume":
        controller.pause()
    expected_type = (
        MessageType.WORKER_PAUSE if operation == "pause" else MessageType.WORKER_RESUME
    )
    original_send = runtime.send

    def fail_control(
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        if target is failed_target and envelope.message_type is expected_type:
            raise RuntimeError("closed control queue")
        original_send(target, envelope)

    runtime.send = fail_control  # type: ignore[method-assign]

    getattr(controller, operation)()

    assert controller.state is SessionState.ERROR
    assert controller.snapshot().recording_status == "Error"
    assert runtime.shutdown_count == 1


def test_base_exception_during_lifecycle_broadcast_propagates(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)

    class Abort(BaseException):
        pass

    def abort(
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        del target, envelope
        raise Abort()

    runtime.send = abort  # type: ignore[method-assign]

    with pytest.raises(Abort):
        controller.pause()

    assert controller.state is SessionState.RECORDING
    assert runtime.shutdown_count == 0


def stopped_envelope(worker: ProcessSource) -> MessageEnvelope[object]:
    payload: object
    if worker is ProcessSource.AUDIO:
        payload = {
            "worker": "AUDIO",
            "drained": True,
            "writer_frames": 10,
            "asr_frames": 10,
        }
    elif worker is ProcessSource.ASR:
        payload = {"worker": "ASR", "drained": True, "committed_count": 3}
    else:
        payload = DiscussionStoppedPayload("DISCUSSION", True, 1, 0)
    return worker_envelope(worker, MessageType.WORKER_STOPPED, 2, payload)


def stopped_envelope_with_sequence(
    worker: ProcessSource,
    sequence: int,
) -> MessageEnvelope[object]:
    envelope = stopped_envelope(worker)
    return worker_envelope(worker, envelope.message_type, sequence, envelope.payload)


def test_completion_summary_contract_validates_authoritative_values(
    tmp_path: Path,
) -> None:
    summary_class = getattr(controller_models, "CompletionSummary", None)

    assert summary_class is not None
    with pytest.raises(Exception, match="duration_ms"):
        summary_class(-1, 0, tmp_path.resolve())
    with pytest.raises(Exception, match="transcript_count"):
        summary_class(0, -1, tmp_path.resolve())
    with pytest.raises(Exception, match="save_path"):
        summary_class(0, 0, Path("relative"))

    summary = summary_class(10_000, 1, tmp_path.resolve())
    assert summary.duration_ms == 10_000
    assert summary.transcript_count == 1
    assert summary.save_path == tmp_path.resolve()


def test_completed_snapshot_contains_durable_session_completion_summary(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            2,
            TranscriptCommitted(make_transcript_record()),
        )
    )
    clock.ms = 11_000
    controller.request_stop()
    controller.confirm_stop()
    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    controller.handle_message(stopped_envelope_with_sequence(ProcessSource.ASR, 3))
    controller.handle_message(stopped_envelope(ProcessSource.DISCUSSION))
    finalize = sent_to(runtime, ProcessSource.WRITER)[-1]

    controller.handle_message(
        worker_envelope(
            ProcessSource.WRITER,
            MessageType.WRITER_ACK,
            2,
            WriterAck(finalize.sequence, NOW),
        )
    )

    completion = controller.snapshot().completion
    assert completion is not None
    assert completion.duration_ms == 10_000
    assert completion.transcript_count == 1
    assert completion.save_path == tmp_path.resolve() / SESSION_ID

    controller.enter_preflight()
    assert controller.snapshot().completion is None


@pytest.mark.parametrize("worker", list(_START_WORKERS))
def test_stopping_health_ignores_only_workers_with_drain_ack(
    tmp_path: Path,
    worker: ProcessSource,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.handle_message(stopped_envelope(worker))
    controller.acknowledge_expected_stop(worker)
    runtime.health_values[worker] = False
    clock.ms = 1_250

    controller.tick()

    assert controller.state is SessionState.STOPPING
    assert runtime.shutdown_count == 0
    assert runtime.restarted == []
    controller.handle_worker_exit(worker)
    assert runtime.shutdown_count == 0


@pytest.mark.parametrize("worker", [ProcessSource.ASR, ProcessSource.DISCUSSION])
def test_unsolicited_stopped_payload_does_not_mask_health_failure(
    tmp_path: Path,
    worker: ProcessSource,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.handle_message(stopped_envelope(worker))
    runtime.health_values[worker] = False
    clock.ms = 1_250

    controller.tick()

    assert controller.state is SessionState.ERROR
    assert runtime.restarted == []
    assert runtime.shutdown_count == 1


def test_stopping_still_treats_writer_failure_as_fatal(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()

    controller.handle_worker_exit(ProcessSource.WRITER)

    assert controller.state is SessionState.ERROR
    assert runtime.shutdown_count == 1


@pytest.mark.parametrize("lag_paused", [False, True])
def test_second_discussion_exit_persists_explicit_exit_reason(
    tmp_path: Path,
    lag_paused: bool,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    if lag_paused:
        controller.handle_message(asr_status(2, 5_001, 5_001))
    controller.handle_worker_exit(ProcessSource.DISCUSSION)
    controller.handle_worker_exit(ProcessSource.DISCUSSION)

    events = [
        message.payload.record
        for message in sent_to(runtime, ProcessSource.WRITER)
        if isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type is EventType.ANALYSIS_FAILED
    ]
    assert events[-1].details == {"error_code": "EXIT"}


@pytest.mark.parametrize("lag_paused", [False, True])
def test_gpu_oom_persists_explicit_gpu_reason(
    tmp_path: Path,
    lag_paused: bool,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    if lag_paused:
        controller.handle_message(asr_status(2, 5_001, 5_001))
    controller.handle_message(
        worker_envelope(
            ProcessSource.DISCUSSION,
            MessageType.DISCUSSION_STATUS,
            2,
            DiscussionStatusPayload("FAILED", 0, 0, "GPU_OOM"),
        )
    )

    events = [
        message.payload.record
        for message in sent_to(runtime, ProcessSource.WRITER)
        if isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type is EventType.ANALYSIS_FAILED
    ]
    assert events[-1].details == {"error_code": "GPU_OOM"}


def test_source_connectivity_logs_only_genuine_transition_cycles(
    tmp_path: Path,
) -> None:
    controller, runtime, _, announcer = recording_controller(tmp_path)
    payload = {"source": "ME", "device_id": "mic-1"}

    controller.handle_message(
        worker_envelope(
            ProcessSource.AUDIO, MessageType.SOURCE_DISCONNECTED, 2, payload
        )
    )
    controller.handle_message(
        worker_envelope(
            ProcessSource.AUDIO, MessageType.SOURCE_DISCONNECTED, 3, payload
        )
    )
    controller.handle_message(
        worker_envelope(ProcessSource.AUDIO, MessageType.SOURCE_RECONNECTED, 4, payload)
    )
    controller.handle_message(
        worker_envelope(ProcessSource.AUDIO, MessageType.SOURCE_RECONNECTED, 5, payload)
    )
    controller.handle_message(
        worker_envelope(
            ProcessSource.AUDIO, MessageType.SOURCE_DISCONNECTED, 6, payload
        )
    )

    events = [
        message.payload.record.event_type
        for message in sent_to(runtime, ProcessSource.WRITER)
        if isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type
        in {EventType.SOURCE_DISCONNECTED, EventType.SOURCE_RECONNECTED}
    ]
    assert events == [
        EventType.SOURCE_DISCONNECTED,
        EventType.SOURCE_RECONNECTED,
        EventType.SOURCE_DISCONNECTED,
    ]
    source_announcements = [
        message for message, _ in announcer.messages if "ME source" in message
    ]
    assert len(source_announcements) == 3


def test_inconsistent_asr_status_is_rejected_without_commands_or_sequence_use(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    before_discussion = len(sent_to(runtime, ProcessSource.DISCUSSION))

    controller.handle_message(
        asr_status(2, 2_000, 2_000, state="DELAYED", analysis_paused=False)
    )
    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == before_discussion
    assert controller.snapshot().asr_backlog_ms == 0

    controller.handle_message(
        asr_status(2, 2_000, 2_000, state="RUNNING", analysis_paused=False)
    )
    assert controller.snapshot().asr_backlog_ms == 2_000
    controller.handle_message(
        asr_status(3, 5_001, 5_001, state="DELAYED", analysis_paused=False)
    )
    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == before_discussion
    controller.handle_message(
        asr_status(3, 5_001, 5_001, state="DELAYED", analysis_paused=True)
    )
    assert (
        sent_to(runtime, ProcessSource.DISCUSSION)[-1].message_type
        is MessageType.WORKER_PAUSE
    )


@pytest.mark.parametrize("kind", ["transcript", "status"])
def test_gap_event_persistence_failure_aborts_routing_after_shutdown(
    tmp_path: Path,
    kind: str,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    original_send = runtime.send
    before_snapshot = controller.snapshot()
    before_discussion = len(sent_to(runtime, ProcessSource.DISCUSSION))

    def fail_gap_event(
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        if (
            target is ProcessSource.WRITER
            and envelope.message_type is MessageType.EVENT_APPENDED
        ):
            raise OSError("storage closed")
        original_send(target, envelope)

    runtime.send = fail_gap_event  # type: ignore[method-assign]
    incoming = (
        worker_envelope(
            ProcessSource.ASR,
            MessageType.TRANSCRIPT_COMMITTED,
            3,
            make_transcript_record().to_dict(),
        )
        if kind == "transcript"
        else asr_status(3, 2_001, 2_001)
    )

    controller.handle_message(incoming)

    assert controller.state is SessionState.ERROR
    assert runtime.shutdown_count == 1
    assert controller.snapshot().transcript == before_snapshot.transcript
    assert controller.snapshot().asr_backlog_ms == before_snapshot.asr_backlog_ms
    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == before_discussion
    assert not any(
        message.message_type is MessageType.TRANSCRIPT_COMMITTED
        for message in sent_to(runtime, ProcessSource.WRITER)
    )


def test_stopped_asr_status_never_resumes_discussion_during_stopping(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 5_001, 5_001))
    controller.request_stop()
    controller.confirm_stop()
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.WORKER_STOPPED,
            3,
            {"worker": "ASR", "drained": True, "committed_count": 0},
        )
    )
    before_resume = sum(
        message.message_type is MessageType.WORKER_RESUME
        for message in sent_to(runtime, ProcessSource.DISCUSSION)
    )

    controller.handle_message(
        asr_status(
            4,
            0,
            5_001,
            state="STOPPED",
            analysis_paused=False,
        )
    )

    assert controller.snapshot().asr_status == "Stopped"
    assert (
        sum(
            message.message_type is MessageType.WORKER_RESUME
            for message in sent_to(runtime, ProcessSource.DISCUSSION)
        )
        == before_resume
    )


def test_paused_session_status_updates_display_without_analysis_resume(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 5_001, 5_001))
    controller.pause()
    before_resume = sum(
        message.message_type is MessageType.WORKER_RESUME
        for message in sent_to(runtime, ProcessSource.DISCUSSION)
    )

    controller.handle_message(
        asr_status(
            3,
            1_000,
            5_001,
            state="RUNNING",
            analysis_paused=False,
        )
    )

    assert controller.snapshot().asr_status == "Running"
    assert controller.snapshot().asr_backlog_ms == 1_000
    assert (
        sum(
            message.message_type is MessageType.WORKER_RESUME
            for message in sent_to(runtime, ProcessSource.DISCUSSION)
        )
        == before_resume
    )


def test_user_pause_resume_preserves_active_lag_pause_reason(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 5_001, 5_001))
    discussion_before = len(sent_to(runtime, ProcessSource.DISCUSSION))

    controller.pause()
    controller.resume()

    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == discussion_before
    assert controller.snapshot().analysis_status == "Paused for ASR delay"
    for target in (ProcessSource.AUDIO, ProcessSource.ASR):
        assert [message.message_type for message in sent_to(runtime, target)[-2:]] == [
            MessageType.WORKER_PAUSE,
            MessageType.WORKER_RESUME,
        ]

    controller.handle_message(asr_status(3, 1_000, 5_001))

    assert (
        sent_to(runtime, ProcessSource.DISCUSSION)[-1].message_type
        is MessageType.WORKER_RESUME
    )


def test_lag_clear_during_user_pause_syncs_without_command_then_resumes_once(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 5_001, 5_001))
    controller.pause()
    discussion_before = len(sent_to(runtime, ProcessSource.DISCUSSION))
    lag_events_before = tuple(
        message.payload.record.event_type
        for message in sent_to(runtime, ProcessSource.WRITER)
        if isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type
        in {EventType.ASR_LAG_STARTED, EventType.ASR_LAG_ENDED}
    )

    controller.handle_message(asr_status(3, 1_000, 5_001))

    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == discussion_before
    assert controller.snapshot().analysis_status == "Paused"
    assert (
        tuple(
            message.payload.record.event_type
            for message in sent_to(runtime, ProcessSource.WRITER)
            if isinstance(message.payload, WriterAppendEvent)
            and message.payload.record.event_type
            in {EventType.ASR_LAG_STARTED, EventType.ASR_LAG_ENDED}
        )
        == lag_events_before
    )

    controller.resume()

    discussion_controls = sent_to(runtime, ProcessSource.DISCUSSION)
    assert len(discussion_controls) == discussion_before + 1
    assert discussion_controls[-1].message_type is MessageType.WORKER_RESUME
    assert controller.snapshot().analysis_status == "Running"


def test_lag_start_during_user_pause_keeps_discussion_paused_on_resume(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.pause()
    discussion_before = len(sent_to(runtime, ProcessSource.DISCUSSION))
    lag_events_before = tuple(
        message.payload.record.event_type
        for message in sent_to(runtime, ProcessSource.WRITER)
        if isinstance(message.payload, WriterAppendEvent)
        and message.payload.record.event_type
        in {EventType.ASR_LAG_STARTED, EventType.ASR_LAG_ENDED}
    )

    controller.handle_message(asr_status(2, 5_001, 5_001))
    controller.resume()

    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == discussion_before
    assert controller.snapshot().analysis_status == "Paused for ASR delay"
    assert (
        tuple(
            message.payload.record.event_type
            for message in sent_to(runtime, ProcessSource.WRITER)
            if isinstance(message.payload, WriterAppendEvent)
            and message.payload.record.event_type
            in {EventType.ASR_LAG_STARTED, EventType.ASR_LAG_ENDED}
        )
        == lag_events_before
    )
    for target in (ProcessSource.AUDIO, ProcessSource.ASR):
        assert [message.message_type for message in sent_to(runtime, target)[-2:]] == [
            MessageType.WORKER_PAUSE,
            MessageType.WORKER_RESUME,
        ]


def test_disabled_analysis_stays_disabled_across_user_pause_resume(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(
        worker_envelope(
            ProcessSource.DISCUSSION,
            MessageType.DISCUSSION_STATUS,
            2,
            DiscussionStatusPayload("FAILED", 0, 0, "GPU_OOM"),
        )
    )
    discussion_before = len(sent_to(runtime, ProcessSource.DISCUSSION))

    controller.pause()
    controller.resume()

    assert len(sent_to(runtime, ProcessSource.DISCUSSION)) == discussion_before
    assert controller.snapshot().analysis_status == "Unavailable"
    for target in (ProcessSource.AUDIO, ProcessSource.ASR):
        assert [message.message_type for message in sent_to(runtime, target)[-2:]] == [
            MessageType.WORKER_PAUSE,
            MessageType.WORKER_RESUME,
        ]
