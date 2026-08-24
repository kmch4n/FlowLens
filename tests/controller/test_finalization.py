"""Acknowledgement-driven finalization state-machine tests."""

import itertools
import pickle
import random
from datetime import datetime
from pathlib import Path

import pytest

from flowlens.controller.finalization import (
    FinalizationCoordinator,
    FinalizationStep,
)
from flowlens.controller.session_controller import SessionController, SessionState
from flowlens.discussion.contracts import DiscussionStoppedPayload
from flowlens.domain.enums import EventType, MessageType, ProcessSource
from flowlens.domain.messages import (
    EventRecord,
    MessageEnvelope,
    WriterAck,
    WriterFinalize,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterForceCloseResult,
)
from tests.controller.test_session_controller import (
    FakeRuntime,
    recording_controller,
    sent_to,
    stopped_envelope,
    worker_envelope,
)
from tests.factories import make_discussion_state

SESSION_ID = "01J00000000000000000000000"
NOW = datetime.fromisoformat("2026-08-19T12:30:00+09:00")


class Harness:
    """Strict fake transport for observable finalization effects."""

    def __init__(self) -> None:
        self.sequences = {source: 0 for source in ProcessSource}
        self.sent: list[tuple[ProcessSource, MessageEnvelope[object]]] = []
        self.acknowledged: list[ProcessSource] = []
        self.shutdown_count = 0
        self.fail_at_send: int | None = None
        self.force_requests: list[WriterForceCloseRequest] = []
        self.force_result: WriterForceCloseResult | None = None
        self.fail_force_request = False

    def send(
        self,
        target: ProcessSource,
        message_type: MessageType,
        payload: object,
    ) -> MessageEnvelope[object]:
        if self.fail_at_send == len(self.sent) + 1:
            raise OSError("closed control queue")
        self.sequences[target] += 1
        envelope = MessageEnvelope(
            1,
            SESSION_ID,
            message_type,
            self.sequences[target],
            ProcessSource.GUI,
            1_000,
            payload,
        )
        self.sent.append((target, envelope))
        return envelope

    def acknowledge(self, worker: ProcessSource) -> None:
        self.acknowledged.append(worker)

    def shutdown(self) -> None:
        self.shutdown_count += 1

    def request_force_close(
        self,
        request: WriterForceCloseRequest,
        timeout_seconds: float,
    ) -> WriterForceCloseResult | None:
        if self.fail_force_request:
            raise OSError("force-close channel closed")
        assert timeout_seconds == 0.25
        self.force_requests.append(request)
        return self.force_result


def finalize_payload() -> WriterFinalize:
    return WriterFinalize(
        NOW,
        1_800_000,
        (),
        make_discussion_state(),
        EventRecord(
            1,
            SESSION_ID,
            4,
            EventType.SESSION_COMPLETED,
            ProcessSource.GUI,
            1_800_000,
            NOW,
            {},
        ),
    )


def force_event() -> EventRecord:
    return EventRecord(
        1,
        SESSION_ID,
        4,
        EventType.FORCE_CLOSE_REQUESTED,
        ProcessSource.GUI,
        1_000,
        NOW,
        {"finalization_step": "DRAIN_AUDIO"},
    )


def make_finalizer(
    harness: Harness | None = None,
) -> tuple[FinalizationCoordinator, Harness]:
    active = Harness() if harness is None else harness
    coordinator = FinalizationCoordinator(
        session_id=SESSION_ID,
        send=active.send,
        writer_finalize_factory=finalize_payload,
        force_close_event_factory=force_event,
        acknowledge_expected_stop=active.acknowledge,
        request_writer_force_close=active.request_force_close,
        shutdown=active.shutdown,
    )
    return coordinator, active


def current_step(coordinator: FinalizationCoordinator) -> FinalizationStep:
    """Read a step without leaking assertion narrowing across mutations."""

    return coordinator.step


def last_target(runtime: FakeRuntime) -> ProcessSource:
    """Read the last fake-runtime route without cross-assertion narrowing."""

    return runtime.sent[-1][0]


def controller_state(controller: SessionController) -> SessionState:
    """Read lifecycle state without cross-assertion narrowing."""

    return controller.state


def ack(
    source: ProcessSource,
    sequence: int,
    payload: object,
    *,
    session_id: str = SESSION_ID,
    schema_version: int = 1,
) -> MessageEnvelope[object]:
    return MessageEnvelope(
        schema_version,
        session_id,
        (
            MessageType.WRITER_FORCE_CLOSE_RESULT
            if type(payload) is WriterForceCloseResult
            else (
                MessageType.WRITER_ACK
                if source is ProcessSource.WRITER
                else MessageType.WORKER_STOPPED
            )
        ),
        sequence,
        source,
        2_000,
        payload,
    )


def test_exact_acknowledgements_drive_ordered_finalization() -> None:
    coordinator, harness = make_finalizer()

    first = coordinator.begin(100)
    assert [(target, item.message_type, item.payload) for target, item in first] == [
        (ProcessSource.AUDIO, MessageType.WORKER_STOP, {"worker": "AUDIO"})
    ]
    assert current_step(coordinator) is FinalizationStep.DRAIN_AUDIO

    second = coordinator.acknowledge(
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": 10,
                "asr_frames": 10,
            },
        )
    )
    assert second[0][0] is ProcessSource.ASR
    assert second[0][1].payload == {"worker": "ASR", "finalize": True}
    assert current_step(coordinator) is FinalizationStep.FINALIZE_ASR

    third = coordinator.acknowledge(
        ack(
            ProcessSource.ASR,
            2,
            {"worker": "ASR", "drained": True, "committed_count": 3},
        )
    )
    assert third[0][0] is ProcessSource.DISCUSSION
    assert third[0][1].payload == {"worker": "DISCUSSION", "finalize": True}
    assert current_step(coordinator) is FinalizationStep.FINAL_ANALYSIS

    fourth = coordinator.acknowledge(
        ack(
            ProcessSource.DISCUSSION,
            2,
            DiscussionStoppedPayload("DISCUSSION", True, 0, 0),
        )
    )
    finalize = fourth[0][1]
    assert fourth[0][0] is ProcessSource.WRITER
    assert finalize.message_type is MessageType.WRITER_FINALIZE
    assert isinstance(finalize.payload, WriterFinalize)
    assert current_step(coordinator) is FinalizationStep.FINALIZE_WRITER
    assert harness.acknowledged == [
        ProcessSource.AUDIO,
        ProcessSource.ASR,
        ProcessSource.DISCUSSION,
    ]

    assert (
        coordinator.acknowledge(
            ack(ProcessSource.WRITER, 2, WriterAck(finalize.sequence, NOW))
        )
        == ()
    )
    assert current_step(coordinator) is FinalizationStep.COMPLETE
    assert coordinator.completed is True
    assert harness.shutdown_count == 1


def test_thirty_seconds_only_offers_choices_once() -> None:
    coordinator, harness = make_finalizer()
    coordinator.begin(100)

    assert coordinator.tick(30_099).show_slow_message is False
    assert coordinator.tick(30_100).show_slow_message is True
    assert coordinator.tick(30_101).show_slow_message is False
    coordinator.keep_waiting()
    assert coordinator.snapshot().slow_visible is False
    assert coordinator.tick(60_099).show_slow_message is False
    assert coordinator.tick(60_100).show_slow_message is True
    assert harness.shutdown_count == 0


def test_force_close_uses_out_of_band_request_and_waits_for_writer_result() -> None:
    coordinator, harness = make_finalizer()
    coordinator.begin(0)

    request = coordinator.force_close(1_000)

    assert type(request) is WriterForceCloseRequest
    assert harness.force_requests == [request]
    assert harness.sent == [(ProcessSource.AUDIO, harness.sent[0][1])]
    assert coordinator.snapshot().incomplete is False
    assert coordinator.completed is False
    assert harness.shutdown_count == 0

    coordinator.acknowledge(
        ack(
            ProcessSource.WRITER,
            2,
            WriterForceCloseResult(WriterForceCloseOutcome.INCOMPLETE, NOW),
        )
    )

    assert coordinator.snapshot().incomplete is True
    assert harness.shutdown_count == 1


def valid_acknowledgements() -> tuple[MessageEnvelope[object], ...]:
    return (
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": 10,
                "asr_frames": 10,
            },
        ),
        ack(
            ProcessSource.ASR,
            2,
            {"worker": "ASR", "drained": True, "committed_count": 3},
        ),
        ack(
            ProcessSource.DISCUSSION,
            2,
            DiscussionStoppedPayload("DISCUSSION", True, 0, 0),
        ),
    )


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
def test_all_worker_ack_permutations_advance_only_current_step(
    permutation: tuple[int, ...],
) -> None:
    coordinator, harness = make_finalizer()
    coordinator.begin(0)
    acknowledgements = valid_acknowledgements()

    for index in permutation:
        before = coordinator.step
        coordinator.acknowledge(acknowledgements[index])
        if index != {
            FinalizationStep.DRAIN_AUDIO: 0,
            FinalizationStep.FINALIZE_ASR: 1,
            FinalizationStep.FINAL_ANALYSIS: 2,
        }.get(before):
            assert coordinator.step is before

    for item in acknowledgements:
        coordinator.acknowledge(item)

    assert coordinator.step is FinalizationStep.FINALIZE_WRITER
    assert [worker.value for worker in harness.acknowledged] == [
        "AUDIO",
        "ASR",
        "DISCUSSION",
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": False,
                "writer_frames": 10,
                "asr_frames": 10,
            },
        ),
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": True,
                "asr_frames": 10,
            },
        ),
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": 10,
                "asr_frames": 10,
                "extra": 1,
            },
        ),
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": 10,
                "asr_frames": 10,
            },
            session_id="01J00000000000000000000001",
        ),
        ack(
            ProcessSource.AUDIO,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": 10,
                "asr_frames": 10,
            },
            schema_version=2,
        ),
    ],
)
def test_malformed_or_foreign_ack_is_transactionally_rejected(
    invalid: MessageEnvelope[object],
) -> None:
    coordinator, harness = make_finalizer()
    coordinator.begin(0)
    before = coordinator.snapshot()

    assert coordinator.acknowledge(invalid) == ()
    assert coordinator.snapshot() == before
    assert harness.acknowledged == []
    assert len(harness.sent) == 1


def test_duplicate_replay_and_unrelated_writer_ack_never_complete() -> None:
    harness = Harness()
    harness.sequences[ProcessSource.WRITER] = 3
    coordinator, harness = make_finalizer(harness)
    coordinator.begin(0)
    audio, asr, discussion = valid_acknowledgements()

    coordinator.acknowledge(audio)
    sent_after_audio = len(harness.sent)
    assert coordinator.acknowledge(audio) == ()
    assert len(harness.sent) == sent_after_audio
    coordinator.acknowledge(asr)
    coordinator.acknowledge(discussion)
    finalize_sequence = harness.sent[-1][1].sequence

    assert (
        coordinator.acknowledge(
            ack(ProcessSource.WRITER, 2, WriterAck(finalize_sequence + 1, NOW))
        )
        == ()
    )
    assert coordinator.completed is False
    assert (
        coordinator.acknowledge(
            ack(ProcessSource.WRITER, 3, WriterAck(finalize_sequence - 1, NOW))
        )
        == ()
    )
    assert coordinator.completed is False


@pytest.mark.parametrize("failed_send", [1, 2, 3, 4])
def test_send_failure_keeps_current_step_and_emits_no_later_command(
    failed_send: int,
) -> None:
    harness = Harness()
    harness.fail_at_send = failed_send
    coordinator, _ = make_finalizer(harness)
    acknowledgements = valid_acknowledgements()

    if failed_send == 1:
        with pytest.raises(OSError, match="closed control queue"):
            coordinator.begin(0)
        assert coordinator.step is FinalizationStep.STOP_AUDIO
        assert harness.sent == []
        return

    coordinator.begin(0)
    for item in acknowledgements[: failed_send - 2]:
        coordinator.acknowledge(item)
    before = coordinator.step
    with pytest.raises(OSError, match="closed control queue"):
        coordinator.acknowledge(acknowledgements[failed_send - 2])
    assert coordinator.step is before
    assert len(harness.sent) == failed_send - 1
    assert harness.shutdown_count == 0


def test_force_close_request_failure_emits_no_fifo_writer_command() -> None:
    harness = Harness()
    harness.fail_force_request = True
    coordinator, _ = make_finalizer(harness)
    coordinator.begin(0)

    with pytest.raises(OSError, match="force-close channel closed"):
        coordinator.force_close(1_000)

    assert len(harness.sent) == 1
    assert harness.force_requests == []
    assert coordinator.snapshot().incomplete is False
    assert harness.shutdown_count == 0


def test_begin_force_close_and_completion_are_idempotent() -> None:
    coordinator, harness = make_finalizer()
    assert len(coordinator.begin(0)) == 1
    assert coordinator.begin(1) == ()
    assert coordinator.force_close(1_000)
    assert coordinator.force_close(1_000) is None
    assert len(harness.force_requests) == 1
    assert harness.shutdown_count == 0


def test_finalization_values_are_spawn_picklable() -> None:
    coordinator, _ = make_finalizer()
    coordinator.begin(0)
    restored = pickle.loads(pickle.dumps(coordinator.snapshot()))

    assert restored == coordinator.snapshot()
    assert pickle.loads(pickle.dumps(FinalizationStep.COMPLETE)) is (
        FinalizationStep.COMPLETE
    )


def test_randomized_ack_stream_never_skips_or_repeats_a_step() -> None:
    generator = random.Random(7)
    for _ in range(100):
        coordinator, harness = make_finalizer()
        coordinator.begin(0)
        valid = valid_acknowledgements()
        stream = [generator.choice(valid) for _ in range(30)]
        stream.extend(valid)
        for item in stream:
            coordinator.acknowledge(item)
        assert coordinator.step is FinalizationStep.FINALIZE_WRITER
        assert harness.acknowledged == [
            ProcessSource.AUDIO,
            ProcessSource.ASR,
            ProcessSource.DISCUSSION,
        ]
        assert (
            sum(
                envelope.message_type is MessageType.WRITER_FINALIZE
                for _, envelope in harness.sent
            )
            == 1
        )


def test_controller_confirmation_starts_once_and_matching_writer_ack_completes(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    before = len(runtime.sent)

    controller.request_stop()
    controller.confirm_stop()
    controller.confirm_stop()

    assert controller_state(controller) is SessionState.STOPPING
    assert controller.snapshot().recording_status == "Finalizing"
    assert len(runtime.sent) == before + 1
    assert last_target(runtime) is ProcessSource.AUDIO
    assert runtime.sent[-1][1].payload == {"worker": "AUDIO"}

    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    assert last_target(runtime) is ProcessSource.ASR
    assert runtime.sent[-1][1].payload == {"worker": "ASR", "finalize": True}
    controller.handle_message(stopped_envelope(ProcessSource.ASR))
    assert last_target(runtime) is ProcessSource.DISCUSSION
    controller.handle_message(stopped_envelope(ProcessSource.DISCUSSION))
    finalize = runtime.sent[-1][1]
    assert last_target(runtime) is ProcessSource.WRITER
    assert type(finalize.payload) is WriterFinalize
    assert controller_state(controller) is SessionState.STOPPING

    controller.handle_message(
        worker_envelope(
            ProcessSource.WRITER,
            MessageType.WRITER_ACK,
            2,
            WriterAck(finalize.sequence, NOW),
        )
    )

    assert controller_state(controller) is SessionState.COMPLETED
    assert controller.snapshot().recording_status == "Completed"
    assert runtime.shutdown_count == 1


def test_controller_rejects_future_ack_without_consuming_current_sequence(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    before = len(runtime.sent)
    future = stopped_envelope(ProcessSource.AUDIO)
    future = MessageEnvelope(
        future.schema_version,
        future.session_id,
        future.message_type,
        3,
        future.source,
        future.created_monotonic_ms,
        future.payload,
    )

    controller.handle_message(future)

    assert len(runtime.sent) == before
    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    assert runtime.sent[-1][0] is ProcessSource.ASR


def test_controller_rejects_malformed_current_ack_without_writer_side_effect(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    before = len(runtime.sent)

    controller.handle_message(
        worker_envelope(
            ProcessSource.AUDIO,
            MessageType.WORKER_STOPPED,
            2,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": True,
                "asr_frames": 10,
            },
        )
    )

    assert len(runtime.sent) == before
    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    assert last_target(runtime) is ProcessSource.ASR


@pytest.mark.parametrize("elapsed", [29_999, 30_000, 30_001])
def test_controller_slow_prompt_has_exact_hard_boundary(
    tmp_path: Path,
    elapsed: int,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    clock.ms = 1_000 + elapsed

    controller.tick()

    assert controller.snapshot().slow_finalization_visible is (elapsed >= 30_000)
    assert runtime.shutdown_count == 0
    if elapsed >= 30_000:
        controller.keep_waiting()
        assert controller.snapshot().slow_finalization_visible is False
        clock.ms = 60_999
        controller.tick()
        assert controller.snapshot().slow_finalization_visible is False
        clock.ms = 61_000
        controller.tick()
        assert controller.snapshot().slow_finalization_visible is True


@pytest.mark.parametrize(
    "acknowledged_workers",
    [
        (),
        (ProcessSource.AUDIO,),
        (ProcessSource.AUDIO, ProcessSource.ASR),
        (
            ProcessSource.AUDIO,
            ProcessSource.ASR,
            ProcessSource.DISCUSSION,
        ),
    ],
)
def test_controller_force_close_preserves_incomplete_state_at_every_step(
    tmp_path: Path,
    acknowledged_workers: tuple[ProcessSource, ...],
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    for worker in acknowledged_workers:
        controller.handle_message(stopped_envelope(worker))
    writer_before = len(sent_to(runtime, ProcessSource.WRITER))

    controller.force_close()

    assert len(sent_to(runtime, ProcessSource.WRITER)) == writer_before
    assert len(runtime.force_close_requests) == 1
    request = runtime.force_close_requests[0]
    assert request.event.event_type is EventType.FORCE_CLOSE_REQUESTED
    assert "transcript" not in str(request.event.details).lower()
    assert controller.state is SessionState.STOPPING
    assert controller.snapshot().recording_status == "Resolving force close"
    assert runtime.shutdown_count == 0

    controller.handle_message(
        ack(
            ProcessSource.WRITER,
            2,
            WriterForceCloseResult(WriterForceCloseOutcome.INCOMPLETE, NOW),
        )
    )

    assert controller.snapshot().recording_status == "Incomplete"
    assert runtime.shutdown_count == 1


def test_force_close_losing_finalize_race_resolves_completed(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    for worker in (
        ProcessSource.AUDIO,
        ProcessSource.ASR,
        ProcessSource.DISCUSSION,
    ):
        controller.handle_message(stopped_envelope(worker))
    writer_before = len(sent_to(runtime, ProcessSource.WRITER))

    controller.force_close()
    controller.handle_message(
        ack(
            ProcessSource.WRITER,
            2,
            WriterForceCloseResult(WriterForceCloseOutcome.COMPLETED, NOW),
        )
    )

    assert len(sent_to(runtime, ProcessSource.WRITER)) == writer_before
    assert controller_state(controller) is SessionState.COMPLETED
    assert controller.snapshot().recording_status == "Completed"
    assert runtime.shutdown_count == 1


def test_completion_and_force_share_one_unconsumed_terminal_candidate(
    tmp_path: Path,
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    for worker in (
        ProcessSource.AUDIO,
        ProcessSource.ASR,
        ProcessSource.DISCUSSION,
    ):
        controller.handle_message(stopped_envelope(worker))
    finalize = sent_to(runtime, ProcessSource.WRITER)[-1].payload
    assert type(finalize) is WriterFinalize

    controller.force_close()

    request = runtime.force_close_requests[-1]
    assert request.event.sequence == finalize.completion_event.sequence


def test_force_result_shared_state_fallback_survives_lost_response(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.force_close()
    runtime.force_close_result_value = WriterForceCloseResult(
        WriterForceCloseOutcome.INCOMPLETE,
        NOW,
    )
    clock.ms = 1_001

    controller.tick()

    assert controller.snapshot().recording_status == "Incomplete"
    assert controller.snapshot().fatal_error is None
    assert runtime.shutdown_count == 1


def test_force_result_hard_deadline_fails_honestly_when_outcome_is_unknown(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.force_close()
    clock.ms = 5_999
    controller.tick()
    assert controller_state(controller) is SessionState.STOPPING

    clock.ms = 6_000
    controller.tick()

    assert controller_state(controller) is SessionState.ERROR
    assert controller.snapshot().recording_status == "Error"
    assert controller.snapshot().fatal_error == (
        "Writer force-close outcome could not be resolved."
    )
    assert runtime.shutdown_count == 1


def test_writer_death_during_force_wait_fails_before_deadline_without_false_status(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.force_close()
    runtime.health_values[ProcessSource.WRITER] = False
    clock.ms = 1_001

    controller.tick()

    assert controller_state(controller) is SessionState.ERROR
    assert controller.snapshot().recording_status == "Error"
    assert runtime.shutdown_count == 1


def test_stopping_health_failure_aborts_without_restart_or_later_command(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    sent_before = len(runtime.sent)
    runtime.health_values[ProcessSource.ASR] = False
    clock.ms = 1_250

    controller.tick()

    assert controller_state(controller) is SessionState.ERROR
    assert runtime.restarted == []
    assert len(runtime.sent) == sent_before
    assert runtime.shutdown_count == 1


def test_base_exception_from_finalization_send_propagates_without_state_claim(
    tmp_path: Path,
) -> None:
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
    controller.request_stop()

    with pytest.raises(Abort):
        controller.confirm_stop()

    assert controller_state(controller) is SessionState.RECORDING
    assert runtime.shutdown_count == 0


def test_tick_returns_immediately_after_polled_completion_ack(tmp_path: Path) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    controller.handle_message(stopped_envelope(ProcessSource.ASR))
    controller.handle_message(stopped_envelope(ProcessSource.DISCUSSION))
    finalize = sent_to(runtime, ProcessSource.WRITER)[-1]
    runtime.polled = (
        worker_envelope(
            ProcessSource.WRITER,
            MessageType.WRITER_ACK,
            2,
            WriterAck(finalize.sequence, NOW),
        ),
    )
    runtime.health_values = {source: False for source in runtime.health_values}
    clock.ms = 1_250

    controller.tick()

    assert controller_state(controller) is SessionState.COMPLETED
    assert runtime.restarted == []
    assert runtime.shutdown_count == 1


def test_writer_finalize_duration_is_frozen_at_stop_confirmation(
    tmp_path: Path,
) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    clock.ms = 11_000
    controller.request_stop()
    controller.confirm_stop()
    clock.ms = 41_000
    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    controller.handle_message(stopped_envelope(ProcessSource.ASR))
    controller.handle_message(stopped_envelope(ProcessSource.DISCUSSION))

    payload = sent_to(runtime, ProcessSource.WRITER)[-1].payload

    assert type(payload) is WriterFinalize
    assert payload.active_duration_ms == 10_000
    assert payload.completion_event.session_time_ms == 10_000


def test_tick_after_force_close_preserves_incomplete_snapshot(tmp_path: Path) -> None:
    controller, runtime, clock, _ = recording_controller(tmp_path)
    controller.request_stop()
    controller.confirm_stop()
    controller.force_close()
    runtime.polled = (
        ack(
            ProcessSource.WRITER,
            2,
            WriterForceCloseResult(WriterForceCloseOutcome.INCOMPLETE, NOW),
        ),
    )
    runtime.health_values = {source: False for source in runtime.health_values}
    clock.ms = 1_250

    controller.tick()

    assert controller_state(controller) is SessionState.STOPPING
    assert controller.snapshot().recording_status == "Incomplete"
    assert runtime.restarted == []
    assert runtime.shutdown_count == 1


@pytest.mark.parametrize(
    ("failed_target", "acknowledged_workers"),
    [
        (ProcessSource.AUDIO, ()),
        (ProcessSource.ASR, (ProcessSource.AUDIO,)),
        (
            ProcessSource.DISCUSSION,
            (ProcessSource.AUDIO, ProcessSource.ASR),
        ),
        (
            ProcessSource.WRITER,
            (
                ProcessSource.AUDIO,
                ProcessSource.ASR,
                ProcessSource.DISCUSSION,
            ),
        ),
    ],
)
def test_controller_send_failure_aborts_without_later_finalization_commands(
    tmp_path: Path,
    failed_target: ProcessSource,
    acknowledged_workers: tuple[ProcessSource, ...],
) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    original_send = runtime.send

    def fail_target(
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        if target is failed_target and envelope.message_type in {
            MessageType.WORKER_STOP,
            MessageType.WRITER_FINALIZE,
        }:
            raise OSError("closed finalization queue")
        original_send(target, envelope)

    runtime.send = fail_target  # type: ignore[method-assign]
    controller.request_stop()
    if failed_target is ProcessSource.AUDIO:
        controller.confirm_stop()
    else:
        controller.confirm_stop()
        for worker in acknowledged_workers:
            controller.handle_message(stopped_envelope(worker))

    assert controller.state is SessionState.ERROR
    assert runtime.shutdown_count == 1
    assert not any(
        item.message_type is MessageType.WRITER_FINALIZE
        for item in sent_to(runtime, ProcessSource.WRITER)
    )
