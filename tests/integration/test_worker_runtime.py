"""Integration boundary tests for the five-process runtime."""

from __future__ import annotations

import queue
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

from flowlens.asr.types import AsrWorkerConfig
from flowlens.audio.types import AudioFrame, AudioWorkerConfig
from flowlens.controller.session_controller import SessionLaunch
from flowlens.discussion.llama_cpp_adapter import DiscussionModelConfig
from flowlens.discussion.worker import DiscussionWorkerConfig
from flowlens.domain.enums import AudioSource, EventType, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    MessageEnvelope,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterShutdown,
)
from tests.factories import make_discussion_state, make_event_record, make_manifest

SESSION_ID = "01J00000000000000000000000"


def _worker_target(*args: object) -> None:
    del args


@dataclass
class FakeQueue:
    maxsize: int
    items: list[object]
    fail_get: bool = False
    fail_put: bool = False
    closed: bool = False
    joined: bool = False
    get_nowait_calls: int = 0

    def put(
        self, item: object, block: bool = True, timeout: float | None = None
    ) -> None:
        del block, timeout
        self.put_nowait(item)

    def put_nowait(self, item: object) -> None:
        if self.fail_put:
            raise OSError("queue is hostile")
        if self.maxsize > 0 and len(self.items) >= self.maxsize:
            raise queue.Full
        self.items.append(item)

    def get_nowait(self) -> object:
        self.get_nowait_calls += 1
        if self.fail_get:
            raise OSError("queue is hostile")
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        self.joined = True


class FakeEvent:
    def __init__(self) -> None:
        self.set_count = 0

    def set(self) -> None:
        self.set_count += 1

    def is_set(self) -> bool:
        return self.set_count > 0


@dataclass
class FakeRawValue:
    value: int


class FakeProcess:
    def __init__(
        self,
        *,
        name: str,
        target: Callable[..., object],
        args: tuple[object, ...],
        context: FakeMultiprocessingContext,
    ) -> None:
        self.name = name
        self.target = target
        self.args = args
        self.daemon = True
        self.started = False
        self.alive = False
        self.terminated = False
        self.closed = False
        self.join_calls: list[float] = []
        self._context = context

    def start(self) -> None:
        self._context.events.append(f"start:{self.name}")
        if self.name in self._context.fail_start_names:
            raise OSError(f"cannot start {self.name}")
        self.started = True
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None
        self.join_calls.append(timeout)
        if self.name in self._context.fail_join_names:
            raise OSError("cannot join process")
        if self._context.exit_on_join:
            self.alive = False

    def terminate(self) -> None:
        if self.name in self._context.fail_terminate_names:
            raise OSError("cannot terminate process")
        self.terminated = True
        self.alive = self.name in self._context.remain_alive_after_terminate

    def close(self) -> None:
        if self.name in self._context.fail_close_names:
            raise OSError("cannot close process")
        self.closed = True


class FakeMultiprocessingContext:
    def __init__(self, *, start_method: str = "spawn") -> None:
        self.start_method = start_method
        self.queues: list[FakeQueue] = []
        self.processes: list[FakeProcess] = []
        self.events: list[str] = []
        self.fail_start_names: set[str] = set()
        self.fail_join_names: set[str] = set()
        self.fail_terminate_names: set[str] = set()
        self.fail_close_names: set[str] = set()
        self.remain_alive_after_terminate: set[str] = set()
        self.exit_on_join = False

    def get_start_method(self) -> str:
        return self.start_method

    def Queue(self, maxsize: int = 0) -> FakeQueue:
        result = FakeQueue(maxsize=maxsize, items=[])
        self.queues.append(result)
        self.events.append(f"queue:{maxsize}")
        return result

    def Event(self) -> FakeEvent:
        self.events.append("event")
        return FakeEvent()

    def Lock(self) -> Lock:
        return Lock()

    def RawValue(self, typecode: str, value: int) -> FakeRawValue:
        del typecode
        return FakeRawValue(value)

    def RawArray(self, typecode: str, size: int) -> bytearray:
        del typecode
        return bytearray(size)

    def Process(
        self,
        *,
        name: str,
        target: Callable[..., object],
        args: tuple[object, ...],
    ) -> FakeProcess:
        result = FakeProcess(name=name, target=target, args=args, context=self)
        self.processes.append(result)
        self.events.append(f"process:{name}")
        return result


def fake_targets() -> Any:
    from flowlens.integration.worker_runtime import WorkerTargets

    return WorkerTargets(
        writer=_worker_target,
        audio=_worker_target,
        asr=_worker_target,
        discussion=_worker_target,
    )


def make_launch(session_id: str = SESSION_ID) -> SessionLaunch:
    initial_state = make_discussion_state()
    model = object.__new__(DiscussionModelConfig)
    object.__setattr__(model, "model_path", Path("C:/models/qwen.gguf"))
    object.__setattr__(model, "sha256", "a" * 64)
    object.__setattr__(model, "n_ctx", 8192)
    object.__setattr__(model, "n_gpu_layers", -1)
    object.__setattr__(model, "temperature", 0.0)
    object.__setattr__(model, "max_tokens", 512)
    return SessionLaunch(
        session_id=session_id,
        session_dir=Path("C:/FlowLens/sessions/01J").resolve(strict=False),
        manifest=make_manifest(session_id=session_id),
        initial_state=initial_state,
        audio_config=AudioWorkerConfig(session_id, "mic-1", "out-1", 1_000, 4, 5),
        asr_config=AsrWorkerConfig(session_id, Path("C:/models/asr")),
        discussion_config=DiscussionWorkerConfig(session_id, model, initial_state),
    )


def make_runtime(
    *,
    context: FakeMultiprocessingContext | None = None,
    poll_budget: int = 64,
) -> Any:
    from flowlens.integration.worker_runtime import MultiprocessingWorkerRuntime

    selected = context if context is not None else FakeMultiprocessingContext()
    return MultiprocessingWorkerRuntime(
        selected,
        fake_targets(),
        poll_budget=poll_budget,
        join_timeout_seconds=0.25,
    )


def make_audio_frame() -> AudioFrame:
    return AudioFrame(
        source=AudioSource.ME,
        pcm_s16le=b"\x00\x00" * 320,
        source_start_sample=0,
        source_end_sample=320,
        session_start_ms=0,
        captured_monotonic_ms=1_000,
    )


def make_audio_write_command(frame: AudioFrame) -> AudioWriteCommand:
    return AudioWriteCommand(
        source=frame.source,
        pcm_s16le=frame.pcm_s16le,
        source_start_sample=frame.source_start_sample,
        source_end_sample=frame.source_end_sample,
        session_start_ms=frame.session_start_ms,
        captured_monotonic_ms=frame.captured_monotonic_ms,
    )


def worker_envelope(
    source: ProcessSource,
    sequence: int,
) -> MessageEnvelope[object]:
    return MessageEnvelope(
        schema_version=1,
        session_id=SESSION_ID,
        message_type=MessageType.WORKER_READY,
        sequence=sequence,
        source=source,
        created_monotonic_ms=1_000 + sequence,
        payload={"worker": source.value},
    )


def force_close_request(session_id: str = SESSION_ID) -> WriterForceCloseRequest:
    return WriterForceCloseRequest(
        replace(
            make_event_record(sequence=1),
            session_id=session_id,
            event_type=EventType.FORCE_CLOSE_REQUESTED,
        )
    )


def completion_event(session_id: str = SESSION_ID) -> object:
    return replace(
        make_event_record(sequence=1),
        session_id=session_id,
        event_type=EventType.SESSION_COMPLETED,
    )


def test_runtime_starts_exactly_four_children_with_spawn_context() -> None:
    context = FakeMultiprocessingContext(start_method="spawn")
    runtime = make_runtime(context=context)

    runtime.start_all(make_launch())

    assert [process.name for process in context.processes] == [
        "FlowLens-Writer",
        "FlowLens-Audio",
        "FlowLens-ASR",
        "FlowLens-Discussion",
    ]
    assert all(process.daemon is False for process in context.processes)
    first_process = context.events.index("process:FlowLens-Writer")
    assert all(
        not item.startswith("process:") for item in context.events[:first_process]
    )


def test_worker_process_arguments_match_target_signatures() -> None:
    runtime = make_runtime()
    launch = make_launch()

    runtime.start_all(launch)

    writer, audio, asr, discussion = runtime.context.processes
    assert writer.args == (
        runtime.control_queues[ProcessSource.WRITER],
        runtime.writer_audio_queue,
        runtime.response_queue,
        runtime.writer_stop_event,
        runtime.writer_finalization_gate,
    )
    assert audio.args == (
        launch.audio_config,
        runtime.control_queues[ProcessSource.AUDIO],
        runtime.response_queue,
        runtime.writer_audio_queue,
        runtime.asr_audio_queue,
    )
    assert asr.args == (
        launch.asr_config,
        runtime.asr_audio_queue,
        runtime.control_queues[ProcessSource.ASR],
        runtime.response_queue,
    )
    assert discussion.args == (
        launch.discussion_config,
        runtime.control_queues[ProcessSource.DISCUSSION],
        runtime.response_queue,
    )


def test_runtime_owns_a_fresh_writer_gate_and_reports_normal_completion() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())

    first_gate = runtime.writer_finalization_gate
    first_claim = first_gate.claim_terminal(completion_event())
    first_result = first_gate.publish_result(
        first_claim.outcome,
        datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
    )

    assert first_result.outcome is WriterForceCloseOutcome.COMPLETED
    assert runtime.writer_force_close_result() == first_result

    runtime.shutdown()
    runtime.start_all(make_launch("02J00000000000000000000000"))

    assert runtime.writer_finalization_gate is not first_gate
    assert runtime.writer_force_close_result() is None


def test_runtime_delegates_force_first_and_finalize_first_gate_outcomes() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    gate = runtime.writer_finalization_gate

    assert runtime.request_writer_force_close(force_close_request(), 0.1) is None
    force_claim = gate.claim_force_if_requested()
    assert force_claim is not None
    force_result = gate.publish_result(
        force_claim.outcome,
        datetime(2026, 8, 26, 12, 1, tzinfo=UTC),
    )
    assert runtime.writer_force_close_result() == force_result

    runtime.shutdown()
    runtime.start_all(make_launch("02J00000000000000000000000"))
    fresh_gate = runtime.writer_finalization_gate
    completion_claim = fresh_gate.claim_terminal(
        completion_event("02J00000000000000000000000")
    )
    completion_result = fresh_gate.publish_result(
        completion_claim.outcome,
        datetime(2026, 8, 26, 12, 2, tzinfo=UTC),
    )

    assert (
        runtime.request_writer_force_close(
            force_close_request("02J00000000000000000000000"),
            0.1,
        )
        == completion_result
    )
    assert runtime.writer_force_close_result() == completion_result


def test_runtime_fails_closed_for_stale_force_requests_and_dead_writer() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch("02J00000000000000000000000"))

    with pytest.raises(RuntimeError, match="does not match the active session"):
        runtime.request_writer_force_close(force_close_request(), 0.1)

    assert runtime.shutdown_report is not None

    restarted = make_runtime()
    restarted.start_all(make_launch())
    assert restarted.request_writer_force_close(force_close_request(), 0.1) is None
    restarted.context.processes[0].alive = False

    with pytest.raises(RuntimeError, match="could not be resolved"):
        restarted.writer_force_close_result()

    assert restarted.shutdown_report is not None


def test_runtime_rejects_force_close_queries_without_an_active_gate() -> None:
    with pytest.raises(RuntimeError, match="finalization gate became unavailable"):
        make_runtime().writer_force_close_result()


def test_audio_payload_and_writer_fence_never_enter_general_control_queue() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    frame = make_audio_frame()
    bindings = runtime.audio_bindings()

    bindings.asr_audio_out.put(frame)
    bindings.asr_audio_out.put(AudioDrainFence())
    bindings.writer_audio_out.put(make_audio_write_command(frame))
    bindings.writer_audio_out.put(AudioDrainFence())

    assert runtime.asr_audio_queue.items == [frame, AudioDrainFence()]
    assert runtime.writer_audio_queue.items[0].pcm_s16le == frame.pcm_s16le
    assert isinstance(runtime.writer_audio_queue.items[1], AudioDrainFence)
    assert runtime.general_queues_contain_bytes() is False
    assert runtime.general_queues_contain(AudioDrainFence) is False


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00\x01",
        AudioDrainFence(),
        {"nested": [bytearray(b"\x00\x01")]},
        {"nested": memoryview(b"\x00\x01")},
        AudioWriteCommand(AudioSource.ME, b"\x00\x00", 0, 1, 0, 1),
    ],
)
def test_runtime_rejects_audio_payloads_on_general_control_queue(
    payload: object,
) -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    envelope = MessageEnvelope(
        schema_version=1,
        session_id=SESSION_ID,
        message_type=MessageType.WORKER_START,
        sequence=1,
        source=ProcessSource.GUI,
        created_monotonic_ms=1_000,
        payload=payload,
    )

    with pytest.raises(ValueError, match="dedicated audio queues"):
        runtime.send(ProcessSource.AUDIO, envelope)

    assert runtime.general_queues_contain_bytes() is False
    assert runtime.general_queues_contain(AudioDrainFence) is False


def test_runtime_rejects_cyclic_and_hostile_audio_payload_containers() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    cyclic: list[object] = []
    cyclic.extend([cyclic, {"fence": AudioDrainFence()}])

    class HostileMapping(dict[str, object]):
        def items(self) -> Any:
            raise RuntimeError("hostile mapping")

    for payload in (cyclic, HostileMapping(payload="safe-looking")):
        envelope = MessageEnvelope(
            schema_version=1,
            session_id=SESSION_ID,
            message_type=MessageType.WORKER_START,
            sequence=1,
            source=ProcessSource.GUI,
            created_monotonic_ms=1_000,
            payload=payload,
        )
        with pytest.raises(ValueError, match="dedicated audio queues"):
            runtime.send(ProcessSource.AUDIO, envelope)

    assert runtime.control_queues[ProcessSource.AUDIO].items == []


def test_runtime_keeps_legitimate_control_envelopes_usable() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    envelope = MessageEnvelope(
        schema_version=1,
        session_id=SESSION_ID,
        message_type=MessageType.WORKER_START,
        sequence=1,
        source=ProcessSource.GUI,
        created_monotonic_ms=1_000,
        payload={"worker": "AUDIO"},
    )

    runtime.send(ProcessSource.AUDIO, envelope)

    assert runtime.control_queues[ProcessSource.AUDIO].items == [envelope]


def test_poll_drains_with_fixed_nonblocking_budget() -> None:
    runtime = make_runtime(poll_budget=2)
    runtime.start_all(make_launch())
    runtime.response_queue.items.extend(
        [
            worker_envelope(ProcessSource.AUDIO, 1),
            worker_envelope(ProcessSource.ASR, 1),
            worker_envelope(ProcessSource.DISCUSSION, 1),
        ]
    )

    polled = runtime.poll()

    assert [item.source for item in polled] == [ProcessSource.AUDIO, ProcessSource.ASR]
    assert [item.source for item in runtime.response_queue.items] == [
        ProcessSource.DISCUSSION
    ]
    assert runtime.response_queue.get_nowait_calls == 2


def test_poll_hostile_queue_exception_fails_closed() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    runtime.response_queue.fail_get = True

    with pytest.raises(RuntimeError, match="response queue"):
        runtime.poll()

    assert runtime.shutdown_report is not None
    assert runtime.shutdown_report.completed is False
    assert runtime.writer_stop_event.is_set() is True


def test_restart_is_limited_to_asr_and_discussion_with_bounded_old_process_stop() -> (
    None
):
    runtime = make_runtime()
    runtime.start_all(make_launch())
    old_asr_process = runtime.processes[ProcessSource.ASR]
    old_asr_queue = runtime.control_queues[ProcessSource.ASR]

    runtime.restart(ProcessSource.ASR)

    assert old_asr_queue.items[-1].message_type is MessageType.WORKER_STOP
    assert old_asr_queue.items[-1].payload == {"worker": "ASR", "finalize": True}
    assert old_asr_process.join_calls == [0.25, 0.25]
    assert old_asr_process.terminated is True
    assert old_asr_process.closed is True
    assert runtime.restart_reports[-1].terminated == (ProcessSource.ASR,)
    fresh_asr_process = runtime.processes[ProcessSource.ASR]
    assert fresh_asr_process is not old_asr_process
    assert fresh_asr_process.args[0] == make_launch().asr_config
    assert fresh_asr_process.args[1] is runtime.asr_audio_queue
    assert fresh_asr_process.args[2] is runtime.control_queues[ProcessSource.ASR]
    assert fresh_asr_process.args[2] is not old_asr_queue
    assert fresh_asr_process.args[3] is runtime.response_queue

    with pytest.raises(ValueError, match="only ASR and Discussion"):
        runtime.restart(ProcessSource.AUDIO)


def test_shutdown_sends_typed_controls_joins_bounded_and_never_completes() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())

    report = runtime.shutdown()
    second = runtime.shutdown()

    assert report is second
    assert report.completed is False
    assert set(report.terminated) == {
        ProcessSource.WRITER,
        ProcessSource.AUDIO,
        ProcessSource.ASR,
        ProcessSource.DISCUSSION,
    }
    assert runtime.writer_stop_event.is_set() is True
    writer_control = runtime.control_queues[ProcessSource.WRITER].items[-1]
    assert writer_control.message_type is MessageType.WRITER_SHUTDOWN
    assert isinstance(writer_control.payload, WriterShutdown)
    assert runtime.control_queues[ProcessSource.AUDIO].items[-1].payload == {
        "worker": "AUDIO"
    }
    assert runtime.control_queues[ProcessSource.ASR].items[-1].payload == {
        "worker": "ASR",
        "finalize": True,
    }
    assert runtime.control_queues[ProcessSource.DISCUSSION].items[-1].payload == {
        "worker": "DISCUSSION",
        "finalize": True,
    }
    assert all(
        process.join_calls == [0.25, 0.25] for process in runtime.processes.values()
    )
    assert all(process.closed is True for process in runtime.processes.values())
    assert all(queue.closed and queue.joined for queue in runtime.context.queues)


def test_runtime_recreates_queues_events_and_processes_after_shutdown() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    first_processes = tuple(runtime.processes.values())
    first_queues = tuple(runtime.context.queues)
    first_gate = runtime.writer_finalization_gate

    first_report = runtime.shutdown()
    runtime.start_all(make_launch("02J00000000000000000000000"))
    second_report = runtime.shutdown()

    assert first_report is not second_report
    assert all(queue.closed and queue.joined for queue in first_queues)
    assert not any(process in runtime.processes.values() for process in first_processes)
    assert runtime.writer_finalization_gate is not first_gate


def test_runtime_allows_retry_after_partial_start_failure() -> None:
    context = FakeMultiprocessingContext()
    context.fail_start_names.add("FlowLens-ASR")
    runtime = make_runtime(context=context)

    with pytest.raises(RuntimeError, match="FlowLens-ASR"):
        runtime.start_all(make_launch())

    failed_queues = tuple(context.queues)
    context.fail_start_names.clear()
    runtime.start_all(make_launch("02J00000000000000000000000"))

    assert len(runtime.processes) == 4
    assert all(queue.closed and queue.joined for queue in failed_queues)


def test_shutdown_reports_a_process_that_survives_termination_without_closing_it() -> (
    None
):
    context = FakeMultiprocessingContext()
    context.remain_alive_after_terminate.add("FlowLens-ASR")
    runtime = make_runtime(context=context)
    runtime.start_all(make_launch())

    report = runtime.shutdown()
    asr = runtime.processes[ProcessSource.ASR]

    assert ProcessSource.ASR not in report.joined
    assert ProcessSource.ASR in report.terminated
    assert any("ASR remains alive" in error for error in report.errors)
    assert asr.closed is False


def test_shutdown_records_process_operation_errors_without_raising() -> None:
    context = FakeMultiprocessingContext()
    context.fail_join_names.add("FlowLens-ASR")
    runtime = make_runtime(context=context)
    runtime.start_all(make_launch())

    report = runtime.shutdown()

    assert any("ASR join: OSError" in error for error in report.errors)


def test_start_failure_cleans_up_already_started_children_without_starting_later() -> (
    None
):
    context = FakeMultiprocessingContext()
    context.fail_start_names.add("FlowLens-ASR")
    runtime = make_runtime(context=context)

    with pytest.raises(RuntimeError, match="FlowLens-ASR"):
        runtime.start_all(make_launch())

    assert [process.name for process in context.processes] == [
        "FlowLens-Writer",
        "FlowLens-Audio",
        "FlowLens-ASR",
    ]
    assert runtime.shutdown_report is not None
    assert runtime.shutdown_report.completed is False
    assert context.processes[0].terminated is True
    assert context.processes[1].terminated is True
