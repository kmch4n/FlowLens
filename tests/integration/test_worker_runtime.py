"""Integration boundary tests for the five-process runtime."""

from __future__ import annotations

import multiprocessing
import queue
import time
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
    DiscussionStateReplaced,
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterShutdown,
)
from tests.factories import (
    make_discussion_state,
    make_event_record,
    make_manifest,
    make_transcript_record,
)

SESSION_ID = "01J00000000000000000000000"


def _worker_target(*args: object) -> None:
    del args


def _queue_worker(*args: Any) -> None:
    control: Any = args[0]
    if not hasattr(control, "get"):
        control = args[1]
    control.get(timeout=10)


def _restart_asr_worker(
    config: AsrWorkerConfig,
    audio: Any,
    control: Any,
    output: Any,
) -> None:
    del audio
    output.put(worker_envelope(ProcessSource.ASR, 1))
    output.put(
        MessageEnvelope(
            1,
            config.session_id,
            MessageType.ASR_STATUS,
            2,
            ProcessSource.ASR,
            1_002,
            {
                "state": "READY",
                "backlog_ms": 0,
                "maximum_backlog_ms": 0,
                "analysis_paused": False,
            },
        )
    )
    control.get(timeout=10)
    output.put(
        MessageEnvelope(
            1,
            config.session_id,
            MessageType.TRANSCRIPT_COMMITTED,
            3,
            ProcessSource.ASR,
            1_003,
            make_transcript_record(config.initial_transcript_sequence).to_dict(),
        )
    )


def _restart_discussion_worker(config: Any, control: Any, output: Any) -> None:
    control.get(timeout=10)
    output.put(worker_envelope(ProcessSource.DISCUSSION, 1))
    committed = control.get(timeout=10)
    assert isinstance(committed.payload, TranscriptCommitted)
    state = replace(
        config.initial_state,
        revision=config.initial_state.revision + 1,
        updated_at=committed.payload.record.committed_at,
    )
    output.put(
        MessageEnvelope(
            1,
            config.session_id,
            MessageType.DISCUSSION_STATE_REPLACED,
            2,
            ProcessSource.DISCUSSION,
            1_002,
            DiscussionStateReplaced(config.initial_state.revision, state),
        )
    )


@dataclass
class FakeQueue:
    maxsize: int
    items: list[object]
    fail_get: bool = False
    fail_put: bool = False
    fail_close: bool = False
    fail_join: bool = False
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
        if self.fail_close:
            raise OSError("cannot close queue")
        self.closed = True

    def join_thread(self) -> None:
        if self.fail_join:
            raise OSError("cannot join queue thread")
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
        index: int,
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
        self._index = index

    def start(self) -> None:
        self._context.events.append(f"start:{self.name}")
        if self.name in self._context.fail_start_names:
            raise OSError(f"cannot start {self.name}")
        self.started = True
        self.alive = True
        if self.name in self._context.fail_start_after_start_names:
            raise OSError(f"cannot finish starting {self.name}")

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None
        self.join_calls.append(timeout)
        if not self.started:
            self._context.unstarted_lifecycle_calls.append(f"join:{self.name}")
            if self._context.reject_unstarted_lifecycle:
                raise AssertionError("join is invalid before start")
        if self.name in self._context.fail_join_names:
            raise OSError("cannot join process")
        if self._context.exit_on_join:
            self.alive = False

    def terminate(self) -> None:
        if not self.started:
            self._context.unstarted_lifecycle_calls.append(f"terminate:{self.name}")
            if self._context.reject_unstarted_lifecycle:
                raise AssertionError("terminate is invalid before start")
        if self.name in self._context.fail_terminate_names:
            raise OSError("cannot terminate process")
        self.terminated = True
        self.alive = (
            self.name in self._context.remain_alive_after_terminate
            or self._index in self._context.remain_alive_after_terminate_indices
        )

    def close(self) -> None:
        if self._index in self._context.fail_process_close_indices:
            raise OSError("cannot close process")
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
        self.fail_start_after_start_names: set[str] = set()
        self.fail_join_names: set[str] = set()
        self.fail_terminate_names: set[str] = set()
        self.fail_close_names: set[str] = set()
        self.fail_process_close_indices: set[int] = set()
        self.fail_queue_close_indices: set[int] = set()
        self.remain_alive_after_terminate: set[str] = set()
        self.remain_alive_after_terminate_indices: set[int] = set()
        self.reject_unstarted_lifecycle = False
        self.unstarted_lifecycle_calls: list[str] = []
        self.exit_on_join = False

    def get_start_method(self) -> str:
        return self.start_method

    def Queue(self, maxsize: int = 0) -> FakeQueue:
        result = FakeQueue(maxsize=maxsize, items=[])
        result.fail_close = len(self.queues) in self.fail_queue_close_indices
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
        result = FakeProcess(
            name=name,
            target=target,
            args=args,
            context=self,
            index=len(self.processes),
        )
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
    assert fresh_asr_process.args[3] is not runtime.response_queue

    runtime.response_queue.items.append(worker_envelope(ProcessSource.ASR, 2))
    fresh_asr_process.args[3].items.append(worker_envelope(ProcessSource.ASR, 1))
    runtime.response_queue.items.append(worker_envelope(ProcessSource.AUDIO, 2))

    polled = runtime.poll()

    assert [(item.source, item.sequence) for item in polled] == [
        (ProcessSource.ASR, 1),
        (ProcessSource.AUDIO, 2),
    ]

    with pytest.raises(ValueError, match="only ASR and Discussion"):
        runtime.restart(ProcessSource.AUDIO)


def test_real_spawn_queues_restore_asr_transcript_and_discussion_update() -> None:
    from flowlens.integration.worker_runtime import (
        MultiprocessingWorkerRuntime,
        WorkerTargets,
    )

    runtime = MultiprocessingWorkerRuntime(
        context=multiprocessing.get_context("spawn"),
        worker_targets=WorkerTargets(
            writer=_queue_worker,
            audio=_queue_worker,
            asr=_restart_asr_worker,
            discussion=_restart_discussion_worker,
        ),
        join_timeout_seconds=0.5,
    )
    launch_value = make_launch()
    runtime.start_all(launch_value)

    def wait_for(
        source: ProcessSource,
        message_type: MessageType,
    ) -> MessageEnvelope[object]:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for message in runtime.poll():
                if message.source is source and message.message_type is message_type:
                    return message
            time.sleep(0.01)
        raise AssertionError(
            f"timed out waiting for {source.value} {message_type.value}"
        )

    wait_for(ProcessSource.ASR, MessageType.WORKER_READY)
    runtime.send(
        ProcessSource.ASR,
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_START,
            1,
            ProcessSource.GUI,
            1_000,
            {"worker": "ASR"},
        ),
    )
    first_transcript = wait_for(ProcessSource.ASR, MessageType.TRANSCRIPT_COMMITTED)
    assert TranscriptRecord.from_dict(first_transcript.payload).sequence == 1

    recovered_asr = replace(
        launch_value.asr_config,
        allow_nonzero_initial_sample=True,
        initial_transcript_sequence=2,
    )
    launch_value = replace(launch_value, asr_config=recovered_asr)
    runtime.restart(ProcessSource.ASR, launch_value)
    wait_for(ProcessSource.ASR, MessageType.WORKER_READY)
    runtime.send(
        ProcessSource.ASR,
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_START,
            1,
            ProcessSource.GUI,
            1_000,
            {"worker": "ASR"},
        ),
    )
    second_transcript = wait_for(ProcessSource.ASR, MessageType.TRANSCRIPT_COMMITTED)
    second_record = TranscriptRecord.from_dict(second_transcript.payload)
    second_payload = TranscriptCommitted(second_record)
    assert second_payload.record.sequence == 2

    runtime.send(
        ProcessSource.DISCUSSION,
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_START,
            1,
            ProcessSource.GUI,
            1_000,
            {"worker": "DISCUSSION"},
        ),
    )
    wait_for(ProcessSource.DISCUSSION, MessageType.WORKER_READY)
    runtime.send(
        ProcessSource.DISCUSSION,
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.TRANSCRIPT_COMMITTED,
            2,
            ProcessSource.GUI,
            1_001,
            second_payload,
        ),
    )
    first_update = wait_for(
        ProcessSource.DISCUSSION,
        MessageType.DISCUSSION_STATE_REPLACED,
    )
    assert isinstance(first_update.payload, DiscussionStateReplaced)
    recovered_state = first_update.payload.state

    recovered_discussion = replace(
        launch_value.discussion_config,
        initial_state=recovered_state,
    )
    launch_value = replace(
        launch_value,
        initial_state=recovered_state,
        discussion_config=recovered_discussion,
    )
    runtime.restart(ProcessSource.DISCUSSION, launch_value)
    runtime.send(
        ProcessSource.DISCUSSION,
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_START,
            1,
            ProcessSource.GUI,
            1_000,
            {"worker": "DISCUSSION"},
        ),
    )
    wait_for(ProcessSource.DISCUSSION, MessageType.WORKER_READY)
    runtime.send(
        ProcessSource.DISCUSSION,
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.TRANSCRIPT_COMMITTED,
            2,
            ProcessSource.GUI,
            1_001,
            second_payload,
        ),
    )
    second_update = wait_for(
        ProcessSource.DISCUSSION,
        MessageType.DISCUSSION_STATE_REPLACED,
    )
    assert isinstance(second_update.payload, DiscussionStateReplaced)
    assert second_update.payload.state.revision == recovered_state.revision + 1

    runtime.shutdown()


@pytest.mark.parametrize(
    ("target", "process_name"),
    [
        (ProcessSource.ASR, "FlowLens-ASR"),
        (ProcessSource.DISCUSSION, "FlowLens-Discussion"),
    ],
)
def test_restart_start_failure_cleans_unstarted_handles_and_allows_new_session(
    target: ProcessSource,
    process_name: str,
) -> None:
    context = FakeMultiprocessingContext()
    context.reject_unstarted_lifecycle = True
    runtime = make_runtime(context=context)
    runtime.start_all(make_launch())
    old_process = runtime.processes[target]
    old_queue = runtime.control_queues[target]
    context.fail_start_names.add(process_name)

    with pytest.raises(RuntimeError, match=f"{target.value} restart failed"):
        runtime.restart(target)

    fresh_process = context.processes[4]
    fresh_queue = context.queues[7]
    report = runtime.shutdown_report
    assert report is not None
    assert runtime.shutdown() is report
    assert target not in runtime.processes
    assert target not in runtime.control_queues
    assert fresh_process.started is False
    assert fresh_process.terminated is False
    assert fresh_process.closed is True
    assert fresh_process.join_calls == []
    assert context.unstarted_lifecycle_calls == []
    assert fresh_queue.closed is True
    assert fresh_queue.joined is True
    assert old_process.closed is True
    assert old_queue.closed is True
    assert any(
        f"{target.value} restart start: OSError" in error for error in report.errors
    )

    context.fail_start_names.clear()
    runtime.start_all(make_launch("02J00000000000000000000000"))

    assert len(runtime.processes) == 4
    assert runtime.processes[target] is not old_process


@pytest.mark.parametrize(
    ("target", "process_name"),
    [
        (ProcessSource.ASR, "FlowLens-ASR"),
        (ProcessSource.DISCUSSION, "FlowLens-Discussion"),
    ],
)
def test_restart_start_failure_reports_fresh_cleanup_exceptions(
    target: ProcessSource,
    process_name: str,
) -> None:
    context = FakeMultiprocessingContext()
    runtime = make_runtime(context=context)
    runtime.start_all(make_launch())
    context.fail_start_names.add(process_name)
    context.fail_process_close_indices.add(4)
    context.fail_queue_close_indices.add(7)

    with pytest.raises(RuntimeError, match=f"{target.value} restart failed"):
        runtime.restart(target)

    report = runtime.shutdown_report
    assert report is not None
    assert target not in runtime.processes
    assert target not in runtime.control_queues
    assert any(
        f"{target.value} restart start: OSError" in error for error in report.errors
    )
    assert any(
        f"{target.value} restart close: OSError" in error for error in report.errors
    )
    assert any("queue close: OSError" in error for error in report.errors)
    context.fail_start_names.clear()
    with pytest.raises(RuntimeError, match="incomplete restart cleanup"):
        runtime.start_all(make_launch("02J00000000000000000000000"))
    assert runtime.shutdown() is report


def test_restart_start_failure_closes_fresh_queue_when_child_remains_alive() -> None:
    context = FakeMultiprocessingContext()
    runtime = make_runtime(context=context)
    runtime.start_all(make_launch())
    context.fail_start_after_start_names.add("FlowLens-ASR")
    context.remain_alive_after_terminate_indices.add(4)

    with pytest.raises(RuntimeError, match="ASR restart failed"):
        runtime.restart(ProcessSource.ASR)

    fresh_process = context.processes[4]
    fresh_queue = context.queues[7]
    report = runtime.shutdown_report
    assert report is not None
    assert fresh_process.started is True
    assert fresh_process.terminated is True
    assert fresh_process.closed is False
    assert fresh_queue.closed is True
    assert ProcessSource.ASR not in runtime.processes
    assert ProcessSource.ASR not in runtime.control_queues
    assert any(
        "ASR remains alive after bounded termination" in error
        for error in report.errors
    )
    assert runtime.shutdown() is report


def test_shutdown_sends_typed_controls_joins_bounded_and_never_completes() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())
    runtime.send(
        ProcessSource.WRITER,
        MessageEnvelope(
            schema_version=1,
            session_id=SESSION_ID,
            message_type=MessageType.WRITER_SHUTDOWN,
            sequence=3,
            source=ProcessSource.GUI,
            created_monotonic_ms=1_000,
            payload=WriterShutdown(),
        ),
    )

    report = runtime.safe_stop()
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
    assert writer_control.sequence == 4
    assert isinstance(runtime.writer_audio_queue.items[-1], AudioDrainFence)
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
    assert runtime.processes[ProcessSource.WRITER].join_calls == [0.25, 0.25, 0.25]
    assert all(
        process.join_calls == [0.25, 0.25]
        for worker, process in runtime.processes.items()
        if worker is not ProcessSource.WRITER
    )
    assert all(process.closed is True for process in runtime.processes.values())
    assert all(queue.closed and queue.joined for queue in runtime.context.queues)


def test_normal_shutdown_does_not_inject_a_duplicate_audio_fence() -> None:
    runtime = make_runtime()
    runtime.start_all(make_launch())

    runtime.shutdown()

    assert not any(
        isinstance(item, AudioDrainFence) for item in runtime.writer_audio_queue.items
    )


def test_safe_stop_relies_on_a_gracefully_stopped_audio_worker_fence() -> None:
    context = FakeMultiprocessingContext()
    context.exit_on_join = True
    runtime = make_runtime(context=context)
    runtime.start_all(make_launch())

    runtime.safe_stop()

    assert not any(
        isinstance(item, AudioDrainFence) for item in runtime.writer_audio_queue.items
    )


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
