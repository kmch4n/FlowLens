"""Discussion worker core and bounded queue-loop tests."""

from __future__ import annotations

import pickle
import queue
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from flowlens.discussion.contracts import (
    ChatMessage,
    DiscussionBackend,
    DiscussionStatusPayload,
    DiscussionStoppedPayload,
)
from flowlens.discussion.llama_cpp_adapter import DiscussionModelConfig
from flowlens.discussion.worker import (
    DiscussionWorkerConfig,
    DiscussionWorkerCore,
    DiscussionWorkerProtocolError,
    _discussion_worker_loop,
    run_discussion_worker,
)
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    DiscussionStateReplaced,
    MessageEnvelope,
    MessageSequenceError,
    TranscriptCommitted,
)
from tests.discussion.factories import NOW, make_record, make_state

SESSION_ID = "01J00000000000000000000000"


class FakeBackend:
    """Deterministic local backend with observable generation calls."""

    def __init__(self, outputs: list[str | BaseException]) -> None:
        self._outputs = outputs
        self.generate_calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text)

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        response_schema: dict[str, object],
    ) -> str:
        del messages, response_schema
        self.generate_calls += 1
        output = self._outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


class FakeClock:
    """Injectable monotonic and aware wall clocks."""

    def __init__(self, monotonic_ms: int = 0) -> None:
        self.monotonic_ms = monotonic_ms

    def monotonic(self) -> int:
        return self.monotonic_ms

    def wall(self) -> datetime:
        return NOW


class FakeQueue:
    """Bounded-get queue double with observable timeout and cleanup."""

    def __init__(self, values: list[object] | None = None) -> None:
        self.values = list(values or [])
        self.put_values: list[object] = []
        self.timeouts: list[float | None] = []
        self.close_calls = 0
        self.join_calls = 0

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        del block
        self.timeouts.append(timeout)
        if not self.values:
            raise queue.Empty
        return self.values.pop(0)

    def put(self, value: object) -> None:
        self.put_values.append(value)

    def close(self) -> None:
        self.close_calls += 1

    def join_thread(self) -> None:
        self.join_calls += 1


def _model_config(tmp_path: Path) -> DiscussionModelConfig:
    model = (tmp_path / "model.gguf").resolve()
    model.write_bytes(b"model")
    return DiscussionModelConfig(model, "0" * 64)


def _config(tmp_path: Path, *, coalesce_ms: int = 500) -> DiscussionWorkerConfig:
    return DiscussionWorkerConfig(
        session_id=SESSION_ID,
        model=_model_config(tmp_path),
        initial_state=make_state(revision=0),
        coalesce_ms=coalesce_ms,
    )


def _valid_output(revision: int = 1) -> str:
    return (
        f'{{"revision":{revision},"mode":"MEETING","current_focus":"Scope",'
        '"key_points":[],"confirmed_outcomes":[],"follow_up_items":[],'
        '"updated_at":"2026-08-19T12:35:02.125+09:00"}'
    )


def _envelope(
    message_type: MessageType,
    sequence: int,
    payload: object,
    *,
    session_id: str = SESSION_ID,
    source: ProcessSource = ProcessSource.GUI,
    created_monotonic_ms: int = 0,
    schema_version: int = 1,
) -> MessageEnvelope[object]:
    return MessageEnvelope(
        schema_version,
        session_id,
        message_type,
        sequence,
        source,
        created_monotonic_ms,
        payload,
    )


def _start(sequence: int = 1) -> MessageEnvelope[object]:
    return _envelope(
        MessageType.WORKER_START,
        sequence,
        {"worker": "DISCUSSION"},
    )


def _commit(
    sequence: int,
    record_sequence: int,
    text: str = "方針を確認します",
    *,
    at_ms: int | None = None,
) -> MessageEnvelope[object]:
    return _envelope(
        MessageType.TRANSCRIPT_COMMITTED,
        sequence,
        TranscriptCommitted(make_record(sequence=record_sequence, text=text)),
        created_monotonic_ms=(record_sequence - 1) * 100 if at_ms is None else at_ms,
    )


def _make_core(
    tmp_path: Path,
    outputs: list[str | BaseException],
) -> tuple[DiscussionWorkerCore, FakeBackend, FakeClock]:
    backend = FakeBackend(outputs)
    clock = FakeClock()
    core = DiscussionWorkerCore(
        _config(tmp_path),
        backend_loader=lambda _config: backend,
        monotonic_ms=clock.monotonic,
        wall_clock=clock.wall,
    )
    return core, backend, clock


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": "bad"},
        {"session_id": True},
        {"model": object()},
        {"initial_state": object()},
        {"coalesce_ms": 0},
        {"coalesce_ms": True},
    ],
)
def test_config_rejects_invalid_contract_values(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "session_id": SESSION_ID,
        "model": _model_config(tmp_path),
        "initial_state": make_state(),
        "coalesce_ms": 500,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        DiscussionWorkerConfig(**values)  # type: ignore[arg-type]


def test_config_core_and_entrypoint_are_spawn_picklable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert pickle.loads(pickle.dumps(config)) == config
    assert pickle.loads(pickle.dumps(run_discussion_worker)).__name__ == (
        "run_discussion_worker"
    )


def test_start_loads_once_and_ready_uses_discussion_sender_sequence(
    tmp_path: Path,
) -> None:
    core, backend, clock = _make_core(tmp_path, [_valid_output()])
    clock.monotonic_ms = 77

    outgoing = core.handle(_start())

    assert len(outgoing) == 1
    assert outgoing[0] == MessageEnvelope(
        1,
        SESSION_ID,
        MessageType.WORKER_READY,
        1,
        ProcessSource.DISCUSSION,
        77,
        {"worker": "DISCUSSION"},
    )
    with pytest.raises(DiscussionWorkerProtocolError, match="state"):
        core.handle(_envelope(MessageType.WORKER_START, 2, {"worker": "DISCUSSION"}))
    assert backend.generate_calls == 0


def test_committed_message_generates_replacement_then_status_after_coalesce(
    tmp_path: Path,
) -> None:
    core, _backend, clock = _make_core(tmp_path, [_valid_output()])
    core.handle(_start())
    core.handle(_commit(2, 1))
    assert core.tick(499, NOW) == ()
    clock.monotonic_ms = 500

    outgoing = core.tick(500, NOW)

    assert [item.message_type for item in outgoing] == [
        MessageType.DISCUSSION_STATE_REPLACED,
        MessageType.DISCUSSION_STATUS,
    ]
    replacement = outgoing[0]
    assert replacement.sequence == 2
    assert replacement.created_monotonic_ms == 500
    assert isinstance(replacement.payload, DiscussionStateReplaced)
    assert replacement.payload.previous_revision == 0
    assert replacement.payload.state.revision == 1
    assert outgoing[1].payload == DiscussionStatusPayload(
        state="UPDATED",
        revision=1,
        pending_count=0,
        error_code=None,
    )


def test_invalid_output_emits_metadata_only_failure_and_retries_after_new_commit(
    tmp_path: Path,
) -> None:
    core, backend, _clock = _make_core(
        tmp_path,
        ["transcript-secret invalid", _valid_output()],
    )
    core.handle(_start())
    core.handle(_commit(2, 1, "transcript-secret"))
    failed = core.tick(500, NOW)

    assert failed[0].payload == DiscussionStatusPayload(
        state="FAILED",
        revision=0,
        pending_count=1,
        error_code="INVALID_OUTPUT",
    )
    assert "transcript-secret" not in repr(failed)
    assert core.state.revision == 0
    assert core.tick(10_000, NOW) == ()
    core.handle(_commit(3, 2, "新しい入力", at_ms=10_000))
    assert core.tick(10_499, NOW) == ()
    assert core.tick(10_500, NOW)[0].message_type is (
        MessageType.DISCUSSION_STATE_REPLACED
    )
    assert backend.generate_calls == 2


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RuntimeError("prompt transcript path C:/secret"), "GENERATION_FAILED"),
        (MemoryError("C:/model.gguf transcript"), "GENERATION_FAILED"),
    ],
)
def test_generation_failures_are_metadata_only(
    tmp_path: Path,
    error: BaseException,
    code: str,
) -> None:
    core, _backend, _clock = _make_core(tmp_path, [error])
    core.handle(_start())
    core.handle(_commit(2, 1, "private transcript"))

    outgoing = core.tick(500, NOW)

    assert outgoing[0].payload == DiscussionStatusPayload(
        state="FAILED",
        revision=0,
        pending_count=1,
        error_code=code,
    )
    assert "private" not in repr(outgoing)
    assert "secret" not in repr(outgoing)


def test_pause_defers_generation_and_resume_preserves_pending(tmp_path: Path) -> None:
    core, _backend, _clock = _make_core(tmp_path, [_valid_output()])
    core.handle(_start())
    core.handle(_envelope(MessageType.WORKER_PAUSE, 2, {"worker": "DISCUSSION"}))
    core.handle(_commit(3, 1, "保留"))
    assert core.tick(5_000, NOW) == ()
    core.handle(_envelope(MessageType.WORKER_RESUME, 4, {"worker": "DISCUSSION"}))
    assert core.tick(5_000, NOW)[0].message_type is (
        MessageType.DISCUSSION_STATE_REPLACED
    )


def test_stop_runs_one_final_request_then_reports_drained(tmp_path: Path) -> None:
    core, backend, _clock = _make_core(tmp_path, [_valid_output()])
    core.handle(_start())
    core.handle(_commit(2, 1, "最終入力"))

    outgoing = core.handle(
        _envelope(
            MessageType.WORKER_STOP,
            3,
            {"worker": "DISCUSSION", "finalize": True},
        )
    )

    assert [message.message_type for message in outgoing] == [
        MessageType.DISCUSSION_STATE_REPLACED,
        MessageType.DISCUSSION_STATUS,
        MessageType.WORKER_STOPPED,
    ]
    assert outgoing[-1].payload == DiscussionStoppedPayload(
        worker="DISCUSSION",
        drained=True,
        final_revision=1,
        pending_count=0,
    )
    assert (
        core.handle(
            _envelope(
                MessageType.WORKER_STOP,
                4,
                {"worker": "DISCUSSION", "finalize": True},
            )
        )
        == ()
    )
    assert backend.generate_calls == 1


def test_stop_after_failure_retains_exact_pending_count(tmp_path: Path) -> None:
    core, backend, _clock = _make_core(tmp_path, ["bad"])
    core.handle(_start())
    core.handle(_commit(2, 1, "one"))
    core.handle(_commit(3, 2, "two"))
    core.tick(600, NOW)

    outgoing = core.handle(
        _envelope(
            MessageType.WORKER_STOP,
            4,
            {"worker": "DISCUSSION", "finalize": True},
        )
    )

    assert outgoing == (
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_STOPPED,
            3,
            ProcessSource.DISCUSSION,
            0,
            DiscussionStoppedPayload("DISCUSSION", True, 0, 2),
        ),
    )
    assert backend.generate_calls == 1


@pytest.mark.parametrize(
    "envelope",
    [
        _envelope(
            MessageType.WORKER_START,
            1,
            {"worker": "DISCUSSION"},
            session_id="01J00000000000000000000001",
        ),
        _envelope(
            MessageType.WORKER_START,
            1,
            {"worker": "DISCUSSION"},
            source=ProcessSource.ASR,
        ),
        _envelope(MessageType.WORKER_START, 1, {"worker": "ASR"}),
        _envelope(MessageType.WORKER_START, 1, {"worker": "DISCUSSION", "extra": True}),
        _envelope(
            MessageType.WORKER_START, 1, {"worker": "DISCUSSION"}, schema_version=2
        ),
        _envelope(MessageType.TRANSCRIPT_COMMITTED, 1, {"record": {}}),
    ],
)
def test_invalid_envelope_does_not_consume_sender_sequence(
    tmp_path: Path,
    envelope: MessageEnvelope[object],
) -> None:
    core, _backend, _clock = _make_core(tmp_path, [_valid_output()])
    with pytest.raises((DiscussionWorkerProtocolError, ValueError)):
        core.handle(envelope)
    assert core.handle(_start())[0].message_type is MessageType.WORKER_READY


@pytest.mark.parametrize(
    "message_type",
    [
        MessageType.WORKER_START,
        MessageType.WORKER_PAUSE,
        MessageType.WORKER_RESUME,
        MessageType.WORKER_STOP,
    ],
)
@pytest.mark.parametrize(
    "malformation",
    ["dict-subclass", "extra-key", "string-subclass", "non-string-worker"],
)
def test_each_lifecycle_payload_requires_exact_builtin_types_and_keys(
    tmp_path: Path,
    message_type: MessageType,
    malformation: str,
) -> None:
    class DictSubclass(dict[str, object]):
        pass

    class StringSubclass(str):
        pass

    core, _backend, _clock = _make_core(tmp_path, [_valid_output()])
    if message_type is MessageType.WORKER_START:
        sequence = 1
    else:
        core.handle(_start())
        sequence = 2
        if message_type is MessageType.WORKER_RESUME:
            core.handle(
                _envelope(
                    MessageType.WORKER_PAUSE,
                    sequence,
                    {"worker": "DISCUSSION"},
                )
            )
            sequence += 1

    valid_payload: dict[str, object] = {"worker": "DISCUSSION"}
    if message_type is MessageType.WORKER_STOP:
        valid_payload["finalize"] = True
    if malformation == "dict-subclass":
        invalid_payload: object = DictSubclass(valid_payload)
    elif malformation == "extra-key":
        invalid_payload = {**valid_payload, "extra": None}
    elif malformation == "string-subclass":
        invalid_payload = {
            **valid_payload,
            "worker": StringSubclass("DISCUSSION"),
        }
    else:
        invalid_payload = {**valid_payload, "worker": 1}

    with pytest.raises(DiscussionWorkerProtocolError, match="payload"):
        core.handle(_envelope(message_type, sequence, invalid_payload))

    core.handle(_envelope(message_type, sequence, valid_payload))


def test_stop_payload_requires_finalize_to_be_the_true_singleton(
    tmp_path: Path,
) -> None:
    core, _backend, _clock = _make_core(tmp_path, [_valid_output()])
    core.handle(_start())

    with pytest.raises(DiscussionWorkerProtocolError, match="payload"):
        core.handle(
            _envelope(
                MessageType.WORKER_STOP,
                2,
                {"worker": "DISCUSSION", "finalize": 1},
            )
        )

    outgoing = core.handle(
        _envelope(
            MessageType.WORKER_STOP,
            2,
            {"worker": "DISCUSSION", "finalize": True},
        )
    )
    assert outgoing[-1].message_type is MessageType.WORKER_STOPPED


def test_duplicate_and_gap_fail_without_corrupting_sequence(tmp_path: Path) -> None:
    core, _backend, _clock = _make_core(tmp_path, [_valid_output()])
    core.handle(_start())
    with pytest.raises(MessageSequenceError, match="expected.*2"):
        core.handle(_commit(1, 1))
    with pytest.raises(MessageSequenceError, match="expected.*2"):
        core.handle(_commit(3, 1))
    core.handle(_commit(2, 1))


def test_record_sequence_failure_does_not_consume_envelope_sequence(
    tmp_path: Path,
) -> None:
    core, _backend, _clock = _make_core(tmp_path, [_valid_output()])
    core.handle(_start())
    core.handle(_commit(2, 2, at_ms=200))
    with pytest.raises(ValueError, match="record sequence"):
        core.handle(_commit(3, 1, at_ms=5_000))
    core.handle(_commit(3, 3, at_ms=300))


@pytest.mark.parametrize(
    ("loader_error", "code"),
    [
        (RuntimeError("C:/models/private.gguf failed"), "MODEL_LOAD_FAILED"),
        (MemoryError("CUDA out of memory at C:/models/private.gguf"), "GPU_OOM"),
        (RuntimeError("CUDA out of memory"), "GPU_OOM"),
    ],
)
def test_load_failure_is_sanitized_and_terminal(
    tmp_path: Path,
    loader_error: BaseException,
    code: str,
) -> None:
    clock = FakeClock(91)

    def fail_loader(_config: DiscussionModelConfig) -> DiscussionBackend:
        raise loader_error

    core = DiscussionWorkerCore(
        _config(tmp_path),
        backend_loader=fail_loader,
        monotonic_ms=clock.monotonic,
        wall_clock=clock.wall,
    )

    outgoing = core.handle(_start())

    assert outgoing[0].message_type is MessageType.WORKER_ERROR
    assert outgoing[0].payload == {"worker": "DISCUSSION", "code": code}
    assert "private" not in repr(outgoing)
    with pytest.raises(DiscussionWorkerProtocolError, match="state"):
        core.handle(_commit(2, 1))


def test_queue_loop_load_failure_emits_once_and_exits(tmp_path: Path) -> None:
    control = FakeQueue([_start()])
    output = FakeQueue()

    def fail_loader(_config: DiscussionModelConfig) -> DiscussionBackend:
        raise RuntimeError("C:/models/private.gguf failed")

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=fail_loader,
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
        parent_alive=lambda: True,
        poll_timeout_seconds=0.01,
    )

    assert result == 1
    assert len(output.put_values) == 1
    fatal = cast(MessageEnvelope[object], output.put_values[0])
    assert fatal.message_type is MessageType.WORKER_ERROR
    assert fatal.payload == {
        "worker": "DISCUSSION",
        "code": "MODEL_LOAD_FAILED",
    }
    assert "private" not in repr(fatal)
    assert output.close_calls == 1
    assert output.join_calls == 1


def test_load_failure_code_does_not_trust_exception_text(tmp_path: Path) -> None:
    class HostileLoadError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("secret path escaped through __str__")

    def fail_loader(_config: DiscussionModelConfig) -> DiscussionBackend:
        raise HostileLoadError()

    core = DiscussionWorkerCore(
        _config(tmp_path),
        backend_loader=fail_loader,
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
    )

    outgoing = core.handle(_start())

    assert outgoing[0].payload == {
        "worker": "DISCUSSION",
        "code": "MODEL_LOAD_FAILED",
    }


@pytest.mark.parametrize("control_error", [SystemExit(17), KeyboardInterrupt()])
def test_model_loader_does_not_capture_process_control_exceptions(
    tmp_path: Path,
    control_error: BaseException,
) -> None:
    def fail_loader(_config: DiscussionModelConfig) -> DiscussionBackend:
        raise control_error

    core = DiscussionWorkerCore(
        _config(tmp_path),
        backend_loader=fail_loader,
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
    )

    with pytest.raises(type(control_error)):
        core.handle(_start())


@pytest.mark.parametrize("control_error", [SystemExit(17), KeyboardInterrupt()])
def test_generation_does_not_capture_process_control_exceptions(
    tmp_path: Path,
    control_error: BaseException,
) -> None:
    core, _backend, _clock = _make_core(tmp_path, [control_error])
    core.handle(_start())
    core.handle(_commit(2, 1))

    with pytest.raises(type(control_error)):
        core.tick(500, NOW)


@pytest.mark.parametrize("control_error", [SystemExit(17), KeyboardInterrupt()])
@pytest.mark.parametrize("failure_stage", ["load", "generation"])
def test_queue_loop_propagates_process_control_exceptions_and_cleans_up(
    tmp_path: Path,
    control_error: BaseException,
    failure_stage: str,
) -> None:
    controls: list[object] = [_start()]
    if failure_stage == "generation":
        controls.append(_commit(2, 1))
    control = FakeQueue(controls)
    output = FakeQueue()

    def fail_loader(_config: DiscussionModelConfig) -> DiscussionBackend:
        if failure_stage == "load":
            raise control_error
        return FakeBackend([control_error])

    with pytest.raises(type(control_error)):
        _discussion_worker_loop(
            _config(tmp_path),
            control,
            output,
            backend_loader=fail_loader,
            monotonic_ms=lambda: 500,
            wall_clock=lambda: NOW,
            parent_alive=lambda: True,
            poll_timeout_seconds=0.01,
        )

    expected_types = [] if failure_stage == "load" else [MessageType.WORKER_READY]
    assert [
        cast(MessageEnvelope[object], item).message_type for item in output.put_values
    ] == expected_types
    assert output.close_calls == 1
    assert output.join_calls == 1


def test_queue_loop_polls_boundedly_and_stops_with_cleanup(tmp_path: Path) -> None:
    control = FakeQueue(
        [
            _start(),
            _envelope(
                MessageType.WORKER_STOP,
                2,
                {"worker": "DISCUSSION", "finalize": True},
            ),
        ]
    )
    output = FakeQueue()
    backend = FakeBackend([_valid_output()])

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=lambda _config: backend,
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
        parent_alive=lambda: True,
        poll_timeout_seconds=0.05,
    )

    assert result == 0
    assert control.timeouts and max(cast(list[float], control.timeouts)) <= 0.05
    assert [
        cast(MessageEnvelope[object], item).message_type for item in output.put_values
    ] == [
        MessageType.WORKER_READY,
        MessageType.WORKER_STOPPED,
    ]
    assert output.close_calls == 1
    assert output.join_calls == 1


def test_queue_loop_parent_death_never_starts_generation(tmp_path: Path) -> None:
    control = FakeQueue([_start(), _commit(2, 1)])
    output = FakeQueue()
    backend = FakeBackend([_valid_output()])
    parent_checks = iter([True, True, False])

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=lambda _config: backend,
        monotonic_ms=lambda: 500,
        wall_clock=lambda: NOW,
        parent_alive=lambda: next(parent_checks, False),
        poll_timeout_seconds=0.01,
    )

    assert result == 0
    assert backend.generate_calls == 0
    assert all(
        cast(MessageEnvelope[object], item).message_type
        is not MessageType.DISCUSSION_STATE_REPLACED
        for item in output.put_values
    )


def test_queue_loop_parent_death_during_get_skips_final_generation(
    tmp_path: Path,
) -> None:
    parent_state = {"alive": True}

    class ParentDeathQueue(FakeQueue):
        def get(self, block: bool = True, timeout: float | None = None) -> object:
            value = super().get(block=block, timeout=timeout)
            if (
                isinstance(value, MessageEnvelope)
                and value.message_type is MessageType.WORKER_STOP
            ):
                parent_state["alive"] = False
            return value

    control = ParentDeathQueue(
        [
            _start(),
            _commit(2, 1),
            _envelope(
                MessageType.WORKER_STOP,
                3,
                {"worker": "DISCUSSION", "finalize": True},
            ),
        ]
    )
    output = FakeQueue()
    backend = FakeBackend([_valid_output()])

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=lambda _config: backend,
        monotonic_ms=lambda: 500,
        wall_clock=lambda: NOW,
        parent_alive=lambda: parent_state["alive"],
        poll_timeout_seconds=0.01,
    )

    assert result == 0
    assert backend.generate_calls == 0
    assert [
        cast(MessageEnvelope[object], item).message_type for item in output.put_values
    ] == [MessageType.WORKER_READY]


def test_queue_loop_emits_replacement_created_during_parent_death(
    tmp_path: Path,
) -> None:
    parent_state = {"alive": True}

    class ParentDeathBackend(FakeBackend):
        def generate(
            self,
            messages: tuple[ChatMessage, ...],
            response_schema: dict[str, object],
        ) -> str:
            result = super().generate(messages, response_schema)
            parent_state["alive"] = False
            return result

    control = FakeQueue([_start(), _commit(2, 1, at_ms=0)])
    output = FakeQueue()
    backend = ParentDeathBackend([_valid_output()])

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=lambda _config: backend,
        monotonic_ms=lambda: 500,
        wall_clock=lambda: NOW,
        parent_alive=lambda: parent_state["alive"],
        poll_timeout_seconds=0.01,
    )

    assert result == 0
    assert backend.generate_calls == 1
    assert [
        cast(MessageEnvelope[object], item).message_type for item in output.put_values
    ] == [
        MessageType.WORKER_READY,
        MessageType.DISCUSSION_STATE_REPLACED,
        MessageType.DISCUSSION_STATUS,
    ]
    assert output.close_calls == 1
    assert output.join_calls == 1


def test_queue_loop_protocol_error_emits_sanitized_worker_error(tmp_path: Path) -> None:
    control = FakeQueue([_start(), _commit(4, 1, "private transcript")])
    output = FakeQueue()

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=lambda _config: FakeBackend([_valid_output()]),
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
        parent_alive=lambda: True,
        poll_timeout_seconds=0.01,
    )

    assert result == 1
    fatal = cast(MessageEnvelope[object], output.put_values[-1])
    assert fatal.message_type is MessageType.WORKER_ERROR
    assert fatal.payload == {"worker": "DISCUSSION", "code": "PROTOCOL_ERROR"}
    assert "private" not in repr(fatal)
    assert output.close_calls == 1
    assert output.join_calls == 1


def test_queue_loop_closed_input_exits_without_busy_spin(tmp_path: Path) -> None:
    class ClosedQueue(FakeQueue):
        def get(self, block: bool = True, timeout: float | None = None) -> object:
            del block, timeout
            raise EOFError("closed")

    output = FakeQueue()
    result = _discussion_worker_loop(
        _config(tmp_path),
        ClosedQueue(),
        output,
        backend_loader=lambda _config: FakeBackend([_valid_output()]),
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
        parent_alive=lambda: True,
        poll_timeout_seconds=0.01,
    )

    assert result == 0
    assert output.put_values == []
    assert output.close_calls == 1
    assert output.join_calls == 1


def test_queue_loop_cleanup_errors_do_not_escape_or_skip_join(tmp_path: Path) -> None:
    class HostileCleanupQueue(FakeQueue):
        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close failed")

        def join_thread(self) -> None:
            self.join_calls += 1
            raise KeyboardInterrupt("join failed")

    control = FakeQueue(
        [
            _start(),
            _envelope(
                MessageType.WORKER_STOP,
                2,
                {"worker": "DISCUSSION", "finalize": True},
            ),
        ]
    )
    output = HostileCleanupQueue()

    result = _discussion_worker_loop(
        _config(tmp_path),
        control,
        output,
        backend_loader=lambda _config: FakeBackend([_valid_output()]),
        monotonic_ms=lambda: 0,
        wall_clock=lambda: NOW,
        parent_alive=lambda: True,
        poll_timeout_seconds=0.01,
    )

    assert result == 0
    assert output.close_calls == 1
    assert output.join_calls == 1
