"""Writer shared-gate terminal linearization tests."""

import os
import random
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent
from threading import Thread

import pytest

from flowlens.domain.enums import EventType, MessageType
from flowlens.domain.messages import (
    AudioDrainFence,
    EventRecord,
    MessageEnvelope,
    WriterFinalize,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterForceCloseResult,
)
from flowlens.domain.session import SessionManifest
from flowlens.persistence.session_writer import PreparedFinalization
from flowlens.workers.finalization_gate import WriterFinalizationGate
from flowlens.workers.writer import run_writer_worker
from tests.factories import make_event_record
from tests.workers.writer_support import (
    _as_process_event,
    _as_process_queue,
    _FakeSessionWriter,
    _install_fake_writer,
    _RecordingResponseQueue,
    make_finalize_envelope,
    make_open_envelope,
    make_shutdown_envelope,
)


def _force_request() -> WriterForceCloseRequest:
    return WriterForceCloseRequest(
        replace(
            make_event_record(sequence=1),
            event_type=EventType.FORCE_CLOSE_REQUESTED,
        )
    )


def _force_results(
    responses: _RecordingResponseQueue,
) -> list[WriterForceCloseResult]:
    return [
        envelope.payload
        for item in responses.items
        if isinstance(item, MessageEnvelope)
        and (envelope := item).message_type is MessageType.WRITER_FORCE_CLOSE_RESULT
        and isinstance(envelope.payload, WriterForceCloseResult)
    ]


def _run_worker(
    control: ThreadQueue[object],
    audio: ThreadQueue[object],
    responses: _RecordingResponseQueue,
    stop: ThreadEvent,
    gate: WriterFinalizationGate,
    errors: list[BaseException],
) -> None:
    try:
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
            gate,
        )
    except BaseException as error:
        errors.append(error)


def _exit_holding_gate_lock(gate: WriterFinalizationGate) -> None:
    gate._lock.acquire()
    os._exit(0)


def _claim_completion_and_exit(
    gate: WriterFinalizationGate,
    completion_event: EventRecord,
) -> None:
    gate.claim_terminal(completion_event)
    os._exit(0)


def _send_force_after(delay: float, gate: WriterFinalizationGate) -> None:
    time.sleep(delay)
    gate.request_force_close(_force_request(), timeout_seconds=0.1)


def test_force_before_terminal_claim_uses_one_candidate_without_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    gate = WriterFinalizationGate.create(get_context("spawn"))
    gate.request_force_close(_force_request(), timeout_seconds=0.1)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_finalize_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(ThreadEvent()),
        gate,
    )

    assert fake.operations == [("event", 1), ("force_sync", 0), ("close", 1)]
    result = gate.result()
    assert result is not None
    assert result.outcome is WriterForceCloseOutcome.INCOMPLETE
    assert [item.outcome for item in _force_results(responses)] == [
        WriterForceCloseOutcome.INCOMPLETE
    ]


def test_finalize_claim_wins_without_persisting_late_force_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = WriterFinalizationGate.create(get_context("spawn"))

    class FinalizeThenForceWriter(_FakeSessionWriter):
        def commit_finalize(self, prepared: PreparedFinalization) -> SessionManifest:
            result = super().commit_finalize(prepared)
            assert (
                gate.request_force_close(_force_request(), timeout_seconds=0.1) is None
            )
            return result

    fake = FinalizeThenForceWriter()
    _install_fake_writer(monkeypatch, fake)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_finalize_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(ThreadEvent()),
        gate,
    )

    assert fake.operations[:2] == [("prepare_finalize", 0), ("finalize", 0)]
    assert not any(item[0] in {"event", "force_sync"} for item in fake.operations)
    result = gate.result()
    assert result is not None
    assert result.outcome is WriterForceCloseOutcome.COMPLETED
    assert _force_results(responses) == []


def test_force_during_slow_prepare_wins_at_short_commit_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preparing = ThreadEvent()
    release = ThreadEvent()

    class SlowPrepareWriter(_FakeSessionWriter):
        def prepare_finalize(self, command: WriterFinalize) -> PreparedFinalization:
            preparing.set()
            assert release.wait(timeout=2)
            return super().prepare_finalize(command)

    fake = SlowPrepareWriter()
    _install_fake_writer(monkeypatch, fake)
    gate = WriterFinalizationGate.create(get_context("spawn"))
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    errors: list[BaseException] = []
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_finalize_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))
    audio.put(AudioDrainFence())
    worker = Thread(
        target=_run_worker,
        args=(control, audio, responses, ThreadEvent(), gate, errors),
    )
    worker.start()
    assert preparing.wait(timeout=2)
    gate.request_force_close(_force_request(), timeout_seconds=0.1)
    release.set()
    worker.join(timeout=2)

    assert errors == []
    assert fake.operations == [
        ("prepare_finalize", 0),
        ("event", 1),
        ("force_sync", 0),
        ("close", 1),
    ]
    result = gate.result()
    assert result is not None
    assert result.outcome is WriterForceCloseOutcome.INCOMPLETE


def test_hung_commit_does_not_hold_gate_or_publish_result_early(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    committing = ThreadEvent()
    release = ThreadEvent()

    class HungCommitWriter(_FakeSessionWriter):
        def commit_finalize(self, prepared: PreparedFinalization) -> SessionManifest:
            committing.set()
            assert release.wait(timeout=2)
            return super().commit_finalize(prepared)

    fake = HungCommitWriter()
    _install_fake_writer(monkeypatch, fake)
    gate = WriterFinalizationGate.create(get_context("spawn"))
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    errors: list[BaseException] = []
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_finalize_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))
    audio.put(AudioDrainFence())
    worker = Thread(
        target=_run_worker,
        args=(control, audio, responses, ThreadEvent(), gate, errors),
    )
    worker.start()
    assert committing.wait(timeout=2)

    assert gate.request_force_close(_force_request(), timeout_seconds=0.05) is None
    assert gate.result() is None
    release.set()
    worker.join(timeout=2)

    assert errors == []
    result = gate.result()
    assert result is not None
    assert result.outcome is WriterForceCloseOutcome.COMPLETED
    assert not any(item[0] in {"event", "force_sync"} for item in fake.operations)


def test_randomized_force_and_finalize_claim_has_one_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = random.Random(19)
    for iteration in range(50):
        fake = _FakeSessionWriter()
        _install_fake_writer(monkeypatch, fake)
        gate = WriterFinalizationGate.create(get_context("spawn"))
        control: ThreadQueue[object] = ThreadQueue()
        audio: ThreadQueue[object] = ThreadQueue()
        responses = _RecordingResponseQueue()
        errors: list[BaseException] = []
        control.put(make_open_envelope(tmp_path / f"session-{iteration}", sequence=1))
        control.put(make_finalize_envelope(sequence=2))
        control.put(make_shutdown_envelope(sequence=3))
        audio.put(AudioDrainFence())
        worker = Thread(
            target=_run_worker,
            args=(control, audio, responses, ThreadEvent(), gate, errors),
        )
        delay = generator.random() / 10_000
        force = Thread(target=_send_force_after, args=(delay, gate))
        worker.start()
        force.start()
        worker.join(timeout=2)
        force.join(timeout=2)

        assert errors == []
        result = gate.result()
        assert result is not None
        terminal_events = [item for item in fake.operations if item[0] == "event"]
        if result.outcome is WriterForceCloseOutcome.INCOMPLETE:
            assert terminal_events == [("event", 1)]
            assert ("finalize", 0) not in fake.operations
        else:
            assert terminal_events == []
            assert ("finalize", 0) in fake.operations


def test_dead_process_holding_short_gate_lock_times_out_boundedly() -> None:
    context = get_context("spawn")
    gate = WriterFinalizationGate.create(context)
    process = context.Process(target=_exit_holding_gate_lock, args=(gate,))
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="gate request timed out"):
        gate.request_force_close(_force_request(), timeout_seconds=0.05)

    assert time.monotonic() - started < 1
    assert gate.result() is None


def test_process_death_after_claim_never_publishes_false_completion() -> None:
    context = get_context("spawn")
    gate = WriterFinalizationGate.create(context)
    finalize = make_finalize_envelope(sequence=2).payload
    assert type(finalize) is WriterFinalize
    process = context.Process(
        target=_claim_completion_and_exit,
        args=(gate, finalize.completion_event),
    )
    process.start()
    process.join(timeout=5)

    assert process.exitcode == 0
    assert gate.request_force_close(_force_request(), timeout_seconds=0.05) is None
    assert gate.result() is None
