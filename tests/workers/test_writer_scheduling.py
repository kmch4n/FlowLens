"""Writer Worker scheduling, fairness, and audio-fence tests."""

from datetime import datetime
from pathlib import Path
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent
from typing import cast

import pytest

from flowlens.domain.enums import AudioSource, MessageType
from flowlens.domain.messages import (
    AudioDrainFence,
    MessageEnvelope,
    WriterAck,
    WriterFatal,
    WriterFlush,
)
from flowlens.persistence.session_writer import PersistenceInvariantError
from flowlens.workers import writer as writer_module
from flowlens.workers.writer import run_writer_worker
from tests.workers.writer_support import (
    _as_process_event,
    _as_process_queue,
    _CrossQueueLagAudioQueue,
    _DelayedFenceQueue,
    _FakeSessionWriter,
    _install_fake_writer,
    _RecordingResponseQueue,
    _ScriptedClock,
    _StopDuringFenceWaitQueue,
    control_envelope,
    make_audio_command,
    make_finalize_envelope,
    make_open_envelope,
    make_shutdown_envelope,
    make_transcript_envelope,
)


def test_transcript_waits_for_cross_queue_audio_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ready transcript must not overtake its earlier audio command."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _CrossQueueLagAudioQueue(
        make_audio_command(source_start_sample=0),
        AudioDrainFence(),
    )
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_transcript_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    audio_index = fake.operations.index(("audio", 0))
    transcript_index = fake.operations.index(("transcript", 1))
    assert audio_index < transcript_index


def test_pending_transcript_runs_when_its_exact_audio_cursor_is_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Later audio must not delay a transcript whose dependency is satisfied."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _CrossQueueLagAudioQueue(
        make_audio_command(source_start_sample=0),
        make_audio_command(source_start_sample=12_800),
        AudioDrainFence(),
    )
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_transcript_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    transcript_index = fake.operations.index(("transcript", 1))
    later_audio_index = fake.operations.index(("audio", 12_800))
    assert transcript_index < later_audio_index


def test_other_source_audio_does_not_release_a_pending_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each transcript dependency must use its own persisted source cursor."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _CrossQueueLagAudioQueue(
        make_audio_command(source_start_sample=0, source=AudioSource.OTHERS),
        make_audio_command(source_start_sample=0, source=AudioSource.ME),
        AudioDrainFence(),
    )
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_transcript_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    transcript_index = fake.operations.index(("transcript", 1))
    audio_before_transcript = [
        operation
        for operation in fake.operations[:transcript_index]
        if operation[0] == "audio"
    ]
    assert len(audio_before_transcript) == 2


def test_pending_transcript_audio_deadline_is_fatal_and_fixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing audio dependency must not wait forever or consume its fence."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    monkeypatch.setattr(writer_module, "_TRANSCRIPT_AUDIO_TIMEOUT_SECONDS", 0.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _CrossQueueLagAudioQueue(AudioDrainFence())
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_transcript_envelope(sequence=2))
    control.put(make_shutdown_envelope(sequence=3))

    with pytest.raises(
        writer_module.WriterWorkerProtocolError,
        match="timed out waiting for transcript audio persistence",
    ):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 2
    assert audio.get() == AudioDrainFence()
    assert ("transcript", 1) not in fake.operations
    assert ("close", 1) in fake.operations


def test_audio_fence_exposes_a_genuinely_missing_transcript_cursor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fence must preserve the persistence invariant for absent audio."""

    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _CrossQueueLagAudioQueue(AudioDrainFence())
    session_dir = tmp_path / "session"
    control.put(make_open_envelope(session_dir, sequence=1))
    control.put(make_transcript_envelope(sequence=2))

    with pytest.raises(
        PersistenceInvariantError,
        match="exceeds persisted audio cursor 0 for ME",
    ):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 2
    assert fatal.payload.error_type == "PersistenceInvariantError"
    assert '"status": "incomplete"' in (session_dir / "session.json").read_text(
        encoding="utf-8"
    )


def test_pending_transcript_remains_responsive_to_lifecycle_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stop during the bounded audio wait must leave the session incomplete."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _StopDuringFenceWaitQueue(stop)
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_transcript_envelope(sequence=2))

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    assert len(responses.items) == 1
    assert ("transcript", 1) not in fake.operations
    assert fake.close_calls == 1


def test_duplicate_control_reuses_latest_successful_save_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A duplicate must ACK without pretending another save succeeded."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    timestamps = iter(
        (
            datetime.fromisoformat("2026-08-20T10:00:00+09:00"),
            datetime.fromisoformat("2026-08-20T10:00:01+09:00"),
            datetime.fromisoformat("2026-08-20T10:00:02+09:00"),
        )
    )
    monkeypatch.setattr(writer_module, "_wall_clock", lambda: next(timestamps))
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    open_envelope = make_open_envelope(tmp_path / "session", sequence=1)
    control.put(open_envelope)
    control.put(open_envelope)
    control.put(make_shutdown_envelope(sequence=2))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    assert len(responses.items) == 3
    first = cast(MessageEnvelope[object], responses.items[0])
    duplicate = cast(MessageEnvelope[object], responses.items[1])
    assert isinstance(first.payload, WriterAck)
    assert isinstance(duplicate.payload, WriterAck)
    assert duplicate.payload.acknowledged_sequence == 1
    assert (
        duplicate.payload.latest_successful_save_at
        == first.payload.latest_successful_save_at
    )
    assert fake.close_calls == 1


def test_writer_limits_audio_batch_before_processing_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A continuously ready audio queue must yield to one control after 64 writes."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(control_envelope(2, MessageType.WRITER_FLUSH, WriterFlush()))
    control.put(make_shutdown_envelope(sequence=3))
    for index in range(65):
        audio.put(make_audio_command(source_start_sample=index * 12_800))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    forced_sync_index = fake.operations.index(("force_sync", 0))
    sixty_fifth_audio_index = fake.operations.index(("audio", 64 * 12_800))
    assert forced_sync_index == 64
    assert forced_sync_index < sixty_fifth_audio_index
    assert fake.close_calls == 1


def test_writer_flush_preserves_the_regular_sync_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A flush at 10.1 must not replace the regular 11.0 deadline tick."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(
        writer_module,
        "_monotonic",
        _ScriptedClock([10.0, 10.1, 10.1, 10.2, 11.0, 11.1]),
    )
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    duplicate_open = make_open_envelope(tmp_path / "session", sequence=1)
    control.put(duplicate_open)
    control.put(control_envelope(2, MessageType.WRITER_FLUSH, WriterFlush()))
    control.put(duplicate_open)
    control.put(make_shutdown_envelope(sequence=3))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    assert fake.operations.count(("force_sync", 0)) == 1
    scheduled_syncs = [
        value for operation, value in fake.operations if operation == "sync"
    ]
    assert scheduled_syncs[:2] == [10.1, 11.0]


@pytest.mark.parametrize(
    "terminal_type",
    [MessageType.WRITER_FINALIZE, MessageType.WRITER_SHUTDOWN],
)
def test_terminal_control_waits_for_prior_audio_to_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal_type: MessageType,
) -> None:
    """Finalize and shutdown must not overtake the 65th queued audio write."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    if terminal_type is MessageType.WRITER_FINALIZE:
        control.put(make_finalize_envelope(sequence=2))
        control.put(make_shutdown_envelope(sequence=3))
        terminal_operation = ("finalize", 0)
    else:
        control.put(make_shutdown_envelope(sequence=2))
        terminal_operation = ("close", 1)
    for index in range(65):
        audio.put(make_audio_command(source_start_sample=index * 12_800))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    sixty_fifth_audio_index = fake.operations.index(("audio", 64 * 12_800))
    terminal_index = fake.operations.index(terminal_operation)
    assert sixty_fifth_audio_index < terminal_index


@pytest.mark.parametrize(
    "terminal_type",
    [MessageType.WRITER_FINALIZE, MessageType.WRITER_SHUTDOWN],
)
def test_terminal_control_waits_for_delayed_audio_fence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal_type: MessageType,
) -> None:
    """An empty poll must not substitute for the Audio Worker's FIFO fence."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio = _DelayedFenceQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    if terminal_type is MessageType.WRITER_FINALIZE:
        control.put(make_finalize_envelope(sequence=2))
        control.put(make_shutdown_envelope(sequence=3))
    else:
        control.put(make_shutdown_envelope(sequence=2))

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    assert audio.fence_observed is True
    response_types = [
        cast(MessageEnvelope[object], item).message_type for item in responses.items
    ]
    assert MessageType.WRITER_FATAL not in response_types


def test_missing_audio_fence_can_be_bypassed_only_by_stop_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lifecycle stop must close incomplete without acknowledging shutdown."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    audio = _StopDuringFenceWaitQueue(stop)
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_shutdown_envelope(sequence=2))

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    assert len(responses.items) == 1
    only_response = cast(MessageEnvelope[object], responses.items[0])
    assert only_response.message_type is MessageType.WRITER_ACK
    assert fake.close_calls == 1


def test_audio_command_after_drain_fence_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The FIFO fence must make every later audio command a protocol failure."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_shutdown_envelope(sequence=2))
    audio.put(AudioDrainFence())
    audio.put(make_audio_command(source_start_sample=0))

    with pytest.raises(
        writer_module.WriterWorkerProtocolError,
        match="after audio drain fence",
    ):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 0
    assert fake.close_calls == 1
