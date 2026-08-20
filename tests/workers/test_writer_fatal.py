"""Writer Worker fatal serialization and queue-boundary tests."""

from pathlib import Path
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent
from typing import cast

import pytest

from flowlens.domain.enums import MessageType
from flowlens.domain.messages import MessageEnvelope, WriterFatal
from flowlens.workers import writer as writer_module
from flowlens.workers.writer import run_writer_worker
from tests.workers.writer_support import (
    _as_process_event,
    _as_process_queue,
    _BrokenAudioQueue,
    _FakeSessionWriter,
    _install_fake_writer,
    _RecordingResponseQueue,
    _StorageError,
    make_audio_command,
    make_open_envelope,
    make_shutdown_envelope,
)


def test_storage_error_reports_only_serializable_fatal_and_cleans_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A storage exception must replace pending work with one flushed fatal."""

    fake = _FakeSessionWriter()
    fake.audio_error = _StorageError()
    fake.close_error = RuntimeError("cleanup failed")
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_shutdown_envelope(sequence=2))
    audio.put(make_audio_command(source_start_sample=0))

    with pytest.raises(_StorageError, match="disk full") as raised:
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    assert len(responses.items) == 2
    response_types = [
        cast(MessageEnvelope[object], item).message_type for item in responses.items
    ]
    assert response_types == [MessageType.WRITER_ACK, MessageType.WRITER_FATAL]
    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 0
    assert fatal.payload.error_type == "_StorageError"
    assert fatal.payload.message == "disk full"
    assert "cleanup failed" in "\n".join(getattr(raised.value, "__notes__", ()))
    assert fake.close_calls == 1
    assert responses.close_calls == 1
    assert responses.join_thread_calls == 1


def test_whitespace_exception_message_uses_nonempty_class_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whitespace-only exception text must not break fatal construction."""

    primary_error = Exception("   ")
    fake = _FakeSessionWriter()
    fake.audio_error = primary_error
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    audio.put(make_audio_command(source_start_sample=0))

    with pytest.raises(Exception) as raised:
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    assert raised.value is primary_error
    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.error_type == "Exception"
    assert fatal.payload.message == "Exception"
    assert fake.close_calls == 1
    assert responses.close_calls == 1
    assert responses.join_thread_calls == 1


def test_fatal_construction_failure_still_flushes_queue_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fatal serialization itself must remain secondary to the storage error."""

    primary_error = _StorageError()
    fake = _FakeSessionWriter()
    fake.audio_error = primary_error
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)

    def fail_fatal_construction(**values: object) -> WriterFatal:
        del values
        raise RuntimeError("fatal serialization failed")

    monkeypatch.setattr(writer_module, "WriterFatal", fail_fatal_construction)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    audio.put(make_audio_command(source_start_sample=0))

    with pytest.raises(_StorageError) as raised:
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    assert raised.value is primary_error
    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    assert "fatal serialization failed" in notes
    assert fake.close_calls == 1
    assert responses.close_calls == 1
    assert responses.join_thread_calls == 1


def test_unknown_audio_item_reports_zero_without_echoing_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-command audio item must fail with the non-control sentinel only."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    audio.put("private transcript text")

    with pytest.raises(writer_module.WriterWorkerProtocolError):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 0
    assert "private transcript text" not in fatal.payload.message
    assert fake.close_calls == 1


def test_audio_queue_exception_reports_zero_and_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A queue boundary exception must close incomplete and be re-raised."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))

    with pytest.raises(EOFError, match="audio queue closed"):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(_BrokenAudioQueue()),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 0
    assert fatal.payload.error_type == "EOFError"
    assert fatal.payload.message == "audio queue closed"
    assert fake.close_calls == 1
    assert responses.close_calls == 1
    assert responses.join_thread_calls == 1
