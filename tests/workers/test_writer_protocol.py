"""Writer Worker control-envelope protocol and lifecycle tests."""

from pathlib import Path
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent
from typing import cast

import pytest

from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    DiscussionStateReplaced,
    MessageEnvelope,
    MessageSequenceError,
    TranscriptCommitted,
    UnknownSchemaVersionError,
    WriterAck,
    WriterAppendEvent,
    WriterFatal,
    WriterFlush,
    WriterOpenSession,
    WriterShutdown,
)
from flowlens.persistence.session_writer import SessionWriter
from flowlens.workers import writer as writer_module
from flowlens.workers.writer import run_writer_worker
from tests.factories import (
    make_discussion_state,
    make_event_record,
    make_transcript_record,
)
from tests.workers.writer_support import (
    SESSION_ID,
    _as_process_event,
    _as_process_queue,
    _FakeSessionWriter,
    _install_fake_writer,
    _RecordingResponseQueue,
    control_envelope,
    make_finalize_envelope,
    make_open_envelope,
    make_shutdown_envelope,
    make_transcript_envelope,
)


def test_writer_dispatches_every_control_payload_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Controller-rewrapped GUI controls must mutate and ACK in local order."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(make_transcript_envelope(sequence=2))
    control.put(
        control_envelope(
            3,
            MessageType.DISCUSSION_STATE_REPLACED,
            DiscussionStateReplaced(0, make_discussion_state(revision=1)),
        )
    )
    control.put(
        control_envelope(
            4,
            MessageType.EVENT_APPENDED,
            WriterAppendEvent(make_event_record()),
        )
    )
    control.put(control_envelope(5, MessageType.WRITER_FLUSH, WriterFlush()))
    control.put(make_finalize_envelope(sequence=6))
    control.put(make_shutdown_envelope(sequence=7))
    audio.put(AudioDrainFence())

    run_writer_worker(
        _as_process_queue(control),
        _as_process_queue(audio),
        _as_process_queue(responses),
        _as_process_event(stop),
    )

    envelopes = [cast(MessageEnvelope[object], item) for item in responses.items]
    assert [envelope.sequence for envelope in envelopes] == list(range(1, 8))
    assert [
        cast(WriterAck, envelope.payload).acknowledged_sequence
        for envelope in envelopes
    ] == list(range(1, 8))
    assert ("transcript", 1) in fake.operations
    assert ("discussion", 0) in fake.operations
    assert ("event", 1) in fake.operations
    assert ("force_sync", 0) in fake.operations
    assert ("finalize", 0) in fake.operations
    assert fake.close_calls == 1


def test_gui_local_sequence_one_transcript_passes_source_allowlist() -> None:
    """Controller rewrap accepts GUI seq1 without claiming post-open validity."""

    envelope = MessageEnvelope(
        1,
        SESSION_ID,
        MessageType.TRANSCRIPT_COMMITTED,
        1,
        ProcessSource.GUI,
        100,
        TranscriptCommitted(make_transcript_record()),
    )

    writer_module._validate_controller_source(envelope)


@pytest.mark.parametrize(
    "source",
    [
        ProcessSource.ASR,
        ProcessSource.AUDIO,
        ProcessSource.DISCUSSION,
        ProcessSource.WRITER,
    ],
)
def test_writer_rejects_non_gui_source_before_sequence_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: ProcessSource,
) -> None:
    """Only controller-rewrapped GUI envelopes enter the Writer sequence domain."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    control.put(
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.TRANSCRIPT_COMMITTED,
            10,
            source,
            1_000,
            TranscriptCommitted(make_transcript_record()),
        )
    )

    with pytest.raises(
        writer_module.WriterWorkerProtocolError,
        match="source must be GUI",
    ):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == 10
    assert fatal.payload.error_type == "WriterWorkerProtocolError"
    assert ("transcript", 1) not in fake.operations
    assert fake.close_calls == 1


@pytest.mark.parametrize(
    ("case", "expected_error", "failed_sequence"),
    [
        ("gap", MessageSequenceError, 3),
        ("unknown_schema", UnknownSchemaVersionError, 2),
        ("wrong_session", writer_module.WriterWorkerProtocolError, 2),
        ("wrong_payload", writer_module.WriterWorkerProtocolError, 2),
    ],
)
def test_control_protocol_errors_report_the_control_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    expected_error: type[BaseException],
    failed_sequence: int,
) -> None:
    """Invalid control metadata must fail closed with its positive sequence."""

    fake = _FakeSessionWriter()
    _install_fake_writer(monkeypatch, fake)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()
    control.put(make_open_envelope(tmp_path / "session", sequence=1))
    if case == "gap":
        invalid = control_envelope(3, MessageType.WRITER_FLUSH, WriterFlush())
    elif case == "unknown_schema":
        invalid = MessageEnvelope(
            2,
            SESSION_ID,
            MessageType.WRITER_FLUSH,
            2,
            ProcessSource.GUI,
            200,
            WriterFlush(),
        )
    elif case == "wrong_session":
        invalid = MessageEnvelope(
            1,
            "01J00000000000000000000009",
            MessageType.WRITER_FLUSH,
            2,
            ProcessSource.GUI,
            200,
            WriterFlush(),
        )
    elif case == "wrong_payload":
        invalid = control_envelope(
            2,
            MessageType.WRITER_FLUSH,
            WriterShutdown(),
        )
    else:
        raise AssertionError(f"unhandled test case {case}")
    control.put(invalid)

    with pytest.raises(expected_error):
        run_writer_worker(
            _as_process_queue(control),
            _as_process_queue(audio),
            _as_process_queue(responses),
            _as_process_event(stop),
        )

    assert len(responses.items) == 2
    fatal = cast(MessageEnvelope[object], responses.items[-1])
    assert fatal.message_type is MessageType.WRITER_FATAL
    assert isinstance(fatal.payload, WriterFatal)
    assert fatal.payload.failed_sequence == failed_sequence
    assert fatal.payload.error_type == expected_error.__name__
    assert fake.close_calls == 1
    assert responses.close_calls == 1
    assert responses.join_thread_calls == 1


def test_stop_event_closes_incomplete_once_without_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A lifecycle stop after open must exit cleanly and preserve recovery state."""

    fake = _FakeSessionWriter()
    control: ThreadQueue[object] = ThreadQueue()
    audio: ThreadQueue[object] = ThreadQueue()
    responses = _RecordingResponseQueue()
    stop = ThreadEvent()

    def open_then_stop(command: WriterOpenSession) -> SessionWriter:
        del command
        stop.set()
        return cast(SessionWriter, fake)

    monkeypatch.setattr(writer_module, "_open_session", open_then_stop)
    monkeypatch.setattr(writer_module, "_monotonic", lambda: 10.0)
    control.put(make_open_envelope(tmp_path / "session", sequence=1))

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
    assert responses.close_calls == 0
    assert responses.join_thread_calls == 0
