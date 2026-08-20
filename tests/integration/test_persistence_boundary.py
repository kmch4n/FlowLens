"""Spawned Writer and offline recovery persistence-boundary proof."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import sys
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, tzinfo
from multiprocessing import get_context
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty
from threading import enumerate as enumerate_threads
from time import monotonic
from typing import Any, NoReturn, cast
from unittest.mock import patch

import pytest

_COMPLETED_SESSION_ID = "01J00000000000000000000010"
_INTERRUPTED_SESSION_ID = "01J00000000000000000000020"
_REQUIRED_ARTIFACT_NAMES = {
    "discussion-state.json",
    "events.jsonl",
    "loopback.wav",
    "mic.wav",
    "session.json",
    "state-history.jsonl",
    "transcript.jsonl",
}
_TORN_FRAGMENT = b'{"torn":'
_QUEUE_MAX_SIZE = 32
_QUEUE_TIMEOUT_SECONDS = 5.0
_QUEUE_CLEANUP_TIMEOUT_SECONDS = 5.0


def _network_forbidden(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise AssertionError("foundation persistence attempted network access")


def _run_offline_writer_worker(
    control_queue: Queue[object],
    audio_queue: Queue[object],
    response_queue: Queue[object],
    stop_event: Event,
) -> None:
    """Install the socket guard before importing any production module."""

    if any(name == "flowlens" or name.startswith("flowlens.") for name in sys.modules):
        raise AssertionError("FlowLens production imported before child socket guard")
    patch.object(socket, "socket", _network_forbidden).start()
    from flowlens.workers.writer import run_writer_worker

    run_writer_worker(control_queue, audio_queue, response_queue, stop_event)


def _wait_forever() -> None:
    """Provide a deliberately stuck spawned child for cleanup verification."""

    get_context("spawn").Event().wait()


class _BrokenTimezone(tzinfo):
    """Timezone whose offset cannot be evaluated."""

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        del value
        raise ValueError("broken timezone")

    def dst(self, value: datetime | None) -> timedelta | None:
        del value
        return None

    def tzname(self, value: datetime | None) -> str | None:
        del value
        return "broken"


class _CleanupQueueProbe:
    """Queue cleanup double supporting close, join, and timeout failures."""

    def __init__(
        self,
        *,
        close_error: BaseException | None = None,
        join_error: BaseException | None = None,
    ) -> None:
        self.close_error = close_error
        self.join_error = join_error
        self.calls: list[str] = []

    def close(self) -> None:
        self.calls.append("close")
        if self.close_error is not None:
            raise self.close_error

    def join_thread(self) -> None:
        self.calls.append("join_thread")
        if self.join_error is not None:
            raise self.join_error

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel_join_thread")


def _make_manifest(session_id: str, started_at: datetime) -> Any:
    from flowlens.domain.enums import SessionMode, SessionStatus
    from flowlens.domain.session import (
        DeviceIdentity,
        ModelIdentity,
        SessionManifest,
    )

    return SessionManifest(
        schema_version=1,
        session_id=session_id,
        status=SessionStatus.INCOMPLETE,
        mode=SessionMode.MEETING,
        started_at=started_at,
        ended_at=None,
        active_duration_ms=0,
        pause_intervals=(),
        microphone=DeviceIdentity("mic-1", "USB Microphone"),
        loopback_output=DeviceIdentity("out-1", "Speakers"),
        asr_model=ModelIdentity("local-asr", "rev-a", "a" * 64),
        discussion_model=ModelIdentity("local-discussion", "rev-b", "b" * 64),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=(),
    )


def _make_state(revision: int, updated_at: datetime) -> Any:
    from flowlens.domain.discussion import DiscussionState
    from flowlens.domain.enums import SessionMode

    return DiscussionState(
        revision=revision,
        mode=SessionMode.MEETING,
        current_focus="永続化境界" if revision else "",
        key_points=("完全ローカル",) if revision else (),
        confirmed_outcomes=("七成果物を保持",) if revision else (),
        follow_up_items=(),
        updated_at=updated_at,
    )


def _make_transcript(
    sequence: int,
    source: Any,
    text: str,
    committed_at: datetime,
) -> Any:
    from flowlens.domain.messages import TranscriptRecord

    return TranscriptRecord(
        schema_version=1,
        segment_id=f"01J000000000000000000000{sequence:02d}",
        sequence=sequence,
        source=source,
        text=text,
        session_start_ms=sequence * 1_000,
        session_end_ms=sequence * 1_000 + 800,
        source_start_sample=0,
        source_end_sample=12_800,
        committed_at=committed_at,
    )


def _make_event(
    session_id: str,
    sequence: int,
    event_type: Any,
    session_time_ms: int,
    created_at: datetime,
) -> Any:
    from flowlens.domain.enums import ProcessSource
    from flowlens.domain.messages import EventRecord

    return EventRecord(
        schema_version=1,
        session_id=session_id,
        sequence=sequence,
        event_type=event_type,
        source=ProcessSource.GUI,
        session_time_ms=session_time_ms,
        created_at=created_at,
        details={},
    )


def _control_envelope(
    session_id: str,
    sequence: int,
    message_type: Any,
    payload: object,
) -> Any:
    from flowlens.domain.enums import ProcessSource
    from flowlens.domain.messages import MessageEnvelope

    return MessageEnvelope(
        schema_version=1,
        session_id=session_id,
        message_type=message_type,
        sequence=sequence,
        source=ProcessSource.GUI,
        created_monotonic_ms=sequence * 100,
        payload=payload,
    )


def _as_queue(value: object) -> Queue[object]:
    return cast(Queue[object], value)


def _as_event(value: object) -> Event:
    return cast(Event, value)


def _queue_failure(queue_name: str, operation_name: str, error: BaseException) -> str:
    return (
        f"{queue_name} queue {operation_name} failed: "
        f"{type(error).__name__}: {error}"
    )


def _safe_diagnostic_field(value: str, *, limit: int = 256) -> str:
    """Bound one contract diagnostic field and keep it on one log line."""

    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _safe_queue_item_metadata(item: object) -> str:
    """Describe a drained item without serializing its potentially large payload."""

    from flowlens.domain.messages import MessageEnvelope, WriterAck, WriterFatal

    if not isinstance(item, MessageEnvelope):
        return type(item).__name__
    message_type = item.message_type.value
    if isinstance(item.payload, WriterFatal):
        return (
            f"MessageEnvelope(message_type={message_type}, "
            f"failed_sequence={item.payload.failed_sequence}, "
            f"error_type={_safe_diagnostic_field(item.payload.error_type, limit=64)}, "
            f"message={_safe_diagnostic_field(item.payload.message)})"
        )
    if isinstance(item.payload, WriterAck):
        return (
            f"MessageEnvelope(message_type={message_type}, "
            f"acknowledged_sequence={item.payload.acknowledged_sequence})"
        )
    return (
        f"MessageEnvelope(message_type={message_type}, "
        f"payload_type={type(item.payload).__name__})"
    )


def _drained_queue_diagnostic(queue_name: str, items: list[str]) -> str | None:
    if not items:
        return None
    noun = "item" if len(items) == 1 else "items"
    return f"{queue_name} queue drained {len(items)} unexpected {noun}: " + ", ".join(
        items
    )


def _drain_real_queue(
    queue_name: str,
    queue: Queue[object],
    timeout: float,
) -> tuple[str, ...]:
    """Drain queued bytes after producer exit so the feeder can make progress."""

    deadline = monotonic() + timeout
    drained_items: list[str] = []
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            result = []
            diagnostic = _drained_queue_diagnostic(queue_name, drained_items)
            if diagnostic is not None:
                result.append(diagnostic)
            result.append(
                f"{queue_name} queue drain timed out after {timeout:g} seconds"
            )
            return tuple(result)
        try:
            item = queue.get(timeout=min(0.1, remaining))
        except Empty:
            diagnostic = _drained_queue_diagnostic(queue_name, drained_items)
            return () if diagnostic is None else (diagnostic,)
        except (EOFError, OSError, ValueError) as error:
            result = []
            diagnostic = _drained_queue_diagnostic(queue_name, drained_items)
            if diagnostic is not None:
                result.append(diagnostic)
            result.append(_queue_failure(queue_name, "drain", error))
            return tuple(result)
        drained_items.append(_safe_queue_item_metadata(item))


def _close_real_queue(
    queue_name: str,
    queue: Queue[object],
    timeout: float,
) -> tuple[str, ...]:
    """Close one drained queue and leave no feeder or helper thread alive."""

    failures = list(_drain_real_queue(queue_name, queue, timeout))
    try:
        queue.close()
    except (OSError, ValueError) as error:
        failures.append(_queue_failure(queue_name, "close", error))

    feeder = cast(Any, getattr(queue, "_thread", None))
    if feeder is not None:
        feeder.join(timeout=timeout)
        if feeder.is_alive():
            for endpoint_name in ("_reader", "_writer"):
                endpoint = cast(Any, getattr(queue, endpoint_name, None))
                if endpoint is None:
                    continue
                try:
                    endpoint.close()
                except (OSError, ValueError) as error:
                    failures.append(
                        _queue_failure(queue_name, f"close {endpoint_name}", error)
                    )
            try:
                queue.cancel_join_thread()
            except (OSError, ValueError) as error:
                failures.append(_queue_failure(queue_name, "cancel_join_thread", error))
            feeder.join(timeout=timeout)
        if feeder.is_alive():
            failures.append(
                f"{queue_name} queue feeder did not exit after {timeout:g} seconds"
            )
            return tuple(failures)
    try:
        queue.join_thread()
    except (AssertionError, OSError, ValueError) as error:
        failures.append(_queue_failure(queue_name, "join_thread", error))
    return tuple(failures)


def _cleanup_queues(
    queues: tuple[tuple[str, Any], ...],
    *,
    join_timeout: float = _QUEUE_CLEANUP_TIMEOUT_SECONDS,
) -> tuple[str, ...]:
    """Attempt bounded close/join cleanup for every queue and retain context."""

    failures: list[str] = []
    for queue_name, queue in queues:
        if isinstance(queue, Queue):
            failures.extend(_close_real_queue(queue_name, queue, timeout=join_timeout))
            continue
        try:
            queue.close()
        except BaseException as error:
            failures.append(_queue_failure(queue_name, "close", error))
        try:
            queue.join_thread()
        except BaseException as error:
            failures.append(_queue_failure(queue_name, "join_thread", error))
    return tuple(failures)


def _bounded_put(queue: Queue[object], item: object) -> None:
    queue.put(item, timeout=_QUEUE_TIMEOUT_SECONDS)


def _join_child(process: BaseProcess, timeout: float = 10.0) -> None:
    """Join one child, terminating only a stuck process and reporting its PID."""

    process.join(timeout=timeout)
    if not process.is_alive():
        return
    pid = process.pid
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    if process.is_alive():
        pytest.fail(f"Writer child process {pid} could not be stopped")
    pytest.fail(f"Writer child process {pid} did not exit")


@contextmanager
def _spawned_writer(
    context: SpawnContext,
) -> Iterator[tuple[BaseProcess, Queue[object], Queue[object], Queue[object]]]:
    control = _as_queue(context.Queue(maxsize=_QUEUE_MAX_SIZE))
    audio = _as_queue(context.Queue(maxsize=_QUEUE_MAX_SIZE))
    responses = _as_queue(context.Queue(maxsize=_QUEUE_MAX_SIZE))
    stop = _as_event(context.Event())
    process = context.Process(
        target=_run_offline_writer_worker,
        args=(control, audio, responses, stop),
    )
    process.start()
    primary_error: BaseException | None = None
    try:
        yield process, control, audio, responses
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_failures: list[str] = []
        try:
            _join_child(process)
        except BaseException as cleanup_error:
            cleanup_failures.append(str(cleanup_error))
        if process.is_alive():
            cleanup_failures.append("queue cleanup skipped while child remains alive")
        else:
            cleanup_failures.extend(
                _cleanup_queues(
                    (
                        ("control", control),
                        ("audio", audio),
                        ("responses", responses),
                    )
                )
            )
        if primary_error is not None:
            for failure in cleanup_failures:
                primary_error.add_note(failure)
        elif cleanup_failures:
            pytest.fail("; ".join(cleanup_failures))


def _assert_ack(responses: Queue[object], expected_sequence: int) -> None:
    from flowlens.domain.enums import MessageType, ProcessSource
    from flowlens.domain.messages import MessageEnvelope, WriterAck

    response = responses.get(timeout=_QUEUE_TIMEOUT_SECONDS)
    assert isinstance(response, MessageEnvelope)
    assert response.message_type is MessageType.WRITER_ACK
    assert response.source is ProcessSource.WRITER
    assert isinstance(response.payload, WriterAck)
    assert response.payload.acknowledged_sequence == expected_sequence


def _assert_queue_empty(queue_name: str, queue: Queue[object]) -> None:
    """Reject an unexpected queued message after the producer process exits."""

    try:
        unexpected = queue.get(timeout=0.1)
    except Empty:
        return
    pytest.fail(
        f"{queue_name} queue retained unexpected item: {type(unexpected).__name__}"
    )


def _run_scripted_writer_session(
    sessions_root: Path,
    session_id: str,
    started_at: datetime,
    *,
    finalize: bool,
) -> Path:
    from flowlens.domain.enums import AudioSource, EventType, MessageType
    from flowlens.domain.messages import (
        AudioDrainFence,
        AudioWriteCommand,
        DiscussionStateReplaced,
        TranscriptCommitted,
        WriterAppendEvent,
        WriterFinalize,
        WriterOpenSession,
        WriterShutdown,
    )

    context = get_context("spawn")
    session_dir = sessions_root / (
        f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_{session_id}"
    )
    initial_state = _make_state(0, started_at)
    final_state = _make_state(
        1,
        datetime.fromisoformat("2026-08-19T12:02:00+09:00"),
    )
    transcripts = (
        _make_transcript(
            1,
            AudioSource.ME,
            "今回の方針を確認します。",
            datetime.fromisoformat("2026-08-19T12:01:00+09:00"),
        ),
        _make_transcript(
            2,
            AudioSource.OTHERS,
            "七つの成果物をローカルに保存します。",
            datetime.fromisoformat("2026-08-19T12:02:00+09:00"),
        ),
    )
    start_event = _make_event(
        session_id,
        1,
        EventType.SESSION_START,
        0,
        started_at,
    )
    completion_event = _make_event(
        session_id,
        2,
        EventType.SESSION_COMPLETED,
        2_800,
        datetime.fromisoformat("2026-08-19T12:03:00+09:00"),
    )

    with _spawned_writer(context) as (process, control, audio, responses):
        _bounded_put(
            audio,
            AudioWriteCommand(
                AudioSource.ME, b"\x01\x00" * 12_800, 0, 12_800, 0, 1_000
            ),
        )
        _bounded_put(
            audio,
            AudioWriteCommand(
                AudioSource.OTHERS,
                b"\x02\x00" * 12_800,
                0,
                12_800,
                0,
                1_000,
            ),
        )
        _bounded_put(audio, AudioDrainFence())
        _bounded_put(
            control,
            _control_envelope(
                session_id,
                1,
                MessageType.WRITER_OPEN_SESSION,
                WriterOpenSession(
                    session_dir,
                    _make_manifest(session_id, started_at),
                    initial_state,
                ),
            ),
        )
        _assert_ack(responses, 1)
        for sequence, transcript in enumerate(transcripts, start=2):
            _bounded_put(
                control,
                _control_envelope(
                    session_id,
                    sequence,
                    MessageType.TRANSCRIPT_COMMITTED,
                    TranscriptCommitted(transcript),
                ),
            )
            _assert_ack(responses, sequence)
        _bounded_put(
            control,
            _control_envelope(
                session_id,
                4,
                MessageType.DISCUSSION_STATE_REPLACED,
                DiscussionStateReplaced(0, final_state),
            ),
        )
        _assert_ack(responses, 4)
        _bounded_put(
            control,
            _control_envelope(
                session_id,
                5,
                MessageType.EVENT_APPENDED,
                WriterAppendEvent(start_event),
            ),
        )
        _assert_ack(responses, 5)
        if finalize:
            terminal_payload: object = WriterFinalize(
                ended_at=datetime.fromisoformat("2026-08-19T12:03:00+09:00"),
                active_duration_ms=2_800,
                pause_intervals=(),
                final_state=final_state,
                completion_event=completion_event,
            )
            terminal_type = MessageType.WRITER_FINALIZE
        else:
            terminal_payload = WriterShutdown()
            terminal_type = MessageType.WRITER_SHUTDOWN
        _bounded_put(
            control,
            _control_envelope(
                session_id,
                6,
                terminal_type,
                terminal_payload,
            ),
        )
        _assert_ack(responses, 6)
        if finalize:
            _bounded_put(
                control,
                _control_envelope(
                    session_id,
                    7,
                    MessageType.WRITER_SHUTDOWN,
                    WriterShutdown(),
                ),
            )
            _assert_ack(responses, 7)
        _join_child(process)
        assert process.exitcode == 0
        for queue_name, queue in (
            ("control", control),
            ("audio", audio),
            ("responses", responses),
        ):
            _assert_queue_empty(queue_name, queue)
    return session_dir


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(isinstance(record, dict) for record in records)
    return records


def _wav_contract(path: Path) -> tuple[int, int, int, int, str]:
    with wave.open(str(path), "rb") as reader:
        frames = reader.readframes(reader.getnframes())
        return (
            reader.getframerate(),
            reader.getsampwidth(),
            reader.getnchannels(),
            reader.getnframes(),
            hashlib.sha256(frames).hexdigest(),
        )


def _pcm_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()[44:]).hexdigest()


def _damage_only_final_jsonl_fragment_and_wav_headers(session_dir: Path) -> None:
    with (session_dir / "events.jsonl").open("ab") as event_log:
        event_log.write(_TORN_FRAGMENT)
    for wav_name in ("mic.wav", "loopback.wav"):
        with (session_dir / wav_name).open("r+b") as wav_file:
            wav_file.seek(4)
            wav_file.write(struct.pack("<I", 0))
            wav_file.seek(40)
            wav_file.write(struct.pack("<I", 0))


def _assert_recovery_report(
    report: Any,
    interrupted_dir: Path,
) -> None:
    assert report.session_id == _INTERRUPTED_SESSION_ID
    assert report.session_dir == interrupted_dir
    assert report.discarded_jsonl_tail_bytes == {"events.jsonl": len(_TORN_FRAGMENT)}
    assert report.repaired_wav_headers == ("loopback.wav", "mic.wav")
    assert report.transcript_entry_count == 2
    assert report.final_discussion_state_revision == 1
    assert report.active_duration_ms == 800


def test_writer_and_recovery_boundary_is_complete_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real spawned Writer must finalize or recover seven local artifacts."""

    from flowlens.persistence.recovery import recover_incomplete_sessions

    monkeypatch.setattr(socket, "socket", _network_forbidden)
    sessions_root = tmp_path / "sessions"
    completed_dir = _run_scripted_writer_session(
        sessions_root,
        _COMPLETED_SESSION_ID,
        datetime.fromisoformat("2026-08-19T12:00:00+09:00"),
        finalize=True,
    )
    assert {path.name for path in completed_dir.iterdir()} == _REQUIRED_ARTIFACT_NAMES
    completed_manifest = _load_json(completed_dir / "session.json")
    assert completed_manifest["status"] == "completed"
    assert completed_manifest["transcript_entry_count"] == 2
    assert completed_manifest["final_discussion_state_revision"] == 1
    completed_transcripts = _load_jsonl(completed_dir / "transcript.jsonl")
    assert [record["sequence"] for record in completed_transcripts] == [1, 2]
    assert [record["text"] for record in completed_transcripts] == [
        "今回の方針を確認します。",
        "七つの成果物をローカルに保存します。",
    ]
    for wav_name in ("mic.wav", "loopback.wav"):
        assert _wav_contract(completed_dir / wav_name)[:4] == (16_000, 2, 1, 12_800)

    interrupted_dir = _run_scripted_writer_session(
        sessions_root,
        _INTERRUPTED_SESSION_ID,
        datetime.fromisoformat("2026-08-19T12:10:00+09:00"),
        finalize=False,
    )
    pcm_digests = {
        name: _pcm_digest(interrupted_dir / name)
        for name in ("mic.wav", "loopback.wav")
    }
    _damage_only_final_jsonl_fragment_and_wav_headers(interrupted_dir)

    with pytest.raises(ValueError, match="must not precede"):
        recover_incomplete_sessions(
            sessions_root,
            datetime.fromisoformat("2026-08-19T11:59:00+09:00"),
        )

    reports = recover_incomplete_sessions(
        sessions_root,
        datetime.fromisoformat("2026-08-19T13:00:00+09:00"),
    )

    assert [report.session_dir for report in reports] == [interrupted_dir]
    _assert_recovery_report(reports[0], interrupted_dir)
    assert {path.name for path in interrupted_dir.iterdir()} == _REQUIRED_ARTIFACT_NAMES
    recovered_manifest = _load_json(interrupted_dir / "session.json")
    assert recovered_manifest["status"] == "recovered"
    assert recovered_manifest["ended_at"] == "2026-08-19T13:00:00.000+09:00"
    assert recovered_manifest["transcript_entry_count"] == 2
    assert recovered_manifest["final_discussion_state_revision"] == 1
    recovered_transcripts = _load_jsonl(interrupted_dir / "transcript.jsonl")
    assert [record["sequence"] for record in recovered_transcripts] == [1, 2]
    assert [record["text"] for record in recovered_transcripts] == [
        "今回の方針を確認します。",
        "七つの成果物をローカルに保存します。",
    ]
    recovered_events = _load_jsonl(interrupted_dir / "events.jsonl")
    assert [record["event_type"] for record in recovered_events] == [
        "SESSION_START",
        "SESSION_RECOVERED",
    ]
    assert recovered_events[-1]["details"] == {
        "discarded_jsonl_tail_bytes": {"events.jsonl": len(_TORN_FRAGMENT)},
        "repaired_wav_headers": ["loopback.wav", "mic.wav"],
    }
    recovered_state = _load_json(interrupted_dir / "discussion-state.json")
    assert recovered_state["revision"] == 1
    assert recovered_state["current_focus"] == "永続化境界"
    for wav_name in ("mic.wav", "loopback.wav"):
        contract = _wav_contract(interrupted_dir / wav_name)
        assert contract[:4] == (16_000, 2, 1, 12_800)
        assert _pcm_digest(interrupted_dir / wav_name) == pcm_digests[wav_name]
    assert (
        recover_incomplete_sessions(
            sessions_root,
            datetime.fromisoformat("2026-08-19T13:00:00+09:00"),
        )
        == ()
    )
    print(f"completed_session={completed_dir}")
    print(f"recovered_session={interrupted_dir}")


def test_recovery_time_validation_precedes_session_root_access(tmp_path: Path) -> None:
    """Invalid recovery clocks must fail before inspecting a sessions root."""

    from flowlens.persistence.recovery import recover_incomplete_sessions

    missing_root = tmp_path / "does-not-exist"
    with pytest.raises(TypeError, match="must be a datetime"):
        recover_incomplete_sessions(missing_root, cast(Any, "not-a-datetime"))
    with pytest.raises(ValueError, match="timezone is invalid"):
        recover_incomplete_sessions(
            missing_root,
            datetime(2026, 8, 19, 13, tzinfo=_BrokenTimezone()),
        )


def test_persistence_wire_contracts_reject_malformed_values(tmp_path: Path) -> None:
    """Persisted boundary records must fail closed on malformed wire values."""

    from flowlens.config.model import AppConfig
    from flowlens.config.store import ConfigStore
    from flowlens.domain._validation import ContractValidationError
    from flowlens.domain.discussion import DiscussionState
    from flowlens.domain.enums import AudioSource
    from flowlens.domain.messages import TranscriptRecord
    from flowlens.domain.session import SessionManifest

    with pytest.raises(ContractValidationError, match="AppConfig must be an object"):
        AppConfig.from_dict([])
    config_value = AppConfig.default().to_dict()
    config_value["last_mode"] = "UNKNOWN"
    with pytest.raises(ContractValidationError, match="supported SessionMode"):
        AppConfig.from_dict(config_value)
    with pytest.raises(TypeError, match="config must be an AppConfig"):
        ConfigStore(tmp_path / "config.json").save(cast(Any, object()))

    with pytest.raises(
        ContractValidationError,
        match="DiscussionState must be an object",
    ):
        DiscussionState.from_dict([])
    state_value = _make_state(
        0,
        datetime.fromisoformat("2026-08-19T12:00:00+09:00"),
    ).to_dict()
    state_value["mode"] = "UNKNOWN"
    with pytest.raises(ContractValidationError, match="supported SessionMode"):
        DiscussionState.from_dict(state_value)

    transcript_value = _make_transcript(
        1,
        AudioSource.ME,
        "境界値を確認します。",
        datetime.fromisoformat("2026-08-19T12:01:00+09:00"),
    ).to_dict()
    transcript_value["source"] = "UNKNOWN"
    with pytest.raises(ContractValidationError, match="supported AudioSource"):
        TranscriptRecord.from_dict(transcript_value)

    manifest_value = _make_manifest(
        _INTERRUPTED_SESSION_ID,
        datetime.fromisoformat("2026-08-19T12:10:00+09:00"),
    ).to_dict()
    manifest_value["status"] = "unknown"
    with pytest.raises(ContractValidationError, match="supported SessionStatus"):
        SessionManifest.from_dict(manifest_value)


def test_child_cleanup_terminates_only_stuck_child_and_reports_pid() -> None:
    """Cleanup must leave exited children alone and identify a terminated child."""

    context = get_context("spawn")
    exited = context.Process(target=tuple)
    exited.start()
    _join_child(exited)
    assert exited.exitcode == 0

    stuck = context.Process(target=_wait_forever)
    stuck.start()
    stuck_pid = stuck.pid
    with pytest.raises(
        pytest.fail.Exception, match=rf"process {stuck_pid} did not exit"
    ):
        _join_child(stuck, timeout=0.1)
    assert not stuck.is_alive()
    assert stuck.exitcode is not None


def test_queue_cleanup_attempts_every_queue_and_retains_failure_context() -> None:
    """Cleanup errors must not skip a later queue boundary."""

    control = _CleanupQueueProbe(close_error=OSError("control close failed"))
    audio = _CleanupQueueProbe()
    responses = _CleanupQueueProbe(join_error=RuntimeError("response join failed"))

    failures = _cleanup_queues(
        (("control", control), ("audio", audio), ("responses", responses)),
        join_timeout=0.05,
    )

    assert control.calls == ["close", "join_thread"]
    assert audio.calls == ["close", "join_thread"]
    assert responses.calls == ["close", "join_thread"]
    assert failures == (
        "control queue close failed: OSError: control close failed",
        "responses queue join_thread failed: RuntimeError: response join failed",
    )


def test_real_queue_cleanup_leaves_no_helper_or_feeder_thread() -> None:
    """A blocked real feeder must be unblocked and joined before cleanup returns."""

    context = get_context("spawn")
    queue = _as_queue(context.Queue(maxsize=1))
    original_thread_ids = {thread.ident for thread in enumerate_threads()}
    queue.put(b"x" * (8 * 1024 * 1024), timeout=_QUEUE_TIMEOUT_SECONDS)

    failures = _cleanup_queues((("blocked", queue),), join_timeout=1.0)

    leaked_threads = [
        thread.name
        for thread in enumerate_threads()
        if thread.ident not in original_thread_ids
        and thread.is_alive()
        and (thread.name == "QueueFeederThread" or "(invoke)" in thread.name)
    ]
    assert failures == ("blocked queue drained 1 unexpected item: bytes",)
    assert "xxxxxxxx" not in failures[0]
    assert leaked_threads == []


def test_pending_writer_fatal_is_recorded_without_masking_parent_primary() -> None:
    """Cleanup must attach a safe pending-fatal diagnostic to the primary error."""

    context = get_context("spawn")
    with pytest.raises(RuntimeError, match="parent boundary failed") as raised:
        with _spawned_writer(context) as (_, control, _, _):
            _bounded_put(control, "malformed-control-secret")
            raise RuntimeError("parent boundary failed")

    assert str(raised.value) == "parent boundary failed"
    diagnostics = "\n".join(raised.value.__notes__)
    assert "responses queue drained 1 unexpected item" in diagnostics
    assert "message_type=WRITER_FATAL" in diagnostics
    assert "failed_sequence=0" in diagnostics
    assert "error_type=WriterWorkerProtocolError" in diagnostics
    assert "message=control queue item must be a MessageEnvelope" in diagnostics
    assert "malformed-control-secret" not in diagnostics


def test_pending_writer_fatal_fails_cleanup_without_a_primary() -> None:
    """A pending fatal must make cleanup-only completion fail safely."""

    context = get_context("spawn")
    with pytest.raises(pytest.fail.Exception) as raised:
        with _spawned_writer(context) as (_, control, _, _):
            _bounded_put(control, "cleanup-only-malformed-secret")

    diagnostic = str(raised.value)
    assert "responses queue drained 1 unexpected item" in diagnostic
    assert "message_type=WRITER_FATAL" in diagnostic
    assert "failed_sequence=0" in diagnostic
    assert "cleanup-only-malformed-secret" not in diagnostic


def test_queue_item_metadata_records_ack_and_redacts_payloads() -> None:
    """Cleanup diagnostics expose schema fields, never queued payload content."""

    from flowlens.domain.enums import MessageType, ProcessSource
    from flowlens.domain.messages import MessageEnvelope, WriterAck

    ack = MessageEnvelope(
        schema_version=1,
        session_id=_COMPLETED_SESSION_ID,
        message_type=MessageType.WRITER_ACK,
        sequence=8,
        source=ProcessSource.WRITER,
        created_monotonic_ms=800,
        payload=WriterAck(
            acknowledged_sequence=7,
            latest_successful_save_at=datetime.fromisoformat(
                "2026-08-19T12:00:00+09:00"
            ),
        ),
    )

    assert _safe_queue_item_metadata(ack) == (
        "MessageEnvelope(message_type=WRITER_ACK, acknowledged_sequence=7)"
    )
    assert _safe_queue_item_metadata(b"pcm-secret") == "bytes"
    assert _safe_queue_item_metadata("transcript-secret") == "str"
