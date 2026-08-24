"""SessionWriter synchronization and ordered-finalization tests."""

import json
import math
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import AudioSource, EventType, SessionMode, SessionStatus
from flowlens.domain.messages import AudioWriteCommand, EventRecord
from flowlens.domain.session import PauseInterval, SessionManifest
from flowlens.persistence.json_files import AtomicJsonFile, JsonlAppender
from flowlens.persistence.session_writer import (
    PersistenceInvariantError,
    SessionWriter,
    WriterOwnershipError,
)
from flowlens.persistence.wav_sink import WavSink
from tests.factories import (
    make_discussion_state,
    make_event_record,
    make_finalize_command,
    make_manifest,
    make_transcript_record,
)


def _load_manifest(writer: SessionWriter) -> dict[str, object]:
    """Load the persisted manifest for behavior-level assertions."""

    value = json.loads(
        (writer.session_dir / "session.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load complete JSONL records from a test artifact path."""

    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        assert isinstance(value, dict)
        records.append(value)
    return records


def test_sync_occurs_only_when_one_second_deadline_is_due(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the deadline comparison off either boundary must fail."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))

    assert open_writer.sync_if_due(open_writer.opened_monotonic + 0.999) is False
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 1.000) is True
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 1.500) is False
    assert calls == ["sync"]


def test_force_sync_does_not_shift_the_regular_deadline(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit flush must not postpone the fixed one-second schedule."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))

    open_writer.force_sync()

    assert open_writer.sync_if_due(open_writer.opened_monotonic + 0.999) is False
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 1.0) is True
    assert calls == ["sync", "sync"]


def test_force_sync_failure_keeps_incomplete_and_terminalizes_writer(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced durability failure must use the same fail-closed path."""

    def fail_sync() -> None:
        raise OSError("forced sync failed")

    monkeypatch.setattr(open_writer, "_sync_all", fail_sync)

    with pytest.raises(OSError, match="forced sync failed"):
        open_writer.force_sync()

    assert _load_manifest(open_writer)["status"] == "incomplete"
    with pytest.raises(RuntimeError, match="failed"):
        open_writer.force_sync()


def test_force_sync_rejects_non_owner_and_closed_writer_before_resource_access(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership and open-state checks must precede every explicit sync."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))
    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.getpid",
        lambda: open_writer.owner_pid + 1,
    )

    with pytest.raises(WriterOwnershipError, match="owner PID"):
        open_writer.force_sync()

    assert calls == []
    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.getpid",
        lambda: open_writer.owner_pid,
    )
    open_writer.close_incomplete()
    with pytest.raises(RuntimeError, match="closed"):
        open_writer.force_sync()
    assert calls == []


def test_late_sync_advances_by_whole_intervals_past_observed_time(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advancing only one interval after a late tick would sync repeatedly."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))

    assert open_writer.sync_if_due(open_writer.opened_monotonic + 3.5) is True
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 3.999) is False
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 4.0) is True
    assert calls == ["sync", "sync"]


def test_fractional_sync_deadline_stays_strictly_after_same_observed_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary rounding must not make one observed time synchronize twice."""

    monkeypatch.setattr(
        "flowlens.persistence.session_writer.time.monotonic",
        lambda: 1.0,
    )
    writer = SessionWriter.open(
        tmp_path / "session",
        make_manifest(),
        make_discussion_state(),
        sync_interval_seconds=0.1,
    )
    calls: list[str] = []
    monkeypatch.setattr(writer, "_sync_all", lambda: calls.append("sync"))
    try:
        assert writer.sync_if_due(1.3) is True
        assert writer._next_sync_deadline > 1.3
        assert math.isclose(writer._next_sync_deadline, 1.4)
        assert writer.sync_if_due(1.3) is False
        assert calls == ["sync"]
    finally:
        writer.close_incomplete()


@pytest.mark.parametrize(
    ("opened_monotonic", "sync_interval_seconds"),
    [(1_000_000.0, 5e-324), (1e308, 1e308)],
)
def test_open_rejects_unrepresentable_initial_sync_deadline_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened_monotonic: float,
    sync_interval_seconds: float,
) -> None:
    """An unusable interval must fail before session directory creation."""

    session_dir = tmp_path / "session"
    io_calls: list[str] = []
    monkeypatch.setattr(
        "flowlens.persistence.session_writer.time.monotonic",
        lambda: opened_monotonic,
    )

    def unexpected_prepare(path: Path) -> None:
        io_calls.append(str(path))
        raise AssertionError("session directory preparation must not run")

    monkeypatch.setattr(
        SessionWriter,
        "_prepare_empty_session_directory",
        staticmethod(unexpected_prepare),
    )

    with pytest.raises(PersistenceInvariantError, match="sync deadline"):
        SessionWriter.open(
            session_dir,
            make_manifest(),
            make_discussion_state(),
            sync_interval_seconds=sync_interval_seconds,
        )

    assert io_calls == []
    assert not session_dir.exists()


def test_open_rejects_huge_integer_sync_interval_before_io(
    tmp_path: Path,
) -> None:
    """Float conversion overflow must be a normal pre-I/O invariant error."""

    session_dir = tmp_path / "session"

    with pytest.raises(PersistenceInvariantError, match="sync_interval_seconds"):
        SessionWriter.open(
            session_dir,
            make_manifest(),
            make_discussion_state(),
            sync_interval_seconds=10**5000,
        )

    assert not session_dir.exists()


def test_sync_rejects_huge_integer_time_before_resource_access(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Float conversion overflow must not call the durable sync operation."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))

    with pytest.raises(PersistenceInvariantError, match="now_monotonic"):
        open_writer.sync_if_due(10**5000)

    assert calls == []
    assert open_writer.sync_if_due(open_writer.opened_monotonic) is False


def test_sync_rejects_time_with_no_finite_next_deadline_before_resource_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A due time at the float ceiling must not synchronize without a future tick."""

    monkeypatch.setattr(
        "flowlens.persistence.session_writer.time.monotonic",
        lambda: 1e308,
    )
    writer = SessionWriter.open(
        tmp_path / "session",
        make_manifest(),
        make_discussion_state(),
        sync_interval_seconds=1e307,
    )
    calls: list[str] = []
    monkeypatch.setattr(writer, "_sync_all", lambda: calls.append("sync"))
    try:
        with pytest.raises(PersistenceInvariantError, match="sync deadline"):
            writer.sync_if_due(sys.float_info.max)
        assert calls == []
        assert writer.sync_if_due(1e308) is False
    finally:
        writer.close_incomplete()


def test_due_sync_durably_synchronizes_both_wavs_and_every_jsonl(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting any owned append resource would leave one artifact vulnerable."""

    calls: list[str] = []
    resources = (
        (open_writer._microphone_sink, "mic.wav"),
        (open_writer._loopback_sink, "loopback.wav"),
        (open_writer._transcript_log, "transcript.jsonl"),
        (open_writer._state_history_log, "state-history.jsonl"),
        (open_writer._event_log, "events.jsonl"),
    )
    for resource, name in resources:
        monkeypatch.setattr(
            resource,
            "sync",
            lambda resource_name=name: calls.append(resource_name),
        )

    assert open_writer.sync_if_due(open_writer.opened_monotonic + 1.0) is True
    assert calls == [
        "mic.wav",
        "loopback.wav",
        "transcript.jsonl",
        "state-history.jsonl",
        "events.jsonl",
    ]


@pytest.mark.parametrize("now_monotonic", [True, math.nan, math.inf, "1.0"])
def test_sync_rejects_invalid_monotonic_time_before_resource_mutation(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
    now_monotonic: object,
) -> None:
    """An invalid scheduling value must not reach a durability operation."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))

    with pytest.raises(PersistenceInvariantError, match="now_monotonic"):
        open_writer.sync_if_due(now_monotonic)  # type: ignore[arg-type]

    assert calls == []


def test_sync_failure_keeps_manifest_incomplete_and_terminalizes_writer(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fsync must never leave the Writer usable or marked completed."""

    def fail_sync() -> None:
        raise OSError("durable sync failed")

    monkeypatch.setattr(open_writer, "_sync_all", fail_sync)

    with pytest.raises(OSError, match="durable sync failed"):
        open_writer.sync_if_due(open_writer.opened_monotonic + 1.0)

    assert _load_manifest(open_writer)["status"] == "incomplete"
    with pytest.raises(RuntimeError, match="failed"):
        open_writer.sync_if_due(open_writer.opened_monotonic + 2.0)


def test_sync_rejects_non_owner_before_deadline_or_resource_access(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child process must not synchronize handles inherited from its parent."""

    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))
    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.getpid",
        lambda: open_writer.owner_pid + 1,
    )

    with pytest.raises(WriterOwnershipError, match="owner PID"):
        open_writer.sync_if_due(open_writer.opened_monotonic + 1.0)

    assert calls == []
    assert _load_manifest(open_writer)["status"] == "incomplete"


def test_finalize_marks_completed_after_resources_are_durable_and_closed(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reordering any final persistent or close step must fail this test."""

    order: list[str] = []
    original_append = JsonlAppender.append
    original_sync = JsonlAppender.sync
    original_jsonl_close = JsonlAppender.close
    original_wav_finalize = WavSink.finalize
    original_replace = AtomicJsonFile.replace

    def record_append(appender: JsonlAppender, value: object) -> None:
        if (
            appender.path.name == "events.jsonl"
            and isinstance(value, dict)
            and value.get("event_type") == "SESSION_COMPLETED"
        ):
            order.append("completion-event")
        original_append(appender, value)

    def record_sync(appender: JsonlAppender) -> None:
        order.append(f"sync-{appender.path.name}")
        original_sync(appender)

    def record_finalize(sink: WavSink) -> None:
        order.append(f"finalize-{sink.path.name}")
        original_wav_finalize(sink)

    def record_replace(file: AtomicJsonFile, value: object) -> None:
        assert isinstance(value, dict)
        if file.path.name == "discussion-state.json":
            order.append("replace-final-state")
        elif file.path.name == "session.json":
            order.append(f"replace-manifest-{value['status']}")
        original_replace(file, value)

    def record_close(appender: JsonlAppender) -> None:
        order.append(f"close-{appender.path.name}")
        original_jsonl_close(appender)

    monkeypatch.setattr(JsonlAppender, "append", record_append)
    monkeypatch.setattr(JsonlAppender, "sync", record_sync)
    monkeypatch.setattr(JsonlAppender, "close", record_close)
    monkeypatch.setattr(WavSink, "finalize", record_finalize)
    monkeypatch.setattr(AtomicJsonFile, "replace", record_replace)

    result = open_writer.finalize(make_finalize_command())

    assert result.status is SessionStatus.COMPLETED
    assert order == [
        "sync-transcript.jsonl",
        "sync-state-history.jsonl",
        "sync-events.jsonl",
        "completion-event",
        "sync-transcript.jsonl",
        "sync-state-history.jsonl",
        "sync-events.jsonl",
        "finalize-mic.wav",
        "finalize-loopback.wav",
        "replace-final-state",
        "close-transcript.jsonl",
        "close-state-history.jsonl",
        "close-events.jsonl",
        "replace-manifest-completed",
    ]
    assert _load_manifest(open_writer)["status"] == "completed"


def test_finalize_persists_canonical_completion_counts_and_timestamps(
    open_writer: SessionWriter,
) -> None:
    """Using bootstrap counts or stale revisions would corrupt final metadata."""

    open_writer.append_audio(
        AudioWriteCommand(
            source=AudioSource.ME,
            pcm_s16le=b"\x00\x00" * 25_600,
            source_start_sample=0,
            source_end_sample=25_600,
            session_start_ms=0,
            captured_monotonic_ms=0,
        )
    )
    open_writer.append_transcript(make_transcript_record(1))
    open_writer.append_transcript(make_transcript_record(2))
    final_state = make_discussion_state(1)
    open_writer.replace_discussion_state(0, final_state)
    open_writer.append_event(make_event_record(1, session_time_ms=0))
    pause = PauseInterval(started_ms=600_000, ended_ms=660_000)
    command = replace(
        make_finalize_command(event_sequence=2),
        pause_intervals=(pause,),
        final_state=final_state,
    )

    result = open_writer.finalize(command)

    persisted = _load_manifest(open_writer)
    assert result.to_dict() == persisted
    assert result.status is SessionStatus.COMPLETED
    assert result.ended_at == datetime.fromisoformat("2026-08-19T12:30:00+09:00")
    assert result.active_duration_ms == 1_800_000
    assert result.pause_intervals == (pause,)
    assert result.transcript_entry_count == 2
    assert result.final_discussion_state_revision == 1
    assert result.recovery_notes == ()
    events = _load_jsonl(open_writer.session_dir / "events.jsonl")
    assert [event["event_type"] for event in events] == [
        "SESSION_START",
        "SESSION_COMPLETED",
    ]
    final_snapshot = json.loads(
        (open_writer.session_dir / "discussion-state.json").read_text(encoding="utf-8")
    )
    assert final_snapshot == final_state.to_dict()


@pytest.mark.parametrize(
    ("persisted_revision", "final_revision"),
    [(0, 1), (1, 0)],
)
def test_finalize_rejects_nonmatching_final_revision_before_event_append(
    open_writer: SessionWriter,
    persisted_revision: int,
    final_revision: int,
) -> None:
    """Final discussion changes must use replace_discussion_state first."""

    if persisted_revision == 1:
        open_writer.replace_discussion_state(0, make_discussion_state(1))
    events_path = open_writer.session_dir / "events.jsonl"
    before = events_path.read_bytes()
    command = replace(
        make_finalize_command(),
        final_state=make_discussion_state(final_revision),
    )

    with pytest.raises(PersistenceInvariantError, match="final discussion revision"):
        open_writer.finalize(command)

    assert events_path.read_bytes() == before
    assert _load_manifest(open_writer)["status"] == "incomplete"
    open_writer.append_event(make_event_record())


def test_finalize_rejects_wrong_final_mode_before_event_append(
    open_writer: SessionWriter,
) -> None:
    """A final state for another mode must not enter this session bundle."""

    events_path = open_writer.session_dir / "events.jsonl"
    wrong_mode = replace(make_discussion_state(), mode=SessionMode.INTERVIEW)
    command = replace(make_finalize_command(), final_state=wrong_mode)

    with pytest.raises(PersistenceInvariantError, match="mode"):
        open_writer.finalize(command)

    assert events_path.read_bytes() == b""
    assert _load_manifest(open_writer)["status"] == "incomplete"


@pytest.mark.parametrize("current_revision", [0, 1])
def test_finalize_rejects_same_revision_with_different_state_content(
    open_writer: SessionWriter,
    current_revision: int,
) -> None:
    """Revision equality must not let unpersisted discussion content finalize."""

    current_state = make_discussion_state(current_revision)
    if current_revision == 1:
        open_writer.replace_discussion_state(0, current_state)
    wrong_state = replace(
        current_state,
        current_focus="永続化されていない内容",
    )
    before = {
        path.name: path.read_bytes() for path in open_writer.session_dir.iterdir()
    }
    invalid_command = replace(
        make_finalize_command(),
        final_state=wrong_state,
    )

    with pytest.raises(PersistenceInvariantError, match="final discussion state"):
        open_writer.finalize(invalid_command)

    assert {
        path.name: path.read_bytes() for path in open_writer.session_dir.iterdir()
    } == before
    valid_command = replace(
        make_finalize_command(),
        final_state=current_state,
    )
    assert open_writer.finalize(valid_command).status is SessionStatus.COMPLETED


@pytest.mark.parametrize(
    ("completion_event", "match"),
    [
        (
            replace(
                make_finalize_command().completion_event,
                event_type=EventType.SESSION_START,
            ),
            "SESSION_COMPLETED",
        ),
        (make_finalize_command(event_sequence=2).completion_event, "event sequence"),
        (
            make_finalize_command(
                session_id="01J00000000000000000000009"
            ).completion_event,
            "session_id",
        ),
    ],
)
def test_finalize_rejects_invalid_completion_event_before_append(
    open_writer: SessionWriter,
    completion_event: EventRecord,
    match: str,
) -> None:
    """Completion identity and sequence errors must not consume the event slot."""

    events_path = open_writer.session_dir / "events.jsonl"
    command = replace(
        make_finalize_command(),
        completion_event=completion_event,
    )

    with pytest.raises(PersistenceInvariantError, match=match):
        open_writer.finalize(command)

    assert events_path.read_bytes() == b""
    assert _load_manifest(open_writer)["status"] == "incomplete"
    open_writer.append_event(make_event_record())


def test_finalize_rejects_completion_event_time_regression_before_append(
    open_writer: SessionWriter,
) -> None:
    """The terminal event must not move the event log's session clock backward."""

    open_writer.append_event(make_event_record(1, session_time_ms=1_800_001))
    events_path = open_writer.session_dir / "events.jsonl"
    before = events_path.read_bytes()

    with pytest.raises(PersistenceInvariantError, match="chronological"):
        open_writer.finalize(make_finalize_command(event_sequence=2))

    assert events_path.read_bytes() == before
    assert _load_manifest(open_writer)["status"] == "incomplete"


@pytest.mark.parametrize("tamper_target", ["duration", "ended_at", "pause"])
def test_finalize_revalidates_tampered_command_before_event_append(
    open_writer: SessionWriter,
    tamper_target: str,
) -> None:
    """Frozen-object tampering must not bypass canonical completion validation."""

    command = make_finalize_command()
    if tamper_target == "duration":
        object.__setattr__(command, "active_duration_ms", -1)
    elif tamper_target == "ended_at":
        object.__setattr__(command, "ended_at", datetime(2026, 8, 19, 12, 30))
    else:
        pause = PauseInterval(1, 2)
        object.__setattr__(pause, "started_ms", -1)
        object.__setattr__(command, "pause_intervals", (pause,))

    with pytest.raises(PersistenceInvariantError, match="finalize command"):
        open_writer.finalize(command)

    assert (open_writer.session_dir / "events.jsonl").read_bytes() == b""
    assert _load_manifest(open_writer)["status"] == "incomplete"


def test_finalize_rejects_end_before_start_before_event_append(
    open_writer: SessionWriter,
) -> None:
    """A completed manifest must not end before its persisted start time."""

    command = replace(
        make_finalize_command(),
        ended_at=datetime.fromisoformat("2026-08-19T11:59:59+09:00"),
    )

    with pytest.raises(PersistenceInvariantError, match="ended_at"):
        open_writer.finalize(command)

    assert (open_writer.session_dir / "events.jsonl").read_bytes() == b""
    assert _load_manifest(open_writer)["status"] == "incomplete"


def test_finalize_preflights_huge_duration_json_before_first_mutation(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late manifest encoding error must not consume or close session state."""

    calls: list[str] = []
    original_append = open_writer._append_completion_event
    original_sync = open_writer._sync_jsonl
    original_finalize_wavs = open_writer._finalize_wavs
    original_replace_state = open_writer._replace_state
    original_close_jsonl = open_writer._close_jsonl_resources
    original_write_manifest = open_writer._write_manifest

    def record_append(event: EventRecord) -> None:
        calls.append("completion-event")
        original_append(event)

    def record_sync() -> None:
        calls.append("sync-jsonl")
        original_sync()

    def record_finalize_wavs() -> None:
        calls.append("finalize-wavs")
        original_finalize_wavs()

    def record_replace_state(state: DiscussionState) -> None:
        calls.append("replace-state")
        original_replace_state(state)

    def record_close_jsonl() -> None:
        calls.append("close-jsonl")
        original_close_jsonl()

    def record_write_manifest(manifest: SessionManifest) -> None:
        calls.append("write-manifest")
        original_write_manifest(manifest)

    monkeypatch.setattr(open_writer, "_append_completion_event", record_append)
    monkeypatch.setattr(open_writer, "_sync_jsonl", record_sync)
    monkeypatch.setattr(open_writer, "_finalize_wavs", record_finalize_wavs)
    monkeypatch.setattr(open_writer, "_replace_state", record_replace_state)
    monkeypatch.setattr(open_writer, "_close_jsonl_resources", record_close_jsonl)
    monkeypatch.setattr(open_writer, "_write_manifest", record_write_manifest)
    before = {
        path.name: path.read_bytes() for path in open_writer.session_dir.iterdir()
    }
    invalid_command = replace(
        make_finalize_command(),
        active_duration_ms=10**5000,
    )

    with pytest.raises(PersistenceInvariantError, match="JSON"):
        open_writer.finalize(invalid_command)

    assert calls == []
    assert {
        path.name: path.read_bytes() for path in open_writer.session_dir.iterdir()
    } == before
    assert open_writer.finalize(make_finalize_command()).status is (
        SessionStatus.COMPLETED
    )


def test_finalize_preflights_completion_event_utf8_before_any_mutation(
    open_writer: SessionWriter,
) -> None:
    """A UTF-8 failure must leave files, resources, and Writer state reusable."""

    before_files = {
        path.name: path.read_bytes() for path in open_writer.session_dir.iterdir()
    }
    before_internal_state = (
        open_writer._manifest,
        open_writer._next_event_sequence,
        open_writer._last_event_session_time_ms,
        open_writer._state,
        open_writer._resources_closed,
        frozenset(open_writer._closed_resource_names),
    )
    jsonl_files = (
        open_writer._transcript_log._file,
        open_writer._state_history_log._file,
        open_writer._event_log._file,
    )
    wav_sinks = (open_writer._microphone_sink, open_writer._loopback_sink)
    valid_command = make_finalize_command()
    invalid_command = replace(
        valid_command,
        completion_event=replace(
            valid_command.completion_event,
            details={"invalid_utf8": "\ud800"},
        ),
    )

    with pytest.raises(PersistenceInvariantError, match="JSON"):
        open_writer.finalize(invalid_command)

    assert {
        path.name: path.read_bytes() for path in open_writer.session_dir.iterdir()
    } == before_files
    assert (
        open_writer._manifest,
        open_writer._next_event_sequence,
        open_writer._last_event_session_time_ms,
        open_writer._state,
        open_writer._resources_closed,
        frozenset(open_writer._closed_resource_names),
    ) == before_internal_state
    assert all(not file.closed for file in jsonl_files)
    assert all(not sink._closed and not sink._file.closed for sink in wav_sinks)
    assert open_writer.finalize(valid_command).status is SessionStatus.COMPLETED


@pytest.mark.parametrize(
    "failure_point",
    [
        "_append_completion_event",
        "_sync_jsonl",
        "_finalize_wavs",
        "_replace_state",
        "_close_jsonl_resources",
        "_write_manifest",
    ],
)
def test_finalize_failure_keeps_incomplete_manifest_and_terminalizes_writer(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """No partial finalization step may publish completed session metadata."""

    def fail_step(*args: object) -> None:
        del args
        raise OSError(f"{failure_point} failed")

    monkeypatch.setattr(open_writer, failure_point, fail_step, raising=False)

    with pytest.raises(OSError, match=f"{failure_point} failed"):
        open_writer.finalize(make_finalize_command())

    assert _load_manifest(open_writer)["status"] == "incomplete"
    with pytest.raises(RuntimeError, match="failed"):
        open_writer.sync_if_due(open_writer.opened_monotonic + 2.0)


def test_jsonl_close_failure_happens_before_completed_manifest_replace(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close failure after manifest replacement would publish false completion."""

    close_calls: list[str] = []
    original_close = JsonlAppender.close

    def fail_first_close(appender: JsonlAppender) -> None:
        close_calls.append(appender.path.name)
        original_close(appender)
        if appender.path.name == "transcript.jsonl":
            raise OSError("transcript close failed")

    monkeypatch.setattr(JsonlAppender, "close", fail_first_close)

    with pytest.raises(OSError, match="transcript close failed"):
        open_writer.finalize(make_finalize_command())

    assert close_calls == [
        "transcript.jsonl",
        "state-history.jsonl",
        "events.jsonl",
    ]
    assert _load_manifest(open_writer)["status"] == "incomplete"


def test_finalize_preserves_primary_error_across_all_resource_close_failures(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failures must be notes, not replacements for the durable-sync error."""

    primary_error = OSError("primary final sync failure")
    cleanup_calls: list[str] = []

    def fail_sync_jsonl() -> None:
        raise primary_error

    monkeypatch.setattr(open_writer, "_sync_jsonl", fail_sync_jsonl, raising=False)
    resources = (
        (open_writer._event_log, "close", "events.jsonl"),
        (open_writer._state_history_log, "close", "state-history.jsonl"),
        (open_writer._transcript_log, "close", "transcript.jsonl"),
        (open_writer._loopback_sink, "close_incomplete", "loopback.wav"),
        (open_writer._microphone_sink, "close_incomplete", "mic.wav"),
    )
    for resource, method_name, resource_name in resources:
        original_operation = getattr(resource, method_name)

        def fail_after_close(
            *,
            name: str = resource_name,
            operation: object = original_operation,
        ) -> None:
            cleanup_calls.append(name)
            assert callable(operation)
            operation()
            raise OSError(f"{name} close failed")

        monkeypatch.setattr(resource, method_name, fail_after_close)

    with pytest.raises(OSError, match="primary final sync failure") as raised:
        open_writer.finalize(make_finalize_command())

    assert raised.value is primary_error
    assert cleanup_calls == [
        "events.jsonl",
        "state-history.jsonl",
        "transcript.jsonl",
        "loopback.wav",
        "mic.wav",
    ]
    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    for resource_name in cleanup_calls:
        assert resource_name in notes
    assert _load_manifest(open_writer)["status"] == "incomplete"


def test_successful_finalize_closes_each_resource_once_and_rejects_reentry(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal API calls must never finalize or close an owned handle twice."""

    close_calls: list[str] = []
    resources = (
        (open_writer._microphone_sink, "finalize", "mic.wav"),
        (open_writer._loopback_sink, "finalize", "loopback.wav"),
        (open_writer._transcript_log, "close", "transcript.jsonl"),
        (open_writer._state_history_log, "close", "state-history.jsonl"),
        (open_writer._event_log, "close", "events.jsonl"),
    )
    for resource, method_name, resource_name in resources:
        original_operation = getattr(resource, method_name)

        def record_close(
            *,
            name: str = resource_name,
            operation: object = original_operation,
        ) -> None:
            close_calls.append(name)
            assert callable(operation)
            operation()

        monkeypatch.setattr(resource, method_name, record_close)

    open_writer.finalize(make_finalize_command())
    open_writer.close_incomplete()
    open_writer.close_incomplete()

    with pytest.raises(RuntimeError, match="closed"):
        open_writer.finalize(make_finalize_command())
    with pytest.raises(RuntimeError, match="closed"):
        open_writer.sync_if_due(open_writer.opened_monotonic + 2.0)

    assert close_calls == [
        "mic.wav",
        "loopback.wav",
        "transcript.jsonl",
        "state-history.jsonl",
        "events.jsonl",
    ]


def test_force_close_path_never_marks_session_completed(
    open_writer: SessionWriter,
) -> None:
    """An abnormal close must remain recoverable and idempotent."""

    open_writer.close_incomplete()
    assert _load_manifest(open_writer)["status"] == "incomplete"
    open_writer.close_incomplete()


def test_finalize_rejects_non_owner_before_validation_or_event_append(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the opening process may publish terminal session metadata."""

    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.getpid",
        lambda: open_writer.owner_pid + 1,
    )

    with pytest.raises(WriterOwnershipError, match="owner PID"):
        open_writer.finalize(make_finalize_command())

    assert (open_writer.session_dir / "events.jsonl").read_bytes() == b""
    assert _load_manifest(open_writer)["status"] == "incomplete"
