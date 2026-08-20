"""SessionWriter append invariant and persistence-order tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from flowlens.domain.discussion import StateHistoryRecord
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import AudioWriteCommand
from flowlens.persistence.json_files import AtomicJsonFile, JsonlAppender
from flowlens.persistence.session_writer import PersistenceInvariantError, SessionWriter
from tests.factories import (
    make_discussion_state,
    make_event_record,
    make_manifest,
    make_transcript_record,
)


def _append_audio_through(
    writer: SessionWriter,
    source: AudioSource,
    end_sample: int,
) -> None:
    writer.append_audio(
        AudioWriteCommand(
            source,
            b"\x00\x00" * end_sample,
            0,
            end_sample,
            0,
            1_800,
        )
    )


def test_append_routes_audio_and_enforces_source_sample_contiguity(
    open_writer: SessionWriter,
) -> None:
    """A source gap must be rejected without extending either WAV."""

    open_writer.append_audio(
        AudioWriteCommand(
            AudioSource.ME,
            b"\x00\x00" * 320,
            0,
            320,
            0,
            15_540,
        )
    )
    with pytest.raises(
        PersistenceInvariantError,
        match="expected source_start_sample 320",
    ):
        open_writer.append_audio(
            AudioWriteCommand(
                AudioSource.ME,
                b"\x00\x00" * 320,
                640,
                960,
                40,
                15_580,
            )
        )
    open_writer.append_audio(
        AudioWriteCommand(
            AudioSource.OTHERS,
            b"\x01\x00" * 160,
            0,
            160,
            0,
            15_540,
        )
    )

    open_writer.close_incomplete()
    assert (open_writer.session_dir / "mic.wav").read_bytes()[44:] == (
        b"\x00\x00" * 320
    )
    assert (open_writer.session_dir / "loopback.wav").read_bytes()[44:] == (
        b"\x01\x00" * 160
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_end_sample", 2, "PCM sample count"),
        ("source_start_sample", "0", "source_start_sample"),
        ("session_start_ms", -1, "session_start_ms"),
        ("captured_monotonic_ms", -1, "captured_monotonic_ms"),
    ],
)
def test_append_audio_defensively_revalidates_command_before_writing(
    open_writer: SessionWriter,
    field: str,
    value: object,
    match: str,
) -> None:
    """Bypassing immutable-record construction must not bypass Writer safety."""

    command = AudioWriteCommand(AudioSource.ME, b"\x00\x00", 0, 1, 0, 0)
    object.__setattr__(command, field, value)
    before = (open_writer.session_dir / "mic.wav").stat().st_size

    with pytest.raises(PersistenceInvariantError, match=match):
        open_writer.append_audio(command)

    assert (open_writer.session_dir / "mic.wav").stat().st_size == before


def test_transcript_sequence_is_strictly_monotonic(
    open_writer: SessionWriter,
) -> None:
    """A rejected duplicate sequence must not consume or duplicate the next slot."""

    _append_audio_through(open_writer, AudioSource.ME, 25_600)
    first = make_transcript_record(sequence=1)
    open_writer.append_transcript(first)
    with pytest.raises(
        PersistenceInvariantError,
        match="expected transcript sequence 2",
    ):
        open_writer.append_transcript(first)
    open_writer.append_transcript(make_transcript_record(sequence=2))
    lines = (
        (open_writer.session_dir / "transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(line)["sequence"] for line in lines] == [1, 2]


def test_transcript_requires_audio_to_be_persisted_first(
    open_writer: SessionWriter,
) -> None:
    """A transcript must not reference source samples absent from its WAV."""

    with pytest.raises(PersistenceInvariantError, match="persisted audio cursor 0"):
        open_writer.append_transcript(make_transcript_record())

    assert not (open_writer.session_dir / "transcript.jsonl").read_bytes()


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("session_start_ms", -1)],
)
def test_transcript_strictly_revalidates_tampered_record_before_writing(
    open_writer: SessionWriter,
    field: str,
    value: int,
) -> None:
    """Invalid persisted transcript fields must not mutate sequence or disk."""

    _append_audio_through(open_writer, AudioSource.ME, 12_800)
    record = make_transcript_record()
    object.__setattr__(record, field, value)

    with pytest.raises(PersistenceInvariantError, match="transcript"):
        open_writer.append_transcript(record)

    assert not (open_writer.session_dir / "transcript.jsonl").read_bytes()
    open_writer.append_transcript(make_transcript_record())


def test_transcript_rejects_duplicate_segment_id_before_writing(
    open_writer: SessionWriter,
) -> None:
    """A new sequence must not reuse a durable transcript segment identity."""

    _append_audio_through(open_writer, AudioSource.ME, 25_600)
    first = make_transcript_record(1)
    duplicate_id = replace(make_transcript_record(2), segment_id=first.segment_id)
    open_writer.append_transcript(first)

    with pytest.raises(PersistenceInvariantError, match="segment_id"):
        open_writer.append_transcript(duplicate_id)

    lines = (
        (open_writer.session_dir / "transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1


def test_transcript_rejects_cross_source_tie_in_wrong_display_order(
    open_writer: SessionWriter,
) -> None:
    """Equal-time OTHERS then ME records would violate the merged source order."""

    _append_audio_through(open_writer, AudioSource.ME, 25_600)
    _append_audio_through(open_writer, AudioSource.OTHERS, 12_800)
    others = replace(make_transcript_record(1), source=AudioSource.OTHERS)
    me = replace(
        make_transcript_record(2),
        session_start_ms=others.session_start_ms,
        session_end_ms=others.session_end_ms,
    )
    open_writer.append_transcript(others)

    with pytest.raises(PersistenceInvariantError, match="chronological order"):
        open_writer.append_transcript(me)

    lines = (
        (open_writer.session_dir / "transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1


def test_event_sequence_session_and_time_are_validated_before_writing(
    open_writer: SessionWriter,
) -> None:
    """Event identity/order mistakes must leave the durable event log unchanged."""

    open_writer.append_event(make_event_record(1, session_time_ms=2_000))
    invalid_records = (
        (make_event_record(1, session_time_ms=2_000), "expected event sequence 2"),
        (
            make_event_record(
                2,
                session_id="01J00000000000000000000009",
                session_time_ms=2_000,
            ),
            "session_id",
        ),
        (make_event_record(2, session_time_ms=1_999), "chronological order"),
    )
    for record, match in invalid_records:
        with pytest.raises(PersistenceInvariantError, match=match):
            open_writer.append_event(record)

    open_writer.append_event(make_event_record(2, session_time_ms=2_000))
    lines = (
        (open_writer.session_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(line)["sequence"] for line in lines] == [1, 2]


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("session_time_ms", -1)],
)
def test_event_strictly_revalidates_tampered_record_before_writing(
    open_writer: SessionWriter,
    field: str,
    value: int,
) -> None:
    """Invalid persisted event fields must not mutate sequence or disk."""

    record = make_event_record()
    object.__setattr__(record, field, value)

    with pytest.raises(PersistenceInvariantError, match="event"):
        open_writer.append_event(record)

    assert not (open_writer.session_dir / "events.jsonl").read_bytes()
    open_writer.append_event(make_event_record())


def test_discussion_replacement_writes_history_then_snapshot(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reversing the crash-recovery order must fail this observable call trace."""

    order: list[str] = []
    original_append = JsonlAppender.append
    original_replace = AtomicJsonFile.replace

    def record_history(appender: JsonlAppender, value: object) -> None:
        if appender.path.name == "state-history.jsonl":
            order.append("history")
        original_append(appender, value)

    def record_snapshot(file: AtomicJsonFile, value: object) -> None:
        if file.path.name == "discussion-state.json":
            order.append("snapshot")
        original_replace(file, value)

    monkeypatch.setattr(JsonlAppender, "append", record_history)
    monkeypatch.setattr(AtomicJsonFile, "replace", record_snapshot)
    state = make_discussion_state(revision=1)

    open_writer.replace_discussion_state(previous_revision=0, state=state)

    snapshot = json.loads(
        (open_writer.session_dir / "discussion-state.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (open_writer.session_dir / "state-history.jsonl").read_text(encoding="utf-8")
    )
    assert order == ["history", "snapshot"]
    assert snapshot["revision"] == 1
    assert history["previous_revision"] == 0
    assert history["new_revision"] == 1


def test_discussion_invariant_failure_does_not_consume_revision(
    open_writer: SessionWriter,
) -> None:
    """A rejected stale revision must leave the valid next replacement usable."""

    with pytest.raises(PersistenceInvariantError, match="previous_revision 0"):
        open_writer.replace_discussion_state(
            previous_revision=1,
            state=make_discussion_state(2),
        )

    open_writer.replace_discussion_state(
        previous_revision=0,
        state=make_discussion_state(1),
    )
    lines = (
        (open_writer.session_dir / "state-history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1


def test_discussion_state_is_strictly_revalidated_before_history_write(
    open_writer: SessionWriter,
) -> None:
    """Invalid nested state values must not reach history or live snapshot."""

    state = make_discussion_state(1)
    object.__setattr__(state, "key_points", (1,))

    with pytest.raises(PersistenceInvariantError, match="discussion state"):
        open_writer.replace_discussion_state(0, state)

    assert not (open_writer.session_dir / "state-history.jsonl").read_bytes()
    snapshot = json.loads(
        (open_writer.session_dir / "discussion-state.json").read_text(encoding="utf-8")
    )
    assert snapshot["revision"] == 0
    open_writer.replace_discussion_state(0, make_discussion_state(1))


def test_state_history_is_strictly_reparsed_before_append(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Writer must validate its constructed history wire object before I/O."""

    original_to_dict = StateHistoryRecord.to_dict
    call_count = 0

    def tamper_first_wire_record(record: StateHistoryRecord) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        value = original_to_dict(record)
        if call_count == 1:
            value["schema_version"] = 2
        return value

    monkeypatch.setattr(StateHistoryRecord, "to_dict", tamper_first_wire_record)

    with pytest.raises(PersistenceInvariantError, match="state history"):
        open_writer.replace_discussion_state(0, make_discussion_state(1))

    assert not (open_writer.session_dir / "state-history.jsonl").read_bytes()
    snapshot = json.loads(
        (open_writer.session_dir / "discussion-state.json").read_text(encoding="utf-8")
    )
    assert snapshot["revision"] == 0


def test_snapshot_failure_leaves_history_and_fails_writer_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed live snapshot must preserve recovery history and stop mutations."""

    writer = SessionWriter.open(
        tmp_path / "session",
        make_manifest(),
        make_discussion_state(),
    )
    original_replace = AtomicJsonFile.replace

    def fail_replacement(file: AtomicJsonFile, value: object) -> None:
        if file.path.name == "discussion-state.json":
            raise OSError("snapshot replace failed")
        original_replace(file, value)

    monkeypatch.setattr(AtomicJsonFile, "replace", fail_replacement)

    with pytest.raises(OSError, match="snapshot replace failed"):
        writer.replace_discussion_state(0, make_discussion_state(1))

    history = json.loads(
        (writer.session_dir / "state-history.jsonl").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (writer.session_dir / "discussion-state.json").read_text(encoding="utf-8")
    )
    assert history["new_revision"] == 1
    assert snapshot["revision"] == 0
    with pytest.raises(RuntimeError, match="failed"):
        writer.append_event(make_event_record())
    for name in ("mic.wav", "loopback.wav", "events.jsonl"):
        path = writer.session_dir / name
        path.unlink()
        assert not path.exists()


def test_jsonl_write_failure_fails_writer_closed_without_snapshot_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A history write error must not be hidden or followed by snapshot publish."""

    writer = SessionWriter.open(
        tmp_path / "session",
        make_manifest(),
        make_discussion_state(),
    )
    original_append = JsonlAppender.append

    def fail_history_append(appender: JsonlAppender, value: object) -> None:
        if appender.path.name == "state-history.jsonl":
            raise OSError("history append failed")
        original_append(appender, value)

    monkeypatch.setattr(JsonlAppender, "append", fail_history_append)

    with pytest.raises(OSError, match="history append failed"):
        writer.replace_discussion_state(0, make_discussion_state(1))

    assert not (writer.session_dir / "state-history.jsonl").read_bytes()
    snapshot = json.loads(
        (writer.session_dir / "discussion-state.json").read_text(encoding="utf-8")
    )
    assert snapshot["revision"] == 0
    with pytest.raises(RuntimeError, match="failed"):
        writer.append_event(make_event_record())
