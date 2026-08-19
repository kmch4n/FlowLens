import math
import pickle
from collections.abc import Callable
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

import flowlens.domain.messages as messages
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import (
    AudioSource,
    EventType,
    MessageType,
    ProcessSource,
    SessionMode,
    SessionStatus,
)
from flowlens.domain.messages import (
    AudioWriteCommand,
    DiscussionStateReplaced,
    EventRecord,
    MessageEnvelope,
    MessageSequenceError,
    SequenceResult,
    SequenceTracker,
    TranscriptCommitted,
    TranscriptRecord,
    UnknownSchemaVersionError,
    WriterAck,
    WriterAppendEvent,
    WriterFatal,
    WriterFinalize,
    WriterFlush,
    WriterOpenSession,
    WriterShutdown,
)
from flowlens.domain.session import (
    DeviceIdentity,
    ModelIdentity,
    PauseInterval,
    SessionManifest,
)

NOW = datetime.fromisoformat("2026-08-19T12:34:56.789+09:00")
SESSION_ID = "01J00000000000000000000000"
SEGMENT_ID = "01J00000000000000000000001"


def make_transcript_record(sequence: int = 1) -> TranscriptRecord:
    return TranscriptRecord(
        schema_version=1,
        segment_id=SEGMENT_ID,
        sequence=sequence,
        source=AudioSource.ME,
        text="今回の方針を確認します。",
        session_start_ms=12_480,
        session_end_ms=15_820,
        source_start_sample=182_400,
        source_end_sample=235_840,
        committed_at=NOW,
    )


def make_event_record(sequence: int = 1) -> EventRecord:
    return EventRecord(
        schema_version=1,
        session_id=SESSION_ID,
        sequence=sequence,
        event_type=EventType.ASR_LAG_STARTED,
        source=ProcessSource.ASR,
        session_time_ms=125_400,
        created_at=NOW,
        details={"backlog_ms": 5_200},
    )


def make_self_cyclic_details() -> dict[str, messages.JsonValue]:
    cycle: list[messages.JsonValue] = []
    cycle.append(cycle)
    return {"cycle": cycle}


def make_mutually_cyclic_details() -> dict[str, messages.JsonValue]:
    cycle_list: list[messages.JsonValue] = []
    cycle_dict: dict[str, messages.JsonValue] = {"list": cycle_list}
    cycle_list.append(cycle_dict)
    return {"cycle": cycle_dict}


def make_discussion_state(revision: int = 1) -> DiscussionState:
    return DiscussionState(
        revision=revision,
        mode=SessionMode.MEETING,
        current_focus="今回の方針",
        key_points=("完全ローカル動作",),
        confirmed_outcomes=(),
        follow_up_items=(),
        updated_at=NOW,
    )


def make_manifest() -> SessionManifest:
    return SessionManifest(
        schema_version=1,
        session_id=SESSION_ID,
        status=SessionStatus.INCOMPLETE,
        mode=SessionMode.MEETING,
        started_at=NOW,
        ended_at=None,
        active_duration_ms=0,
        pause_intervals=(),
        microphone=DeviceIdentity("mic-1", "USB Microphone"),
        loopback_output=DeviceIdentity("out-1", "Speakers"),
        asr_model=ModelIdentity("asr/repository", "rev-a", "a" * 64),
        discussion_model=ModelIdentity(
            "discussion/repository",
            "rev-b",
            "b" * 64,
        ),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=(),
    )


def envelope(
    sequence: int,
    *,
    session_id: str = SESSION_ID,
    source: ProcessSource = ProcessSource.ASR,
) -> MessageEnvelope[TranscriptCommitted]:
    return MessageEnvelope(
        1,
        session_id,
        MessageType.TRANSCRIPT_COMMITTED,
        sequence,
        source,
        15_540,
        TranscriptCommitted(record=make_transcript_record(sequence)),
    )


def test_transcript_record_matches_exact_spec_shape() -> None:
    record = make_transcript_record(42)

    assert TranscriptRecord.from_dict(record.to_dict()) == record
    assert list(record.to_dict()) == [
        "schema_version",
        "segment_id",
        "sequence",
        "source",
        "text",
        "session_start_ms",
        "session_end_ms",
        "source_start_sample",
        "source_end_sample",
        "committed_at",
    ]
    assert record.to_dict()["source"] == "ME"
    assert record.to_dict()["committed_at"] == "2026-08-19T12:34:56.789+09:00"


def test_transcript_record_normalizes_wall_clock_to_milliseconds() -> None:
    record = replace(
        make_transcript_record(),
        committed_at=datetime.fromisoformat("2026-08-19T12:34:56.789999+09:00"),
    )

    assert record.committed_at == NOW
    assert TranscriptRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: replace(value, schema_version=2), "schema_version"),
        (lambda value: replace(value, segment_id="invalid"), "segment_id"),
        (lambda value: replace(value, sequence=0), "sequence"),
        (
            lambda value: replace(value, source=cast(AudioSource, "ME")),
            "source",
        ),
        (lambda value: replace(value, text="  \t"), "text"),
        (lambda value: replace(value, session_start_ms=-1), "session_start_ms"),
        (lambda value: replace(value, session_end_ms=12_479), "session_end_ms"),
        (
            lambda value: replace(value, source_start_sample=-1),
            "source_start_sample",
        ),
        (
            lambda value: replace(value, source_end_sample=182_399),
            "source_end_sample",
        ),
        (
            lambda value: replace(
                value,
                committed_at=datetime.fromisoformat("2026-08-19T12:34:56.789"),
            ),
            "committed_at",
        ),
    ],
)
def test_transcript_record_rejects_invalid_invariants(
    mutate: Callable[[TranscriptRecord], TranscriptRecord],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        mutate(make_transcript_record())


@pytest.mark.parametrize("changed_key", ["missing", "unknown"])
def test_transcript_record_from_dict_requires_exact_keys(changed_key: str) -> None:
    serialized = make_transcript_record().to_dict()
    if changed_key == "missing":
        del serialized["text"]
    else:
        serialized["partial"] = False

    with pytest.raises(ValueError, match=changed_key):
        TranscriptRecord.from_dict(serialized)


def test_event_record_round_trips_operational_json_values() -> None:
    event = EventRecord(
        1,
        SESSION_ID,
        18,
        EventType.ASR_LAG_STARTED,
        ProcessSource.ASR,
        125_400,
        NOW,
        {
            "backlog_ms": 5_200,
            "recovering": True,
            "ratio": 0.25,
            "worker": None,
            "nested": ["ASR", {"attempt": 1}],
        },
    )

    assert EventRecord.from_dict(event.to_dict()) == event
    assert list(event.to_dict()) == [
        "schema_version",
        "session_id",
        "sequence",
        "event_type",
        "source",
        "session_time_ms",
        "created_at",
        "details",
    ]
    assert event.to_dict()["event_type"] == "ASR_LAG_STARTED"


def test_event_details_are_defensively_copied_recursively() -> None:
    nested: list[messages.JsonValue] = ["ASR", {"attempt": 1}]
    details: dict[str, messages.JsonValue] = {"nested": nested}
    event = replace(make_event_record(), details=details)

    nested.append("later input mutation")
    details["later"] = True
    serialized = event.to_dict()
    serialized_details = cast(dict[str, object], serialized["details"])
    cast(list[object], serialized_details["nested"]).append("output mutation")

    assert event.details == {"nested": ["ASR", {"attempt": 1}]}
    assert event.to_dict()["details"] == {"nested": ["ASR", {"attempt": 1}]}


@pytest.mark.parametrize(
    "make_details",
    [make_self_cyclic_details, make_mutually_cyclic_details],
    ids=["self-cyclic-list", "mutually-cyclic-list-dict"],
)
def test_event_details_reject_cyclic_json_containers(
    make_details: Callable[[], dict[str, messages.JsonValue]],
) -> None:
    with pytest.raises(ValueError, match="details.*cycle"):
        replace(make_event_record(), details=make_details())


def test_event_details_accept_repeated_shared_non_cyclic_container() -> None:
    shared: list[messages.JsonValue] = [{"attempt": 1}]
    event = replace(
        make_event_record(),
        details={"first": shared, "second": shared},
    )

    assert event.details == {"first": [{"attempt": 1}], "second": [{"attempt": 1}]}


@pytest.mark.parametrize(
    "invalid_details",
    [
        {"value": math.nan},
        {"value": math.inf},
        {"value": object()},
        {"value": (1, 2)},
        cast(dict[str, messages.JsonValue], {1: "not a string key"}),
        {"nested": [{"value": -math.inf}]},
    ],
)
def test_event_details_reject_non_finite_or_non_json_values(
    invalid_details: dict[str, messages.JsonValue],
) -> None:
    with pytest.raises(ValueError, match="details"):
        replace(make_event_record(), details=invalid_details)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: replace(value, schema_version=2), "schema_version"),
        (lambda value: replace(value, session_id="invalid"), "session_id"),
        (lambda value: replace(value, sequence=0), "sequence"),
        (
            lambda value: replace(value, event_type=cast(EventType, "PAUSE_START")),
            "event_type",
        ),
        (
            lambda value: replace(value, source=cast(ProcessSource, "ASR")),
            "source",
        ),
        (lambda value: replace(value, session_time_ms=-1), "session_time_ms"),
        (
            lambda value: replace(
                value,
                created_at=datetime.fromisoformat("2026-08-19T12:34:56.789"),
            ),
            "created_at",
        ),
    ],
)
def test_event_record_rejects_invalid_invariants(
    mutate: Callable[[EventRecord], EventRecord],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        mutate(make_event_record())


def test_event_record_normalizes_wall_clock_to_milliseconds() -> None:
    event = replace(
        make_event_record(),
        created_at=datetime.fromisoformat("2026-08-19T12:34:56.789999+09:00"),
    )

    assert event.created_at == NOW
    assert EventRecord.from_dict(event.to_dict()) == event


@pytest.mark.parametrize("changed_key", ["missing", "unknown"])
def test_event_record_from_dict_requires_exact_keys(changed_key: str) -> None:
    serialized = make_event_record().to_dict()
    if changed_key == "missing":
        del serialized["details"]
    else:
        serialized["text"] = "must not contain transcript content"

    with pytest.raises(ValueError, match=changed_key):
        EventRecord.from_dict(serialized)


def test_message_envelope_has_exact_generic_field_contract() -> None:
    payload = TranscriptCommitted(record=make_transcript_record())
    value: MessageEnvelope[TranscriptCommitted] = MessageEnvelope(
        1,
        SESSION_ID,
        MessageType.TRANSCRIPT_COMMITTED,
        1,
        ProcessSource.ASR,
        15_540,
        payload,
    )

    assert [field.name for field in fields(MessageEnvelope)] == [
        "schema_version",
        "session_id",
        "message_type",
        "sequence",
        "source",
        "created_monotonic_ms",
        "payload",
    ]
    assert value.payload is payload
    value.validate_schema()


def test_unknown_envelope_schema_is_retained_until_explicit_validation() -> None:
    invalid = MessageEnvelope(
        2,
        SESSION_ID,
        MessageType.TRANSCRIPT_COMMITTED,
        1,
        ProcessSource.ASR,
        100,
        TranscriptCommitted(record=make_transcript_record()),
    )

    assert invalid.schema_version == 2
    with pytest.raises(UnknownSchemaVersionError, match="2"):
        invalid.validate_schema()
    with pytest.raises(UnknownSchemaVersionError, match="2"):
        SequenceTracker().observe(invalid)


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: MessageEnvelope(
                1,
                "invalid",
                MessageType.TRANSCRIPT_COMMITTED,
                1,
                ProcessSource.ASR,
                100,
                TranscriptCommitted(make_transcript_record()),
            ),
            ValueError,
            "session_id",
        ),
        (
            lambda: MessageEnvelope(
                1,
                SESSION_ID,
                cast(MessageType, "TRANSCRIPT_COMMITTED"),
                1,
                ProcessSource.ASR,
                100,
                TranscriptCommitted(make_transcript_record()),
            ),
            ValueError,
            "message_type",
        ),
        (
            lambda: MessageEnvelope(
                1,
                SESSION_ID,
                MessageType.TRANSCRIPT_COMMITTED,
                0,
                ProcessSource.ASR,
                100,
                TranscriptCommitted(make_transcript_record()),
            ),
            MessageSequenceError,
            "sequence",
        ),
        (
            lambda: MessageEnvelope(
                1,
                SESSION_ID,
                MessageType.TRANSCRIPT_COMMITTED,
                1,
                cast(ProcessSource, "ASR"),
                100,
                TranscriptCommitted(make_transcript_record()),
            ),
            ValueError,
            "source",
        ),
        (
            lambda: MessageEnvelope(
                1,
                SESSION_ID,
                MessageType.TRANSCRIPT_COMMITTED,
                1,
                ProcessSource.ASR,
                -1,
                TranscriptCommitted(make_transcript_record()),
            ),
            ValueError,
            "created_monotonic_ms",
        ),
    ],
)
def test_message_envelope_rejects_invalid_non_schema_fields(
    factory: Callable[[], object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()


def test_sequence_tracker_detects_duplicate_and_gap_per_sender() -> None:
    tracker = SequenceTracker()

    assert tracker.expected(ProcessSource.ASR, SESSION_ID) == 1
    assert tracker.observe(envelope(1)) is SequenceResult.ACCEPTED
    assert tracker.observe(envelope(1)) is SequenceResult.DUPLICATE
    assert tracker.observe(envelope(3)) is SequenceResult.GAP
    assert tracker.expected(ProcessSource.ASR, SESSION_ID) == 4
    assert tracker.observe(envelope(2)) is SequenceResult.DUPLICATE
    assert tracker.observe(envelope(4)) is SequenceResult.ACCEPTED


def test_sequence_tracker_keeps_independent_session_and_sender_state() -> None:
    tracker = SequenceTracker()
    other_session = "01J00000000000000000000002"

    assert tracker.observe(envelope(1)) is SequenceResult.ACCEPTED
    assert (
        tracker.observe(envelope(1, source=ProcessSource.DISCUSSION))
        is SequenceResult.ACCEPTED
    )
    assert (
        tracker.observe(envelope(1, session_id=other_session))
        is SequenceResult.ACCEPTED
    )
    assert tracker.expected(ProcessSource.ASR, SESSION_ID) == 2
    assert tracker.expected(ProcessSource.DISCUSSION, SESSION_ID) == 2
    assert tracker.expected(ProcessSource.ASR, other_session) == 2


def test_audio_command_is_picklable_and_has_exact_dedicated_queue_fields() -> None:
    command = AudioWriteCommand(
        AudioSource.ME,
        b"\x00\x00" * 320,
        0,
        320,
        0,
        15_540,
    )

    assert pickle.loads(pickle.dumps(command)) == command
    assert [field.name for field in fields(AudioWriteCommand)] == [
        "source",
        "pcm_s16le",
        "source_start_sample",
        "source_end_sample",
        "session_start_ms",
        "captured_monotonic_ms",
    ]


@pytest.mark.parametrize(
    ("command", "match"),
    [
        (
            lambda: AudioWriteCommand(
                cast(AudioSource, "ME"),
                b"\x00\x00",
                0,
                1,
                0,
                1,
            ),
            "source",
        ),
        (
            lambda: AudioWriteCommand(AudioSource.ME, b"\x00", 0, 0, 0, 1),
            "even number of bytes",
        ),
        (
            lambda: AudioWriteCommand(
                AudioSource.OTHERS,
                b"\x00\x00" * 320,
                0,
                319,
                0,
                15_540,
            ),
            "source sample range",
        ),
        (
            lambda: AudioWriteCommand(AudioSource.ME, b"", -1, -1, 0, 1),
            "source_start_sample",
        ),
        (
            lambda: AudioWriteCommand(AudioSource.ME, b"", 0, 0, -1, 1),
            "session_start_ms",
        ),
        (
            lambda: AudioWriteCommand(AudioSource.ME, b"", 0, 0, 0, -1),
            "captured_monotonic_ms",
        ),
    ],
)
def test_audio_command_rejects_invalid_boundaries(
    command: Callable[[], AudioWriteCommand],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        command()


@pytest.mark.parametrize(
    ("record_type", "expected_fields"),
    [
        (TranscriptCommitted, ["record"]),
        (DiscussionStateReplaced, ["previous_revision", "state"]),
        (WriterOpenSession, ["session_dir", "manifest", "initial_state"]),
        (WriterAppendEvent, ["record"]),
        (WriterFlush, []),
        (
            WriterFinalize,
            [
                "ended_at",
                "active_duration_ms",
                "pause_intervals",
                "final_state",
                "completion_event",
            ],
        ),
        (WriterShutdown, []),
        (WriterAck, ["acknowledged_sequence", "latest_successful_save_at"]),
        (WriterFatal, ["failed_sequence", "error_type", "message"]),
    ],
)
def test_typed_payloads_have_exact_fields(
    record_type: type[object],
    expected_fields: list[str],
) -> None:
    assert [field.name for field in fields(cast(Any, record_type))] == expected_fields


def test_typed_payloads_preserve_valid_contract_values() -> None:
    transcript = make_transcript_record()
    state = make_discussion_state()
    manifest = make_manifest()
    event = make_event_record()

    assert TranscriptCommitted(transcript).record is transcript
    assert DiscussionStateReplaced(0, state).state is state
    assert WriterOpenSession(Path("session"), manifest, state).manifest is manifest
    assert WriterAppendEvent(event).record is event
    assert WriterFlush() == WriterFlush()
    assert WriterShutdown() == WriterShutdown()


def test_writer_finalize_copies_pauses_and_normalizes_wall_clock() -> None:
    pauses = [PauseInterval(100, 200)]
    command = WriterFinalize(
        ended_at=datetime.fromisoformat("2026-08-19T12:34:56.789999+09:00"),
        active_duration_ms=1_000,
        pause_intervals=cast(tuple[PauseInterval, ...], pauses),
        final_state=make_discussion_state(),
        completion_event=replace(
            make_event_record(),
            event_type=EventType.SESSION_COMPLETED,
        ),
    )

    pauses.append(PauseInterval(300, 400))

    assert command.ended_at == NOW
    assert command.pause_intervals == (PauseInterval(100, 200),)


def test_writer_ack_normalizes_wall_clock_to_milliseconds() -> None:
    ack = WriterAck(
        3,
        datetime.fromisoformat("2026-08-19T12:34:56.789999+09:00"),
    )

    assert ack.latest_successful_save_at == NOW


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: TranscriptCommitted(cast(TranscriptRecord, object())), "record"),
        (
            lambda: DiscussionStateReplaced(1, make_discussion_state(revision=1)),
            "previous_revision",
        ),
        (
            lambda: WriterOpenSession(
                cast(Path, "session"),
                make_manifest(),
                make_discussion_state(),
            ),
            "session_dir",
        ),
        (lambda: WriterAppendEvent(cast(EventRecord, object())), "record"),
        (
            lambda: WriterFinalize(
                datetime.fromisoformat("2026-08-19T12:34:56.789"),
                1_000,
                (),
                make_discussion_state(),
                make_event_record(),
            ),
            "ended_at",
        ),
        (
            lambda: WriterFinalize(
                NOW,
                -1,
                (),
                make_discussion_state(),
                make_event_record(),
            ),
            "active_duration_ms",
        ),
        (
            lambda: WriterAck(0, NOW),
            "acknowledged_sequence",
        ),
        (
            lambda: WriterFatal(0, "OSError", "disk full"),
            "failed_sequence",
        ),
        (
            lambda: WriterFatal(1, "", "disk full"),
            "error_type",
        ),
        (
            lambda: WriterFatal(1, "OSError", ""),
            "message",
        ),
    ],
)
def test_typed_payloads_reject_invalid_values(
    factory: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_persistence_contract_does_not_define_partial_transcript_payload() -> None:
    assert not hasattr(messages, "TranscriptPartial")
