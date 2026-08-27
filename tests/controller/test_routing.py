"""Strict sender-local routing tests."""

import random
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from flowlens.controller.routing import (
    PayloadValidationError,
    SequenceTracker,
    rewrap_for_gui,
    validate_worker_payload,
)
from flowlens.discussion.contracts import DiscussionStatusPayload
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    UnknownSchemaVersionError,
)

SESSION_ID = "01J00000000000000000000000"


def envelope(
    *,
    source: ProcessSource = ProcessSource.ASR,
    sequence: int = 1,
    message_type: MessageType = MessageType.WORKER_READY,
    payload: object | None = None,
    schema_version: int = 1,
    session_id: str = SESSION_ID,
) -> MessageEnvelope[object]:
    """Build one deterministic worker envelope."""

    return MessageEnvelope(
        schema_version=schema_version,
        session_id=session_id,
        message_type=message_type,
        sequence=sequence,
        source=source,
        created_monotonic_ms=100,
        payload={"worker": source.value} if payload is None else payload,
    )


def transcript_record() -> TranscriptRecord:
    """Build one valid committed transcript record."""

    return TranscriptRecord(
        schema_version=1,
        segment_id="01J00000000000000000000001",
        sequence=1,
        source=AudioSource.ME,
        text="確認します",
        session_start_ms=100,
        session_end_ms=300,
        source_start_sample=0,
        source_end_sample=3_200,
        committed_at=datetime.fromisoformat("2026-08-19T12:00:00+09:00"),
    )


def test_sequence_tracker_rejects_duplicate_and_reports_exact_gap() -> None:
    tracker = SequenceTracker(SESSION_ID)

    first = tracker.accept(envelope(sequence=1))
    duplicate = tracker.accept(envelope(sequence=1))
    gap = tracker.accept(envelope(sequence=3))

    assert (first.accepted, first.duplicate, first.gap) == (True, False, None)
    assert (duplicate.accepted, duplicate.duplicate, duplicate.gap) == (
        False,
        True,
        None,
    )
    assert (gap.accepted, gap.duplicate, gap.gap) == (True, False, (2, 2))
    assert tracker.expected(ProcessSource.ASR) == 4


def test_sequence_tracker_is_independent_per_sender() -> None:
    tracker = SequenceTracker(SESSION_ID)

    assert tracker.accept(envelope(source=ProcessSource.ASR, sequence=2)).gap == (
        1,
        1,
    )
    audio = tracker.accept(envelope(source=ProcessSource.AUDIO, sequence=1))

    assert audio.accepted is True
    assert audio.gap is None


def test_sequence_tracker_resets_only_the_restarted_sender_generation() -> None:
    tracker = SequenceTracker(SESSION_ID)
    assert tracker.accept(envelope(source=ProcessSource.ASR, sequence=4)).accepted
    assert tracker.accept(envelope(source=ProcessSource.AUDIO, sequence=3)).accepted

    tracker.reset(ProcessSource.ASR)

    assert tracker.expected(ProcessSource.ASR) == 1
    assert tracker.expected(ProcessSource.AUDIO) == 4
    assert tracker.accept(envelope(source=ProcessSource.ASR, sequence=1)).accepted


def test_sequence_tracker_randomized_model_matches_sender_local_oracle() -> None:
    rng = random.Random(20260822)
    tracker = SequenceTracker(SESSION_ID)
    expected = {ProcessSource.ASR: 1, ProcessSource.AUDIO: 1}

    for _ in range(1_000):
        source = rng.choice((ProcessSource.ASR, ProcessSource.AUDIO))
        sequence = rng.randint(1, 150)
        result = tracker.accept(envelope(source=source, sequence=sequence))
        oracle = expected[source]
        if sequence < oracle:
            assert result.duplicate is True
            assert result.accepted is False
        else:
            assert result.accepted is True
            assert result.duplicate is False
            assert result.gap == (
                None if sequence == oracle else (oracle, sequence - 1)
            )
            expected[source] = sequence + 1
        assert tracker.expected(source) == expected[source]


def test_sequence_tracker_rejects_stale_session_and_schema_transactionally() -> None:
    tracker = SequenceTracker(SESSION_ID)

    with pytest.raises(ValueError, match="active session"):
        tracker.accept(envelope(session_id="01J00000000000000000000009"))
    with pytest.raises(UnknownSchemaVersionError):
        tracker.accept(envelope(schema_version=2))

    assert tracker.expected(ProcessSource.ASR) == 1


def test_sequence_tracker_rejects_int_subclasses_before_mutation() -> None:
    class IntegerSubclass(int):
        pass

    hostile = envelope()
    object.__setattr__(hostile, "sequence", IntegerSubclass(1))
    tracker = SequenceTracker(SESSION_ID)

    with pytest.raises(ValueError, match="exact integer"):
        tracker.accept(hostile)

    assert tracker.expected(ProcessSource.ASR) == 1


def test_sequence_tracker_rejects_schema_and_session_subclasses() -> None:
    class IntegerSubclass(int):
        pass

    class StringSubclass(str):
        pass

    tracker = SequenceTracker(SESSION_ID)
    hostile_schema = envelope()
    object.__setattr__(hostile_schema, "schema_version", IntegerSubclass(1))
    hostile_session = envelope()
    object.__setattr__(hostile_session, "session_id", StringSubclass(SESSION_ID))

    with pytest.raises(ValueError, match="schema_version"):
        tracker.accept(hostile_schema)
    with pytest.raises(ValueError, match="session_id"):
        tracker.accept(hostile_session)

    assert tracker.expected(ProcessSource.ASR) == 1


def test_asr_commit_dict_is_validated_into_typed_payload() -> None:
    incoming = envelope(
        message_type=MessageType.TRANSCRIPT_COMMITTED,
        payload=transcript_record().to_dict(),
    )

    payload = validate_worker_payload(incoming)

    assert payload == TranscriptCommitted(transcript_record())


def test_status_payload_requires_exact_shape_and_scalar_types() -> None:
    valid = envelope(
        source=ProcessSource.DISCUSSION,
        message_type=MessageType.DISCUSSION_STATUS,
        payload=DiscussionStatusPayload("RUNNING", 0, 0, None),
    )
    assert validate_worker_payload(valid) == valid.payload

    malformed = envelope(
        message_type=MessageType.ASR_STATUS,
        payload={
            "state": "RUNNING",
            "backlog_ms": True,
            "maximum_backlog_ms": 0,
            "analysis_paused": False,
        },
    )
    with pytest.raises(PayloadValidationError, match="backlog_ms"):
        validate_worker_payload(malformed)

    unsupported_state = envelope(
        message_type=MessageType.ASR_STATUS,
        payload={
            "state": "PAUSED",
            "backlog_ms": 0,
            "maximum_backlog_ms": 0,
            "analysis_paused": False,
        },
    )
    with pytest.raises(PayloadValidationError, match="state"):
        validate_worker_payload(unsupported_state)


def test_unknown_payload_does_not_leak_hostile_text() -> None:
    malformed = envelope(
        source=ProcessSource.AUDIO,
        message_type=MessageType.AUDIO_LEVEL,
        payload={"source": "ME", "peak_dbfs": -10.0, "text": "SECRET"},
    )

    with pytest.raises(PayloadValidationError) as raised:
        validate_worker_payload(malformed)

    assert "SECRET" not in str(raised.value)


def test_rewrap_for_gui_preserves_typed_payload_and_is_immutable() -> None:
    incoming = envelope(
        sequence=10,
        message_type=MessageType.TRANSCRIPT_COMMITTED,
        payload=TranscriptCommitted(transcript_record()),
    )

    outgoing = rewrap_for_gui(incoming, sequence=2)

    assert outgoing is not incoming
    assert outgoing.source is ProcessSource.GUI
    assert outgoing.sequence == 2
    assert outgoing.payload == incoming.payload
    with pytest.raises(FrozenInstanceError):
        outgoing.sequence = 3  # type: ignore[misc]
