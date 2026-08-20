"""Deterministic domain factories shared by FlowLens tests."""

from datetime import datetime

from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import (
    AudioSource,
    EventType,
    ProcessSource,
    SessionMode,
    SessionStatus,
)
from flowlens.domain.messages import EventRecord, TranscriptRecord, WriterFinalize
from flowlens.domain.session import DeviceIdentity, ModelIdentity, SessionManifest


def make_manifest(
    *,
    status: SessionStatus = SessionStatus.INCOMPLETE,
    session_id: str = "01J00000000000000000000000",
) -> SessionManifest:
    """Create a deterministic session manifest."""

    started_at = datetime.fromisoformat("2026-08-19T12:00:00+09:00")
    ended_at = None
    if status is not SessionStatus.INCOMPLETE:
        ended_at = datetime.fromisoformat("2026-08-19T12:30:00+09:00")
    return SessionManifest(
        schema_version=1,
        session_id=session_id,
        status=status,
        mode=SessionMode.MEETING,
        started_at=started_at,
        ended_at=ended_at,
        active_duration_ms=0 if ended_at is None else 1_800_000,
        pause_intervals=(),
        microphone=DeviceIdentity("mic-1", "USB Microphone"),
        loopback_output=DeviceIdentity("out-1", "Speakers"),
        asr_model=ModelIdentity(
            "kotoba-tech/kotoba-whisper-v2.0-faster", "rev-a", "a" * 64
        ),
        discussion_model=ModelIdentity(
            "Qwen/Qwen3-4B-Instruct-2507", "rev-b", "b" * 64
        ),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=(),
    )


def make_discussion_state(revision: int = 0) -> DiscussionState:
    """Create a deterministic discussion-state snapshot."""

    return DiscussionState(
        revision=revision,
        mode=SessionMode.MEETING,
        current_focus="方針" if revision else "",
        key_points=("ローカル保存",) if revision else (),
        confirmed_outcomes=(),
        follow_up_items=(),
        updated_at=datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
    )


def make_transcript_record(sequence: int = 1) -> TranscriptRecord:
    """Create a deterministic committed transcript record."""

    segment_ids = {
        1: "01J00000000000000000000001",
        2: "01J00000000000000000000002",
    }
    return TranscriptRecord(
        1,
        segment_ids[sequence],
        sequence,
        AudioSource.ME,
        "今回の方針を確認します。",
        sequence * 1_000,
        sequence * 1_000 + 800,
        (sequence - 1) * 12_800,
        sequence * 12_800,
        datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
    )


def make_event_record(
    sequence: int = 1,
    *,
    session_id: str = "01J00000000000000000000000",
    session_time_ms: int | None = None,
) -> EventRecord:
    """Create a deterministic operational event record."""

    return EventRecord(
        schema_version=1,
        session_id=session_id,
        sequence=sequence,
        event_type=EventType.SESSION_START,
        source=ProcessSource.GUI,
        session_time_ms=(
            sequence * 1_000 if session_time_ms is None else session_time_ms
        ),
        created_at=datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
        details={},
    )


def make_finalize_command(
    event_sequence: int = 1,
    session_id: str = "01J00000000000000000000000",
) -> WriterFinalize:
    """Create a deterministic normal-finalization command."""

    return WriterFinalize(
        ended_at=datetime.fromisoformat("2026-08-19T12:30:00+09:00"),
        active_duration_ms=1_800_000,
        pause_intervals=(),
        final_state=make_discussion_state(revision=0),
        completion_event=EventRecord(
            schema_version=1,
            session_id=session_id,
            sequence=event_sequence,
            event_type=EventType.SESSION_COMPLETED,
            source=ProcessSource.GUI,
            session_time_ms=1_800_000,
            created_at=datetime.fromisoformat("2026-08-19T12:30:00+09:00"),
            details={},
        ),
    )
