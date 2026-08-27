"""Deterministic factories for discussion component tests."""

from datetime import datetime

from flowlens.discussion.contracts import DiscussionRequest
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import AudioSource, SessionMode
from flowlens.domain.messages import TranscriptRecord

NOW = datetime.fromisoformat("2026-08-19T12:35:02.125+09:00")


def make_state(
    *,
    mode: SessionMode = SessionMode.MEETING,
    revision: int = 0,
    analyzed_through_sequence: int | None = None,
) -> DiscussionState:
    """Build one complete state with Japanese content."""

    return DiscussionState(
        revision=revision,
        mode=mode,
        current_focus="実装方針",
        key_points=("完全ローカルで動作する",),
        confirmed_outcomes=("MVPでは助言を行わない",),
        follow_up_items=("遅延を確認する",),
        updated_at=NOW,
        analyzed_through_sequence=(
            revision if analyzed_through_sequence is None else analyzed_through_sequence
        ),
    )


def make_record(
    *,
    sequence: int = 1,
    text: str = "議論を整理します",
    source: AudioSource = AudioSource.ME,
) -> TranscriptRecord:
    """Build one valid committed transcript record."""

    start_ms = sequence * 1_000
    start_sample = sequence * 16_000
    return TranscriptRecord(
        schema_version=1,
        segment_id=f"01J{sequence:023d}",
        sequence=sequence,
        source=source,
        text=text,
        session_start_ms=start_ms,
        session_end_ms=start_ms + 500,
        source_start_sample=start_sample,
        source_end_sample=start_sample + 8_000,
        committed_at=NOW,
    )


def make_request(
    *,
    mode: SessionMode = SessionMode.MEETING,
    revision: int = 1,
    records: tuple[TranscriptRecord, ...] | None = None,
    updated_at: datetime = NOW,
) -> DiscussionRequest:
    """Build one valid next-revision request."""

    return DiscussionRequest(
        current_state=make_state(mode=mode, revision=revision - 1),
        records=(make_record(),) if records is None else records,
        requested_revision=revision,
        updated_at=updated_at,
    )
