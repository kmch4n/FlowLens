"""Fail-closed validation tests for persisted acceptance sessions."""

import json
import wave
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from flowlens.domain.discussion import StateHistoryRecord
from flowlens.domain.enums import EventType, ProcessSource, SessionMode, SessionStatus
from flowlens.domain.messages import EventRecord
from scripts.validate_session import REQUIRED_ARTIFACTS, validate_session
from tests.factories import (
    make_discussion_state,
    make_event_record,
    make_manifest,
    make_transcript_record,
)


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _jsonl(path: Path, values: Sequence[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
        newline="\n",
    )


def _wav(path: Path, duration_ms: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * (duration_ms * 16))


def make_valid_session(
    root: Path,
    *,
    active_ms: int = 300_000,
    wav_ms: int = 300_000,
    status: SessionStatus = SessionStatus.COMPLETED,
) -> Path:
    session = root / "session"
    session.mkdir(parents=True)
    manifest = replace(
        make_manifest(status=status),
        active_duration_ms=active_ms,
        transcript_entry_count=1,
        final_discussion_state_revision=1,
    )
    _json(session / "session.json", manifest.to_dict())
    _wav(session / "mic.wav", wav_ms)
    _wav(session / "loopback.wav", wav_ms)
    _jsonl(session / "transcript.jsonl", [make_transcript_record().to_dict()])
    state = make_discussion_state(revision=1)
    _json(session / "discussion-state.json", state.to_dict())
    history = StateHistoryRecord(
        schema_version=1,
        session_id=manifest.session_id,
        previous_revision=0,
        new_revision=1,
        state=state,
    )
    _jsonl(session / "state-history.jsonl", [history.to_dict()])
    first = make_event_record(session_time_ms=0)
    final_type = (
        EventType.SESSION_COMPLETED
        if status is SessionStatus.COMPLETED
        else EventType.SESSION_RECOVERED
    )
    assert isinstance(manifest.ended_at, datetime)
    final = EventRecord(
        schema_version=1,
        session_id=manifest.session_id,
        sequence=2,
        event_type=final_type,
        source=ProcessSource.WRITER,
        session_time_ms=active_ms,
        created_at=manifest.ended_at,
        details={},
    )
    _jsonl(session / "events.jsonl", [first.to_dict(), final.to_dict()])
    return session


def test_validator_requires_exact_seven_session_artifacts(tmp_path: Path) -> None:
    session = make_valid_session(tmp_path)
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="completed"
    )
    assert result.errors == ()
    assert tuple(sorted(path.name for path in session.iterdir())) == tuple(
        sorted(REQUIRED_ARTIFACTS)
    )
    (session / "events.jsonl").unlink()
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="completed"
    )
    assert result.errors == ("Missing required artifact: events.jsonl",)


def test_validator_checks_wav_format_and_pause_excluded_duration(
    tmp_path: Path,
) -> None:
    session = make_valid_session(tmp_path, active_ms=300_000, wav_ms=298_800)
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="completed"
    )
    assert result.wav_error_percent == pytest.approx(0.4)
    assert result.wav_error_percent is not None
    assert result.wav_error_percent < 0.5
    assert result.mic_format == (1, 2, 16_000)
    assert result.loopback_format == (1, 2, 16_000)


def test_validator_rejects_unknown_artifact_and_torn_jsonl(tmp_path: Path) -> None:
    session = make_valid_session(tmp_path)
    (session / "unexpected.txt").write_text("x", encoding="utf-8")
    with (session / "transcript.jsonl").open("ab") as output:
        output.write(b'{"schema_version":1')
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="completed"
    )
    assert "Unexpected artifact: unexpected.txt" in result.errors
    assert (
        "transcript.jsonl must end with a complete LF-terminated JSON line"
        in result.errors
    )


def test_validator_rejects_sequence_revision_and_terminal_event_errors(
    tmp_path: Path,
) -> None:
    session = make_valid_session(tmp_path)
    transcript = make_transcript_record().to_dict()
    transcript["sequence"] = 2
    _jsonl(session / "transcript.jsonl", [transcript])
    state = StateHistoryRecord(
        schema_version=1,
        session_id=make_manifest().session_id,
        previous_revision=1,
        new_revision=2,
        state=replace(make_discussion_state(revision=1), revision=2),
    ).to_dict()
    _jsonl(session / "state-history.jsonl", [state])
    events = [make_event_record(session_time_ms=0).to_dict()]
    _jsonl(session / "events.jsonl", events)
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="completed"
    )
    assert "transcript.jsonl sequences must be contiguous from 1" in result.errors
    assert "state-history.jsonl revisions must be contiguous from 1" in result.errors
    assert "events.jsonl must end with exactly one SESSION_COMPLETED" in result.errors


def test_validator_recovered_never_accepts_completed_terminal_event(
    tmp_path: Path,
) -> None:
    session = make_valid_session(tmp_path, status=SessionStatus.RECOVERED)
    events = [make_event_record(session_time_ms=0).to_dict()]
    completed = make_event_record(sequence=2, session_time_ms=300_000).to_dict()
    completed["event_type"] = "SESSION_COMPLETED"
    _jsonl(session / "events.jsonl", [*events, completed])
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="recovered"
    )
    assert "Recovered session must not contain SESSION_COMPLETED" in result.errors


def test_validator_rejects_linked_session_directory(tmp_path: Path) -> None:
    target = make_valid_session(tmp_path / "target")
    link = tmp_path / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    result = validate_session(
        link, minimum_active_seconds=300, expected_status="completed"
    )
    assert result.errors == ("Session directory must not be a link or reparse point",)


def test_validator_rejects_history_mode_that_differs_from_manifest(
    tmp_path: Path,
) -> None:
    session = make_valid_session(tmp_path)
    manifest = make_manifest(status=SessionStatus.COMPLETED)
    wrong_state = replace(
        make_discussion_state(revision=1),
        mode=SessionMode.INTERVIEW,
    )
    history = StateHistoryRecord(
        schema_version=1,
        session_id=manifest.session_id,
        previous_revision=0,
        new_revision=1,
        state=wrong_state,
    )
    _jsonl(session / "state-history.jsonl", [history.to_dict()])
    result = validate_session(
        session, minimum_active_seconds=300, expected_status="completed"
    )
    assert "state-history.jsonl modes must match session.json" in result.errors
