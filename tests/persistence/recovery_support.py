"""Shared deterministic fixtures for recovery tests."""

import json
import struct
from datetime import datetime
from pathlib import Path

from flowlens.domain.enums import AudioSource, EventType, ProcessSource
from flowlens.domain.messages import AudioWriteCommand, EventRecord
from flowlens.persistence.session_writer import SessionWriter
from tests.factories import (
    make_discussion_state,
    make_finalize_command,
    make_manifest,
    make_transcript_record,
)


def create_session_fixture(session_dir: Path, status: str) -> Path:
    """Create one complete seven-artifact session fixture."""

    session_id = session_dir.name[-26:]
    if len(session_id) != 26:
        session_id = "01J00000000000000000000000"
    writer = SessionWriter.open(
        session_dir,
        make_manifest(session_id=session_id),
        make_discussion_state(),
    )
    if status == "completed":
        writer.finalize(make_finalize_command(session_id=session_id))
    elif status == "incomplete":
        writer.close_incomplete()
    else:
        writer.close_incomplete()
        raise ValueError(f"unsupported fixture status: {status}")
    return session_dir


def aware_recovery_time() -> datetime:
    """Return the deterministic recovery timestamp."""

    return datetime.fromisoformat("2026-08-19T13:00:00+09:00")


def create_interrupted_session(sessions_root: Path) -> Path:
    """Create one incomplete session with audio, transcript, and start event."""

    session_dir = sessions_root / "20260819T120000+0900_01J00000000000000000000000"
    writer = SessionWriter.open(
        session_dir,
        make_manifest(),
        make_discussion_state(),
    )
    writer.append_audio(
        AudioWriteCommand(AudioSource.ME, b"\x00\x00" * 12_800, 0, 12_800, 0, 1_800)
    )
    writer.append_audio(
        AudioWriteCommand(
            AudioSource.OTHERS,
            b"\x01\x00" * 12_800,
            0,
            12_800,
            0,
            1_800,
        )
    )
    writer.append_transcript(make_transcript_record(1))
    writer.append_event(
        EventRecord(
            1,
            "01J00000000000000000000000",
            1,
            EventType.SESSION_START,
            ProcessSource.GUI,
            0,
            datetime.fromisoformat("2026-08-19T12:00:00+09:00"),
            {},
        )
    )
    writer.close_incomplete()
    return session_dir


def corrupt_wav_sizes(path: Path) -> None:
    """Zero both mutable RIFF size fields without changing PCM."""

    with path.open("r+b") as file:
        file.seek(4)
        file.write(struct.pack("<I", 0))
        file.seek(40)
        file.write(struct.pack("<I", 0))


def load_manifest_status(session_dir: Path) -> str:
    """Read one manifest status for failure-order assertions."""

    value = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    status = value["status"]
    if not isinstance(status, str):
        raise TypeError("manifest status must be a string")
    return status
