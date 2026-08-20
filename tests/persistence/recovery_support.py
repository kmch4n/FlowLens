"""Shared deterministic fixtures for recovery tests."""

from pathlib import Path

from flowlens.persistence.session_writer import SessionWriter
from tests.factories import (
    make_discussion_state,
    make_finalize_command,
    make_manifest,
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
