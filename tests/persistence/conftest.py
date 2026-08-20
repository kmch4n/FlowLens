"""Shared fixtures for persistence tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from flowlens.persistence.session_writer import SessionWriter
from tests.factories import make_discussion_state, make_manifest


@pytest.fixture
def open_writer(tmp_path: Path) -> Iterator[SessionWriter]:
    """Provide an open session writer and always release its resources."""

    manifest = make_manifest()
    writer = SessionWriter.open(
        tmp_path / "session",
        manifest,
        make_discussion_state(),
    )
    try:
        yield writer
    finally:
        writer.close_incomplete()
