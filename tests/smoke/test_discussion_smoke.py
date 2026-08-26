"""Fixture and CLI contracts for the real local discussion smoke."""

import json
from pathlib import Path
from typing import cast

import pytest

from flowlens.domain.enums import SessionMode
from scripts.smoke_discussion import load_fixture, validate_discussion_smoke


@pytest.mark.parametrize("mode", ["MEETING", "INTERVIEW", "GENERAL"])
def test_discussion_fixture_contract(mode: str) -> None:
    fixture = load_fixture(Path(f"tests/fixtures/discussion/{mode.lower()}.json"))
    state = validate_discussion_smoke(fixture["output"], SessionMode(mode))
    assert state.mode.value == mode
    assert state.revision == 1
    combined = " ".join(
        (
            state.current_focus,
            *state.key_points,
            *state.confirmed_outcomes,
            *state.follow_up_items,
        )
    )
    prohibited_phrases = fixture["prohibited_phrases"]
    assert isinstance(prohibited_phrases, list)
    for prohibited in cast(list[str], prohibited_phrases):
        assert prohibited not in combined


def test_discussion_validation_rejects_extra_fields_and_advice() -> None:
    output = load_fixture(Path("tests/fixtures/discussion/meeting.json"))["output"]
    assert isinstance(output, dict)
    output["advice"] = "この案を選ぶべき"
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_discussion_smoke(output, SessionMode.MEETING)


def test_fixture_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    fixture = tmp_path / "duplicate.json"
    fixture.write_text('{"output": {}, "output": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_fixture(fixture)


def test_fixture_files_are_deterministic_utf8_json() -> None:
    for fixture in Path("tests/fixtures/discussion").glob("*.json"):
        encoded = fixture.read_bytes()
        assert not encoded.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in encoded
        assert json.loads(encoded)["transcript"]
