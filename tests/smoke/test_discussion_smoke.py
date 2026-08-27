"""Fixture and CLI contracts for the real local discussion smoke."""

import json
from pathlib import Path
from typing import cast

import pytest

from flowlens.domain.enums import SessionMode
from scripts import smoke_discussion
from scripts.smoke_discussion import (
    load_fixture,
    run_discussion_smoke,
    validate_discussion_smoke,
)


@pytest.mark.parametrize("mode", ["MEETING", "INTERVIEW", "GENERAL"])
def test_discussion_fixture_contract(mode: str) -> None:
    fixture = load_fixture(Path(f"tests/fixtures/discussion/{mode.lower()}.json"))
    prohibited_phrases = fixture["prohibited_phrases"]
    assert isinstance(prohibited_phrases, list)
    state = validate_discussion_smoke(
        fixture["output"],
        SessionMode(mode),
        cast(list[str], prohibited_phrases),
    )
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
    for prohibited in cast(list[str], prohibited_phrases):
        assert prohibited not in combined


def test_discussion_validation_rejects_extra_fields_and_advice() -> None:
    output = load_fixture(Path("tests/fixtures/discussion/meeting.json"))["output"]
    assert isinstance(output, dict)
    output["advice"] = "この案を選ぶべき"
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_discussion_smoke(output, SessionMode.MEETING)


def test_discussion_validation_requires_meeting_confirmation() -> None:
    output = load_fixture(Path("tests/fixtures/discussion/meeting.json"))["output"]
    assert isinstance(output, dict)
    output["confirmed_outcomes"] = []

    with pytest.raises(ValueError, match="MEETING output must include a confirmation"):
        validate_discussion_smoke(output, SessionMode.MEETING)


def test_discussion_validation_requires_explicit_meeting_confirmation() -> None:
    output = load_fixture(Path("tests/fixtures/discussion/meeting.json"))["output"]
    assert isinstance(output, dict)
    output["confirmed_outcomes"] = ["ローカル保存"]

    with pytest.raises(ValueError, match="explicit confirmation"):
        validate_discussion_smoke(output, SessionMode.MEETING)


def test_discussion_validation_keeps_interview_labels_non_decisional() -> None:
    output = load_fixture(Path("tests/fixtures/discussion/interview.json"))["output"]
    assert isinstance(output, dict)
    output["follow_up_items"] = ["未解決事項を確認"]

    with pytest.raises(ValueError, match="prohibited phrase: 未解決事項"):
        validate_discussion_smoke(output, SessionMode.INTERVIEW)


def test_discussion_validation_keeps_general_mode_neutral() -> None:
    output = load_fixture(Path("tests/fixtures/discussion/general.json"))["output"]
    assert isinstance(output, dict)
    output["key_points"] = ["賛成意見と反対意見"]

    with pytest.raises(ValueError, match="prohibited phrase: 賛成"):
        validate_discussion_smoke(output, SessionMode.GENERAL)


class _FixtureBackend:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = outputs

    def count_tokens(self, text: str) -> int:
        return len(text)

    def generate(self, messages: object, schema: object) -> str:
        del messages
        assert isinstance(schema, dict)
        properties = cast(dict[str, object], schema["properties"])
        timestamp_schema = cast(dict[str, object], properties["updated_at"])
        output = self._outputs.pop(0)
        output["updated_at"] = timestamp_schema["const"]
        return json.dumps(output, ensure_ascii=False)


def test_real_runner_applies_fixture_specific_prohibited_phrases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    outputs: list[dict[str, object]] = []
    for mode in SessionMode:
        fixture = load_fixture(
            Path(f"tests/fixtures/discussion/{mode.value.lower()}.json")
        )
        output = cast(dict[str, object], fixture["output"])
        if mode is SessionMode.MEETING:
            output["current_focus"] = "fixture固有禁止語"
            cast(list[str], fixture["prohibited_phrases"]).append("fixture固有禁止語")
        (fixture_root / f"{mode.value.lower()}.json").write_text(
            json.dumps(fixture, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        outputs.append(output)

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "model.gguf").write_bytes(b"model")
    backend = _FixtureBackend(outputs)
    monkeypatch.setattr(smoke_discussion, "_FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(
        smoke_discussion,
        "parse_manifest_bytes",
        lambda encoded: {
            "models": {
                "qwen3-4b-instruct-2507": {
                    "relative_path": "model.gguf",
                    "sha256": "a" * 64,
                }
            }
        },
    )
    monkeypatch.setattr(
        smoke_discussion,
        "load_llama_cpp_backend",
        lambda config: backend,
    )

    with pytest.raises(ValueError, match="fixture固有禁止語"):
        run_discussion_smoke(manifest)


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
