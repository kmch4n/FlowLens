"""Run one real, local discussion-analysis prompt for each FlowLens mode."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from flowlens.discussion.context import select_recent_records
from flowlens.discussion.contracts import DiscussionRequest
from flowlens.discussion.llama_cpp_adapter import (
    DiscussionModelConfig,
    load_llama_cpp_backend,
)
from flowlens.discussion.model_manifest import parse_manifest_bytes
from flowlens.discussion.prompt import build_messages
from flowlens.discussion.schema import discussion_state_schema, parse_discussion_state
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import AudioSource, SessionMode
from flowlens.domain.messages import TranscriptRecord

_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "discussion"
)
_MODEL_ID = "qwen3-4b-instruct-2507"
_PROHIBITED = (
    "おすすめ",
    "推奨",
    "すべき",
    "賛成材料",
    "反対材料",
    "pros and cons",
    "recommend",
    "you should",
)
_INTERVIEW_PROHIBITED = ("決定事項", "未解決事項", "decisions", "unresolved issues")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_fixture(path: Path) -> dict[str, object]:
    """Load one strict UTF-8 fixture without accepting duplicate keys."""

    encoded = path.read_bytes()
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise ValueError("fixture must be UTF-8 without BOM")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError("fixture must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("fixture must be a JSON object")
    return cast(dict[str, object], value)


def _combined_state_text(state: DiscussionState) -> str:
    return " ".join(
        (
            state.current_focus,
            *state.key_points,
            *state.confirmed_outcomes,
            *state.follow_up_items,
        )
    ).casefold()


def validate_discussion_smoke(
    output: object,
    expected_mode: SessionMode,
) -> DiscussionState:
    """Validate exact state shape, expected mode, and anti-advice rules."""

    if not isinstance(expected_mode, SessionMode):
        raise TypeError("expected_mode must be a SessionMode")
    try:
        state = DiscussionState.from_dict(output)
    except Exception as error:
        raise ValueError(
            f"discussion output has missing or unknown fields: {error}"
        ) from error
    if state.mode is not expected_mode:
        raise ValueError("discussion output mode does not match requested mode")
    if state.revision != 1:
        raise ValueError("discussion smoke output revision must be 1")
    combined = _combined_state_text(state)
    prohibited = list(_PROHIBITED)
    if expected_mode is SessionMode.INTERVIEW:
        prohibited.extend(_INTERVIEW_PROHIBITED)
    for phrase in prohibited:
        if phrase.casefold() in combined:
            raise ValueError(f"discussion output contains prohibited phrase: {phrase}")
    return state


def _transcript_records(
    fixture: Mapping[str, object],
    updated_at: datetime,
) -> tuple[TranscriptRecord, ...]:
    raw = fixture.get("transcript")
    if not isinstance(raw, list) or not raw:
        raise ValueError("fixture transcript must be a non-empty list")
    records: list[TranscriptRecord] = []
    for sequence, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError("fixture transcript entries must be objects")
        source = AudioSource(cast(str, item.get("source")))
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("fixture transcript text must be non-blank")
        start_ms = (sequence - 1) * 1_000
        records.append(
            TranscriptRecord(
                schema_version=1,
                segment_id=f"01J0000000000000000000000{sequence}",
                sequence=sequence,
                source=source,
                text=text,
                session_start_ms=start_ms,
                session_end_ms=start_ms + 800,
                source_start_sample=start_ms * 16,
                source_end_sample=(start_ms + 800) * 16,
                committed_at=updated_at,
            )
        )
    return tuple(records)


def _deny_network(*args: object, **kwargs: object) -> socket.socket:
    del args, kwargs
    raise RuntimeError("network access is forbidden during discussion smoke")


def _write_report(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=4, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_discussion_smoke(model_manifest: Path) -> dict[str, object]:
    """Run all three production prompts using the pinned local model."""

    manifest_path = model_manifest.resolve(strict=True)
    manifest = parse_manifest_bytes(manifest_path.read_bytes())
    entry = manifest["models"].get(_MODEL_ID)
    if entry is None:
        raise ValueError("discussion model is absent from manifest")
    relative = Path(entry["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("discussion model path must remain under the model root")
    model_path = (manifest_path.parent / relative).resolve(strict=True)
    model_path.relative_to(manifest_path.parent)
    backend = load_llama_cpp_backend(
        DiscussionModelConfig(model_path=model_path, sha256=entry["sha256"])
    )
    original_socket = socket.socket
    socket.socket = cast(type[socket.socket], _deny_network)  # type: ignore[misc]
    reports: list[dict[str, object]] = []
    try:
        for offset, mode in enumerate(SessionMode):
            fixture = load_fixture(_FIXTURE_ROOT / f"{mode.value.lower()}.json")
            updated_at = datetime.now().astimezone().replace(microsecond=0) + timedelta(
                milliseconds=offset
            )
            initial = DiscussionState.initial(mode, updated_at - timedelta(seconds=1))
            records = _transcript_records(fixture, updated_at)
            selected = select_recent_records(records, backend.count_tokens)
            request = DiscussionRequest(initial, selected, 1, updated_at)
            raw = backend.generate(
                build_messages(request), discussion_state_schema(request)
            )
            parsed = parse_discussion_state(raw, request)
            state = validate_discussion_smoke(parsed.to_dict(), mode)
            reports.append(
                {
                    "mode": mode.value,
                    "revision": state.revision,
                    "passed": True,
                    "schema_fields": list(state.to_dict()),
                }
            )
    finally:
        socket.socket = original_socket  # type: ignore[misc]
    return {"schema_version": 1, "passed": True, "modes": reports, "local_only": True}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke and write either a passing or fail-closed report."""

    arguments = _build_parser().parse_args(argv)
    report_path = cast(Path, arguments.report)
    try:
        report = run_discussion_smoke(cast(Path, arguments.model_manifest))
        exit_code = 0
    except Exception as error:
        report = {
            "schema_version": 1,
            "passed": False,
            "local_only": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_code = 1
    _write_report(report_path, report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
