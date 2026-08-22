"""Closed discussion schema and parser behavior."""

import json
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime
from typing import Any, cast

import pytest

from flowlens.discussion.contracts import (
    ChatMessage,
    DiscussionRequest,
    DiscussionStatusPayload,
    DiscussionStoppedPayload,
)
from flowlens.discussion.schema import (
    DiscussionOutputError,
    discussion_state_schema,
    parse_discussion_state,
)
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import SessionMode
from flowlens.domain.messages import TranscriptRecord
from tests.discussion.factories import NOW, make_record, make_request, make_state


def _valid_output(*, revision: int = 1) -> dict[str, object]:
    return {
        "revision": revision,
        "mode": "MEETING",
        "current_focus": "設計範囲",
        "key_points": ["ローカル処理"],
        "confirmed_outcomes": ["MVPの範囲を固定した"],
        "follow_up_items": ["遅延を測定する"],
        "updated_at": "2026-08-19T12:35:02.125+09:00",
    }


def test_schema_has_exact_closed_shape_and_request_constants() -> None:
    request = make_request(mode=SessionMode.INTERVIEW, revision=7)

    schema = discussion_state_schema(request)

    assert list(cast(dict[str, object], schema["properties"])) == [
        "revision",
        "mode",
        "current_focus",
        "key_points",
        "confirmed_outcomes",
        "follow_up_items",
        "updated_at",
    ]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    properties = cast(dict[str, object], schema["properties"])
    assert properties["revision"] == {"const": 7}
    assert properties["mode"] == {"const": "INTERVIEW"}
    assert properties["updated_at"] == {"const": "2026-08-19T12:35:02.125+09:00"}
    assert properties["current_focus"] == {"type": "string"}
    assert properties["key_points"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert schema["required"] == list(properties)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: {**value, "revision": 2}, "revision"),
        (lambda value: {**value, "revision": True}, "revision"),
        (lambda value: {**value, "mode": "GENERAL"}, "mode"),
        (
            lambda value: {**value, "updated_at": "2026-08-19T03:35:02.125+00:00"},
            "updated_at",
        ),
        (
            lambda value: {**value, "updated_at": "2026-08-19T12:35:02+09:00"},
            "updated_at",
        ),
        (
            lambda value: {**value, "updated_at": "2026-08-19T12:35:02.125"},
            "updated_at",
        ),
        (lambda value: {**value, "current_focus": 1}, "current_focus"),
        (lambda value: {**value, "key_points": "local"}, "key_points"),
        (lambda value: {**value, "key_points": ["valid", 3]}, "key_points"),
        (lambda value: {**value, "extra": "forbidden"}, "unknown"),
        (
            lambda value: {
                key: item for key, item in value.items() if key != "current_focus"
            },
            "missing",
        ),
    ],
)
def test_parser_rejects_every_contract_violation_without_mutating_state(
    mutate: Any,
    match: str,
) -> None:
    request = make_request()
    old_state = request.current_state
    raw = json.dumps(mutate(_valid_output()), ensure_ascii=False)

    with pytest.raises(DiscussionOutputError, match=match):
        parse_discussion_state(raw, request)

    assert request.current_state is old_state
    assert request.current_state.revision == 0


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "null",
        '{"revision": 1, "revision": 1}',
        '{"revision": NaN}',
    ],
)
def test_parser_rejects_malformed_duplicate_or_non_object_json(raw: str) -> None:
    with pytest.raises(DiscussionOutputError):
        parse_discussion_state(raw, make_request())


def test_parser_returns_new_immutable_normalized_snapshot() -> None:
    raw = json.dumps(_valid_output(), ensure_ascii=False)

    parsed = parse_discussion_state(raw, make_request())

    assert parsed == DiscussionState(
        revision=1,
        mode=SessionMode.MEETING,
        current_focus="設計範囲",
        key_points=("ローカル処理",),
        confirmed_outcomes=("MVPの範囲を固定した",),
        follow_up_items=("遅延を測定する",),
        updated_at=NOW,
    )
    assert isinstance(parsed.key_points, tuple)


def test_public_contracts_are_strict_immutable_and_picklable() -> None:
    request = make_request(
        updated_at=datetime.fromisoformat("2026-08-19T12:35:02.125999+09:00")
    )
    values = (
        ChatMessage("system", "rules"),
        request,
        DiscussionStatusPayload("FAILED", 0, 1, "INVALID_OUTPUT"),
        DiscussionStoppedPayload("DISCUSSION", True, 2, 0),
    )

    assert request.updated_at == NOW
    assert pickle.loads(pickle.dumps(values)) == values
    for value in values:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(value, fields(value)[0].name, None)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ChatMessage(cast(Any, "assistant"), "x"), "role"),
        (lambda: ChatMessage("user", ""), "content"),
        (
            lambda: DiscussionRequest(
                make_state(),
                cast(tuple[TranscriptRecord, ...], [make_record()]),
                1,
                NOW,
            ),
            "records",
        ),
        (
            lambda: replace(make_request(), requested_revision=True),
            "requested_revision",
        ),
        (lambda: replace(make_request(), requested_revision=2), "requested_revision"),
        (
            lambda: replace(
                make_request(),
                updated_at=datetime.fromisoformat("2026-08-19T12:35:02.125"),
            ),
            "updated_at",
        ),
        (lambda: DiscussionStatusPayload("FAILED", True, 0, None), "revision"),
        (lambda: DiscussionStatusPayload("FAILED", 0, True, None), "pending_count"),
        (lambda: DiscussionStatusPayload("", 0, 0, None), "state"),
        (lambda: DiscussionStatusPayload("FAILED", 0, 0, ""), "error_code"),
        (lambda: DiscussionStoppedPayload("AUDIO", True, 0, 0), "worker"),
        (
            lambda: DiscussionStoppedPayload("DISCUSSION", cast(Any, 1), 0, 0),
            "drained",
        ),
        (
            lambda: DiscussionStoppedPayload("DISCUSSION", True, True, 0),
            "final_revision",
        ),
    ],
)
def test_public_contracts_reject_invalid_or_mutable_inputs(
    factory: Any,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()
