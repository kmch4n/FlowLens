"""Closed JSON schema and strict parser for discussion snapshots."""

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import cast

from flowlens.discussion.contracts import DiscussionOutputError, DiscussionRequest
from flowlens.domain.discussion import DiscussionState

__all__ = [
    "DiscussionOutputError",
    "discussion_state_schema",
    "parse_discussion_state",
]

_FIELD_ORDER = (
    "revision",
    "mode",
    "current_focus",
    "key_points",
    "confirmed_outcomes",
    "follow_up_items",
    "updated_at",
    "analyzed_through_sequence",
)
_MILLISECOND_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(?:Z|[+-]\d{2}:\d{2})"
)


def _timestamp_text(request: DiscussionRequest) -> str:
    return request.updated_at.isoformat(timespec="milliseconds")


def discussion_state_schema(request: DiscussionRequest) -> dict[str, object]:
    """Build the exact closed output schema for one request."""

    properties: dict[str, object] = {
        "revision": {"const": request.requested_revision},
        "mode": {"const": request.current_state.mode.value},
        "current_focus": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "confirmed_outcomes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "follow_up_items": {
            "type": "array",
            "items": {"type": "string"},
        },
        "updated_at": {"const": _timestamp_text(request)},
        "analyzed_through_sequence": {"const": request.analyzed_through_sequence},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(_FIELD_ORDER),
        "additionalProperties": False,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DiscussionOutputError(f"duplicate key: {key}")
        result[key] = value
    return result


def _parse_json_object(raw: str) -> Mapping[str, object]:
    if not isinstance(raw, str):
        raise DiscussionOutputError("output must be a JSON string")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_raise_invalid_constant(value)),
        )
    except DiscussionOutputError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise DiscussionOutputError("output must be valid JSON") from error
    if not isinstance(value, dict):
        raise DiscussionOutputError("output must be a JSON object")
    return cast(Mapping[str, object], value)


def _raise_invalid_constant(value: str) -> object:
    raise DiscussionOutputError(f"invalid JSON constant: {value}")


def _require_exact_fields(value: Mapping[str, object]) -> None:
    expected = frozenset(_FIELD_ORDER)
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise DiscussionOutputError(
            f"DiscussionState: missing={missing}, unknown={unknown}"
        )


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise DiscussionOutputError(f"{field_name} must be a string")
    return value


def _require_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DiscussionOutputError(f"{field_name} must be a list of strings")
    return tuple(value)


def _require_timestamp(value: object, request: DiscussionRequest) -> datetime:
    timestamp = _require_string(value, "updated_at")
    if _MILLISECOND_TIMESTAMP.fullmatch(timestamp) is None:
        raise DiscussionOutputError(
            "updated_at must be timezone-aware with millisecond precision"
        )
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise DiscussionOutputError("updated_at must be a valid timestamp") from error
    if parsed.utcoffset() is None or parsed.microsecond % 1_000 != 0:
        raise DiscussionOutputError(
            "updated_at must be timezone-aware with millisecond precision"
        )
    if timestamp != _timestamp_text(request):
        raise DiscussionOutputError("updated_at must match the requested timestamp")
    return parsed


def parse_discussion_state(raw: str, request: DiscussionRequest) -> DiscussionState:
    """Parse a generated complete snapshot without fallback or mutation."""

    value = _parse_json_object(raw)
    _require_exact_fields(value)

    revision = value["revision"]
    if not isinstance(revision, int) or isinstance(revision, bool):
        raise DiscussionOutputError("revision must be an integer")
    if revision != request.requested_revision:
        raise DiscussionOutputError("revision must match requested_revision")

    mode = _require_string(value["mode"], "mode")
    if mode != request.current_state.mode.value:
        raise DiscussionOutputError("mode must match the current session mode")

    analyzed_through_sequence = value["analyzed_through_sequence"]
    if (
        not isinstance(analyzed_through_sequence, int)
        or isinstance(analyzed_through_sequence, bool)
        or analyzed_through_sequence != request.analyzed_through_sequence
    ):
        raise DiscussionOutputError(
            "analyzed_through_sequence must match the request watermark"
        )

    return DiscussionState(
        revision=revision,
        mode=request.current_state.mode,
        current_focus=_require_string(value["current_focus"], "current_focus"),
        key_points=_require_string_list(value["key_points"], "key_points"),
        confirmed_outcomes=_require_string_list(
            value["confirmed_outcomes"],
            "confirmed_outcomes",
        ),
        follow_up_items=_require_string_list(
            value["follow_up_items"],
            "follow_up_items",
        ),
        updated_at=_require_timestamp(value["updated_at"], request),
        analyzed_through_sequence=analyzed_through_sequence,
    )
