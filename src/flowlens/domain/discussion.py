"""Immutable discussion-state persistence contracts."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Self, cast

from flowlens.domain._validation import (
    ContractValidationError,
    parse_timezone_datetime,
    require_exact_keys,
    require_int,
    require_non_negative_int,
    require_str,
    require_str_list,
)
from flowlens.domain.enums import SessionMode

_SESSION_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
_DISCUSSION_STATE_KEYS = frozenset(
    {
        "revision",
        "mode",
        "current_focus",
        "key_points",
        "confirmed_outcomes",
        "follow_up_items",
        "updated_at",
    }
)
_STATE_HISTORY_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "previous_revision",
        "new_revision",
        "state",
    }
)


def _require_mapping(value: object, record_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ContractValidationError(f"{record_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_mode(value: object) -> SessionMode:
    if not isinstance(value, SessionMode):
        raise ContractValidationError("mode must be a SessionMode")
    return value


def _parse_mode(value: object) -> SessionMode:
    parsed = require_str(value, "mode")
    try:
        return SessionMode(parsed)
    except ValueError as error:
        raise ContractValidationError("mode must be a supported SessionMode") from error


def _require_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) for item in value
    ):
        raise ContractValidationError(f"{field_name} must contain only strings")
    return tuple(value)


def _normalize_aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must include a timezone")
    return value.replace(microsecond=(value.microsecond // 1_000) * 1_000)


def _require_session_id(value: object) -> str:
    parsed = require_str(value, "session_id")
    if _SESSION_ID_PATTERN.fullmatch(parsed) is None:
        raise ContractValidationError(
            "session_id must contain 26 uppercase Crockford characters"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class DiscussionState:
    """Complete discussion snapshot persisted after analysis."""

    revision: int
    mode: SessionMode
    current_focus: str
    key_points: tuple[str, ...]
    confirmed_outcomes: tuple[str, ...]
    follow_up_items: tuple[str, ...]
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision",
            require_non_negative_int(self.revision, "revision"),
        )
        object.__setattr__(self, "mode", _require_mode(self.mode))
        object.__setattr__(
            self,
            "current_focus",
            require_str(self.current_focus, "current_focus"),
        )
        object.__setattr__(
            self,
            "key_points",
            _require_string_tuple(self.key_points, "key_points"),
        )
        object.__setattr__(
            self,
            "confirmed_outcomes",
            _require_string_tuple(self.confirmed_outcomes, "confirmed_outcomes"),
        )
        object.__setattr__(
            self,
            "follow_up_items",
            _require_string_tuple(self.follow_up_items, "follow_up_items"),
        )
        object.__setattr__(
            self,
            "updated_at",
            _normalize_aware_datetime(self.updated_at, "updated_at"),
        )

    @classmethod
    def initial(cls, mode: SessionMode, updated_at: datetime) -> Self:
        """Create the empty revision-zero state for a new session."""

        return cls(
            revision=0,
            mode=mode,
            current_focus="",
            key_points=(),
            confirmed_outcomes=(),
            follow_up_items=(),
            updated_at=updated_at,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the snapshot to its strict wire shape."""

        return {
            "revision": self.revision,
            "mode": self.mode.value,
            "current_focus": self.current_focus,
            "key_points": list(self.key_points),
            "confirmed_outcomes": list(self.confirmed_outcomes),
            "follow_up_items": list(self.follow_up_items),
            "updated_at": self.updated_at.isoformat(timespec="milliseconds"),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a snapshot while rejecting missing or unknown fields."""

        mapping = _require_mapping(value, "DiscussionState")
        require_exact_keys(mapping, _DISCUSSION_STATE_KEYS, "DiscussionState")
        return cls(
            revision=require_non_negative_int(mapping["revision"], "revision"),
            mode=_parse_mode(mapping["mode"]),
            current_focus=require_str(mapping["current_focus"], "current_focus"),
            key_points=tuple(require_str_list(mapping["key_points"], "key_points")),
            confirmed_outcomes=tuple(
                require_str_list(mapping["confirmed_outcomes"], "confirmed_outcomes")
            ),
            follow_up_items=tuple(
                require_str_list(mapping["follow_up_items"], "follow_up_items")
            ),
            updated_at=parse_timezone_datetime(mapping["updated_at"], "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class StateHistoryRecord:
    """One successful, sequential discussion-state replacement."""

    schema_version: int
    session_id: str
    previous_revision: int
    new_revision: int
    state: DiscussionState

    def __post_init__(self) -> None:
        schema_version = require_int(self.schema_version, "schema_version")
        if schema_version != 1:
            raise ContractValidationError("schema_version must be 1")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "session_id", _require_session_id(self.session_id))
        previous_revision = require_non_negative_int(
            self.previous_revision,
            "previous_revision",
        )
        new_revision = require_non_negative_int(self.new_revision, "new_revision")
        if previous_revision + 1 != new_revision:
            raise ContractValidationError(
                "new_revision must equal previous_revision plus one"
            )
        if not isinstance(self.state, DiscussionState):
            raise ContractValidationError("state must be a DiscussionState")
        if new_revision != self.state.revision:
            raise ContractValidationError("new_revision must equal state.revision")
        object.__setattr__(self, "previous_revision", previous_revision)
        object.__setattr__(self, "new_revision", new_revision)

    def to_dict(self) -> dict[str, object]:
        """Serialize the history record to its strict wire shape."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "previous_revision": self.previous_revision,
            "new_revision": self.new_revision,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a history record while rejecting missing or unknown fields."""

        mapping = _require_mapping(value, "StateHistoryRecord")
        require_exact_keys(mapping, _STATE_HISTORY_RECORD_KEYS, "StateHistoryRecord")
        return cls(
            schema_version=require_int(mapping["schema_version"], "schema_version"),
            session_id=_require_session_id(mapping["session_id"]),
            previous_revision=require_non_negative_int(
                mapping["previous_revision"],
                "previous_revision",
            ),
            new_revision=require_non_negative_int(
                mapping["new_revision"],
                "new_revision",
            ),
            state=DiscussionState.from_dict(mapping["state"]),
        )
