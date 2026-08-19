"""Immutable session-manifest persistence contracts."""

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
    require_sha256,
    require_str,
    require_str_list,
)
from flowlens.domain.enums import SessionMode, SessionStatus

_SESSION_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
_DEVICE_IDENTITY_KEYS = frozenset({"device_id", "display_name"})
_MODEL_IDENTITY_KEYS = frozenset({"repository", "revision", "sha256"})
_PAUSE_INTERVAL_KEYS = frozenset({"started_ms", "ended_ms"})
_SESSION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "status",
        "mode",
        "started_at",
        "ended_at",
        "active_duration_ms",
        "pause_intervals",
        "microphone",
        "loopback_output",
        "asr_model",
        "discussion_model",
        "application_version",
        "transcript_entry_count",
        "final_discussion_state_revision",
        "recovery_notes",
    }
)


def _require_mapping(value: object, record_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ContractValidationError(f"{record_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_non_empty_str(value: object, field_name: str) -> str:
    parsed = require_str(value, field_name)
    if not parsed:
        raise ContractValidationError(f"{field_name} must be non-empty")
    return parsed


def _require_session_id(value: object) -> str:
    parsed = require_str(value, "session_id")
    if _SESSION_ID_PATTERN.fullmatch(parsed) is None:
        raise ContractValidationError(
            "session_id must contain 26 uppercase Crockford characters"
        )
    return parsed


def _normalize_aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must include a timezone")
    return value.replace(microsecond=(value.microsecond // 1_000) * 1_000)


def _require_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) for item in value
    ):
        raise ContractValidationError(f"{field_name} must contain only strings")
    return tuple(value)


def _parse_mode(value: object) -> SessionMode:
    parsed = require_str(value, "mode")
    try:
        return SessionMode(parsed)
    except ValueError as error:
        raise ContractValidationError("mode must be a supported SessionMode") from error


def _parse_status(value: object) -> SessionStatus:
    parsed = require_str(value, "status")
    try:
        return SessionStatus(parsed)
    except ValueError as error:
        raise ContractValidationError(
            "status must be a supported SessionStatus"
        ) from error


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Stable audio-device identifier and user-facing display name."""

    device_id: str
    display_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "device_id",
            _require_non_empty_str(self.device_id, "device_id"),
        )
        object.__setattr__(
            self,
            "display_name",
            _require_non_empty_str(self.display_name, "display_name"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the identity to its strict wire shape."""

        return {
            "device_id": self.device_id,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse an identity while rejecting missing or unknown fields."""

        mapping = _require_mapping(value, "DeviceIdentity")
        require_exact_keys(mapping, _DEVICE_IDENTITY_KEYS, "DeviceIdentity")
        return cls(
            device_id=_require_non_empty_str(mapping["device_id"], "device_id"),
            display_name=_require_non_empty_str(
                mapping["display_name"],
                "display_name",
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Pinned model source and verified artifact checksum."""

    repository: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository",
            _require_non_empty_str(self.repository, "repository"),
        )
        object.__setattr__(
            self,
            "revision",
            _require_non_empty_str(self.revision, "revision"),
        )
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "sha256"))

    def to_dict(self) -> dict[str, object]:
        """Serialize the identity to its strict wire shape."""

        return {
            "repository": self.repository,
            "revision": self.revision,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse an identity while rejecting missing or unknown fields."""

        mapping = _require_mapping(value, "ModelIdentity")
        require_exact_keys(mapping, _MODEL_IDENTITY_KEYS, "ModelIdentity")
        return cls(
            repository=_require_non_empty_str(mapping["repository"], "repository"),
            revision=_require_non_empty_str(mapping["revision"], "revision"),
            sha256=require_sha256(mapping["sha256"], "sha256"),
        )


@dataclass(frozen=True, slots=True)
class PauseInterval:
    """One closed pause interval on the monotonic session clock."""

    started_ms: int
    ended_ms: int

    def __post_init__(self) -> None:
        started_ms = require_non_negative_int(self.started_ms, "started_ms")
        ended_ms = require_non_negative_int(self.ended_ms, "ended_ms")
        if ended_ms < started_ms:
            raise ContractValidationError("ended_ms must not precede started_ms")
        object.__setattr__(self, "started_ms", started_ms)
        object.__setattr__(self, "ended_ms", ended_ms)

    def to_dict(self) -> dict[str, object]:
        """Serialize the interval to its strict wire shape."""

        return {
            "started_ms": self.started_ms,
            "ended_ms": self.ended_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse an interval while rejecting missing or unknown fields."""

        mapping = _require_mapping(value, "PauseInterval")
        require_exact_keys(mapping, _PAUSE_INTERVAL_KEYS, "PauseInterval")
        return cls(
            started_ms=require_non_negative_int(mapping["started_ms"], "started_ms"),
            ended_ms=require_non_negative_int(mapping["ended_ms"], "ended_ms"),
        )


@dataclass(frozen=True, slots=True)
class SessionManifest:
    """Complete persisted metadata for one capture session."""

    schema_version: int
    session_id: str
    status: SessionStatus
    mode: SessionMode
    started_at: datetime
    ended_at: datetime | None
    active_duration_ms: int
    pause_intervals: tuple[PauseInterval, ...]
    microphone: DeviceIdentity
    loopback_output: DeviceIdentity
    asr_model: ModelIdentity
    discussion_model: ModelIdentity
    application_version: str
    transcript_entry_count: int
    final_discussion_state_revision: int
    recovery_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        schema_version = require_int(self.schema_version, "schema_version")
        if schema_version != 1:
            raise ContractValidationError("schema_version must be 1")
        if not isinstance(self.status, SessionStatus):
            raise ContractValidationError("status must be a SessionStatus")
        if not isinstance(self.mode, SessionMode):
            raise ContractValidationError("mode must be a SessionMode")
        started_at = _normalize_aware_datetime(self.started_at, "started_at")
        ended_at = self.ended_at
        if ended_at is not None:
            ended_at = _normalize_aware_datetime(ended_at, "ended_at")
        if self.status in {SessionStatus.COMPLETED, SessionStatus.RECOVERED}:
            if ended_at is None:
                raise ContractValidationError(
                    "ended_at is required for completed or recovered sessions"
                )
        if not isinstance(self.pause_intervals, list | tuple) or not all(
            isinstance(interval, PauseInterval) for interval in self.pause_intervals
        ):
            raise ContractValidationError(
                "pause_intervals must contain only PauseInterval values"
            )
        if not isinstance(self.microphone, DeviceIdentity):
            raise ContractValidationError("microphone must be a DeviceIdentity")
        if not isinstance(self.loopback_output, DeviceIdentity):
            raise ContractValidationError("loopback_output must be a DeviceIdentity")
        if not isinstance(self.asr_model, ModelIdentity):
            raise ContractValidationError("asr_model must be a ModelIdentity")
        if not isinstance(self.discussion_model, ModelIdentity):
            raise ContractValidationError("discussion_model must be a ModelIdentity")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "session_id", _require_session_id(self.session_id))
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(
            self,
            "active_duration_ms",
            require_non_negative_int(self.active_duration_ms, "active_duration_ms"),
        )
        object.__setattr__(self, "pause_intervals", tuple(self.pause_intervals))
        object.__setattr__(
            self,
            "application_version",
            _require_non_empty_str(self.application_version, "application_version"),
        )
        object.__setattr__(
            self,
            "transcript_entry_count",
            require_non_negative_int(
                self.transcript_entry_count,
                "transcript_entry_count",
            ),
        )
        object.__setattr__(
            self,
            "final_discussion_state_revision",
            require_non_negative_int(
                self.final_discussion_state_revision,
                "final_discussion_state_revision",
            ),
        )
        object.__setattr__(
            self,
            "recovery_notes",
            _require_string_tuple(self.recovery_notes, "recovery_notes"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the manifest in its specified field order."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "status": self.status.value,
            "mode": self.mode.value,
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "ended_at": (
                None
                if self.ended_at is None
                else self.ended_at.isoformat(timespec="milliseconds")
            ),
            "active_duration_ms": self.active_duration_ms,
            "pause_intervals": [
                interval.to_dict() for interval in self.pause_intervals
            ],
            "microphone": self.microphone.to_dict(),
            "loopback_output": self.loopback_output.to_dict(),
            "asr_model": self.asr_model.to_dict(),
            "discussion_model": self.discussion_model.to_dict(),
            "application_version": self.application_version,
            "transcript_entry_count": self.transcript_entry_count,
            "final_discussion_state_revision": (self.final_discussion_state_revision),
            "recovery_notes": list(self.recovery_notes),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a manifest while rejecting missing or unknown fields."""

        mapping = _require_mapping(value, "SessionManifest")
        require_exact_keys(mapping, _SESSION_MANIFEST_KEYS, "SessionManifest")

        pause_values = mapping["pause_intervals"]
        if not isinstance(pause_values, list):
            raise ContractValidationError("pause_intervals must be a list")
        ended_value = mapping["ended_at"]
        ended_at = (
            None
            if ended_value is None
            else parse_timezone_datetime(ended_value, "ended_at")
        )

        return cls(
            schema_version=require_int(mapping["schema_version"], "schema_version"),
            session_id=_require_session_id(mapping["session_id"]),
            status=_parse_status(mapping["status"]),
            mode=_parse_mode(mapping["mode"]),
            started_at=parse_timezone_datetime(mapping["started_at"], "started_at"),
            ended_at=ended_at,
            active_duration_ms=require_non_negative_int(
                mapping["active_duration_ms"],
                "active_duration_ms",
            ),
            pause_intervals=tuple(
                PauseInterval.from_dict(interval) for interval in pause_values
            ),
            microphone=DeviceIdentity.from_dict(mapping["microphone"]),
            loopback_output=DeviceIdentity.from_dict(mapping["loopback_output"]),
            asr_model=ModelIdentity.from_dict(mapping["asr_model"]),
            discussion_model=ModelIdentity.from_dict(mapping["discussion_model"]),
            application_version=_require_non_empty_str(
                mapping["application_version"],
                "application_version",
            ),
            transcript_entry_count=require_non_negative_int(
                mapping["transcript_entry_count"],
                "transcript_entry_count",
            ),
            final_discussion_state_revision=require_non_negative_int(
                mapping["final_discussion_state_revision"],
                "final_discussion_state_revision",
            ),
            recovery_notes=tuple(
                require_str_list(mapping["recovery_notes"], "recovery_notes")
            ),
        )
