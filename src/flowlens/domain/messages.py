"""Transcript, event, and inter-process message contracts."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Self, cast

from flowlens.domain._validation import (
    ContractValidationError,
    parse_timezone_datetime,
    require_exact_keys,
    require_int,
    require_non_negative_int,
    require_str,
)
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import AudioSource, EventType, MessageType, ProcessSource
from flowlens.domain.session import PauseInterval, SessionManifest

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
_TRANSCRIPT_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "segment_id",
        "sequence",
        "source",
        "text",
        "session_start_ms",
        "session_end_ms",
        "source_start_sample",
        "source_end_sample",
        "committed_at",
    }
)
_EVENT_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "sequence",
        "event_type",
        "source",
        "session_time_ms",
        "created_at",
        "details",
    }
)


class UnknownSchemaVersionError(ContractValidationError):
    """Raised when an IPC receiver cannot handle an envelope schema."""


class MessageSequenceError(ContractValidationError):
    """Raised when a message carries an invalid sender-local sequence."""


class SequenceResult(str, Enum):
    """Outcome of comparing a message with a sender's expected sequence."""

    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    GAP = "GAP"


def _require_mapping(value: object, record_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ContractValidationError(f"{record_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_id(value: object, field_name: str) -> str:
    parsed = require_str(value, field_name)
    if _ID_PATTERN.fullmatch(parsed) is None:
        raise ContractValidationError(
            f"{field_name} must contain 26 uppercase Crockford characters"
        )
    return parsed


def _require_schema_version(value: object) -> int:
    parsed = require_int(value, "schema_version")
    if parsed != 1:
        raise ContractValidationError("schema_version must be 1")
    return parsed


def _require_positive_int(value: object, field_name: str) -> int:
    parsed = require_int(value, field_name)
    if parsed <= 0:
        raise ContractValidationError(f"{field_name} must be positive")
    return parsed


def _require_enum[EnumT: Enum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if not isinstance(value, enum_type):
        raise ContractValidationError(f"{field_name} must be a {enum_type.__name__}")
    return value


def _parse_enum[EnumT: Enum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    parsed = require_str(value, field_name)
    try:
        return enum_type(parsed)
    except ValueError as error:
        raise ContractValidationError(
            f"{field_name} must be a supported {enum_type.__name__}"
        ) from error


def _normalize_aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must include a timezone")
    return value.replace(microsecond=(value.microsecond // 1_000) * 1_000)


def _copy_json_value(
    value: object,
    field_name: str,
    active_container_ids: set[int] | None = None,
) -> JsonValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(
                f"{field_name} must contain finite JSON values"
            )
        return value
    if isinstance(value, list | dict):
        active_ids = set() if active_container_ids is None else active_container_ids
        container_id = id(value)
        if container_id in active_ids:
            raise ContractValidationError(
                f"{field_name} contains a JSON container cycle"
            )
        active_ids.add(container_id)
        try:
            if isinstance(value, list):
                return [
                    _copy_json_value(item, f"{field_name}[{index}]", active_ids)
                    for index, item in enumerate(value)
                ]
            if not all(isinstance(key, str) for key in value):
                raise ContractValidationError(
                    f"{field_name} must use string object keys"
                )
            return {
                key: _copy_json_value(item, f"{field_name}.{key}", active_ids)
                for key, item in value.items()
            }
        finally:
            active_ids.remove(container_id)
    raise ContractValidationError(f"{field_name} must contain only JSON values")


def _copy_json_object(value: object, field_name: str) -> dict[str, JsonValue]:
    copied = _copy_json_value(value, field_name)
    if not isinstance(copied, dict):
        raise ContractValidationError(f"{field_name} must be a JSON object")
    return copied


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    """One immutable committed transcript segment."""

    schema_version: int
    segment_id: str
    sequence: int
    source: AudioSource
    text: str
    session_start_ms: int
    session_end_ms: int
    source_start_sample: int
    source_end_sample: int
    committed_at: datetime

    def __post_init__(self) -> None:
        schema_version = _require_schema_version(self.schema_version)
        segment_id = _require_id(self.segment_id, "segment_id")
        sequence = _require_positive_int(self.sequence, "sequence")
        source = _require_enum(self.source, AudioSource, "source")
        text = require_str(self.text, "text")
        if not text.strip():
            raise ContractValidationError("text must not be empty or whitespace")
        session_start_ms = require_non_negative_int(
            self.session_start_ms,
            "session_start_ms",
        )
        session_end_ms = require_non_negative_int(
            self.session_end_ms,
            "session_end_ms",
        )
        if session_end_ms < session_start_ms:
            raise ContractValidationError(
                "session_end_ms must not precede session_start_ms"
            )
        source_start_sample = require_non_negative_int(
            self.source_start_sample,
            "source_start_sample",
        )
        source_end_sample = require_non_negative_int(
            self.source_end_sample,
            "source_end_sample",
        )
        if source_end_sample < source_start_sample:
            raise ContractValidationError(
                "source_end_sample must not precede source_start_sample"
            )

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "session_start_ms", session_start_ms)
        object.__setattr__(self, "session_end_ms", session_end_ms)
        object.__setattr__(self, "source_start_sample", source_start_sample)
        object.__setattr__(self, "source_end_sample", source_end_sample)
        object.__setattr__(
            self,
            "committed_at",
            _normalize_aware_datetime(self.committed_at, "committed_at"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the record in the persistence field order."""

        return {
            "schema_version": self.schema_version,
            "segment_id": self.segment_id,
            "sequence": self.sequence,
            "source": self.source.value,
            "text": self.text,
            "session_start_ms": self.session_start_ms,
            "session_end_ms": self.session_end_ms,
            "source_start_sample": self.source_start_sample,
            "source_end_sample": self.source_end_sample,
            "committed_at": self.committed_at.isoformat(timespec="milliseconds"),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a transcript record with an exact wire shape."""

        mapping = _require_mapping(value, "TranscriptRecord")
        require_exact_keys(mapping, _TRANSCRIPT_RECORD_KEYS, "TranscriptRecord")
        return cls(
            schema_version=require_int(mapping["schema_version"], "schema_version"),
            segment_id=_require_id(mapping["segment_id"], "segment_id"),
            sequence=require_int(mapping["sequence"], "sequence"),
            source=_parse_enum(mapping["source"], AudioSource, "source"),
            text=require_str(mapping["text"], "text"),
            session_start_ms=require_int(
                mapping["session_start_ms"],
                "session_start_ms",
            ),
            session_end_ms=require_int(mapping["session_end_ms"], "session_end_ms"),
            source_start_sample=require_int(
                mapping["source_start_sample"],
                "source_start_sample",
            ),
            source_end_sample=require_int(
                mapping["source_end_sample"],
                "source_end_sample",
            ),
            committed_at=parse_timezone_datetime(
                mapping["committed_at"],
                "committed_at",
            ),
        )


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One operational session event persisted to the event log."""

    schema_version: int
    session_id: str
    sequence: int
    event_type: EventType
    source: ProcessSource
    session_time_ms: int
    created_at: datetime
    details: dict[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _require_schema_version(self.schema_version),
        )
        object.__setattr__(
            self,
            "session_id",
            _require_id(self.session_id, "session_id"),
        )
        object.__setattr__(
            self,
            "sequence",
            _require_positive_int(self.sequence, "sequence"),
        )
        object.__setattr__(
            self,
            "event_type",
            _require_enum(self.event_type, EventType, "event_type"),
        )
        object.__setattr__(
            self,
            "source",
            _require_enum(self.source, ProcessSource, "source"),
        )
        object.__setattr__(
            self,
            "session_time_ms",
            require_non_negative_int(self.session_time_ms, "session_time_ms"),
        )
        object.__setattr__(
            self,
            "created_at",
            _normalize_aware_datetime(self.created_at, "created_at"),
        )
        object.__setattr__(
            self,
            "details",
            _copy_json_object(self.details, "details"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the event in the persistence field order."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "source": self.source.value,
            "session_time_ms": self.session_time_ms,
            "created_at": self.created_at.isoformat(timespec="milliseconds"),
            "details": _copy_json_object(self.details, "details"),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse an event record with an exact wire shape."""

        mapping = _require_mapping(value, "EventRecord")
        require_exact_keys(mapping, _EVENT_RECORD_KEYS, "EventRecord")
        return cls(
            schema_version=require_int(mapping["schema_version"], "schema_version"),
            session_id=_require_id(mapping["session_id"], "session_id"),
            sequence=require_int(mapping["sequence"], "sequence"),
            event_type=_parse_enum(mapping["event_type"], EventType, "event_type"),
            source=_parse_enum(mapping["source"], ProcessSource, "source"),
            session_time_ms=require_int(
                mapping["session_time_ms"],
                "session_time_ms",
            ),
            created_at=parse_timezone_datetime(mapping["created_at"], "created_at"),
            details=_copy_json_object(mapping["details"], "details"),
        )


@dataclass(frozen=True, slots=True)
class TranscriptCommitted:
    """Control-queue payload carrying one committed transcript record."""

    record: TranscriptRecord

    def __post_init__(self) -> None:
        if not isinstance(self.record, TranscriptRecord):
            raise ContractValidationError("record must be a TranscriptRecord")


@dataclass(frozen=True, slots=True)
class DiscussionStateReplaced:
    """Control-queue payload carrying a complete discussion replacement."""

    previous_revision: int
    state: DiscussionState

    def __post_init__(self) -> None:
        previous_revision = require_non_negative_int(
            self.previous_revision,
            "previous_revision",
        )
        if not isinstance(self.state, DiscussionState):
            raise ContractValidationError("state must be a DiscussionState")
        if previous_revision + 1 != self.state.revision:
            raise ContractValidationError(
                "previous_revision must immediately precede state.revision"
            )
        object.__setattr__(self, "previous_revision", previous_revision)


@dataclass(frozen=True, slots=True)
class WriterOpenSession:
    """Writer command that creates an incomplete session."""

    session_dir: Path
    manifest: SessionManifest
    initial_state: DiscussionState

    def __post_init__(self) -> None:
        if not isinstance(self.session_dir, Path):
            raise ContractValidationError("session_dir must be a Path")
        if not isinstance(self.manifest, SessionManifest):
            raise ContractValidationError("manifest must be a SessionManifest")
        if not isinstance(self.initial_state, DiscussionState):
            raise ContractValidationError("initial_state must be a DiscussionState")


@dataclass(frozen=True, slots=True)
class WriterAppendEvent:
    """Writer command that appends one operational event."""

    record: EventRecord

    def __post_init__(self) -> None:
        if not isinstance(self.record, EventRecord):
            raise ContractValidationError("record must be an EventRecord")


@dataclass(frozen=True, slots=True)
class WriterFlush:
    """Writer command that synchronizes pending session data."""


@dataclass(frozen=True, slots=True)
class WriterFinalize:
    """Writer command containing all values needed for final persistence."""

    ended_at: datetime
    active_duration_ms: int
    pause_intervals: tuple[PauseInterval, ...]
    final_state: DiscussionState
    completion_event: EventRecord

    def __post_init__(self) -> None:
        ended_at = _normalize_aware_datetime(self.ended_at, "ended_at")
        active_duration_ms = require_non_negative_int(
            self.active_duration_ms,
            "active_duration_ms",
        )
        if not isinstance(self.pause_intervals, list | tuple) or not all(
            isinstance(interval, PauseInterval) for interval in self.pause_intervals
        ):
            raise ContractValidationError(
                "pause_intervals must contain only PauseInterval values"
            )
        if not isinstance(self.final_state, DiscussionState):
            raise ContractValidationError("final_state must be a DiscussionState")
        if not isinstance(self.completion_event, EventRecord):
            raise ContractValidationError("completion_event must be an EventRecord")

        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "active_duration_ms", active_duration_ms)
        object.__setattr__(self, "pause_intervals", tuple(self.pause_intervals))


@dataclass(frozen=True, slots=True)
class WriterShutdown:
    """Writer command that exits after prior work has drained."""


class WriterForceCloseOutcome(str, Enum):
    """Writer-owned result of the force-close/finalize linearization race."""

    INCOMPLETE = "INCOMPLETE"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True, slots=True)
class WriterForceCloseRequest:
    """Out-of-band request carrying metadata-only incomplete evidence."""

    event: EventRecord

    def __post_init__(self) -> None:
        if type(self.event) is not EventRecord:
            raise ContractValidationError("event must be an exact EventRecord")
        if self.event.event_type is not EventType.FORCE_CLOSE_REQUESTED:
            raise ContractValidationError("event must record force close")


@dataclass(frozen=True, slots=True)
class WriterForceCloseResult:
    """Writer-owned force-close result emitted after durable linearization."""

    outcome: WriterForceCloseOutcome
    latest_successful_save_at: datetime

    def __post_init__(self) -> None:
        if type(self.outcome) is not WriterForceCloseOutcome:
            raise ContractValidationError(
                "outcome must be an exact WriterForceCloseOutcome"
            )
        object.__setattr__(
            self,
            "latest_successful_save_at",
            _normalize_aware_datetime(
                self.latest_successful_save_at,
                "latest_successful_save_at",
            ),
        )


@dataclass(frozen=True, slots=True)
class WriterAck:
    """Writer response acknowledging a successful control mutation."""

    acknowledged_sequence: int
    latest_successful_save_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acknowledged_sequence",
            _require_positive_int(
                self.acknowledged_sequence,
                "acknowledged_sequence",
            ),
        )
        object.__setattr__(
            self,
            "latest_successful_save_at",
            _normalize_aware_datetime(
                self.latest_successful_save_at,
                "latest_successful_save_at",
            ),
        )


@dataclass(frozen=True, slots=True)
class WriterFatal:
    """Writer response describing a fail-closed persistence error.

    A failed sequence of zero identifies audio-queue or lifecycle work that did
    not originate from a control envelope.
    """

    failed_sequence: int
    error_type: str
    message: str

    def __post_init__(self) -> None:
        failed_sequence = require_non_negative_int(
            self.failed_sequence,
            "failed_sequence",
        )
        error_type = require_str(self.error_type, "error_type")
        message = require_str(self.message, "message")
        if not error_type.strip():
            raise ContractValidationError("error_type must be non-empty")
        if not message.strip():
            raise ContractValidationError("message must be non-empty")
        object.__setattr__(self, "failed_sequence", failed_sequence)
        object.__setattr__(self, "error_type", error_type)
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class AudioWriteCommand:
    """Dedicated-audio-queue payload containing canonical PCM samples."""

    source: AudioSource
    pcm_s16le: bytes
    source_start_sample: int
    source_end_sample: int
    session_start_ms: int
    captured_monotonic_ms: int

    def __post_init__(self) -> None:
        source = _require_enum(self.source, AudioSource, "source")
        if not isinstance(self.pcm_s16le, bytes):
            raise ContractValidationError("pcm_s16le must be bytes")
        if len(self.pcm_s16le) % 2 != 0:
            raise ContractValidationError(
                "pcm_s16le must contain an even number of bytes"
            )
        source_start_sample = require_non_negative_int(
            self.source_start_sample,
            "source_start_sample",
        )
        source_end_sample = require_non_negative_int(
            self.source_end_sample,
            "source_end_sample",
        )
        if source_end_sample - source_start_sample != len(self.pcm_s16le) // 2:
            raise ContractValidationError(
                "source sample range must match the PCM sample count"
            )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_start_sample", source_start_sample)
        object.__setattr__(self, "source_end_sample", source_end_sample)
        object.__setattr__(
            self,
            "session_start_ms",
            require_non_negative_int(self.session_start_ms, "session_start_ms"),
        )
        object.__setattr__(
            self,
            "captured_monotonic_ms",
            require_non_negative_int(
                self.captured_monotonic_ms,
                "captured_monotonic_ms",
            ),
        )


@dataclass(frozen=True, slots=True)
class AudioDrainFence:
    """Repeatable ordered pause/stop marker on a dedicated audio queue."""


@dataclass(frozen=True, slots=True)
class MessageEnvelope[PayloadT]:
    """Generic control-queue envelope with deferred schema validation."""

    schema_version: int
    session_id: str
    message_type: MessageType
    sequence: int
    source: ProcessSource
    created_monotonic_ms: int
    payload: PayloadT

    def __post_init__(self) -> None:
        schema_version = require_int(self.schema_version, "schema_version")
        session_id = _require_id(self.session_id, "session_id")
        message_type = _require_enum(self.message_type, MessageType, "message_type")
        sequence = require_int(self.sequence, "sequence")
        if sequence <= 0:
            raise MessageSequenceError("sequence must be positive")
        source = _require_enum(self.source, ProcessSource, "source")
        created_monotonic_ms = require_non_negative_int(
            self.created_monotonic_ms,
            "created_monotonic_ms",
        )

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "created_monotonic_ms", created_monotonic_ms)

    def validate_schema(self) -> None:
        """Reject an envelope schema only at the receiver dispatch boundary."""

        if self.schema_version != 1:
            raise UnknownSchemaVersionError(
                f"unsupported message schema version {self.schema_version}"
            )


class SequenceTracker:
    """Track the next sequence independently for every session and sender."""

    def __init__(self) -> None:
        self._expected: dict[tuple[str, ProcessSource], int] = {}

    def observe[EnvelopePayloadT](
        self,
        envelope: MessageEnvelope[EnvelopePayloadT],
    ) -> SequenceResult:
        """Validate and classify one sender-local message sequence."""

        envelope.validate_schema()
        key = (envelope.session_id, envelope.source)
        expected = self._expected.get(key, 1)
        if envelope.sequence < expected:
            return SequenceResult.DUPLICATE
        self._expected[key] = envelope.sequence + 1
        if envelope.sequence > expected:
            return SequenceResult.GAP
        return SequenceResult.ACCEPTED

    def expected(self, source: ProcessSource, session_id: str) -> int:
        """Return the next expected sequence for a session and sender."""

        validated_source = _require_enum(source, ProcessSource, "source")
        validated_session_id = _require_id(session_id, "session_id")
        return self._expected.get((validated_session_id, validated_source), 1)
