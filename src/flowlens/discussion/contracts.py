"""Immutable contracts for local discussion analysis."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from flowlens.domain._validation import (
    ContractValidationError,
    require_non_negative_int,
    require_str,
)
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.messages import TranscriptRecord


def _require_nonblank_str(value: object, field_name: str) -> str:
    parsed = require_str(value, field_name)
    if not parsed.strip():
        raise ContractValidationError(f"{field_name} must be non-blank")
    return parsed


def _normalize_aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must include a timezone")
    return value.replace(microsecond=(value.microsecond // 1_000) * 1_000)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One deterministic local-model chat message."""

    role: Literal["system", "user"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user"):
            raise ContractValidationError("role must be system or user")
        _require_nonblank_str(self.content, "content")


@dataclass(frozen=True, slots=True)
class DiscussionRequest:
    """One complete-state discussion analysis request."""

    current_state: DiscussionState
    records: tuple[TranscriptRecord, ...]
    requested_revision: int
    updated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.current_state, DiscussionState):
            raise ContractValidationError("current_state must be a DiscussionState")
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, TranscriptRecord) for record in self.records
        ):
            raise ContractValidationError(
                "records must be a tuple containing only TranscriptRecord values"
            )
        requested_revision = require_non_negative_int(
            self.requested_revision,
            "requested_revision",
        )
        if requested_revision != self.current_state.revision + 1:
            raise ContractValidationError(
                "requested_revision must equal current_state.revision plus one"
            )
        object.__setattr__(
            self,
            "updated_at",
            _normalize_aware_datetime(self.updated_at, "updated_at"),
        )

    @property
    def analyzed_through_sequence(self) -> int:
        """Return the greatest global transcript sequence in this request."""

        return max(
            (record.sequence for record in self.records),
            default=self.current_state.analyzed_through_sequence,
        )


@dataclass(frozen=True, slots=True)
class DiscussionStatusPayload:
    """Metadata-only discussion worker status."""

    state: str
    revision: int
    pending_count: int
    error_code: str | None
    analyzed_through_sequence: int = 0

    def __post_init__(self) -> None:
        _require_nonblank_str(self.state, "state")
        require_non_negative_int(self.revision, "revision")
        require_non_negative_int(self.pending_count, "pending_count")
        require_non_negative_int(
            self.analyzed_through_sequence,
            "analyzed_through_sequence",
        )
        if self.error_code is not None:
            _require_nonblank_str(self.error_code, "error_code")


@dataclass(frozen=True, slots=True)
class DiscussionStoppedPayload:
    """Discussion worker drain acknowledgement."""

    worker: str
    drained: bool
    final_revision: int
    pending_count: int
    analyzed_through_sequence: int = 0

    def __post_init__(self) -> None:
        if self.worker != "DISCUSSION":
            raise ContractValidationError("worker must be DISCUSSION")
        if self.drained is not True:
            raise ContractValidationError("drained must be true")
        require_non_negative_int(self.final_revision, "final_revision")
        require_non_negative_int(self.pending_count, "pending_count")
        require_non_negative_int(
            self.analyzed_through_sequence,
            "analyzed_through_sequence",
        )


class DiscussionBackend(Protocol):
    """Local structured-generation backend boundary."""

    def count_tokens(self, text: str) -> int:
        """Count tokens without contacting or loading a remote model."""

        ...

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        response_schema: dict[str, object],
    ) -> str:
        """Return one grammar-constrained JSON object."""

        ...


class DiscussionOutputError(ValueError):
    """Raised when generated output violates the closed state contract."""


class DiscussionGenerationError(RuntimeError):
    """Raised when the local model cannot generate a response."""


class DiscussionContextError(ValueError):
    """Raised when transcript context cannot be bounded safely."""
