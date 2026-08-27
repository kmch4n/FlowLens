"""Deterministic coalescing for local discussion-analysis requests."""

import re
from datetime import datetime

from flowlens.discussion.contracts import DiscussionRequest
from flowlens.domain._validation import ContractValidationError
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.messages import TranscriptRecord

_DEFAULT_COALESCE_MS = 500
_HESITATION_ONLY = frozenset({"えー", "えっと", "あー", "あの"})
_NON_SPEECH_MARKERS = frozenset(
    {
        "[音楽]",
        "[無音]",
        "(無音)",
        "（無音）",  # noqa: RUF001
        "【音楽】",
        "【無音】",
    }
)
_TRAILING_PUNCTUATION = re.compile(r"[。．.!！?？、，,…・:：;；]+$")  # noqa: RUF001


def _require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be an integer")
    return value


def _require_now_ms(value: object) -> int:
    parsed = _require_int(value, "now_ms")
    if parsed < 0:
        raise ContractValidationError("now_ms must be non-negative")
    return parsed


def _require_updated_at(value: object) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ContractValidationError("updated_at must include a timezone")
    return value


def _is_meaningful(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    while normalized:
        without_punctuation = _TRAILING_PUNCTUATION.sub("", normalized).rstrip()
        if without_punctuation == normalized:
            break
        normalized = without_punctuation
    if not normalized or normalized in _NON_SPEECH_MARKERS:
        return False
    return normalized not in _HESITATION_ONLY


class DiscussionScheduler:
    """Coalesce committed transcripts into one failure-safe request at a time."""

    def __init__(
        self,
        initial_state: DiscussionState,
        coalesce_ms: int = _DEFAULT_COALESCE_MS,
    ) -> None:
        if not isinstance(initial_state, DiscussionState):
            raise ContractValidationError("initial_state must be a DiscussionState")
        parsed_coalesce_ms = _require_int(coalesce_ms, "coalesce_ms")
        if parsed_coalesce_ms <= 0:
            raise ContractValidationError("coalesce_ms must be positive")

        self._current_state = initial_state
        self._coalesce_ms = parsed_coalesce_ms
        self._pending: list[TranscriptRecord] = []
        self._in_flight: DiscussionRequest | None = None
        self._paused = False
        self._coalesce_deadline_ms: int | None = None
        self._needs_new_commit_after_failure = False
        self._last_sequence = initial_state.analyzed_through_sequence
        self._last_now_ms: int | None = None

    @property
    def current_state(self) -> DiscussionState:
        """Return the current immutable discussion snapshot."""

        return self._current_state

    @property
    def has_pending(self) -> bool:
        """Report whether meaningful committed transcript remains queued."""

        return bool(self._pending)

    def add(self, record: TranscriptRecord, now_ms: int) -> None:
        """Queue one meaningful committed transcript in strict sequence order."""

        if not isinstance(record, TranscriptRecord):
            raise ContractValidationError("record must be a TranscriptRecord")
        parsed_now_ms = _require_now_ms(now_ms)
        self._check_monotonic(parsed_now_ms)
        if record.sequence <= self._last_sequence:
            raise ContractValidationError(
                "record sequence must be greater than every prior sequence"
            )

        self._last_now_ms = parsed_now_ms
        self._last_sequence = record.sequence
        if not _is_meaningful(record.text):
            return

        self._pending.append(record)
        self._coalesce_deadline_ms = parsed_now_ms + self._coalesce_ms
        self._needs_new_commit_after_failure = False

    def set_paused(self, paused: bool) -> None:
        """Set launch suspension without changing pending timing."""

        if not isinstance(paused, bool):
            raise ContractValidationError("paused must be a boolean")
        self._paused = paused

    def next_request(
        self,
        now_ms: int,
        updated_at: datetime,
    ) -> DiscussionRequest | None:
        """Launch a coalesced request when its exact deadline has elapsed."""

        parsed_now_ms = _require_now_ms(now_ms)
        parsed_updated_at = _require_updated_at(updated_at)
        self._check_monotonic(parsed_now_ms)
        self._last_now_ms = parsed_now_ms

        if (
            self._paused
            or self._in_flight is not None
            or not self._pending
            or self._needs_new_commit_after_failure
            or self._coalesce_deadline_ms is None
            or parsed_now_ms < self._coalesce_deadline_ms
        ):
            return None
        return self._launch(parsed_updated_at)

    def succeed(
        self,
        request: DiscussionRequest,
        new_state: DiscussionState,
    ) -> None:
        """Accept a valid replacement and remove exactly its transcript batch."""

        self._require_active_request(request)
        if not isinstance(new_state, DiscussionState):
            self._invalidate_active_request("replacement must be a DiscussionState")
        try:
            canonical_state = DiscussionState(
                revision=new_state.revision,
                mode=new_state.mode,
                current_focus=new_state.current_focus,
                key_points=new_state.key_points,
                confirmed_outcomes=new_state.confirmed_outcomes,
                follow_up_items=new_state.follow_up_items,
                updated_at=new_state.updated_at,
                analyzed_through_sequence=new_state.analyzed_through_sequence,
            )
        except (TypeError, ValueError):
            self._invalidate_active_request(
                "replacement must satisfy the DiscussionState contract"
            )
        if canonical_state != new_state:
            self._invalidate_active_request(
                "replacement must be a canonical DiscussionState"
            )
        if (
            new_state.revision != request.requested_revision
            or new_state.mode is not request.current_state.mode
            or new_state.updated_at != request.updated_at
            or new_state.analyzed_through_sequence != request.analyzed_through_sequence
        ):
            self._invalidate_active_request(
                "replacement must match the request revision, mode, and timestamp"
            )

        batch_size = len(request.records)
        pending_batch = self._pending[:batch_size]
        if len(pending_batch) != batch_size or any(
            pending is not requested
            for pending, requested in zip(pending_batch, request.records, strict=True)
        ):
            self._invalidate_active_request(
                "replacement request does not match the pending batch"
            )

        del self._pending[:batch_size]
        self._current_state = new_state
        self._in_flight = None
        self._needs_new_commit_after_failure = False
        if not self._pending:
            self._coalesce_deadline_ms = None

    def fail(self, request: DiscussionRequest) -> None:
        """Retain a failed batch and block retries until meaningful new input."""

        self._require_active_request(request)
        self._in_flight = None
        self._needs_new_commit_after_failure = True

    def final_request(self, updated_at: datetime) -> DiscussionRequest | None:
        """Launch all pending input immediately when no launch is already active."""

        parsed_updated_at = _require_updated_at(updated_at)
        if (
            self._paused
            or self._in_flight is not None
            or not self._pending
            or self._needs_new_commit_after_failure
        ):
            return None
        return self._launch(parsed_updated_at)

    def _check_monotonic(self, now_ms: int) -> None:
        if self._last_now_ms is not None and now_ms < self._last_now_ms:
            raise ContractValidationError("now_ms must be monotonic")

    def _launch(self, updated_at: datetime) -> DiscussionRequest:
        request = DiscussionRequest(
            current_state=self._current_state,
            records=tuple(self._pending),
            requested_revision=self._current_state.revision + 1,
            updated_at=updated_at,
        )
        self._in_flight = request
        return request

    def _require_active_request(self, request: object) -> None:
        if not isinstance(request, DiscussionRequest):
            raise ContractValidationError("request must be a DiscussionRequest")
        if request is not self._in_flight:
            raise ContractValidationError(
                "request must be the active in-flight request"
            )

    def _invalidate_active_request(self, message: str) -> None:
        self._in_flight = None
        self._needs_new_commit_after_failure = True
        raise ContractValidationError(message)
