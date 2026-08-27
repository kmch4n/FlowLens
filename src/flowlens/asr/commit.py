"""Stable transcript commits and deterministic chronological sequencing."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from flowlens.asr.types import CommitCandidate, DecodedToken, DecodeHypothesis
from flowlens.domain._validation import (
    ContractValidationError,
    require_non_negative_int,
)
from flowlens.domain.enums import AudioSource
from flowlens.domain.ids import new_ulid
from flowlens.domain.messages import TranscriptRecord

_NON_SPEECH_MARKER = re.compile(
    r"^(?:\[(?:music|applause|laughter|silence|noise|音楽|拍手|笑い|無音|雑音)\]"
    r"|\((?:music|applause|laughter|silence|noise|音楽|拍手|笑い|無音|雑音)\))$",
    re.IGNORECASE,
)
_SPLIT_PUNCTUATION = ("。", "！", "？", "、", ".", "!", "?")  # noqa: RUF001
_SOURCE_RANK = {AudioSource.ME: 0, AudioSource.OTHERS: 1}
_TokenIdentity = tuple[str, int, int]
_CommitKey = tuple[int, int]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _token_identity(token: DecodedToken) -> _TokenIdentity:
    return (token.text, token.start_ms, token.end_ms)


def _require_source(source: object) -> AudioSource:
    if not isinstance(source, AudioSource):
        raise ContractValidationError("source must be an AudioSource")
    return source


def _require_positive_int(value: object, field_name: str) -> int:
    parsed = require_non_negative_int(value, field_name)
    if parsed == 0:
        raise ContractValidationError(f"{field_name} must be positive")
    return parsed


@dataclass(frozen=True, slots=True)
class _ObservedToken:
    identity: _TokenIdentity
    first_seen_ms: int
    consecutive_count: int


class StablePrefixTracker:
    """Return decoder tokens only after their prefix becomes stable."""

    def __init__(self, stable_age_ms: int) -> None:
        self._stable_age_ms = _require_positive_int(stable_age_ms, "stable_age_ms")
        self._previous: tuple[_ObservedToken, ...] = ()
        self._committed_token_count = 0
        self._last_decoded_at_ms: int | None = None

    def observe(
        self,
        hypothesis: DecodeHypothesis,
        decoded_at_ms: int,
        final: bool,
    ) -> tuple[DecodedToken, ...]:
        """Observe one decode and return its newly committable tokens."""

        if not isinstance(hypothesis, DecodeHypothesis):
            raise ContractValidationError("hypothesis must be a DecodeHypothesis")
        decoded_at = require_non_negative_int(decoded_at_ms, "decoded_at_ms")
        if (
            self._last_decoded_at_ms is not None
            and decoded_at < self._last_decoded_at_ms
        ):
            raise ContractValidationError("decoded_at_ms must not move backwards")
        if not isinstance(final, bool):
            raise ContractValidationError("final must be a boolean")

        observed: list[_ObservedToken] = []
        for index, token in enumerate(hypothesis.tokens):
            identity = _token_identity(token)
            if (
                index < len(self._previous)
                and self._previous[index].identity == identity
            ):
                previous = self._previous[index]
                item = _ObservedToken(
                    identity=identity,
                    first_seen_ms=previous.first_seen_ms,
                    consecutive_count=previous.consecutive_count + 1,
                )
            else:
                item = _ObservedToken(
                    identity=identity,
                    first_seen_ms=decoded_at,
                    consecutive_count=1,
                )
            observed.append(item)

        self._previous = tuple(observed)
        self._last_decoded_at_ms = decoded_at

        committed_now: list[DecodedToken] = []
        remaining_tokens = hypothesis.tokens[self._committed_token_count :]
        remaining_observed = observed[self._committed_token_count :]
        for token, item in zip(remaining_tokens, remaining_observed, strict=True):
            if not final and (
                item.consecutive_count < 2
                or decoded_at - item.first_seen_ms < self._stable_age_ms
            ):
                break
            committed_now.append(token)

        self._committed_token_count += len(committed_now)
        return tuple(committed_now)


def is_transcript_content(text: str) -> bool:
    """Return whether text is content rather than a full non-speech marker."""

    if not isinstance(text, str):
        raise ContractValidationError("text must be a string")
    normalized = text.strip()
    return bool(normalized) and _NON_SPEECH_MARKER.fullmatch(normalized) is None


def choose_split_ms(hypothesis: DecodeHypothesis) -> int:
    """Choose the latest permitted punctuation split on a 20 ms boundary."""

    if not isinstance(hypothesis, DecodeHypothesis):
        raise ContractValidationError("hypothesis must be a DecodeHypothesis")
    permitted_ends = [
        token.end_ms
        for token in hypothesis.tokens
        if 10_000 <= token.end_ms <= 12_000 and token.text.endswith(_SPLIT_PUNCTUATION)
    ]
    if not permitted_ends:
        return 12_000
    return max(permitted_ends) // 20 * 20


class ChronologicalCommitBuffer:
    """Release candidates only when active source frontiers make order certain."""

    def __init__(
        self,
        segment_id_factory: Callable[[], str] = new_ulid,
        now: Callable[[], datetime] | None = None,
        *,
        initial_sequence: int = 1,
    ) -> None:
        if not callable(segment_id_factory):
            raise ContractValidationError("segment_id_factory must be callable")
        if now is not None and not callable(now):
            raise ContractValidationError("now must be callable")
        self._segment_id_factory = segment_id_factory
        self._now = _utc_now if now is None else now
        self._frontiers: dict[AudioSource, _CommitKey | None] = {
            AudioSource.ME: None,
            AudioSource.OTHERS: None,
        }
        self._frontier_floors: dict[AudioSource, int | None] = {
            AudioSource.ME: None,
            AudioSource.OTHERS: None,
        }
        self._candidates: list[tuple[_CommitKey, CommitCandidate]] = []
        self._last_candidate_start: dict[AudioSource, int | None] = {
            AudioSource.ME: None,
            AudioSource.OTHERS: None,
        }
        self._last_released_key: _CommitKey | None = None
        self._next_sequence = require_non_negative_int(
            initial_sequence,
            "initial_sequence",
        )
        if self._next_sequence == 0:
            raise ContractValidationError("initial_sequence must be positive")
        self._finalized = False

    def set_frontier(self, source: AudioSource, start_ms: int | None) -> None:
        """Set or clear one source's earliest uncommitted start key."""

        parsed_source = _require_source(source)
        if self._finalized:
            raise ContractValidationError("cannot set a frontier after finalize")
        if start_ms is None:
            self._frontiers[parsed_source] = None
            return
        parsed_start = require_non_negative_int(start_ms, "start_ms")
        floor = self._frontier_floors[parsed_source]
        if floor is not None and parsed_start < floor:
            raise ContractValidationError("frontier must not move backwards")
        key = (parsed_start, _SOURCE_RANK[parsed_source])
        if self._last_released_key is not None and key <= self._last_released_key:
            raise ContractValidationError("frontier must follow released candidates")
        self._frontier_floors[parsed_source] = parsed_start
        self._frontiers[parsed_source] = key

    def push(self, candidate: CommitCandidate) -> None:
        """Store one candidate, rejecting duplicate or retrograde source order."""

        if not isinstance(candidate, CommitCandidate):
            raise ContractValidationError("candidate must be a CommitCandidate")
        if self._finalized:
            raise ContractValidationError("cannot push after finalize")
        source = candidate.source
        start_ms = candidate.session_start_ms
        prior_start = self._last_candidate_start[source]
        if prior_start is not None and start_ms <= prior_start:
            raise ContractValidationError(
                "candidate start must advance within each source"
            )
        key = (start_ms, _SOURCE_RANK[source])
        if self._last_released_key is not None and key <= self._last_released_key:
            raise ContractValidationError("candidate must follow released candidates")
        self._last_candidate_start[source] = start_ms
        self._candidates.append((key, candidate))

    def release_ready(self) -> tuple[TranscriptRecord, ...]:
        """Release the candidates proven earlier than every active frontier."""

        active_frontiers = tuple(
            frontier for frontier in self._frontiers.values() if frontier is not None
        )
        release_before = min(active_frontiers) if active_frontiers else None
        ordered = sorted(self._candidates, key=lambda item: item[0])
        ready = [
            item
            for item in ordered
            if release_before is None or item[0] < release_before
        ]
        records: list[TranscriptRecord] = []
        for offset, (_key, item) in enumerate(ready):
            record = TranscriptRecord(
                schema_version=1,
                segment_id=self._segment_id_factory(),
                sequence=self._next_sequence + offset,
                source=item.source,
                text=item.text,
                session_start_ms=item.session_start_ms,
                session_end_ms=item.session_end_ms,
                source_start_sample=item.source_start_sample,
                source_end_sample=item.source_end_sample,
                committed_at=self._now(),
            )
            records.append(record)
        if ready:
            ready_keys = {key for key, _candidate in ready}
            self._candidates = [
                item for item in self._candidates if item[0] not in ready_keys
            ]
            self._next_sequence += len(ready)
            self._last_released_key = ready[-1][0]
        return tuple(records)

    def finalize(self) -> tuple[TranscriptRecord, ...]:
        """Clear all frontiers and release every remaining candidate exactly once."""

        if self._finalized:
            return ()
        previous_frontiers = self._frontiers
        self._frontiers = {AudioSource.ME: None, AudioSource.OTHERS: None}
        try:
            records = self.release_ready()
        except Exception:
            self._frontiers = previous_frontiers
            raise
        self._finalized = True
        return records
