"""Immutable ASR-local records shared by workers and adapters."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flowlens.domain._validation import (
    ContractValidationError,
    require_non_negative_int,
    require_str,
)
from flowlens.domain.enums import AudioSource

_SESSION_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")


def _require_positive_int(value: object, field_name: str) -> int:
    parsed = require_non_negative_int(value, field_name)
    if parsed == 0:
        raise ContractValidationError(f"{field_name} must be positive")
    return parsed


def _require_audio_source(value: object) -> AudioSource:
    if not isinstance(value, AudioSource):
        raise ContractValidationError("source must be an AudioSource")
    return value


def _require_nonblank_text(value: object, field_name: str) -> str:
    parsed = require_str(value, field_name)
    if not parsed.strip():
        raise ContractValidationError(f"{field_name} must be non-blank")
    return parsed


def _require_normalized_text(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> str:
    parsed = require_str(value, field_name)
    if parsed != parsed.strip() or (not allow_empty and not parsed):
        qualifier = "normalized or empty" if allow_empty else "non-empty and normalized"
        raise ContractValidationError(f"{field_name} must be {qualifier}")
    return parsed


def _validate_transcript_span(
    session_start_ms: object,
    session_end_ms: object,
    source_start_sample: object,
    source_end_sample: object,
) -> None:
    session_start = require_non_negative_int(session_start_ms, "session_start_ms")
    session_end = require_non_negative_int(session_end_ms, "session_end_ms")
    source_start = require_non_negative_int(
        source_start_sample,
        "source_start_sample",
    )
    source_end = require_non_negative_int(source_end_sample, "source_end_sample")
    if session_end <= session_start:
        raise ContractValidationError("session_end_ms must follow session_start_ms")
    if source_end <= source_start:
        raise ContractValidationError(
            "source_end_sample must follow source_start_sample"
        )


@dataclass(frozen=True, slots=True)
class DecodedToken:
    """One decoder token with utterance-relative millisecond bounds."""

    text: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _require_nonblank_text(self.text, "text")
        start_ms = require_non_negative_int(self.start_ms, "start_ms")
        end_ms = require_non_negative_int(self.end_ms, "end_ms")
        if end_ms < start_ms:
            raise ContractValidationError("end_ms must not precede start_ms")


@dataclass(frozen=True, slots=True)
class DecodeHypothesis:
    """One immutable ordered decoder result."""

    tokens: tuple[DecodedToken, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or not all(
            isinstance(token, DecodedToken) for token in self.tokens
        ):
            raise ContractValidationError(
                "tokens must be a tuple containing only DecodedToken values"
            )
        previous_end_ms = 0
        for token in self.tokens:
            if token.start_ms < previous_end_ms:
                raise ContractValidationError("tokens must have ordered spans")
            previous_end_ms = token.end_ms

    @property
    def text(self) -> str:
        """Join token text and normalize only the hypothesis boundary."""

        return "".join(token.text for token in self.tokens).strip()


@dataclass(frozen=True, slots=True)
class PartialTranscript:
    """Ephemeral per-source transcript text for the UI."""

    source: AudioSource
    text: str
    session_start_ms: int
    session_end_ms: int
    source_start_sample: int
    source_end_sample: int

    def __post_init__(self) -> None:
        _require_audio_source(self.source)
        _require_normalized_text(self.text, "text", allow_empty=True)
        _validate_transcript_span(
            self.session_start_ms,
            self.session_end_ms,
            self.source_start_sample,
            self.source_end_sample,
        )


@dataclass(frozen=True, slots=True)
class CommitCandidate:
    """Validated transcript candidate awaiting chronological release."""

    source: AudioSource
    text: str
    session_start_ms: int
    session_end_ms: int
    source_start_sample: int
    source_end_sample: int
    committed_at: datetime

    def __post_init__(self) -> None:
        _require_audio_source(self.source)
        _require_normalized_text(self.text, "text", allow_empty=False)
        _validate_transcript_span(
            self.session_start_ms,
            self.session_end_ms,
            self.source_start_sample,
            self.source_end_sample,
        )
        if not isinstance(self.committed_at, datetime):
            raise ContractValidationError("committed_at must be a datetime")
        if self.committed_at.utcoffset() is None:
            raise ContractValidationError("committed_at must include a timezone")
        if self.committed_at.microsecond % 1_000 != 0:
            raise ContractValidationError(
                "committed_at must have millisecond precision"
            )


@dataclass(frozen=True, slots=True)
class AsrWorkerConfig:
    """Immutable startup configuration for the ASR worker."""

    session_id: str
    model_path: Path
    partial_interval_ms: int = 500
    silence_end_ms: int = 450
    stable_age_ms: int = 1_200
    max_utterance_ms: int = 12_000
    delayed_threshold_ms: int = 2_000
    analysis_pause_threshold_ms: int = 5_000

    def __post_init__(self) -> None:
        session_id = require_str(self.session_id, "session_id")
        if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ContractValidationError(
                "session_id must contain 26 uppercase Crockford characters"
            )
        if not isinstance(self.model_path, Path) or not self.model_path.is_absolute():
            raise ContractValidationError("model_path must be an absolute local Path")
        _require_positive_int(self.partial_interval_ms, "partial_interval_ms")
        silence_end_ms = _require_positive_int(
            self.silence_end_ms,
            "silence_end_ms",
        )
        _require_positive_int(self.stable_age_ms, "stable_age_ms")
        max_utterance_ms = _require_positive_int(
            self.max_utterance_ms,
            "max_utterance_ms",
        )
        delayed_threshold_ms = _require_positive_int(
            self.delayed_threshold_ms,
            "delayed_threshold_ms",
        )
        analysis_pause_threshold_ms = _require_positive_int(
            self.analysis_pause_threshold_ms,
            "analysis_pause_threshold_ms",
        )
        if silence_end_ms > max_utterance_ms:
            raise ContractValidationError(
                "silence_end_ms must not exceed max_utterance_ms"
            )
        if analysis_pause_threshold_ms < delayed_threshold_ms:
            raise ContractValidationError(
                "analysis_pause_threshold_ms must not precede delayed_threshold_ms"
            )
