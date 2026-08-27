"""Strict sender-local validation and controller envelope rewrapping."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from flowlens.asr.types import PartialTranscript
from flowlens.discussion.contracts import (
    DiscussionStatusPayload,
    DiscussionStoppedPayload,
)
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    DiscussionStateReplaced,
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    WriterAck,
    WriterFatal,
    WriterForceCloseResult,
)


class PayloadValidationError(ValueError):
    """Raised when a worker payload does not match its closed contract."""


@dataclass(frozen=True, slots=True)
class SequenceAcceptance:
    """Transaction-safe result for one sender-local sequence."""

    accepted: bool
    duplicate: bool
    gap: tuple[int, int] | None


class SequenceTracker:
    """Track sequences for exactly one active session and every sender."""

    def __init__(self, session_id: str) -> None:
        if type(session_id) is not str or not session_id:
            raise ValueError("session_id must be a non-empty exact string")
        self._session_id = session_id
        self._expected: dict[ProcessSource, int] = {}

    def accept(self, envelope: MessageEnvelope[object]) -> SequenceAcceptance:
        """Validate and classify without mutating on rejected input."""

        if not isinstance(envelope, MessageEnvelope):
            raise TypeError("envelope must be a MessageEnvelope")
        if type(envelope.schema_version) is not int:
            raise ValueError("schema_version must be an exact integer")
        envelope.validate_schema()
        if type(envelope.session_id) is not str:
            raise ValueError("session_id must be an exact string")
        if envelope.session_id != self._session_id:
            raise ValueError("message does not target the active session")
        if type(envelope.source) is not ProcessSource:
            raise ValueError("source must be a ProcessSource")
        if type(envelope.sequence) is not int or envelope.sequence <= 0:
            raise ValueError("sequence must be a positive exact integer")
        expected = self._expected.get(envelope.source, 1)
        if envelope.sequence < expected:
            return SequenceAcceptance(False, True, None)
        gap = (
            None if envelope.sequence == expected else (expected, envelope.sequence - 1)
        )
        self._expected[envelope.source] = envelope.sequence + 1
        return SequenceAcceptance(True, False, gap)

    def expected(self, source: ProcessSource) -> int:
        """Return one sender's next expected sequence."""

        if not isinstance(source, ProcessSource):
            raise ValueError("source must be a ProcessSource")
        return self._expected.get(source, 1)

    def reset(self, source: ProcessSource) -> None:
        """Begin a fresh sequence generation for exactly one restarted sender."""

        if not isinstance(source, ProcessSource):
            raise ValueError("source must be a ProcessSource")
        self._expected.pop(source, None)


def validate_worker_payload(envelope: MessageEnvelope[object]) -> object:
    """Return one defensively validated and normalized worker payload."""

    if not isinstance(envelope, MessageEnvelope):
        raise TypeError("envelope must be a MessageEnvelope")
    envelope.validate_schema()
    validators = {
        ProcessSource.AUDIO: _validate_audio_payload,
        ProcessSource.ASR: _validate_asr_payload,
        ProcessSource.DISCUSSION: _validate_discussion_payload,
        ProcessSource.WRITER: _validate_writer_payload,
    }
    validator = validators.get(envelope.source)
    if validator is None:
        raise PayloadValidationError("worker source is unsupported")
    try:
        return validator(envelope.message_type, envelope.payload)
    except PayloadValidationError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError):
        raise PayloadValidationError("worker payload is invalid") from None


def rewrap_for_gui(
    envelope: MessageEnvelope[object],
    *,
    sequence: int,
    payload: object | None = None,
) -> MessageEnvelope[object]:
    """Create one GUI-local envelope without forwarding worker metadata."""

    if type(sequence) is not int or sequence <= 0:
        raise ValueError("sequence must be a positive exact integer")
    return MessageEnvelope(
        schema_version=1,
        session_id=envelope.session_id,
        message_type=envelope.message_type,
        sequence=sequence,
        source=ProcessSource.GUI,
        created_monotonic_ms=envelope.created_monotonic_ms,
        payload=envelope.payload if payload is None else payload,
    )


def _mapping(value: object, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != keys:
        raise PayloadValidationError("worker payload shape is invalid")
    if not all(type(key) is str for key in value):
        raise PayloadValidationError("worker payload keys are invalid")
    return cast(Mapping[str, object], value)


def _exact_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise PayloadValidationError(f"{field_name} is invalid")
    return value


def _exact_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise PayloadValidationError(f"{field_name} is invalid")
    return value


def _exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise PayloadValidationError(f"{field_name} is invalid")
    return value


def _worker_dict(
    value: object,
    *,
    worker: str,
    extra: frozenset[str] = frozenset(),
) -> dict[str, object]:
    mapping = _mapping(value, frozenset({"worker"}) | extra)
    if mapping["worker"] != worker or type(mapping["worker"]) is not str:
        raise PayloadValidationError("worker is invalid")
    return dict(mapping)


def _validate_audio_payload(message_type: MessageType, payload: object) -> object:
    if message_type is MessageType.WORKER_READY:
        return _worker_dict(payload, worker="AUDIO")
    if message_type is MessageType.AUDIO_LEVEL:
        mapping = _mapping(payload, frozenset({"source", "peak_dbfs"}))
        _audio_source(mapping["source"])
        peak = mapping["peak_dbfs"]
        if type(peak) is not float or not math.isfinite(peak):
            raise PayloadValidationError("peak_dbfs is invalid")
        return dict(mapping)
    if message_type in {
        MessageType.SOURCE_DISCONNECTED,
        MessageType.SOURCE_RECONNECTED,
    }:
        mapping = _mapping(payload, frozenset({"source", "device_id"}))
        _audio_source(mapping["source"])
        _exact_string(mapping["device_id"], "device_id")
        return dict(mapping)
    if message_type is MessageType.WORKER_STOPPED:
        mapping = _worker_dict(
            payload,
            worker="AUDIO",
            extra=frozenset({"drained", "writer_frames", "asr_frames"}),
        )
        if _exact_bool(mapping["drained"], "drained") is not True:
            raise PayloadValidationError("drained is invalid")
        _exact_int(mapping["writer_frames"], "writer_frames")
        _exact_int(mapping["asr_frames"], "asr_frames")
        return mapping
    if message_type is MessageType.WORKER_ERROR:
        return _error_payload(payload, "AUDIO")
    raise PayloadValidationError("audio message type is unsupported")


def _validate_asr_payload(message_type: MessageType, payload: object) -> object:
    if message_type is MessageType.WORKER_READY:
        return _worker_dict(payload, worker="ASR")
    if message_type is MessageType.TRANSCRIPT_COMMITTED:
        if isinstance(payload, TranscriptCommitted):
            return payload
        return TranscriptCommitted(TranscriptRecord.from_dict(payload))
    if message_type is MessageType.TRANSCRIPT_PARTIAL:
        mapping = _mapping(
            payload,
            frozenset(
                {
                    "source",
                    "text",
                    "session_start_ms",
                    "session_end_ms",
                    "source_start_sample",
                    "source_end_sample",
                }
            ),
        )
        return PartialTranscript(
            source=_audio_source(mapping["source"]),
            text=_exact_string_allow_empty(mapping["text"], "text"),
            session_start_ms=_exact_int(
                mapping["session_start_ms"], "session_start_ms"
            ),
            session_end_ms=_exact_int(mapping["session_end_ms"], "session_end_ms"),
            source_start_sample=_exact_int(
                mapping["source_start_sample"], "source_start_sample"
            ),
            source_end_sample=_exact_int(
                mapping["source_end_sample"], "source_end_sample"
            ),
        )
    if message_type is MessageType.ASR_STATUS:
        mapping = _mapping(
            payload,
            frozenset({"state", "backlog_ms", "maximum_backlog_ms", "analysis_paused"}),
        )
        state = _exact_string(mapping["state"], "state")
        if state not in {"READY", "RUNNING", "DELAYED", "STOPPED"}:
            raise PayloadValidationError("state is invalid")
        backlog = _exact_int(mapping["backlog_ms"], "backlog_ms")
        maximum = _exact_int(mapping["maximum_backlog_ms"], "maximum_backlog_ms")
        if maximum < backlog:
            raise PayloadValidationError("maximum_backlog_ms is invalid")
        _exact_bool(mapping["analysis_paused"], "analysis_paused")
        return dict(mapping)
    if message_type is MessageType.WORKER_STOPPED:
        mapping = _worker_dict(
            payload,
            worker="ASR",
            extra=frozenset({"drained", "committed_count"}),
        )
        if _exact_bool(mapping["drained"], "drained") is not True:
            raise PayloadValidationError("drained is invalid")
        _exact_int(mapping["committed_count"], "committed_count")
        return mapping
    if message_type is MessageType.WORKER_ERROR:
        return _error_payload(payload, "ASR")
    raise PayloadValidationError("ASR message type is unsupported")


def _validate_discussion_payload(
    message_type: MessageType,
    payload: object,
) -> object:
    if message_type is MessageType.WORKER_READY:
        return _worker_dict(payload, worker="DISCUSSION")
    if message_type is MessageType.DISCUSSION_STATE_REPLACED and isinstance(
        payload, DiscussionStateReplaced
    ):
        return payload
    if message_type is MessageType.DISCUSSION_STATUS and isinstance(
        payload, DiscussionStatusPayload
    ):
        return payload
    if message_type is MessageType.WORKER_STOPPED and isinstance(
        payload, DiscussionStoppedPayload
    ):
        return payload
    if message_type is MessageType.WORKER_ERROR:
        return _error_payload(payload, "DISCUSSION", allow_detail=False)
    raise PayloadValidationError("discussion payload is invalid")


def _validate_writer_payload(message_type: MessageType, payload: object) -> object:
    if message_type is MessageType.WRITER_ACK and isinstance(payload, WriterAck):
        return payload
    if message_type is MessageType.WRITER_FATAL and isinstance(payload, WriterFatal):
        return payload
    if (
        message_type is MessageType.WRITER_FORCE_CLOSE_RESULT
        and type(payload) is WriterForceCloseResult
    ):
        return payload
    raise PayloadValidationError("Writer payload is invalid")


def _error_payload(
    payload: object,
    worker: str,
    *,
    allow_detail: bool = True,
) -> dict[str, object]:
    extra = frozenset({"code", "detail"} if allow_detail else {"code"})
    mapping = _worker_dict(payload, worker=worker, extra=extra)
    _exact_string(mapping["code"], "code")
    if allow_detail:
        _exact_string(mapping["detail"], "detail")
    return mapping


def _audio_source(value: object) -> AudioSource:
    if type(value) is not str:
        raise PayloadValidationError("source is invalid")
    try:
        return AudioSource(value)
    except ValueError:
        raise PayloadValidationError("source is invalid") from None


def _exact_string_allow_empty(value: object, field_name: str) -> str:
    if type(value) is not str or value != value.strip():
        raise PayloadValidationError(f"{field_name} is invalid")
    return value
