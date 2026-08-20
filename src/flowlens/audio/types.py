"""Capture-local immutable audio types."""

import re
from dataclasses import dataclass

from flowlens.domain._validation import (
    ContractValidationError,
    require_non_negative_int,
    require_str,
)
from flowlens.domain.enums import AudioSource

CANONICAL_RATE_HZ = 16_000
FRAME_SAMPLES = 320
FRAME_DURATION_MS = 20
FRAME_BYTES = 640

_SESSION_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")


def _require_audio_source(value: object, field_name: str) -> AudioSource:
    if not isinstance(value, AudioSource):
        raise ContractValidationError(f"{field_name} must be an AudioSource")
    return value


def _require_non_empty_str(value: object, field_name: str) -> str:
    parsed = require_str(value, field_name)
    if not parsed.strip():
        raise ContractValidationError(f"{field_name} must be non-empty")
    return parsed


def _require_positive_int(value: object, field_name: str) -> int:
    parsed = require_non_negative_int(value, field_name)
    if parsed == 0:
        raise ContractValidationError(f"{field_name} must be positive")
    return parsed


def _require_bytes(value: object, field_name: str) -> bytes:
    if not isinstance(value, bytes):
        raise ContractValidationError(f"{field_name} must be bytes")
    return value


@dataclass(frozen=True, slots=True)
class CaptureDevice:
    """One capture device exposed by the selected backend."""

    device_id: str
    display_name: str
    input_device_index: int
    sample_rate_hz: int
    channels: int
    is_loopback: bool

    def __post_init__(self) -> None:
        _require_non_empty_str(self.device_id, "device_id")
        _require_non_empty_str(self.display_name, "display_name")
        require_non_negative_int(self.input_device_index, "input_device_index")
        _require_positive_int(self.sample_rate_hz, "sample_rate_hz")
        _require_positive_int(self.channels, "channels")
        if not isinstance(self.is_loopback, bool):
            raise ContractValidationError("is_loopback must be a boolean")


@dataclass(frozen=True, slots=True)
class RawAudioChunk:
    """One immutable native-format chunk produced by a capture callback."""

    source: AudioSource
    pcm_s16le_interleaved: bytes
    sample_rate_hz: int
    channels: int
    captured_monotonic_ms: int

    def __post_init__(self) -> None:
        _require_audio_source(self.source, "source")
        payload = _require_bytes(
            self.pcm_s16le_interleaved,
            "pcm_s16le_interleaved",
        )
        _require_positive_int(self.sample_rate_hz, "sample_rate_hz")
        channels = _require_positive_int(self.channels, "channels")
        require_non_negative_int(
            self.captured_monotonic_ms,
            "captured_monotonic_ms",
        )
        if len(payload) % (2 * channels) != 0:
            raise ContractValidationError(
                "pcm_s16le_interleaved must contain complete int16 channel frames"
            )


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One canonical mono 16 kHz, 20 ms audio frame."""

    source: AudioSource
    pcm_s16le: bytes
    source_start_sample: int
    source_end_sample: int
    session_start_ms: int
    captured_monotonic_ms: int

    def __post_init__(self) -> None:
        _require_audio_source(self.source, "source")
        payload = _require_bytes(self.pcm_s16le, "pcm_s16le")
        source_start_sample = require_non_negative_int(
            self.source_start_sample,
            "source_start_sample",
        )
        source_end_sample = require_non_negative_int(
            self.source_end_sample,
            "source_end_sample",
        )
        require_non_negative_int(self.session_start_ms, "session_start_ms")
        require_non_negative_int(
            self.captured_monotonic_ms,
            "captured_monotonic_ms",
        )
        if len(payload) != FRAME_BYTES or (
            source_end_sample - source_start_sample != FRAME_SAMPLES
        ):
            raise ContractValidationError(
                "AudioFrame must contain 320 mono int16 samples"
            )

    @property
    def duration_ms(self) -> int:
        """Return the fixed canonical frame duration."""

        return FRAME_DURATION_MS

    def queue_age_ms(self, now_monotonic_ms: int) -> int:
        """Return a deterministic nonnegative queue age."""

        now = require_non_negative_int(now_monotonic_ms, "now_monotonic_ms")
        return max(0, now - self.captured_monotonic_ms)


@dataclass(frozen=True, slots=True)
class AudioWorkerConfig:
    """Immutable startup configuration for the Audio Worker."""

    session_id: str
    microphone_device_id: str
    loopback_output_device_id: str
    session_started_monotonic_ms: int
    writer_queue_max_frames: int
    asr_queue_max_frames: int
    asr_spool_max_frames: int = 3_000
    reconnect_interval_ms: int = 2_000

    def __post_init__(self) -> None:
        session_id = _require_non_empty_str(self.session_id, "session_id")
        if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ContractValidationError(
                "session_id must contain 26 uppercase Crockford characters"
            )
        _require_non_empty_str(
            self.microphone_device_id,
            "microphone_device_id",
        )
        _require_non_empty_str(
            self.loopback_output_device_id,
            "loopback_output_device_id",
        )
        require_non_negative_int(
            self.session_started_monotonic_ms,
            "session_started_monotonic_ms",
        )
        _require_positive_int(
            self.writer_queue_max_frames,
            "writer_queue_max_frames",
        )
        _require_positive_int(self.asr_queue_max_frames, "asr_queue_max_frames")
        _require_positive_int(self.asr_spool_max_frames, "asr_spool_max_frames")
        _require_positive_int(self.reconnect_interval_ms, "reconnect_interval_ms")
