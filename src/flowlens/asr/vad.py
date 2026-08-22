"""WebRTC speech detection and deterministic utterance boundaries."""

from collections.abc import Callable
from typing import Literal, Protocol, cast

import webrtcvad  # type: ignore[import-untyped]

from flowlens.audio.types import (
    CANONICAL_RATE_HZ,
    FRAME_BYTES,
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    AudioFrame,
)
from flowlens.domain._validation import (
    ContractValidationError,
    require_non_negative_int,
)
from flowlens.domain.enums import AudioSource

BoundaryState = Literal["INACTIVE", "ACTIVE", "END", "HARD_SPLIT"]


class _VadPort(Protocol):
    def is_speech(self, pcm_s16le: bytes, sample_rate_hz: int) -> bool: ...


VadFactory = Callable[[int], _VadPort]


def _default_vad_factory(mode: int) -> _VadPort:
    return cast(_VadPort, webrtcvad.Vad(mode))


def _require_positive_int(value: object, field_name: str) -> int:
    parsed = require_non_negative_int(value, field_name)
    if parsed == 0:
        raise ContractValidationError(f"{field_name} must be positive")
    return parsed


class WebRtcSpeechDetector:
    """Apply the fixed WebRTC VAD contract to canonical audio frames."""

    def __init__(
        self,
        mode: int = 2,
        vad_factory: VadFactory = _default_vad_factory,
    ) -> None:
        if not isinstance(mode, int) or isinstance(mode, bool) or mode != 2:
            raise ContractValidationError("mode must be 2")
        self._vad = vad_factory(mode)

    def is_speech(self, frame: AudioFrame) -> bool:
        """Return whether one canonical mono 16 kHz frame contains speech."""

        if not isinstance(frame, AudioFrame):
            raise ContractValidationError("frame must be an AudioFrame")
        if (
            not isinstance(frame.source, AudioSource)
            or not isinstance(frame.pcm_s16le, bytes)
            or len(frame.pcm_s16le) != FRAME_BYTES
            or not isinstance(frame.source_start_sample, int)
            or isinstance(frame.source_start_sample, bool)
            or not isinstance(frame.source_end_sample, int)
            or isinstance(frame.source_end_sample, bool)
            or frame.source_end_sample - frame.source_start_sample != FRAME_SAMPLES
        ):
            raise ContractValidationError("frame must be a canonical 640-byte frame")
        return self._vad.is_speech(frame.pcm_s16le, CANONICAL_RATE_HZ)


class UtteranceBoundaryTracker:
    """Track exact frame-count silence and hard-split transitions."""

    def __init__(self, silence_end_ms: int, max_utterance_ms: int) -> None:
        parsed_silence_end_ms = _require_positive_int(
            silence_end_ms,
            "silence_end_ms",
        )
        parsed_max_utterance_ms = _require_positive_int(
            max_utterance_ms,
            "max_utterance_ms",
        )
        if parsed_silence_end_ms > parsed_max_utterance_ms:
            raise ContractValidationError(
                "silence_end_ms must not exceed max_utterance_ms"
            )
        self._silence_end_frames = (
            parsed_silence_end_ms + FRAME_DURATION_MS - 1
        ) // FRAME_DURATION_MS
        self._max_utterance_frames = (
            parsed_max_utterance_ms + FRAME_DURATION_MS - 1
        ) // FRAME_DURATION_MS
        self._active_frames = 0
        self._consecutive_silent_frames = 0

    @property
    def active_duration_ms(self) -> int:
        """Return the current utterance duration in exact frame units."""

        return self._active_frames * FRAME_DURATION_MS

    def observe(self, is_speech: bool) -> BoundaryState:
        """Observe one 20 ms classification and return its boundary state."""

        if not isinstance(is_speech, bool):
            raise ContractValidationError("is_speech must be a boolean")
        if self._active_frames == 0 and not is_speech:
            return "INACTIVE"

        self._active_frames += 1
        if is_speech:
            self._consecutive_silent_frames = 0
        else:
            self._consecutive_silent_frames += 1

        if self._consecutive_silent_frames >= self._silence_end_frames:
            self._reset()
            return "END"
        if self._active_frames >= self._max_utterance_frames:
            self._reset()
            return "HARD_SPLIT"
        return "ACTIVE"

    def _reset(self) -> None:
        self._active_frames = 0
        self._consecutive_silent_frames = 0
