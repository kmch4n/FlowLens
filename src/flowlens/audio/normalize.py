"""Stateful native PCM normalization into canonical audio frames."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import soxr  # type: ignore[import-untyped]

from flowlens.audio.types import (
    CANONICAL_RATE_HZ,
    FRAME_SAMPLES,
    AudioFrame,
    RawAudioChunk,
)
from flowlens.domain.enums import AudioSource

Float32Array = npt.NDArray[np.float32]


@dataclass(slots=True)
class _TimelineSpan:
    """Canonical sample extent associated with one capture timestamp."""

    first_sample_monotonic_ms: int
    sample_count: int
    consumed_samples: int = 0


class SoxrAudioNormalizer:
    """Convert one native-format source into contiguous canonical frames."""

    def __init__(
        self,
        source: AudioSource,
        input_rate_hz: int,
        input_channels: int,
        session_started_monotonic_ms: int,
    ) -> None:
        if not isinstance(source, AudioSource):
            raise ValueError("source must be an AudioSource")
        if (
            not isinstance(input_rate_hz, int)
            or isinstance(input_rate_hz, bool)
            or input_rate_hz <= 0
        ):
            raise ValueError("input_rate_hz must be a positive integer")
        if (
            not isinstance(input_channels, int)
            or isinstance(input_channels, bool)
            or input_channels <= 0
        ):
            raise ValueError("input_channels must be a positive integer")
        if (
            not isinstance(session_started_monotonic_ms, int)
            or isinstance(session_started_monotonic_ms, bool)
            or session_started_monotonic_ms < 0
        ):
            raise ValueError(
                "session_started_monotonic_ms must be a nonnegative integer"
            )

        self._source = source
        self._input_rate_hz = input_rate_hz
        self._input_channels = input_channels
        self._session_started_monotonic_ms = session_started_monotonic_ms
        self._resampler = (
            None
            if input_rate_hz == CANONICAL_RATE_HZ
            else soxr.ResampleStream(
                input_rate_hz,
                CANONICAL_RATE_HZ,
                1,
                dtype="float32",
                quality="HQ",
            )
        )
        self._sample_fifo = np.empty(0, dtype=np.float32)
        self._timeline_spans: deque[_TimelineSpan] = deque()
        self._total_input_samples = 0
        self._accounted_output_samples = 0
        self._next_source_sample = 0
        self._last_captured_monotonic_ms: int | None = None
        self._next_input_monotonic_ms = session_started_monotonic_ms
        self._flushed = False

    def push(self, chunk: RawAudioChunk) -> tuple[AudioFrame, ...]:
        """Normalize one native chunk and emit all newly complete frames."""

        self._require_open()
        self._validate_chunk(chunk)
        mono = self._decode_mono_float32(chunk)
        output = self._append_resampled(
            mono,
            captured_monotonic_ms=chunk.captured_monotonic_ms,
            last=False,
        )
        self._last_captured_monotonic_ms = chunk.captured_monotonic_ms
        input_samples = len(chunk.pcm_s16le_interleaved) // (2 * self._input_channels)
        self._next_input_monotonic_ms = (
            chunk.captured_monotonic_ms + input_samples * 1_000 // self._input_rate_hz
        )
        return output

    def flush(self) -> tuple[AudioFrame, ...]:
        """Flush SoXR once, emit complete frames, and discard any fragment."""

        self._require_open()
        captured_monotonic_ms = (
            self._last_captured_monotonic_ms
            if self._last_captured_monotonic_ms is not None
            else self._session_started_monotonic_ms
        )
        output = self._append_resampled(
            np.empty(0, dtype=np.float32),
            captured_monotonic_ms=captured_monotonic_ms,
            last=True,
        )
        self._sample_fifo = np.empty(0, dtype=np.float32)
        self._timeline_spans.clear()
        self._flushed = True
        return output

    def _decode_mono_float32(self, chunk: RawAudioChunk) -> Float32Array:
        """Decode little-endian interleaved int16 PCM and downmix in float32."""

        decoded = np.frombuffer(chunk.pcm_s16le_interleaved, dtype="<i2")
        channels = decoded.reshape(-1, self._input_channels)
        normalized = channels.astype(np.float32) / np.float32(32_768.0)
        return np.mean(normalized, axis=1, dtype=np.float32)

    def _append_resampled(
        self,
        mono: Float32Array,
        captured_monotonic_ms: int,
        last: bool,
    ) -> tuple[AudioFrame, ...]:
        """Append converted samples and consume aligned audio/timeline FIFOs."""

        input_samples = int(mono.size)
        self._total_input_samples += input_samples
        target_output_samples = self._rounded_output_samples(self._total_input_samples)
        span_samples = target_output_samples - self._accounted_output_samples
        self._accounted_output_samples = target_output_samples
        if span_samples:
            self._timeline_spans.append(
                _TimelineSpan(captured_monotonic_ms, span_samples)
            )

        if self._resampler is None:
            resampled = mono
        else:
            resampled = self._resampler.resample_chunk(mono, last=last)
        if resampled.size:
            self._sample_fifo = np.concatenate((self._sample_fifo, resampled))

        frames: list[AudioFrame] = []
        while self._sample_fifo.size >= FRAME_SAMPLES:
            frame_samples = self._sample_fifo[:FRAME_SAMPLES]
            self._sample_fifo = self._sample_fifo[FRAME_SAMPLES:]
            first_input_monotonic_ms = self._consume_timeline(FRAME_SAMPLES)
            clipped = np.clip(
                frame_samples,
                np.float32(-1.0),
                np.float32(32_767.0 / 32_768.0),
            )
            pcm = np.rint(clipped * np.float32(32_768.0)).astype("<i2", copy=False)
            frames.append(
                AudioFrame(
                    source=self._source,
                    pcm_s16le=pcm.tobytes(),
                    source_start_sample=self._next_source_sample,
                    source_end_sample=self._next_source_sample + FRAME_SAMPLES,
                    session_start_ms=(
                        first_input_monotonic_ms - self._session_started_monotonic_ms
                    ),
                    captured_monotonic_ms=first_input_monotonic_ms,
                )
            )
            self._next_source_sample += FRAME_SAMPLES
        return tuple(frames)

    def _require_open(self) -> None:
        if self._flushed:
            raise RuntimeError("normalizer has already been flushed")

    def _validate_chunk(self, chunk: RawAudioChunk) -> None:
        if not isinstance(chunk, RawAudioChunk):
            raise ValueError("chunk must be a RawAudioChunk")
        if (
            chunk.source is not self._source
            or chunk.sample_rate_hz != self._input_rate_hz
            or chunk.channels != self._input_channels
        ):
            raise ValueError("chunk format does not match normalizer")
        frame_bytes = 2 * self._input_channels
        if len(chunk.pcm_s16le_interleaved) % frame_bytes:
            raise ValueError(
                "pcm_s16le_interleaved must contain complete int16 channel frames"
            )
        if chunk.captured_monotonic_ms < self._session_started_monotonic_ms:
            raise ValueError("captured_monotonic_ms must not precede the session start")
        if (
            self._last_captured_monotonic_ms is not None
            and chunk.captured_monotonic_ms < self._next_input_monotonic_ms
        ):
            raise ValueError(
                "captured_monotonic_ms must not overlap the previous chunk"
            )

    def _rounded_output_samples(self, input_samples: int) -> int:
        numerator = 2 * input_samples * CANONICAL_RATE_HZ
        return (numerator + self._input_rate_hz) // (2 * self._input_rate_hz)

    def _consume_timeline(self, sample_count: int) -> int:
        if not self._timeline_spans:
            raise RuntimeError("audio samples have no input timeline")
        first_span = self._timeline_spans[0]
        first_input_monotonic_ms = (
            first_span.first_sample_monotonic_ms
            + first_span.consumed_samples * 1_000 // CANONICAL_RATE_HZ
        )
        remaining = sample_count
        while remaining:
            if not self._timeline_spans:
                raise RuntimeError("audio samples exceed the input timeline")
            span = self._timeline_spans[0]
            available = span.sample_count - span.consumed_samples
            consumed = min(remaining, available)
            span.consumed_samples += consumed
            remaining -= consumed
            if span.consumed_samples == span.sample_count:
                self._timeline_spans.popleft()
        return first_input_monotonic_ms
