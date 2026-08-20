"""Numerical and timeline tests for stateful audio normalization."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from flowlens.audio.normalize import SoxrAudioNormalizer
from flowlens.audio.types import AudioFrame, RawAudioChunk
from flowlens.domain.enums import AudioSource


def _pcm(values: np.ndarray) -> bytes:
    """Encode sample frames as little-endian signed 16-bit PCM."""

    return values.astype("<i2", copy=False).tobytes()


def _stereo_constant(left: int, right: int, frames: int) -> bytes:
    """Return constant stereo PCM without native-endian assumptions."""

    return _pcm(
        np.column_stack(
            (
                np.full(frames, left, dtype=np.int16),
                np.full(frames, right, dtype=np.int16),
            )
        )
    )


def _chunk(
    pcm: bytes,
    *,
    source: AudioSource = AudioSource.ME,
    rate: int = 16_000,
    channels: int = 1,
    captured_ms: int = 1_000,
) -> RawAudioChunk:
    """Build one native PCM chunk."""

    return RawAudioChunk(source, pcm, rate, channels, captured_ms)


def test_normalizer_downmixes_and_emits_exact_resampled_frames() -> None:
    normalizer = SoxrAudioNormalizer(
        source=AudioSource.OTHERS,
        input_rate_hz=48_000,
        input_channels=2,
        session_started_monotonic_ms=1_000,
    )

    output = (
        normalizer.push(
            _chunk(
                _stereo_constant(2_000, 6_000, 1_920),
                source=AudioSource.OTHERS,
                rate=48_000,
                channels=2,
                captured_ms=1_100,
            )
        )
        + normalizer.flush()
    )

    assert len(output) == 2
    assert [len(frame.pcm_s16le) for frame in output] == [640, 640]
    assert [
        (frame.source_start_sample, frame.source_end_sample) for frame in output
    ] == [
        (0, 320),
        (320, 640),
    ]
    assert [frame.session_start_ms for frame in output] == [100, 120]
    assert [frame.captured_monotonic_ms for frame in output] == [1_100, 1_120]
    samples = np.frombuffer(output[1].pcm_s16le, dtype="<i2")
    assert 3_950 <= int(np.median(samples)) <= 4_050


def test_native_framer_preserves_pause_gap_after_partial_crossing_frame() -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=1_000,
    )

    first = normalizer.push(_chunk(bytes(960), captured_ms=1_020))
    second = normalizer.push(_chunk(bytes(960), captured_ms=3_020))

    assert len(first) == 1
    assert len(second) == 2
    assert [frame.source_start_sample for frame in first + second] == [0, 320, 640]
    assert [frame.captured_monotonic_ms for frame in first + second] == [
        1_020,
        1_040,
        3_030,
    ]
    assert [frame.session_start_ms for frame in first + second] == [20, 40, 2_030]


def test_streaming_resampling_is_invariant_to_input_chunk_boundaries() -> None:
    native = np.rint(
        np.sin(np.arange(4_410, dtype=np.float64) * 0.071) * 20_000
    ).astype(np.int16)

    def normalize(parts: tuple[int, ...]) -> bytes:
        normalizer = SoxrAudioNormalizer(
            AudioSource.ME,
            input_rate_hz=44_100,
            input_channels=1,
            session_started_monotonic_ms=5_000,
        )
        frames: tuple[AudioFrame, ...] = ()
        start = 0
        for end in parts:
            frames += normalizer.push(
                _chunk(
                    _pcm(native[start:end]),
                    rate=44_100,
                    captured_ms=5_000 + round(start * 1_000 / 44_100),
                )
            )
            start = end
        frames += normalizer.flush()
        return b"".join(frame.pcm_s16le for frame in frames)

    one_chunk = normalize((4_410,))
    many_chunks = normalize((1, 138, 777, 2_003, 4_409, 4_410))

    assert len(one_chunk) == 3_200
    assert many_chunks == one_chunk


def test_float32_downmix_rounds_and_avoids_int16_overflow() -> None:
    pattern = np.array(
        [
            [32_767, 32_767],
            [-32_768, -32_768],
            [32_767, -32_768],
            [1, 0],
            [3, 0],
        ],
        dtype=np.int16,
    )
    stereo = np.tile(pattern, (64, 1))
    normalizer = SoxrAudioNormalizer(
        AudioSource.OTHERS,
        input_rate_hz=16_000,
        input_channels=2,
        session_started_monotonic_ms=0,
    )

    (frame,) = normalizer.push(
        _chunk(
            _pcm(stereo),
            source=AudioSource.OTHERS,
            channels=2,
            captured_ms=0,
        )
    )

    actual = np.frombuffer(frame.pcm_s16le, dtype="<i2").reshape(-1, 5)
    expected = np.array([32_767, -32_768, 0, 0, 2], dtype=np.int16)
    assert np.all(actual == expected)


def test_flush_emits_resampler_latency_with_original_timeline() -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=48_000,
        input_channels=1,
        session_started_monotonic_ms=1_000,
    )

    live = normalizer.push(_chunk(bytes(1_920), rate=48_000, captured_ms=1_100))
    flushed = normalizer.flush()

    assert live == ()
    assert len(flushed) == 1
    assert flushed[0].source_start_sample == 0
    assert flushed[0].captured_monotonic_ms == 1_100
    assert flushed[0].session_start_ms == 100


def test_flush_discards_fragment_without_padding_and_is_terminal() -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=0,
    )

    assert normalizer.push(_chunk(bytes(638))) == ()
    assert normalizer.flush() == ()
    with pytest.raises(RuntimeError, match="flushed"):
        normalizer.flush()
    with pytest.raises(RuntimeError, match="flushed"):
        normalizer.push(_chunk(bytes(640), captured_ms=1_020))


@pytest.mark.parametrize(
    "invalid_chunk",
    [
        _chunk(bytes(640), source=AudioSource.OTHERS),
        _chunk(bytes(1_920), rate=48_000),
        _chunk(bytes(1_280), channels=2),
    ],
)
def test_mismatched_chunk_is_rejected_without_mutating_state(
    invalid_chunk: RawAudioChunk,
) -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=1_000,
    )

    with pytest.raises(ValueError, match="does not match normalizer"):
        normalizer.push(invalid_chunk)
    (frame,) = normalizer.push(_chunk(bytes(640), captured_ms=1_100))

    assert frame.source_start_sample == 0
    assert frame.captured_monotonic_ms == 1_100


def test_invalid_timestamp_is_rejected_before_buffer_mutation() -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=1_000,
    )

    assert normalizer.push(_chunk(bytes(200), captured_ms=1_100)) == ()
    with pytest.raises(ValueError, match="captured_monotonic_ms"):
        normalizer.push(_chunk(bytes(200), captured_ms=1_099))
    (frame,) = normalizer.push(_chunk(bytes(440), captured_ms=1_120))

    assert frame.source_start_sample == 0
    assert frame.captured_monotonic_ms == 1_100


def test_overlapping_chunk_timeline_is_rejected_without_timestamp_regression() -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=1_000,
    )

    first = normalizer.push(_chunk(bytes(1_280), captured_ms=1_100))
    with pytest.raises(ValueError, match="captured_monotonic_ms"):
        normalizer.push(_chunk(bytes(640), captured_ms=1_101))
    second = normalizer.push(_chunk(bytes(640), captured_ms=1_140))

    assert [frame.captured_monotonic_ms for frame in first + second] == [
        1_100,
        1_120,
        1_140,
    ]


def test_truncated_payload_is_rejected_before_buffer_mutation() -> None:
    normalizer = SoxrAudioNormalizer(
        AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=0,
    )
    malformed = _chunk(bytes(2))
    object.__setattr__(malformed, "pcm_s16le_interleaved", bytes(1))

    with pytest.raises(ValueError, match="complete int16 channel frames"):
        normalizer.push(malformed)
    (frame,) = normalizer.push(_chunk(bytes(640)))

    assert frame.source_start_sample == 0


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SoxrAudioNormalizer(cast(AudioSource, "ME"), 16_000, 1, 0), "source"),
        (lambda: SoxrAudioNormalizer(AudioSource.ME, 0, 1, 0), "input_rate_hz"),
        (lambda: SoxrAudioNormalizer(AudioSource.ME, 16_000, 0, 0), "input_channels"),
        (
            lambda: SoxrAudioNormalizer(AudioSource.ME, 16_000, 1, -1),
            "session_started_monotonic_ms",
        ),
    ],
)
def test_constructor_rejects_invalid_configuration(
    factory: Callable[[], SoxrAudioNormalizer],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
