"""Contract tests for capture-local audio types."""

import pickle
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from typing import cast

import pytest

from flowlens.audio.ports import (
    CaptureBackendPort,
    CaptureCallback,
    CaptureStreamPort,
    StreamingNormalizerPort,
)
from flowlens.audio.types import (
    CANONICAL_RATE_HZ,
    FRAME_BYTES,
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    AudioFrame,
    AudioWorkerConfig,
    CaptureDevice,
    RawAudioChunk,
)
from flowlens.domain.enums import AudioSource


def make_frame() -> AudioFrame:
    """Build one valid canonical frame."""

    return AudioFrame(
        source=AudioSource.ME,
        pcm_s16le=bytes(640),
        source_start_sample=320,
        source_end_sample=640,
        session_start_ms=20,
        captured_monotonic_ms=1_020,
    )


def test_canonical_audio_constants_match_twenty_milliseconds() -> None:
    assert CANONICAL_RATE_HZ == 16_000
    assert FRAME_SAMPLES == 320
    assert FRAME_DURATION_MS == 20
    assert FRAME_BYTES == 640


def test_audio_frame_requires_exactly_twenty_milliseconds() -> None:
    frame = make_frame()

    assert frame.duration_ms == 20
    assert frame.queue_age_ms(now_monotonic_ms=1_075) == 55
    assert frame.queue_age_ms(now_monotonic_ms=1_000) == 0


@pytest.mark.parametrize(
    ("pcm_s16le", "source_start_sample", "source_end_sample"),
    [
        (bytes(638), 0, 319),
        (bytes(642), 0, 321),
        (bytes(640), 0, 319),
        (bytes(640), 0, 321),
    ],
)
def test_audio_frame_rejects_noncanonical_payload(
    pcm_s16le: bytes,
    source_start_sample: int,
    source_end_sample: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="AudioFrame must contain 320 mono int16 samples",
    ):
        AudioFrame(
            source=AudioSource.OTHERS,
            pcm_s16le=pcm_s16le,
            source_start_sample=source_start_sample,
            source_end_sample=source_end_sample,
            session_start_ms=0,
            captured_monotonic_ms=1_000,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", "ME"),
        ("pcm_s16le", bytearray(640)),
        ("source_start_sample", -1),
        ("source_end_sample", -1),
        ("session_start_ms", -1),
        ("captured_monotonic_ms", -1),
        ("source_start_sample", True),
        ("source_end_sample", True),
        ("session_start_ms", True),
        ("captured_monotonic_ms", True),
    ],
)
def test_audio_frame_rejects_invalid_field_values(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "source": AudioSource.ME,
        "pcm_s16le": bytes(640),
        "source_start_sample": 0,
        "source_end_sample": 320,
        "session_start_ms": 0,
        "captured_monotonic_ms": 1_000,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        AudioFrame(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("now_monotonic_ms", [-1, True, 1.5, "1075"])
def test_queue_age_rejects_invalid_monotonic_time(
    now_monotonic_ms: object,
) -> None:
    with pytest.raises(ValueError, match="now_monotonic_ms"):
        make_frame().queue_age_ms(cast(int, now_monotonic_ms))


def test_capture_device_preserves_valid_values() -> None:
    device = CaptureDevice(
        device_id="wasapi-output:7",
        display_name="Speakers",
        input_device_index=11,
        sample_rate_hz=48_000,
        channels=2,
        is_loopback=True,
    )

    assert pickle.loads(pickle.dumps(device)) == device


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("device_id", ""),
        ("device_id", 7),
        ("display_name", ""),
        ("display_name", None),
        ("input_device_index", -1),
        ("input_device_index", True),
        ("sample_rate_hz", 0),
        ("sample_rate_hz", True),
        ("channels", 0),
        ("channels", True),
        ("is_loopback", 1),
    ],
)
def test_capture_device_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "device_id": "input:3",
        "display_name": "USB Mic",
        "input_device_index": 3,
        "sample_rate_hz": 48_000,
        "channels": 1,
        "is_loopback": False,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        CaptureDevice(**values)  # type: ignore[arg-type]


def test_raw_audio_chunk_preserves_immutable_complete_samples() -> None:
    chunk = RawAudioChunk(
        source=AudioSource.OTHERS,
        pcm_s16le_interleaved=bytes(1_280),
        sample_rate_hz=48_000,
        channels=2,
        captured_monotonic_ms=1_000,
    )

    assert isinstance(chunk.pcm_s16le_interleaved, bytes)
    assert pickle.loads(pickle.dumps(chunk)) == chunk


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", "OTHERS"),
        ("pcm_s16le_interleaved", bytearray(4)),
        ("sample_rate_hz", 0),
        ("sample_rate_hz", True),
        ("channels", 0),
        ("channels", True),
        ("captured_monotonic_ms", -1),
        ("captured_monotonic_ms", True),
    ],
)
def test_raw_audio_chunk_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "source": AudioSource.ME,
        "pcm_s16le_interleaved": bytes(4),
        "sample_rate_hz": 48_000,
        "channels": 2,
        "captured_monotonic_ms": 1_000,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        RawAudioChunk(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [bytes(1), bytes(2), bytes(6)])
def test_raw_audio_chunk_requires_complete_interleaved_frames(payload: bytes) -> None:
    with pytest.raises(ValueError, match="pcm_s16le_interleaved"):
        RawAudioChunk(AudioSource.OTHERS, payload, 48_000, 2, 1_000)


def test_audio_worker_config_preserves_defaults() -> None:
    config = AudioWorkerConfig(
        session_id="01J00000000000000000000000",
        microphone_device_id="input:3",
        loopback_output_device_id="wasapi-output:7",
        session_started_monotonic_ms=1_000,
        writer_queue_max_frames=100,
        asr_queue_max_frames=200,
    )

    assert config.asr_spool_max_frames == 3_000
    assert config.reconnect_interval_ms == 2_000
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("session_id", ""),
        ("session_id", 1),
        ("microphone_device_id", ""),
        ("loopback_output_device_id", ""),
        ("session_started_monotonic_ms", -1),
        ("session_started_monotonic_ms", True),
        ("writer_queue_max_frames", 0),
        ("writer_queue_max_frames", True),
        ("asr_queue_max_frames", 0),
        ("asr_spool_max_frames", 0),
        ("reconnect_interval_ms", 0),
    ],
)
def test_audio_worker_config_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "session_id": "01J00000000000000000000000",
        "microphone_device_id": "input:3",
        "loopback_output_device_id": "wasapi-output:7",
        "session_started_monotonic_ms": 1_000,
        "writer_queue_max_frames": 100,
        "asr_queue_max_frames": 200,
        "asr_spool_max_frames": 3_000,
        "reconnect_interval_ms": 2_000,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        AudioWorkerConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [make_frame(), RawAudioChunk(AudioSource.ME, bytes(2), 16_000, 1, 0)],
)
def test_capture_records_are_frozen_and_slotted(value: object) -> None:
    with pytest.raises((FrozenInstanceError, TypeError)):
        value.extra = True  # type: ignore[attr-defined]


def test_capture_records_have_exact_field_contracts() -> None:
    assert [field.name for field in fields(CaptureDevice)] == [
        "device_id",
        "display_name",
        "input_device_index",
        "sample_rate_hz",
        "channels",
        "is_loopback",
    ]
    assert [field.name for field in fields(RawAudioChunk)] == [
        "source",
        "pcm_s16le_interleaved",
        "sample_rate_hz",
        "channels",
        "captured_monotonic_ms",
    ]
    assert [field.name for field in fields(AudioFrame)] == [
        "source",
        "pcm_s16le",
        "source_start_sample",
        "source_end_sample",
        "session_start_ms",
        "captured_monotonic_ms",
    ]
    assert [field.name for field in fields(AudioWorkerConfig)] == [
        "session_id",
        "microphone_device_id",
        "loopback_output_device_id",
        "session_started_monotonic_ms",
        "writer_queue_max_frames",
        "asr_queue_max_frames",
        "asr_spool_max_frames",
        "reconnect_interval_ms",
    ]


def test_capture_ports_are_importable_with_exact_callback_alias() -> None:
    def callback_impl(chunk: RawAudioChunk) -> None:
        del chunk

    callback: CaptureCallback = callback_impl
    callable_callback: Callable[[RawAudioChunk], None] = callback

    assert callable_callback is callback
    assert CaptureStreamPort is not None
    assert CaptureBackendPort is not None
    assert StreamingNormalizerPort is not None
