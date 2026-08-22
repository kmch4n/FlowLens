"""Fake-API tests for the PyAudioWPatch capture adapter."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from flowlens.audio.pyaudiowpatch_backend import (
    DeviceUnavailableError,
    PyAudioCaptureStream,
    PyAudioWPatchBackend,
)
from flowlens.audio.types import RawAudioChunk
from flowlens.domain.enums import AudioSource


class FakeStream:
    """Record lifecycle calls made by the wrapper."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active = False

    def start_stream(self) -> None:
        self.calls.append("start")
        self.active = True

    def stop_stream(self) -> None:
        self.calls.append("stop")
        self.active = False

    def close(self) -> None:
        self.calls.append("close")

    def is_active(self) -> bool:
        return self.active


class FakePyAudio:
    """Small PyAudioWPatch double with deterministic devices."""

    def __init__(self) -> None:
        self.open_kwargs: dict[str, object] = {}
        self.stream = FakeStream()
        self.terminated = 0
        self.devices: tuple[dict[str, object], ...] = (
            {
                "index": 3,
                "name": "USB Mic",
                "maxInputChannels": 1,
                "defaultSampleRate": 48_000.0,
                "isLoopbackDevice": False,
            },
            {
                "index": 5,
                "name": "Stereo Input",
                "maxInputChannels": 2,
                "defaultSampleRate": 44_100.0,
                "isLoopbackDevice": False,
            },
            {
                "index": 7,
                "name": "Speakers",
                "maxInputChannels": 0,
                "defaultSampleRate": 48_000.0,
                "isLoopbackDevice": False,
            },
            {
                "index": 11,
                "name": "Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48_000.0,
                "isLoopbackDevice": True,
            },
            {
                "index": 12,
                "name": "Unavailable Output",
                "maxInputChannels": 0,
                "defaultSampleRate": 96_000.0,
                "isLoopbackDevice": False,
            },
        )
        self.loopbacks: dict[int, dict[str, object]] = {
            7: dict(self.devices[3]),
        }

    def get_device_info_generator(self) -> Iterator[dict[str, object]]:
        return iter(self.devices)

    def get_wasapi_loopback_analogue_by_index(
        self,
        index: int,
    ) -> dict[str, object]:
        try:
            return self.loopbacks[index]
        except KeyError as exc:
            raise LookupError(index) from exc

    def open(self, **kwargs: object) -> FakeStream:
        self.open_kwargs = kwargs
        return self.stream

    def terminate(self) -> None:
        self.terminated += 1


def _backend(
    api: FakePyAudio,
    *,
    monotonic_ms: Callable[[], int] = lambda: 9_000,
) -> PyAudioWPatchBackend:
    return PyAudioWPatchBackend(lambda: api, monotonic_ms=monotonic_ms)


def test_enumerates_microphones_and_resolved_outputs_in_vendor_order() -> None:
    api = FakePyAudio()
    backend = _backend(api)

    microphones = backend.list_microphones()
    outputs = backend.list_loopback_outputs()

    assert [(item.device_id, item.input_device_index) for item in microphones] == [
        ("input:3", 3),
        ("input:5", 5),
    ]
    assert [item.display_name for item in microphones] == ["USB Mic", "Stereo Input"]
    assert [(item.device_id, item.input_device_index) for item in outputs] == [
        ("wasapi-output:7", 11),
    ]
    assert outputs[0].display_name == "Speakers"
    assert outputs[0].is_loopback is True


@pytest.mark.parametrize(
    ("source", "device_id", "expected_index", "expected_channels", "expected_rate"),
    [
        (AudioSource.ME, "input:3", 3, 1, 48_000),
        (AudioSource.OTHERS, "wasapi-output:7", 11, 2, 48_000),
    ],
)
def test_open_uses_exact_selected_native_device(
    source: AudioSource,
    device_id: str,
    expected_index: int,
    expected_channels: int,
    expected_rate: int,
) -> None:
    api = FakePyAudio()
    backend = _backend(api)

    stream = backend.open_stream(source, device_id, lambda chunk: None)

    assert isinstance(stream, PyAudioCaptureStream)
    assert api.open_kwargs == {
        "format": backend.pa_int16,
        "channels": expected_channels,
        "rate": expected_rate,
        "input": True,
        "input_device_index": expected_index,
        "frames_per_buffer": 960,
        "start": False,
        "stream_callback": api.open_kwargs["stream_callback"],
    }


def test_portaudio_callback_only_builds_native_chunk_and_backdates_capture() -> None:
    api = FakePyAudio()
    ticks = iter((9_000,))
    backend = _backend(api, monotonic_ms=lambda: next(ticks))
    chunks: list[RawAudioChunk] = []
    backend.open_stream(AudioSource.OTHERS, "wasapi-output:7", chunks.append)
    callback = api.open_kwargs["stream_callback"]
    assert callable(callback)

    result = callback(bytes(3_840), 960, {}, 0)

    assert result == (None, backend.pa_continue)
    assert chunks == [
        RawAudioChunk(
            AudioSource.OTHERS,
            bytes(3_840),
            48_000,
            2,
            8_980,
        )
    ]


@pytest.mark.parametrize(
    ("source", "device_id"),
    [
        (AudioSource.ME, "input:999"),
        (AudioSource.ME, "wasapi-output:7"),
        (AudioSource.OTHERS, "input:3"),
        (AudioSource.OTHERS, "wasapi-output:12"),
    ],
)
def test_unknown_or_wrong_kind_device_never_falls_back(
    source: AudioSource,
    device_id: str,
) -> None:
    api = FakePyAudio()
    backend = _backend(api)

    with pytest.raises(DeviceUnavailableError, match=device_id):
        backend.open_stream(source, device_id, lambda chunk: None)

    assert api.open_kwargs == {}


def test_capture_stream_lifecycle_delegates_and_close_is_idempotent() -> None:
    raw = FakeStream()
    stream = PyAudioCaptureStream(raw)

    stream.start()
    assert stream.is_active() is True
    stream.stop()
    stream.close()
    stream.close()

    assert raw.calls == ["start", "stop", "close"]


def test_backend_close_terminates_once_and_rejects_later_operations() -> None:
    api = FakePyAudio()
    backend = _backend(api)

    backend.close()
    backend.close()

    assert api.terminated == 1
    with pytest.raises(RuntimeError, match="closed"):
        backend.list_microphones()
    with pytest.raises(RuntimeError, match="closed"):
        backend.open_stream(AudioSource.ME, "input:3", lambda chunk: None)


@pytest.mark.parametrize(
    "broken_device",
    [
        {
            "index": True,
            "name": "Mic",
            "maxInputChannels": 1,
            "defaultSampleRate": 48_000.0,
        },
        {
            "index": 1,
            "name": " ",
            "maxInputChannels": 1,
            "defaultSampleRate": 48_000.0,
        },
        {
            "index": 1,
            "name": "Mic",
            "maxInputChannels": -1,
            "defaultSampleRate": 48_000.0,
        },
        {
            "index": 1,
            "name": "Mic",
            "maxInputChannels": 1,
            "defaultSampleRate": float("nan"),
        },
    ],
)
def test_malformed_vendor_device_is_skipped_without_breaking_valid_devices(
    broken_device: dict[str, Any],
) -> None:
    api = FakePyAudio()
    api.devices = (broken_device, *api.devices)
    backend = _backend(api)

    assert [item.device_id for item in backend.list_microphones()] == [
        "input:3",
        "input:5",
    ]
