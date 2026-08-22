"""PyAudioWPatch/WASAPI capture adapter behind a fakeable seam."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Protocol

import pyaudiowpatch  # type: ignore[import-untyped]

from flowlens.audio.ports import CaptureCallback
from flowlens.audio.types import CaptureDevice, RawAudioChunk
from flowlens.domain.enums import AudioSource

PyAudioFactory = Callable[[], "PyAudioApi"]
MonotonicClock = Callable[[], int]


class DeviceUnavailableError(RuntimeError):
    """Raised when an exact configured capture device cannot be resolved."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        super().__init__(f"capture device is unavailable: {device_id}")


class _NativeStream(Protocol):
    def start_stream(self) -> None: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...

    def is_active(self) -> bool: ...


class PyAudioApi(Protocol):
    """Minimal PyAudioWPatch surface used by the adapter."""

    def get_device_info_generator(self) -> Iterator[Mapping[str, object]]: ...

    def get_wasapi_loopback_analogue_by_index(
        self,
        index: int,
    ) -> Mapping[str, object]: ...

    def open(self, **kwargs: object) -> _NativeStream: ...

    def terminate(self) -> None: ...


class PyAudioCaptureStream:
    """Lifecycle adapter for one native PortAudio stream."""

    def __init__(self, stream: _NativeStream) -> None:
        self._stream = stream
        self._closed = False

    def start(self) -> None:
        """Start capture unless it is already active."""

        self._require_open()
        if not self._stream.is_active():
            self._stream.start_stream()

    def stop(self) -> None:
        """Stop capture when active."""

        self._require_open()
        if self._stream.is_active():
            self._stream.stop_stream()

    def close(self) -> None:
        """Close the native stream exactly once."""

        if self._closed:
            return
        self._stream.close()
        self._closed = True

    def is_active(self) -> bool:
        """Return whether the native stream is currently active."""

        return False if self._closed else bool(self._stream.is_active())

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("capture stream is closed")


class PyAudioWPatchBackend:
    """Enumerate exact Windows devices and open native capture streams."""

    pa_int16 = int(pyaudiowpatch.paInt16)
    pa_continue = int(pyaudiowpatch.paContinue)

    def __init__(
        self,
        py_audio_factory: PyAudioFactory,
        monotonic_ms: MonotonicClock,
    ) -> None:
        if not callable(py_audio_factory):
            raise ValueError("py_audio_factory must be callable")
        if not callable(monotonic_ms):
            raise ValueError("monotonic_ms must be callable")
        self._api = py_audio_factory()
        self._monotonic_ms = monotonic_ms
        self._closed = False

    def list_microphones(self) -> tuple[CaptureDevice, ...]:
        """Return selectable non-loopback input devices in vendor order."""

        self._require_open()
        devices: list[CaptureDevice] = []
        for raw in self._api.get_device_info_generator():
            parsed = _parse_device(raw)
            if parsed is None:
                continue
            index, name, channels, sample_rate_hz, is_loopback = parsed
            if is_loopback or channels == 0:
                continue
            devices.append(
                CaptureDevice(
                    device_id=f"input:{index}",
                    display_name=name,
                    input_device_index=index,
                    sample_rate_hz=sample_rate_hz,
                    channels=channels,
                    is_loopback=False,
                )
            )
        return tuple(devices)

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
        """Resolve selectable outputs to their WASAPI loopback analogues."""

        self._require_open()
        devices: list[CaptureDevice] = []
        for raw in self._api.get_device_info_generator():
            parsed = _parse_device(raw)
            if parsed is None:
                continue
            output_index, output_name, channels, _, is_loopback = parsed
            if is_loopback or channels != 0:
                continue
            try:
                analogue_raw = self._api.get_wasapi_loopback_analogue_by_index(
                    output_index
                )
            except (LookupError, OSError, ValueError):
                continue
            analogue = _parse_device(analogue_raw)
            if analogue is None:
                continue
            input_index, _, input_channels, sample_rate_hz, analogue_is_loopback = (
                analogue
            )
            if not analogue_is_loopback or input_channels <= 0:
                continue
            devices.append(
                CaptureDevice(
                    device_id=f"wasapi-output:{output_index}",
                    display_name=output_name,
                    input_device_index=input_index,
                    sample_rate_hz=sample_rate_hz,
                    channels=input_channels,
                    is_loopback=True,
                )
            )
        return tuple(devices)

    def open_stream(
        self,
        source: AudioSource,
        device_id: str,
        callback: CaptureCallback,
    ) -> PyAudioCaptureStream:
        """Open the exact configured device without fallback selection."""

        self._require_open()
        if not isinstance(source, AudioSource):
            raise ValueError("source must be an AudioSource")
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id must be a non-empty string")
        if not callable(callback):
            raise ValueError("callback must be callable")
        candidates = (
            self.list_microphones()
            if source is AudioSource.ME
            else self.list_loopback_outputs()
        )
        device = next(
            (item for item in candidates if item.device_id == device_id),
            None,
        )
        if device is None:
            raise DeviceUnavailableError(device_id)

        def stream_callback(
            in_data: bytes,
            frame_count: int,
            time_info: object,
            status_flags: int,
        ) -> tuple[None, int]:
            del time_info, status_flags
            captured_monotonic_ms = self._monotonic_ms() - round(
                frame_count * 1_000 / device.sample_rate_hz
            )
            callback(
                RawAudioChunk(
                    source=source,
                    pcm_s16le_interleaved=in_data,
                    sample_rate_hz=device.sample_rate_hz,
                    channels=device.channels,
                    captured_monotonic_ms=captured_monotonic_ms,
                )
            )
            return (None, self.pa_continue)

        stream = self._api.open(
            format=self.pa_int16,
            channels=device.channels,
            rate=device.sample_rate_hz,
            input=True,
            input_device_index=device.input_device_index,
            frames_per_buffer=round(device.sample_rate_hz * 0.020),
            start=False,
            stream_callback=stream_callback,
        )
        return PyAudioCaptureStream(stream)

    def close(self) -> None:
        """Terminate the owned PyAudio instance exactly once."""

        if self._closed:
            return
        self._api.terminate()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("capture backend is closed")


def _parse_device(
    raw: Mapping[str, object],
) -> tuple[int, str, int, int, bool] | None:
    """Parse one vendor mapping without allowing malformed devices through."""

    try:
        index_value = raw["index"]
        name_value = raw["name"]
        channels_value = raw["maxInputChannels"]
        rate_value = raw["defaultSampleRate"]
        loopback_value = raw.get("isLoopbackDevice", False)
    except (KeyError, TypeError):
        return None
    if not isinstance(index_value, int) or isinstance(index_value, bool):
        return None
    if index_value < 0 or not isinstance(name_value, str) or not name_value.strip():
        return None
    if not isinstance(channels_value, int) or isinstance(channels_value, bool):
        return None
    if channels_value < 0 or not isinstance(loopback_value, bool):
        return None
    if not isinstance(rate_value, int | float) or isinstance(rate_value, bool):
        return None
    rate_float = float(rate_value)
    if not math.isfinite(rate_float) or rate_float <= 0:
        return None
    sample_rate_hz = round(rate_float)
    if sample_rate_hz <= 0:
        return None
    return (
        index_value,
        name_value,
        channels_value,
        sample_rate_hz,
        loopback_value,
    )
