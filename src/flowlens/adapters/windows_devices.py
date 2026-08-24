"""Vendor-independent Windows device discovery and meter adapter."""

import math
from collections.abc import Callable
from typing import Protocol

from flowlens.audio.types import CaptureDevice
from flowlens.controller.models import DeviceOption
from flowlens.domain.enums import AudioSource


class CaptureDiscovery(Protocol):
    def list_microphones(self) -> tuple[CaptureDevice, ...]: ...

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]: ...


LevelReader = Callable[[AudioSource, str], float]


class WindowsDeviceCatalog:
    """Adapt the existing PyAudioWPatch seam to controller-only values."""

    def __init__(self, backend: CaptureDiscovery, level_reader: LevelReader) -> None:
        if not callable(level_reader):
            raise ValueError("level_reader must be callable")
        self._backend = backend
        self._level_reader = level_reader

    def list_microphones(self) -> tuple[DeviceOption, ...]:
        """Return exact non-loopback microphone IDs in stable display order."""

        try:
            return _options(self._backend.list_microphones(), loopback=False)
        except Exception:
            return ()

    def list_loopback_outputs(self) -> tuple[DeviceOption, ...]:
        """Return only loopback-capable output IDs in stable display order."""

        try:
            return _options(self._backend.list_loopback_outputs(), loopback=True)
        except Exception:
            return ()

    def read_level(self, source: AudioSource, device_id: str) -> float:
        """Return a sanitized meter value without exposing backend objects."""

        if not isinstance(source, AudioSource):
            raise ValueError("source must be an AudioSource")
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id must be a non-empty string")
        try:
            level = self._level_reader(source, device_id)
        except Exception:
            return 0.0
        if not isinstance(level, int | float) or isinstance(level, bool):
            return 0.0
        value = float(level)
        if not math.isfinite(value):
            return 0.0
        return min(1.0, max(0.0, value))


def _options(
    devices: tuple[CaptureDevice, ...],
    *,
    loopback: bool,
) -> tuple[DeviceOption, ...]:
    seen: set[str] = set()
    options: list[DeviceOption] = []
    for device in devices:
        if device.is_loopback is not loopback or device.device_id in seen:
            continue
        seen.add(device.device_id)
        options.append(
            DeviceOption(
                id=device.device_id,
                display_name=device.display_name,
                loopback_capable=loopback,
            )
        )
    return tuple(
        sorted(options, key=lambda item: (item.display_name.casefold(), item.id))
    )
