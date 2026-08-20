"""Hardware-independent ports for audio capture and normalization."""

from collections.abc import Callable
from typing import Protocol

from flowlens.audio.types import AudioFrame, CaptureDevice, RawAudioChunk
from flowlens.domain.enums import AudioSource

CaptureCallback = Callable[[RawAudioChunk], None]


class CaptureStreamPort(Protocol):
    """Lifecycle contract for one opened capture stream."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def is_active(self) -> bool: ...


class CaptureBackendPort(Protocol):
    """Device discovery and stream-opening contract for capture backends."""

    def list_microphones(self) -> tuple[CaptureDevice, ...]: ...

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]: ...

    def open_stream(
        self,
        source: AudioSource,
        device_id: str,
        callback: CaptureCallback,
    ) -> CaptureStreamPort: ...

    def close(self) -> None: ...


class StreamingNormalizerPort(Protocol):
    """Stateful native-to-canonical audio conversion contract."""

    def push(self, chunk: RawAudioChunk) -> tuple[AudioFrame, ...]: ...

    def flush(self) -> tuple[AudioFrame, ...]: ...
