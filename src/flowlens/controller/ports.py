"""Runtime-checkable composition ports owned by the controller layer."""

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from flowlens.controller.models import DeviceOption, ModelCheck, StorageCheck
from flowlens.domain.enums import AudioSource, ProcessSource
from flowlens.domain.messages import (
    MessageEnvelope,
    WriterForceCloseRequest,
    WriterForceCloseResult,
)


@runtime_checkable
class DeviceCatalog(Protocol):
    def list_microphones(self) -> tuple[DeviceOption, ...]: ...

    def list_loopback_outputs(self) -> tuple[DeviceOption, ...]: ...

    def read_level(self, source: AudioSource, device_id: str) -> float: ...


@runtime_checkable
class ModelReadiness(Protocol):
    def check_required(self) -> Mapping[str, ModelCheck]: ...


@runtime_checkable
class StorageReadiness(Protocol):
    def check(self, root: Path, required_bytes: int) -> StorageCheck: ...


@runtime_checkable
class Clock(Protocol):
    def monotonic_ms(self) -> int: ...

    def now(self) -> datetime: ...


@runtime_checkable
class WorkerRuntime(Protocol):
    def start_all(self, launch: object) -> None: ...

    def send(
        self, target: ProcessSource, envelope: MessageEnvelope[object]
    ) -> None: ...

    def poll(self) -> tuple[MessageEnvelope[object], ...]: ...

    def restart(self, target: ProcessSource, launch: object | None = None) -> None: ...

    def health(self) -> Mapping[ProcessSource, bool]: ...

    def shutdown(self) -> object: ...

    def safe_stop(self, *, audio_fence_required: bool = False) -> object: ...

    def request_writer_force_close(
        self,
        request: WriterForceCloseRequest,
        timeout_seconds: float,
    ) -> WriterForceCloseResult | None: ...

    def writer_force_close_result(self) -> WriterForceCloseResult | None: ...


@runtime_checkable
class FolderOpener(Protocol):
    def open(self, path: Path) -> None: ...


@runtime_checkable
class MotionPreferences(Protocol):
    def reduced_motion(self) -> bool: ...


@runtime_checkable
class AccessibilityAnnouncer(Protocol):
    def announce(
        self, widget: object, message: str, assertive: bool = False
    ) -> None: ...
