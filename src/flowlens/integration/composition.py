"""Production composition root for the local FlowLens application."""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flowlens.adapters.local_models import LocalModelReadiness
from flowlens.adapters.storage import LocalStorageReadiness
from flowlens.adapters.windows_devices import WindowsDeviceCatalog
from flowlens.asr.types import AsrWorkerConfig
from flowlens.audio.types import AudioWorkerConfig, CaptureDevice
from flowlens.controller.preflight import PreflightService
from flowlens.controller.session_controller import SessionController, SessionLaunch
from flowlens.discussion.llama_cpp_adapter import DiscussionModelConfig
from flowlens.discussion.model_manifest import ModelEntry, parse_manifest_bytes
from flowlens.discussion.worker import DiscussionWorkerConfig
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import SessionStatus
from flowlens.domain.ids import new_ulid
from flowlens.domain.session import DeviceIdentity, ModelIdentity, SessionManifest
from flowlens.integration.worker_runtime import (
    MultiprocessingWorkerRuntime,
    production_worker_targets,
)
from flowlens.persistence.paths import AppPaths, session_directory_name

_ASR_KEY = "asr"
_DISCUSSION_KEY = "discussion"
_ASR_MODEL_ID = "kotoba-whisper-v2.0-faster"
_DISCUSSION_MODEL_ID = "qwen3-4b-instruct-2507"
_ADAPTER_MODULES = (
    "flowlens.adapters.windows_devices",
    "flowlens.adapters.local_models",
    "flowlens.adapters.storage",
    "flowlens.adapters.windows_shell",
    "flowlens.audio.pyaudiowpatch_backend",
    "flowlens.asr.kotoba_whisper",
    "flowlens.discussion.llama_cpp_adapter",
    "flowlens.integration.worker_runtime",
)


@dataclass(frozen=True, slots=True)
class AppOptions:
    """Runtime options that must not alter session semantics."""

    acceptance_report: Path | None = None


@dataclass(slots=True)
class ControllerComposition:
    """Expose inspectable production ports while delegating controller behavior."""

    session: SessionController
    preflight: PreflightService
    runtime: MultiprocessingWorkerRuntime

    def __getattr__(self, name: str) -> object:
        return getattr(self.session, name)


@dataclass(frozen=True, slots=True)
class ApplicationGraph:
    """The production object graph up to, but not including, Qt widgets."""

    paths: AppPaths
    options: AppOptions
    controller: ControllerComposition

    def production_adapter_modules(self) -> tuple[str, ...]:
        """Return production adapter module names for static dependency checks."""

        return _ADAPTER_MODULES


class SystemClock:
    """Controller clock backed by local monotonic and wall clocks."""

    def monotonic_ms(self) -> int:
        """Return monotonic milliseconds for controller scheduling."""

        return int(time.monotonic() * 1_000)

    def now(self) -> datetime:
        """Return an aware local timestamp."""

        return datetime.now().astimezone()


class NoOpAccessibilityAnnouncer:
    """Controller-safe announcer used before a concrete Qt widget exists."""

    def announce(
        self,
        widget: object,
        message: str,
        assertive: bool = False,
    ) -> None:
        del widget, message, assertive


class _LazyPyAudioDiscovery:
    """Open the Windows capture backend only while serving a discovery call."""

    def list_microphones(self) -> tuple[CaptureDevice, ...]:
        return self._with_backend("list_microphones")

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
        return self._with_backend("list_loopback_outputs")

    @staticmethod
    def _with_backend(method_name: str) -> tuple[CaptureDevice, ...]:
        import pyaudiowpatch  # type: ignore[import-untyped]

        from flowlens.audio.pyaudiowpatch_backend import PyAudioWPatchBackend

        backend = PyAudioWPatchBackend(
            pyaudiowpatch.PyAudio,
            lambda: int(time.monotonic() * 1_000),
        )
        try:
            method = getattr(backend, method_name)
            return tuple(method())
        finally:
            backend.close()


def build_application(paths: AppPaths, options: AppOptions) -> ApplicationGraph:
    """Build the only production controller composition for local execution."""

    if not isinstance(paths, AppPaths):
        raise TypeError("paths must be an AppPaths")
    if not isinstance(options, AppOptions):
        raise TypeError("options must be an AppOptions")

    model_readiness = LocalModelReadiness(paths.models, paths.models / "manifest.json")
    device_catalog = WindowsDeviceCatalog(
        _LazyPyAudioDiscovery(),
        lambda source, device_id: _zero_level(source, device_id),
    )
    storage_readiness = LocalStorageReadiness()
    preflight = PreflightService(
        device_catalog,
        model_readiness,
        storage_readiness,
        paths.sessions,
    )
    runtime = MultiprocessingWorkerRuntime(
        worker_targets=production_worker_targets(),
    )
    controller = SessionController(
        preflight=preflight,
        runtime=runtime,
        clock=SystemClock(),
        announcer=NoOpAccessibilityAnnouncer(),
        launch_factory=lambda report, now, now_ms: _build_launch(
            paths,
            report,
            now,
            now_ms,
            model_readiness,
        ),
        acceptance_enabled=options.acceptance_report is not None,
    )
    return ApplicationGraph(
        paths=paths,
        options=options,
        controller=ControllerComposition(controller, preflight, runtime),
    )


def _build_launch(
    paths: AppPaths,
    report: Any,
    started_at: datetime,
    started_monotonic_ms: int,
    model_readiness: LocalModelReadiness,
) -> SessionLaunch:
    session_id = new_ulid()
    session_dir = paths.sessions / session_directory_name(started_at, session_id)
    microphone_id = report.selection.microphone_id
    loopback_id = report.selection.loopback_output_id
    microphone = _selected_device(report.microphones, microphone_id, "microphone")
    loopback = _selected_device(report.loopbacks, loopback_id, "loopback")
    checks = model_readiness.check_required()
    asr_path = _ready_model_path(checks, _ASR_KEY)
    discussion_path = _ready_model_path(checks, _DISCUSSION_KEY)
    manifest = parse_manifest_bytes((paths.models / "manifest.json").read_bytes())
    models = manifest["models"]
    asr_entry = _model_entry(models, _ASR_MODEL_ID)
    discussion_entry = _model_entry(models, _DISCUSSION_MODEL_ID)
    initial_state = DiscussionState.initial(report.selection.mode, started_at)
    return SessionLaunch(
        session_id=session_id,
        session_dir=session_dir,
        manifest=SessionManifest(
            schema_version=1,
            session_id=session_id,
            status=SessionStatus.INCOMPLETE,
            mode=report.selection.mode,
            started_at=started_at,
            ended_at=None,
            active_duration_ms=0,
            pause_intervals=(),
            microphone=DeviceIdentity(microphone.id, microphone.display_name),
            loopback_output=DeviceIdentity(loopback.id, loopback.display_name),
            asr_model=_identity(asr_entry),
            discussion_model=_identity(discussion_entry),
            application_version=_application_version(),
            transcript_entry_count=0,
            final_discussion_state_revision=0,
            recovery_notes=(),
        ),
        initial_state=initial_state,
        audio_config=AudioWorkerConfig(
            session_id=session_id,
            microphone_device_id=microphone.id,
            loopback_output_device_id=loopback.id,
            session_started_monotonic_ms=started_monotonic_ms,
            writer_queue_max_frames=3_000,
            asr_queue_max_frames=3_000,
        ),
        asr_config=AsrWorkerConfig(
            session_id=session_id,
            model_path=asr_path.parent,
        ),
        discussion_config=DiscussionWorkerConfig(
            session_id=session_id,
            model=DiscussionModelConfig(
                model_path=discussion_path,
                sha256=discussion_entry["sha256"],
            ),
            initial_state=initial_state,
        ),
    )


def _selected_device(
    devices: tuple[Any, ...],
    device_id: str | None,
    label: str,
) -> Any:
    if device_id is None:
        raise RuntimeError(f"{label} device is not selected")
    for device in devices:
        if device.id == device_id:
            return device
    raise RuntimeError(f"{label} device is unavailable")


def _ready_model_path(
    checks: Mapping[str, object],
    key: str,
) -> Path:
    check = checks.get(key)
    path = getattr(check, "path", None)
    ready = getattr(check, "ready", False)
    if ready is not True or not isinstance(path, Path):
        raise RuntimeError(f"{key} model is not ready")
    return path


def _model_entry(models: Mapping[str, ModelEntry], model_id: str) -> ModelEntry:
    entry = models.get(model_id)
    if entry is None:
        raise RuntimeError(f"{model_id} is missing from model manifest")
    return entry


def _identity(entry: ModelEntry) -> ModelIdentity:
    return ModelIdentity(
        repository=entry["repository"],
        revision=entry["source_revision"],
        sha256=entry["sha256"],
    )


def _application_version() -> str:
    try:
        return importlib.metadata.version("flowlens")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _zero_level(source: object, device_id: str) -> float:
    del source, device_id
    return 0.0


def utc_now_for_acceptance() -> datetime:
    """Return an aware UTC timestamp for future acceptance-report records."""

    return datetime.now(UTC)
