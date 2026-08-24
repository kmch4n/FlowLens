"""Pure preflight evaluation with exact user-facing blockers."""

from collections.abc import Callable, Mapping
from pathlib import Path

from flowlens.controller.models import (
    BlockingIssue,
    DeviceOption,
    ModelCheck,
    PreflightReport,
    PreflightSelection,
    StorageCheck,
)
from flowlens.controller.ports import DeviceCatalog, ModelReadiness, StorageReadiness
from flowlens.domain.enums import AudioSource

REQUIRED_FREE_BYTES = 500 * 1024 * 1024
_MODEL_ORDER = ("asr", "discussion")


class PreflightService:
    """Evaluate current local readiness without mutating saved preferences."""

    def __init__(
        self,
        device_catalog: DeviceCatalog,
        model_readiness: ModelReadiness,
        storage_readiness: StorageReadiness,
        sessions_root: Path,
    ) -> None:
        self.device_catalog = device_catalog
        self.model_readiness = model_readiness
        self.storage_readiness = storage_readiness
        if not isinstance(sessions_root, Path) or not sessions_root.is_absolute():
            raise ValueError("sessions_root must be an absolute Path")
        if sessions_root.resolve(strict=False) != sessions_root:
            raise ValueError("sessions_root must be normalized")
        self._sessions_root = sessions_root

    def evaluate(self, selection: PreflightSelection) -> PreflightReport:
        """Return every blocker in stable control order."""

        microphones = _discover(self.device_catalog.list_microphones)
        loopbacks = tuple(
            option
            for option in _discover(self.device_catalog.list_loopback_outputs)
            if option.loopback_capable
        )
        microphone_ids = {option.id for option in microphones}
        loopback_ids = {option.id for option in loopbacks}
        microphone_id = (
            selection.microphone_id
            if selection.microphone_id in microphone_ids
            else None
        )
        loopback_id = (
            selection.loopback_output_id
            if selection.loopback_output_id in loopback_ids
            else None
        )
        normalized = PreflightSelection(selection.mode, microphone_id, loopback_id)
        issues: list[BlockingIssue] = []
        if microphone_id is None:
            issues.append(
                BlockingIssue("microphone", "Select an available microphone.")
            )
        if loopback_id is None:
            issues.append(
                BlockingIssue(
                    "loopback",
                    "Select a loopback-capable Windows output device.",
                )
            )

        checks = _models(self.model_readiness)
        models = tuple(checks[key] for key in _MODEL_ORDER)
        if not checks["asr"].ready:
            issues.append(
                BlockingIssue("asr_model", _model_message("asr", checks["asr"]))
            )
        if not checks["discussion"].ready:
            issues.append(
                BlockingIssue(
                    "discussion_model",
                    _model_message("discussion", checks["discussion"]),
                )
            )

        storage = _storage(
            self.storage_readiness,
            self._sessions_root,
            REQUIRED_FREE_BYTES,
        )
        if not storage.writable:
            issues.append(
                BlockingIssue("storage", "FlowLens cannot create the session folder.")
            )
        if storage.free_bytes < REQUIRED_FREE_BYTES:
            issues.append(
                BlockingIssue("storage", "At least 500 MB of free space is required.")
            )

        mic_level = _level(self.device_catalog, AudioSource.ME, microphone_id)
        loopback_level = _level(
            self.device_catalog,
            AudioSource.OTHERS,
            loopback_id,
        )
        return PreflightReport(
            selection=normalized,
            microphones=microphones,
            loopbacks=loopbacks,
            mic_level=mic_level,
            loopback_level=loopback_level,
            models=models,
            storage=storage,
            destination=self._sessions_root,
            issues=tuple(issues),
            can_start=not issues,
        )


def _level(catalog: DeviceCatalog, source: AudioSource, device_id: str | None) -> float:
    if device_id is None:
        return 0.0
    try:
        value = catalog.read_level(source, device_id)
    except Exception:
        return 0.0
    if not isinstance(value, float):
        return 0.0
    return min(1.0, max(0.0, value))


def _discover(
    operation: Callable[[], tuple[DeviceOption, ...]],
) -> tuple[DeviceOption, ...]:
    try:
        values = operation()
    except Exception:
        return ()
    if not isinstance(values, tuple) or not all(
        isinstance(item, DeviceOption) for item in values
    ):
        return ()
    return values


def _models(readiness: ModelReadiness) -> dict[str, ModelCheck]:
    fallback = {
        "asr": ModelCheck("kotoba-whisper-v2.0-faster", None, False, "invalid"),
        "discussion": ModelCheck(
            "qwen3-4b-instruct-2507",
            None,
            False,
            "invalid",
        ),
    }
    try:
        checks = readiness.check_required()
    except Exception:
        return fallback
    if not isinstance(checks, Mapping):
        return fallback
    asr = checks.get("asr")
    discussion = checks.get("discussion")
    if not isinstance(asr, ModelCheck) or asr.model_id != fallback["asr"].model_id:
        asr = fallback["asr"]
    if (
        not isinstance(discussion, ModelCheck)
        or discussion.model_id != fallback["discussion"].model_id
    ):
        discussion = fallback["discussion"]
    return {"asr": asr, "discussion": discussion}


def _storage(
    readiness: StorageReadiness,
    root: Path,
    required_bytes: int,
) -> StorageCheck:
    fallback = StorageCheck(root, 0, False, "unavailable")
    try:
        check = readiness.check(root, required_bytes)
    except Exception:
        return fallback
    if not isinstance(check, StorageCheck) or check.root != root:
        return fallback
    return check


def _model_message(kind: str, check: ModelCheck) -> str:
    if kind == "asr":
        if check.reason == "missing":
            return "Kotoba-Whisper model files are missing."
        if check.reason == "checksum":
            return "Kotoba-Whisper model checksum does not match."
        return "Kotoba-Whisper model files are invalid."
    if check.reason == "missing":
        return "Qwen discussion model files are missing."
    if check.reason == "checksum":
        return "Qwen discussion model checksum does not match."
    return "Qwen discussion model files are invalid."
