"""Immutable controller-facing value objects."""

from dataclasses import dataclass
from pathlib import Path

from flowlens.domain._validation import (
    ContractValidationError,
)
from flowlens.domain.enums import SessionMode


def _non_blank(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-blank string")
    return value


def _optional_non_blank(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_blank(value, field_name)


def _absolute_path(value: object, field_name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ContractValidationError(f"{field_name} must be an absolute Path")
    if value.resolve(strict=False) != value:
        raise ContractValidationError(f"{field_name} must be normalized")
    return value


@dataclass(frozen=True, slots=True)
class DeviceOption:
    """One vendor-independent selectable audio device."""

    id: str
    display_name: str
    loopback_capable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _non_blank(self.id, "id"))
        object.__setattr__(
            self,
            "display_name",
            _non_blank(self.display_name, "display_name"),
        )
        if type(self.loopback_capable) is not bool:
            raise ContractValidationError("loopback_capable must be a boolean")


@dataclass(frozen=True, slots=True)
class ModelCheck:
    """Readiness of one required local model."""

    model_id: str
    path: Path | None
    ready: bool
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _non_blank(self.model_id, "model_id"))
        if self.path is not None:
            object.__setattr__(self, "path", _absolute_path(self.path, "path"))
        if type(self.ready) is not bool:
            raise ContractValidationError("ready must be a boolean")
        object.__setattr__(self, "reason", _optional_non_blank(self.reason, "reason"))
        if self.ready and self.reason is not None:
            raise ContractValidationError("a ready model cannot have a failure reason")
        if not self.ready and self.reason is None:
            raise ContractValidationError("an unready model requires a failure reason")


@dataclass(frozen=True, slots=True)
class StorageCheck:
    """Readiness of the local session root."""

    root: Path
    free_bytes: int
    writable: bool
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _absolute_path(self.root, "root"))
        object.__setattr__(
            self,
            "free_bytes",
            _non_negative_int(self.free_bytes, "free_bytes"),
        )
        if type(self.writable) is not bool:
            raise ContractValidationError("writable must be a boolean")
        object.__setattr__(self, "reason", _optional_non_blank(self.reason, "reason"))
        if self.writable and self.reason is not None:
            raise ContractValidationError(
                "writable storage cannot have a failure reason"
            )
        if not self.writable and self.reason is None:
            raise ContractValidationError(
                "unwritable storage requires a failure reason"
            )


@dataclass(frozen=True, slots=True)
class PreflightSelection:
    """Selections the user has explicitly confirmed for a session."""

    mode: SessionMode
    microphone_id: str | None
    loopback_output_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SessionMode):
            raise ContractValidationError("mode must be a SessionMode")
        object.__setattr__(
            self,
            "microphone_id",
            _optional_non_blank(self.microphone_id, "microphone_id"),
        )
        object.__setattr__(
            self,
            "loopback_output_id",
            _optional_non_blank(self.loopback_output_id, "loopback_output_id"),
        )


@dataclass(frozen=True, slots=True)
class BlockingIssue:
    """One exact reason attached to a preflight control."""

    control_id: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "control_id", _non_blank(self.control_id, "control_id")
        )
        object.__setattr__(self, "message", _non_blank(self.message, "message"))


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Complete immutable preflight snapshot for the UI."""

    selection: PreflightSelection
    microphones: tuple[DeviceOption, ...]
    loopbacks: tuple[DeviceOption, ...]
    mic_level: float
    loopback_level: float
    models: tuple[ModelCheck, ...]
    storage: StorageCheck
    destination: Path
    issues: tuple[BlockingIssue, ...]
    can_start: bool

    def __post_init__(self) -> None:
        if not isinstance(self.selection, PreflightSelection):
            raise ContractValidationError("selection must be a PreflightSelection")
        for field_name in ("microphones", "loopbacks"):
            values = getattr(self, field_name)
            if type(values) is not tuple or not all(
                isinstance(item, DeviceOption) for item in values
            ):
                raise ContractValidationError(
                    f"{field_name} must contain DeviceOption values"
                )
            if len({item.id for item in values}) != len(values):
                raise ContractValidationError(
                    f"{field_name} must not contain duplicate IDs"
                )
        for field_name in ("mic_level", "loopback_level"):
            value = getattr(self, field_name)
            if type(value) is not float or not 0.0 <= value <= 1.0:
                raise ContractValidationError(
                    f"{field_name} must be a float from 0 to 1"
                )
        if type(self.models) is not tuple or not all(
            isinstance(item, ModelCheck) for item in self.models
        ):
            raise ContractValidationError("models must contain ModelCheck values")
        if len({item.model_id for item in self.models}) != len(self.models):
            raise ContractValidationError("models must not contain duplicate IDs")
        if not isinstance(self.storage, StorageCheck):
            raise ContractValidationError("storage must be a StorageCheck")
        object.__setattr__(
            self, "destination", _absolute_path(self.destination, "destination")
        )
        if type(self.issues) is not tuple or not all(
            isinstance(item, BlockingIssue) for item in self.issues
        ):
            raise ContractValidationError("issues must contain BlockingIssue values")
        if type(self.can_start) is not bool or self.can_start != (not self.issues):
            raise ContractValidationError("can_start must equal not issues")


def _non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")
    return value
