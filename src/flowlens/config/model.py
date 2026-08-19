"""Immutable local configuration contracts and window geometry helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

from flowlens.domain._validation import (
    ContractValidationError,
    require_exact_keys,
    require_int,
    require_str,
)
from flowlens.domain.enums import SessionMode

_APP_CONFIG_KEYS = frozenset({"schema_version", "window", "devices", "last_mode"})
_DEVICE_PREFERENCES_KEYS = frozenset({"microphone_id", "loopback_output_id"})
_WINDOW_PREFERENCES_KEYS = frozenset(
    {"x", "y", "width", "height", "maximized", "always_on_top"}
)


def _require_mapping(value: object, record_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ContractValidationError(f"{record_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    parsed = require_int(value, field_name)
    if parsed <= 0:
        raise ContractValidationError(f"{field_name} must be positive")
    return parsed


def _parse_mode(value: object) -> SessionMode:
    parsed = require_str(value, "last_mode")
    try:
        return SessionMode(parsed)
    except ValueError as error:
        raise ContractValidationError(
            "last_mode must be a supported SessionMode"
        ) from error


@dataclass(frozen=True, slots=True)
class Rect:
    """An available display's logical-pixel rectangle."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", require_int(self.x, "x"))
        object.__setattr__(self, "y", require_int(self.y, "y"))
        object.__setattr__(self, "width", _require_positive_int(self.width, "width"))
        object.__setattr__(
            self,
            "height",
            _require_positive_int(self.height, "height"),
        )


@dataclass(frozen=True, slots=True)
class WindowPreferences:
    """Persisted window state without any display-specific identifier."""

    x: int
    y: int
    width: int
    height: int
    maximized: bool
    always_on_top: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", require_int(self.x, "x"))
        object.__setattr__(self, "y", require_int(self.y, "y"))
        object.__setattr__(self, "width", _require_positive_int(self.width, "width"))
        object.__setattr__(
            self,
            "height",
            _require_positive_int(self.height, "height"),
        )
        object.__setattr__(
            self,
            "maximized",
            _require_bool(self.maximized, "maximized"),
        )
        object.__setattr__(
            self,
            "always_on_top",
            _require_bool(self.always_on_top, "always_on_top"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the preferences to their strict JSON shape."""

        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "maximized": self.maximized,
            "always_on_top": self.always_on_top,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse preferences while rejecting missing and unknown keys."""

        mapping = _require_mapping(value, "WindowPreferences")
        require_exact_keys(mapping, _WINDOW_PREFERENCES_KEYS, "WindowPreferences")
        return cls(
            x=require_int(mapping["x"], "x"),
            y=require_int(mapping["y"], "y"),
            width=_require_positive_int(mapping["width"], "width"),
            height=_require_positive_int(mapping["height"], "height"),
            maximized=_require_bool(mapping["maximized"], "maximized"),
            always_on_top=_require_bool(
                mapping["always_on_top"],
                "always_on_top",
            ),
        )

    def with_geometry(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Self:
        """Return a preference with updated geometry and unchanged window flags."""

        return type(self)(
            x=x,
            y=y,
            width=width,
            height=height,
            maximized=self.maximized,
            always_on_top=self.always_on_top,
        )


@dataclass(frozen=True, slots=True)
class DevicePreferences:
    """Saved audio-device identities; empty strings mean no selection."""

    microphone_id: str
    loopback_output_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microphone_id",
            require_str(self.microphone_id, "microphone_id"),
        )
        object.__setattr__(
            self,
            "loopback_output_id",
            require_str(self.loopback_output_id, "loopback_output_id"),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the preferences to their strict JSON shape."""

        return {
            "microphone_id": self.microphone_id,
            "loopback_output_id": self.loopback_output_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse preferences while rejecting missing and unknown keys."""

        mapping = _require_mapping(value, "DevicePreferences")
        require_exact_keys(mapping, _DEVICE_PREFERENCES_KEYS, "DevicePreferences")
        return cls(
            microphone_id=require_str(mapping["microphone_id"], "microphone_id"),
            loopback_output_id=require_str(
                mapping["loopback_output_id"],
                "loopback_output_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """The complete, versioned non-session local configuration."""

    schema_version: int
    window: WindowPreferences
    devices: DevicePreferences
    last_mode: SessionMode

    def __post_init__(self) -> None:
        schema_version = require_int(self.schema_version, "schema_version")
        if schema_version != 1:
            raise ContractValidationError("schema_version must be 1")
        if not isinstance(self.window, WindowPreferences):
            raise ContractValidationError("window must be a WindowPreferences")
        if not isinstance(self.devices, DevicePreferences):
            raise ContractValidationError("devices must be a DevicePreferences")
        if not isinstance(self.last_mode, SessionMode):
            raise ContractValidationError("last_mode must be a SessionMode")
        object.__setattr__(self, "schema_version", schema_version)

    @classmethod
    def default(cls) -> Self:
        """Create the specification's initial local configuration."""

        return cls(
            schema_version=1,
            window=WindowPreferences(
                x=100,
                y=100,
                width=1280,
                height=800,
                maximized=False,
                always_on_top=False,
            ),
            devices=DevicePreferences(microphone_id="", loopback_output_id=""),
            last_mode=SessionMode.MEETING,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the configuration in its specified field order."""

        return {
            "schema_version": self.schema_version,
            "window": self.window.to_dict(),
            "devices": self.devices.to_dict(),
            "last_mode": self.last_mode.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse configuration while rejecting missing and unknown keys."""

        mapping = _require_mapping(value, "AppConfig")
        schema_version = require_int(mapping.get("schema_version"), "schema_version")
        if schema_version != 1:
            raise ContractValidationError("schema_version must be 1")
        require_exact_keys(mapping, _APP_CONFIG_KEYS, "AppConfig")
        return cls(
            schema_version=schema_version,
            window=WindowPreferences.from_dict(mapping["window"]),
            devices=DevicePreferences.from_dict(mapping["devices"]),
            last_mode=_parse_mode(mapping["last_mode"]),
        )


def _intersection_area(first: Rect, second: Rect) -> int:
    """Return the overlap area of two rectangles, or zero when disjoint."""

    overlap_width = max(
        0,
        min(first.x + first.width, second.x + second.width) - max(first.x, second.x),
    )
    overlap_height = max(
        0,
        min(first.y + first.height, second.y + second.height) - max(first.y, second.y),
    )
    return overlap_width * overlap_height


def clamp_window(
    window: WindowPreferences,
    displays: Sequence[Rect],
) -> WindowPreferences:
    """Fit saved window geometry entirely into its most-overlapped display."""

    if not displays:
        raise ValueError("displays must not be empty")
    if not all(isinstance(display, Rect) for display in displays):
        raise TypeError("displays must contain only Rect values")

    saved_rect = Rect(window.x, window.y, window.width, window.height)
    selected_display = displays[0]
    largest_intersection = _intersection_area(saved_rect, selected_display)
    for display in displays[1:]:
        intersection = _intersection_area(saved_rect, display)
        if intersection > largest_intersection:
            selected_display = display
            largest_intersection = intersection

    width = min(max(window.width, 900), selected_display.width)
    height = min(max(window.height, 600), selected_display.height)
    maximum_x = selected_display.x + selected_display.width - width
    maximum_y = selected_display.y + selected_display.height - height
    x = min(max(window.x, selected_display.x), maximum_x)
    y = min(max(window.y, selected_display.y), maximum_y)
    return window.with_geometry(x=x, y=y, width=width, height=height)
