"""Stateful Qt widgets for the FlowLens Workbench surface."""

from collections.abc import Mapping
from typing import ClassVar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QPushButton, QWidget


class StatefulButton(QPushButton):
    """Button with the complete FlowLens eight-state contract."""

    supported_states: ClassVar[tuple[str, ...]] = (
        "default",
        "hover",
        "focus",
        "active",
        "disabled",
        "loading",
        "error",
        "success",
    )

    _STATE_PREFIXES: ClassVar[Mapping[str, str]] = {
        "default": "",
        "hover": "",
        "focus": "",
        "active": "",
        "disabled": "",
        "loading": "Checking",
        "error": "! Error",
        "success": "✓ Ready",
    }

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._base_text = text
        self.setMinimumSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("flowlensRole", "stateful")
        self.set_ui_state("default")

    def set_ui_state(self, state: str, description: str | None = None) -> None:
        """Apply a named UI state and accessible state description."""

        if state not in self.supported_states:
            raise ValueError(f"Unsupported button UI state: {state}")
        self.setProperty("uiState", state)
        self.setEnabled(state not in {"disabled", "loading"})
        self.setText(self._state_text(state, description))
        self.setAccessibleDescription(description or state)
        _refresh_style(self)

    def _state_text(self, state: str, description: str | None) -> str:
        if state == "loading" and description:
            return f"Checking · {self._base_text}"
        if state in {"error", "success"}:
            prefix = self._STATE_PREFIXES[state]
            if description:
                return f"{prefix} · {self._base_text}"
            return f"{prefix} · {self._base_text}"
        return self._base_text


class StatusIndicator(QLabel):
    """Accessible text-first status label with non-color state cues."""

    supported_states: ClassVar[tuple[str, ...]] = (
        "default",
        "loading",
        "error",
        "success",
    )

    _STATE_LABELS: ClassVar[Mapping[str, tuple[str, str]]] = {
        "default": ("", "Status"),
        "loading": ("…", "Checking"),
        "error": ("!", "Error"),
        "success": ("✓", "Ready"),
    }

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self.setProperty("flowlensRole", "status")
        self.set_status("default", "Waiting")

    def set_status(self, state: str, message: str) -> None:
        """Render a status with icon, text, and accessible description."""

        if state not in self.supported_states:
            raise ValueError(f"Unsupported status UI state: {state}")
        icon, label = self._STATE_LABELS[state]
        prefix = f"{icon} {label}" if icon else label
        self.setProperty("uiState", state)
        self.setText(f"{prefix}: {self._label} · {message}")
        self.setAccessibleDescription(f"{label}: {message}")
        _refresh_style(self)


class InputMeter(QProgressBar):
    """Level meter for microphone and loopback input surfaces."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self.setRange(0, 100)
        self.setMinimumHeight(44)
        self.setTextVisible(False)
        self.setProperty("flowlensRole", "inputMeter")
        self.setAccessibleName(f"{label} input level")
        self.setAccessibleDescription("Input level: 0 percent")

    def set_level(self, level: float) -> None:
        """Set the current level as a 0.0 to 1.0 normalized value."""

        clamped = min(1.0, max(0.0, level))
        percent = round(clamped * 100)
        self.setValue(percent)
        self.setAccessibleDescription(f"Input level: {percent} percent")


def _refresh_style(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
