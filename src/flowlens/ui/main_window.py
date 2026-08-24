from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QStackedWidget,
    QTextEdit,
    QWidget,
)

from flowlens.config.model import WindowPreferences
from flowlens.controller.models import CompletionSummary, PreflightSelection
from flowlens.ui.completion_page import CompletionPage
from flowlens.ui.dialogs import SlowFinalizationDialog, StopConfirmationDialog
from flowlens.ui.live_page import LivePage
from flowlens.ui.preflight_page import PreflightPage


def clamp_geometry(
    window: WindowPreferences,
    screens: Sequence[QRect],
    *,
    minimum: QSize | None = None,
) -> QRect:
    """Clamp saved geometry to the current display set."""

    if not screens:
        raise ValueError("screens must not be empty")
    if not all(isinstance(screen, QRect) for screen in screens):
        raise TypeError("screens must contain QRect values")
    resolved_minimum = QSize(900, 600) if minimum is None else minimum
    if not isinstance(resolved_minimum, QSize):
        raise TypeError("minimum must be a QSize")

    saved = QRect(window.x, window.y, window.width, window.height)
    selected = max(screens, key=lambda screen: _intersection_area(saved, screen))
    width = min(max(window.width, resolved_minimum.width()), selected.width())
    height = min(max(window.height, resolved_minimum.height()), selected.height())
    x = min(max(window.x, selected.x()), selected.x() + selected.width() - width)
    y = min(max(window.y, selected.y()), selected.y() + selected.height() - height)
    return QRect(x, y, width, height)


class MainWindow(QMainWindow):
    """Compose Task 10 pages and expose only shell-level UI events."""

    selection_changed = Signal(PreflightSelection)
    start_requested = Signal()
    pause_resume_requested = Signal()
    stop_requested = Signal()
    stop_confirmed_requested = Signal()
    stop_cancel_requested = Signal()
    keep_waiting_requested = Signal()
    force_close_requested = Signal()
    always_on_top_changed = Signal(bool)
    active_close_requested = Signal()
    orderly_close_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.stack = QStackedWidget()
        self.preflight_page = PreflightPage()
        self.live_page = LivePage()
        self.completion_page = CompletionPage()
        self.stop_dialog = StopConfirmationDialog()
        self.slow_finalization_dialog = SlowFinalizationDialog()
        self._preflight_can_start = False
        self._active_close_requires_stop = False
        self._always_on_top = False
        self._event_filter_installed = False

        self._build()
        self._connect_static_signals()
        self._connect_live_page()
        self._install_application_filter()

    def show_preflight(self) -> None:
        """Show the preflight page without changing controller state."""

        self.stack.setCurrentWidget(self.preflight_page)

    def show_live(self) -> None:
        """Show the live session page without changing controller state."""

        self.stack.setCurrentWidget(self.live_page)

    def show_completion(self, summary: CompletionSummary | None = None) -> None:
        """Show the completion page, optionally refreshing its summary."""

        if summary is not None:
            self.completion_page.render(summary)
        self.stack.setCurrentWidget(self.completion_page)

    def current_page(self) -> QWidget:
        """Return the currently visible page widget."""

        current = self.stack.currentWidget()
        if current is None:
            raise RuntimeError("no current page")
        return current

    def reset_live_page(self) -> None:
        """Replace live UI state so another session starts with no ephemera."""

        index = self.stack.indexOf(self.live_page)
        if index < 0:
            raise RuntimeError("live page is not in the stack")
        old_page = self.live_page
        self.stack.removeWidget(old_page)
        old_page.deleteLater()
        self.live_page = LivePage()
        self.stack.insertWidget(index, self.live_page)
        self._connect_live_page()

    def set_preflight_can_start(self, enabled: bool) -> None:
        """Record whether the presenter permits Ctrl+Enter start."""

        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        self._preflight_can_start = enabled

    def set_active_close_requires_stop(self, enabled: bool) -> None:
        """Record whether close should request the stop-confirmation path."""

        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        self._active_close_requires_stop = enabled

    def set_stop_confirmation_visible(self, visible: bool) -> None:
        """Synchronize the non-destructive stop dialog visibility."""

        if visible and not self.stop_dialog.isVisible():
            self.stop_dialog.show()
        elif not visible and self.stop_dialog.isVisible():
            self.stop_dialog.hide()

    def set_slow_finalization_visible(self, visible: bool) -> None:
        """Synchronize the slow-finalization dialog visibility."""

        if visible and not self.slow_finalization_dialog.isVisible():
            self.slow_finalization_dialog.show()
        elif not visible and self.slow_finalization_dialog.isVisible():
            self.slow_finalization_dialog.hide()

    def set_always_on_top(self, enabled: bool) -> None:
        """Apply the WindowStaysOnTopHint flag and mirror the live toggle."""

        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self._always_on_top == enabled:
            self._sync_always_on_top_toggle()
            return
        self._always_on_top = enabled
        was_visible = self.isVisible()
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self._sync_always_on_top_toggle()
        if was_visible:
            self.show()

    def is_always_on_top(self) -> bool:
        """Return the present always-on-top preference."""

        return self._always_on_top

    def apply_window_preferences(
        self,
        preferences: WindowPreferences,
        screens: Sequence[QRect],
    ) -> None:
        """Restore clamped geometry and window flags from preferences."""

        self.setMinimumSize(900, 600)
        self.setGeometry(clamp_geometry(preferences, screens))
        self.set_always_on_top(preferences.always_on_top)
        if preferences.maximized:
            self.showMaximized()

    def window_preferences(self) -> WindowPreferences:
        """Return the exact non-session window preferences shape."""

        geometry = self.normalGeometry() if self.isMaximized() else self.geometry()
        return WindowPreferences(
            x=geometry.x(),
            y=geometry.y(),
            width=max(1, geometry.width()),
            height=max(1, geometry.height()),
            maximized=self.isMaximized(),
            always_on_top=self._always_on_top,
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Handle window-scoped shortcuts before child widgets consume them."""

        if (
            not isinstance(event, QKeyEvent)
            or event.type() is not QEvent.Type.KeyPress
            or not self._event_targets_this_window(watched)
        ):
            return super().eventFilter(watched, event)
        if self._handle_key_press(event):
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle direct key delivery to the main window."""

        if self._handle_key_press(event):
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Keep active sessions open and emit an orderly-close signal otherwise."""

        if self._active_close_requires_stop:
            event.ignore()
            self.active_close_requested.emit()
            return
        self.orderly_close_requested.emit()
        self._remove_application_filter()
        super().closeEvent(event)

    def _build(self) -> None:
        self.setWindowTitle("FlowLens")
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self.preflight_page)
        self.stack.addWidget(self.live_page)
        self.stack.addWidget(self.completion_page)
        self.show_preflight()

    def _connect_static_signals(self) -> None:
        self.preflight_page.selection_changed.connect(self.selection_changed.emit)
        self.preflight_page.start_requested.connect(self.start_requested.emit)
        self.completion_page.close_requested.connect(self.close)
        self.stop_dialog.stop_confirmed.connect(self.stop_dialog_confirmed)
        self.stop_dialog.keep_recording_requested.connect(self.stop_dialog_cancelled)
        self.slow_finalization_dialog.keep_waiting_requested.connect(
            self.slow_finalization_keep_waiting,
        )
        self.slow_finalization_dialog.force_close_requested.connect(
            self.slow_finalization_force_close,
        )

    def _connect_live_page(self) -> None:
        self.live_page.pause_requested.connect(self.pause_resume_requested.emit)
        self.live_page.resume_requested.connect(self.pause_resume_requested.emit)
        self.live_page.stop_requested.connect(self.stop_requested.emit)
        self.live_page.always_on_top_changed.connect(self.always_on_top_changed.emit)
        self._sync_always_on_top_toggle()

    def _install_application_filter(self) -> None:
        app = QApplication.instance()
        if app is None or self._event_filter_installed:
            return
        app.installEventFilter(self)
        self._event_filter_installed = True

    def _remove_application_filter(self) -> None:
        app = QApplication.instance()
        if app is None or not self._event_filter_installed:
            return
        app.removeEventFilter(self)
        self._event_filter_installed = False

    def _event_targets_this_window(self, watched: QObject) -> bool:
        if not self.isVisible() and watched is not self:
            return False
        if isinstance(watched, QWidget):
            return watched is self or watched.window() is self
        return False

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        key = event.key()
        modifiers = event.modifiers()
        is_global_shortcut = self._is_global_shortcut(key, modifiers)
        if not is_global_shortcut:
            return False
        if event.isAutoRepeat():
            event.accept()
            return True
        if key == Qt.Key.Key_Escape:
            return False
        if (
            key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}
            and modifiers == Qt.KeyboardModifier.ControlModifier
        ):
            if self.current_page() is self.preflight_page and self._preflight_can_start:
                self.start_requested.emit()
            event.accept()
            return True
        if key == Qt.Key.Key_Space and modifiers == Qt.KeyboardModifier.NoModifier:
            if self._space_is_blocked_by_focus():
                return False
            self.pause_resume_requested.emit()
            event.accept()
            return True
        if key == Qt.Key.Key_T and modifiers == Qt.KeyboardModifier.ControlModifier:
            next_value = not self._always_on_top
            self.set_always_on_top(next_value)
            self.always_on_top_changed.emit(next_value)
            event.accept()
            return True
        if key == Qt.Key.Key_S and modifiers == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ):
            self.stop_requested.emit()
            event.accept()
            return True
        return False

    @staticmethod
    def _is_global_shortcut(
        key: int,
        modifiers: Qt.KeyboardModifier,
    ) -> bool:
        return (
            (
                key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}
                and modifiers == Qt.KeyboardModifier.ControlModifier
            )
            or (key == Qt.Key.Key_Space and modifiers == Qt.KeyboardModifier.NoModifier)
            or (
                key == Qt.Key.Key_T and modifiers == Qt.KeyboardModifier.ControlModifier
            )
            or (
                key == Qt.Key.Key_S
                and modifiers
                == (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.ShiftModifier
                )
            )
            or (key == Qt.Key.Key_Escape)
        )

    def _space_is_blocked_by_focus(self) -> bool:
        popup = QApplication.activePopupWidget()
        if popup is not None and self._popup_belongs_to_window(popup):
            return True
        widget = QApplication.focusWidget()
        if widget is None or widget.window() is not self:
            return False
        blocked_types = (
            QAbstractSpinBox,
            QComboBox,
            QLineEdit,
            QMenu,
            QPlainTextEdit,
            QTextEdit,
        )
        while widget is not None:
            if isinstance(widget, blocked_types) and widget.isVisible():
                return True
            widget = widget.parentWidget()
        return False

    def _popup_belongs_to_window(self, popup: QWidget) -> bool:
        parent = popup.parentWidget()
        return (
            popup.window() is self
            or parent is self
            or (parent is not None and parent.window() is self)
        )

    def _sync_always_on_top_toggle(self) -> None:
        toggle = self.live_page.always_on_top_toggle
        previous = toggle.blockSignals(True)
        try:
            toggle.setChecked(self._always_on_top)
        finally:
            toggle.blockSignals(previous)

    def stop_dialog_confirmed(self) -> None:
        """Forward an explicit stop confirmation without duplicate signals."""

        self.stop_confirmed_requested.emit()

    def stop_dialog_cancelled(self) -> None:
        """Forward the non-destructive stop-dialog choice."""

        self.stop_cancel_requested.emit()

    def slow_finalization_keep_waiting(self) -> None:
        """Forward the non-destructive slow-finalization choice."""

        self.keep_waiting_requested.emit()

    def slow_finalization_force_close(self) -> None:
        """Forward the explicit force-close choice."""

        self.force_close_requested.emit()


def _intersection_area(first: QRect, second: QRect) -> int:
    width = max(
        0,
        min(first.right(), second.right()) - max(first.left(), second.left()),
    )
    height = max(
        0,
        min(first.bottom(), second.bottom()) - max(first.top(), second.top()),
    )
    return width * height
