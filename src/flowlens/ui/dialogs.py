"""Stop and slow-finalization dialogs."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout

from flowlens.ui.widgets import StatefulButton


class StopConfirmationDialog(QDialog):
    """Ask for explicit confirmation before ending live capture."""

    stop_confirmed = Signal()
    keep_recording_requested = Signal()

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.message_label = QLabel("Stop this session?")
        self.confirm_button = StatefulButton("Stop and finalize")
        self.cancel_button = StatefulButton("Keep recording")
        self._choice_emitted = False
        self._build_layout()
        self._connect()

    def text(self) -> str:
        """Return the primary dialog message."""

        return self.message_label.text()

    def _build_layout(self) -> None:
        self.setModal(True)
        self.setWindowTitle("Stop session")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        actions = QHBoxLayout()
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.confirm_button)
        layout.addLayout(actions)
        self.cancel_button.setDefault(True)
        self.confirm_button.setProperty("uiState", "error")

    def _connect(self) -> None:
        self.confirm_button.clicked.connect(self._choose_stop)
        self.cancel_button.clicked.connect(self._choose_keep_recording)

    def _choose_stop(self) -> None:
        self._choice_emitted = True
        self.stop_confirmed.emit()
        self.accept()

    def _choose_keep_recording(self) -> None:
        self._emit_keep_recording_default()
        self.reject()

    def _emit_keep_recording_default(self) -> None:
        if self._choice_emitted:
            return
        self._choice_emitted = True
        self.keep_recording_requested.emit()

    def reject(self) -> None:
        """Treat Escape as the non-destructive keep-recording choice."""

        self._emit_keep_recording_default()
        super().reject()

    def showEvent(self, event: QShowEvent) -> None:
        """Reset completion state each time the dialog is shown."""

        self._choice_emitted = False
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Treat window close as the non-destructive keep-recording choice."""

        self._emit_keep_recording_default()
        super().closeEvent(event)


class SlowFinalizationDialog(QDialog):
    """Offer the slow-finalization choice without selecting force close."""

    keep_waiting_requested = Signal()
    force_close_requested = Signal()

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.message_label = QLabel("Finalization is taking longer than expected")
        self.keep_waiting_button = StatefulButton("Keep waiting")
        self.force_close_button = StatefulButton("Force close")
        self._choice_emitted = False
        self._build_layout()
        self._connect()

    def text(self) -> str:
        """Return the primary dialog message."""

        return self.message_label.text()

    def _build_layout(self) -> None:
        self.setModal(True)
        self.setWindowTitle("Finalization")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        actions = QHBoxLayout()
        actions.addWidget(self.keep_waiting_button)
        actions.addWidget(self.force_close_button)
        layout.addLayout(actions)
        self.keep_waiting_button.setDefault(True)
        self.force_close_button.setDefault(False)
        self.force_close_button.setProperty("uiState", "error")

    def _connect(self) -> None:
        self.keep_waiting_button.clicked.connect(self._choose_keep_waiting)
        self.force_close_button.clicked.connect(self._choose_force_close)

    def _choose_keep_waiting(self) -> None:
        self._emit_keep_waiting_default()
        self.reject()

    def _choose_force_close(self) -> None:
        self._choice_emitted = True
        self.force_close_requested.emit()
        self.accept()

    def _emit_keep_waiting_default(self) -> None:
        if self._choice_emitted:
            return
        self._choice_emitted = True
        self.keep_waiting_requested.emit()

    def reject(self) -> None:
        """Treat Escape as the non-destructive keep-waiting choice."""

        self._emit_keep_waiting_default()
        super().reject()

    def showEvent(self, event: QShowEvent) -> None:
        """Reset completion state each time the dialog is shown."""

        self._choice_emitted = False
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Treat window close as the non-destructive keep-waiting choice."""

        self._emit_keep_waiting_default()
        super().closeEvent(event)
