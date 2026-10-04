"""Small, locally persisted recognition and readability settings."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from flowlens.config.user_settings import SettingsStore, UserSettings


class SettingsDialog(QDialog):
    """Keep edits local until Save succeeds; cancellation has no side effects."""

    saved = Signal(UserSettings)

    def __init__(self, store: SettingsStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        current, warning = store.load_for_use()
        self.setWindowTitle("FlowLens settings")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Recognition & readability")
        title.setProperty("flowlensRole", "sectionTitle")
        layout.addWidget(title)
        explanation = QLabel(
            "Recognition changes apply to the next session.\n"
            "Lower sensitivity ignores quieter sounds, including quiet speech."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.sensitivity = QComboBox()
        for label, value in (
            ("Low — reduce background noise", "low"),
            ("Standard", "normal"),
            ("High — quiet voices", "high"),
        ):
            self.sensitivity.addItem(label, value)
        self.sensitivity.setCurrentIndex(self.sensitivity.findData(current.sensitivity))
        self.silence = QSpinBox()
        self.silence.setRange(300, 1200)
        self.silence.setSingleStep(50)
        self.silence.setSuffix(" ms")
        self.silence.setValue(current.silence_end_ms)
        self.text_size = QSpinBox()
        self.text_size.setRange(14, 22)
        self.text_size.setSuffix(" px")
        self.text_size.setValue(current.text_size)
        form = QFormLayout()
        form.setSpacing(16)
        form.addRow("Speech sensitivity", self.sensitivity)
        form.addRow("Wait after speech", self.silence)
        form.addRow("Transcript text size", self.text_size)
        for widget in (self.sensitivity, self.silence, self.text_size):
            widget.setMinimumHeight(36)
        layout.addLayout(form)
        self.error = QLabel()
        if warning:
            self.error.setText(warning)
        self.error.setWordWrap(True)
        self.error.setProperty("uiState", "error")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self) -> None:
        settings = UserSettings(
            str(self.sensitivity.currentData()),
            self.silence.value(),
            self.text_size.value(),
        )
        try:
            self._store.save(settings)
        except (OSError, ValueError) as error:
            self.error.setText(f"Could not save settings: {error}")
            return
        self.saved.emit(settings)
        self.accept()
