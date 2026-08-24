"""Preflight controls for the FlowLens desktop workbench."""

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from flowlens.controller.models import DeviceOption, PreflightReport, PreflightSelection
from flowlens.domain.enums import SessionMode
from flowlens.ui.widgets import InputMeter, StatefulButton


class TooltipHelper(QObject):
    """Show control tooltips immediately on focus and after hover delay."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._target: QWidget | None = None
        self._timer.timeout.connect(self._show_hover_tooltip)

    def attach(self, widget: QWidget) -> None:
        """Install the common focus and hover tooltip behavior on one control."""

        widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Display the watched control's tooltip at the specified interaction time."""

        if not isinstance(watched, QWidget) or not watched.toolTip():
            return super().eventFilter(watched, event)
        if event.type() is QEvent.Type.FocusIn:
            self._show_tooltip(watched)
        elif event.type() is QEvent.Type.Enter:
            self._target = watched
            self._timer.start(800)
        elif event.type() in {QEvent.Type.Leave, QEvent.Type.FocusOut}:
            if self._target is watched:
                self._timer.stop()
                self._target = None
            QToolTip.hideText()
        return super().eventFilter(watched, event)

    def _show_hover_tooltip(self) -> None:
        if self._target is not None:
            self._show_tooltip(self._target)

    @staticmethod
    def _show_tooltip(widget: QWidget) -> None:
        position = widget.mapToGlobal(QPoint(widget.width() // 2, widget.height()))
        QToolTip.showText(position, widget.toolTip(), widget)


class PreflightPage(QWidget):
    """Render immutable preflight readiness and emit complete user selections."""

    selection_changed = Signal(PreflightSelection)
    start_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selection = PreflightSelection(SessionMode.MEETING, None, None)
        self._can_start = False
        self._rendering = False
        self._tooltips = TooltipHelper(self)

        self.meeting_radio = QRadioButton("Meeting / discussion")
        self.interview_radio = QRadioButton("Interview")
        self.general_radio = QRadioButton("General conversation")
        self.microphone_combo = QComboBox()
        self.loopback_combo = QComboBox()
        self.mic_meter = InputMeter("Microphone")
        self.loopback_meter = InputMeter("PC audio")
        self.model_status = QLabel()
        self.storage_status = QLabel()
        self.destination_summary = QLabel()
        self.start_button = StatefulButton("Start session")

        self.mode_error = self._stable_error_label()
        self.microphone_error = self._stable_error_label()
        self.loopback_error = self._stable_error_label()
        self.model_error = self._stable_error_label()
        self.storage_error = self._stable_error_label()

        self._build_layout()
        self._configure_controls()
        self._connect_signals()

    def render(self, report: PreflightReport) -> None:  # type: ignore[override]
        """Render one complete immutable preflight report without emitting changes."""

        if not isinstance(report, PreflightReport):
            raise TypeError("report must be a PreflightReport")
        self._rendering = True
        try:
            self._selection = report.selection
            self._set_mode(report.selection.mode)
            self._populate_devices(
                self.microphone_combo,
                report.microphones,
                report.selection.microphone_id,
            )
            self._populate_devices(
                self.loopback_combo,
                report.loopbacks,
                report.selection.loopback_output_id,
            )
            self.mic_meter.set_level(report.mic_level)
            self.loopback_meter.set_level(report.loopback_level)
            self._render_readiness(report)
            self._render_issues(report)
            self.destination_summary.setText(
                f"Sessions are saved to {report.destination}"
            )
            self.destination_summary.setAccessibleDescription(
                f"Session destination: {report.destination}"
            )
            self._can_start = report.can_start
            self.start_button.set_ui_state(
                "default" if report.can_start else "disabled",
                "Start session" if report.can_start else "Resolve the listed blockers",
            )
        finally:
            self._rendering = False

    def focus_chain(self) -> list[QWidget]:
        """Return the intentional keyboard sequence for preflight controls."""

        return [
            self.meeting_radio,
            self.interview_radio,
            self.general_radio,
            self.microphone_combo,
            self.loopback_combo,
            self.start_button,
        ]

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Request start through Ctrl+Enter only when preflight is valid."""

        if (
            event.key() == Qt.Key.Key_Enter
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            if self._can_start:
                self.start_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)

        title = QLabel("Preflight")
        title.setProperty("flowlensRole", "metric")
        layout.addWidget(title)
        layout.addWidget(self._separator())

        layout.addWidget(QLabel("Session mode"))
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(12)
        for radio in self._mode_radios():
            mode_layout.addWidget(radio)
        mode_layout.addStretch(1)
        layout.addLayout(mode_layout)
        layout.addWidget(self.mode_error)

        layout.addWidget(self._separator())
        layout.addWidget(QLabel("Microphone"))
        layout.addWidget(self.microphone_combo)
        layout.addWidget(self.microphone_error)
        layout.addWidget(QLabel("Microphone activity"))
        layout.addWidget(self.mic_meter)

        layout.addWidget(self._separator())
        layout.addWidget(QLabel("PC audio output"))
        layout.addWidget(self.loopback_combo)
        layout.addWidget(self.loopback_error)
        layout.addWidget(QLabel("PC audio activity"))
        layout.addWidget(self.loopback_meter)

        layout.addWidget(self._separator())
        layout.addWidget(QLabel("Local models"))
        layout.addWidget(self.model_status)
        layout.addWidget(self.model_error)
        layout.addWidget(QLabel("Storage"))
        layout.addWidget(self.storage_status)
        layout.addWidget(self.storage_error)
        layout.addWidget(self.destination_summary)
        layout.addStretch(1)
        layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignRight)

    def _configure_controls(self) -> None:
        self.setProperty("flowlensRole", "canvas")
        self.mic_meter.setAccessibleName("Microphone activity")
        self.loopback_meter.setAccessibleName("PC audio activity")
        self.microphone_combo.setMinimumHeight(44)
        self.loopback_combo.setMinimumHeight(44)
        for radio in self._mode_radios():
            radio.setMinimumHeight(44)
            radio.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.microphone_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.loopback_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.start_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.meeting_radio.setToolTip("Choose the meeting or discussion mode.")
        self.interview_radio.setToolTip("Choose the interview mode.")
        self.general_radio.setToolTip("Choose the general conversation mode.")
        self.microphone_combo.setToolTip("Choose the microphone for your voice.")
        self.loopback_combo.setToolTip("Choose the Windows output device for PC audio.")
        self.start_button.setToolTip("Start a session after all checks are ready.")
        controls = (
            *self._mode_radios(),
            self.microphone_combo,
            self.loopback_combo,
            self.start_button,
        )
        for control in controls:
            self._tooltips.attach(control)

        group = QButtonGroup(self)
        group.addButton(self.meeting_radio)
        group.addButton(self.interview_radio)
        group.addButton(self.general_radio)
        self._mode_group = group
        self.meeting_radio.setChecked(True)
        QWidget.setTabOrder(self.meeting_radio, self.interview_radio)
        QWidget.setTabOrder(self.interview_radio, self.general_radio)
        QWidget.setTabOrder(self.general_radio, self.microphone_combo)
        QWidget.setTabOrder(self.microphone_combo, self.loopback_combo)
        QWidget.setTabOrder(self.loopback_combo, self.start_button)

    def _connect_signals(self) -> None:
        for radio in self._mode_radios():
            radio.toggled.connect(self._on_mode_toggled)
        self.microphone_combo.currentIndexChanged.connect(self._on_microphone_changed)
        self.loopback_combo.currentIndexChanged.connect(self._on_loopback_changed)
        self.start_button.clicked.connect(self._request_start)

    def _render_readiness(self, report: PreflightReport) -> None:
        if all(check.ready for check in report.models):
            self.model_status.setText("Local models ready")
            self.model_status.setProperty("uiState", "success")
        else:
            self.model_status.setText("Local models need attention")
            self.model_status.setProperty("uiState", "error")
        storage_ready = (
            report.storage.writable and report.storage.free_bytes >= 500 * 1024 * 1024
        )
        if storage_ready:
            self.storage_status.setText("Storage ready: 500 MB minimum satisfied")
            self.storage_status.setProperty("uiState", "success")
        else:
            self.storage_status.setText("Storage needs attention")
            self.storage_status.setProperty("uiState", "error")

    def _render_issues(self, report: PreflightReport) -> None:
        error_slots = (
            (self.mode_error, ("mode",)),
            (self.microphone_error, ("microphone",)),
            (self.loopback_error, ("loopback",)),
            (self.model_error, ("asr_model", "discussion_model")),
            (self.storage_error, ("storage",)),
        )
        for label, control_ids in error_slots:
            text = " ".join(
                issue.message
                for issue in report.issues
                if issue.control_id in control_ids
            )
            label.setText(text)
            label.setAccessibleDescription(text or "No blocking issue")
            label.setProperty("uiState", "error" if text else "default")

    def _set_mode(self, mode: SessionMode) -> None:
        radio = {
            SessionMode.MEETING: self.meeting_radio,
            SessionMode.INTERVIEW: self.interview_radio,
            SessionMode.GENERAL: self.general_radio,
        }[mode]
        radio.setChecked(True)

    @staticmethod
    def _populate_devices(
        combo: QComboBox,
        devices: tuple[DeviceOption, ...],
        selected_id: str | None,
    ) -> None:
        combo.clear()
        for device in devices:
            combo.addItem(device.display_name, device.id)
        selected_index = combo.findData(selected_id, Qt.ItemDataRole.UserRole)
        combo.setCurrentIndex(selected_index if selected_index >= 0 else -1)

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked or self._rendering:
            return
        for radio, mode in self._mode_mapping().items():
            if radio.isChecked():
                self._selection = PreflightSelection(
                    mode,
                    self._selection.microphone_id,
                    self._selection.loopback_output_id,
                )
                self.selection_changed.emit(self._selection)
                return

    def _on_microphone_changed(self, _: int) -> None:
        if self._rendering:
            return
        self._selection = PreflightSelection(
            self._selection.mode,
            self._selected_device_id(self.microphone_combo),
            self._selection.loopback_output_id,
        )
        self.selection_changed.emit(self._selection)

    def _on_loopback_changed(self, _: int) -> None:
        if self._rendering:
            return
        self._selection = PreflightSelection(
            self._selection.mode,
            self._selection.microphone_id,
            self._selected_device_id(self.loopback_combo),
        )
        self.selection_changed.emit(self._selection)

    def _request_start(self) -> None:
        if self._can_start:
            self.start_requested.emit()

    @staticmethod
    def _selected_device_id(combo: QComboBox) -> str | None:
        value = combo.currentData(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else None

    @staticmethod
    def _stable_error_label() -> QLabel:
        label = QLabel()
        label.setMinimumHeight(label.fontMetrics().height())
        label.setProperty("flowlensRole", "helper")
        label.setWordWrap(False)
        return label

    @staticmethod
    def _separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        return separator

    def _mode_radios(self) -> tuple[QRadioButton, QRadioButton, QRadioButton]:
        return self.meeting_radio, self.interview_radio, self.general_radio

    def _mode_mapping(self) -> dict[QRadioButton, SessionMode]:
        return {
            self.meeting_radio: SessionMode.MEETING,
            self.interview_radio: SessionMode.INTERVIEW,
            self.general_radio: SessionMode.GENERAL,
        }
