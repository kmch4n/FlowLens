"""Live session page for transcript, discussion, and status state."""

from datetime import datetime
from typing import ClassVar

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from flowlens.controller.session_controller import ControllerSnapshot, SessionState
from flowlens.domain.enums import AudioSource, SessionMode
from flowlens.domain.messages import TranscriptRecord
from flowlens.ui.discussion_panel import DiscussionPanel, labels_for
from flowlens.ui.status_strip import StatusSnapshot, StatusStrip
from flowlens.ui.transcript_model import ImmutableTranscriptError
from flowlens.ui.transcript_view import TranscriptView
from flowlens.ui.widgets import StatefulButton


class LivePage(QWidget):
    """Render a controller snapshot without owning lifecycle rules."""

    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    always_on_top_changed = Signal(bool)

    _MODE_LABELS: ClassVar[dict[SessionMode, str]] = {
        SessionMode.MEETING: "Meeting / discussion",
        SessionMode.INTERVIEW: "Interview",
        SessionMode.GENERAL: "General conversation",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_state = SessionState.RECORDING
        self._known_segments: set[str] = set()
        self.product_label = QLabel("FlowLens")
        self.mode_label = QLabel()
        self.recording_state = QLabel()
        self.elapsed_timer = QLabel("00:00")
        self.pause_resume_button = StatefulButton("Pause")
        self.stop_button = StatefulButton("Stop")
        self.always_on_top_toggle = QCheckBox("Always on top")
        self.banner = QLabel()
        self.transcript_view = TranscriptView()
        self.discussion_panel = DiscussionPanel()
        self.status_strip = StatusStrip()
        self.main_splitter = QSplitter()
        self.desktop_slot = QWidget()
        self.narrow_scroll_area = QScrollArea()
        self.narrow_content = QWidget()
        self._build_layout()
        self._configure()
        self._connect()
        self.reflow()

    def render(self, snapshot: ControllerSnapshot) -> None:  # type: ignore[override]
        """Render one immutable controller snapshot."""

        if not isinstance(snapshot, ControllerSnapshot):
            raise TypeError("snapshot must be a ControllerSnapshot")
        self._current_state = snapshot.state
        mode = self._snapshot_mode(snapshot)
        self.mode_label.setText(self._MODE_LABELS[mode])
        self.recording_state.setText(snapshot.recording_status)
        self.pause_resume_button.setText(
            "Resume" if snapshot.state is SessionState.PAUSED else "Pause"
        )
        self._render_banner(snapshot)
        self._render_transcript(snapshot)
        if snapshot.discussion_state is not None:
            self.discussion_panel.render(snapshot.discussion_state, labels_for(mode))
        self.status_strip.render(
            StatusSnapshot(
                microphone_level=snapshot.microphone_level,
                loopback_level=snapshot.loopback_level,
                asr=snapshot.asr_status,
                delay_ms=snapshot.asr_backlog_ms,
                analysis=snapshot.analysis_status,
                saved=self._format_saved(snapshot.latest_successful_save_at),
            )
        )

    def reflow(self) -> None:
        """Apply the exact desktop/narrow breakpoint layout."""

        narrow = self.width() < 1000
        self.main_splitter.setOrientation(
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        )
        if narrow:
            self._place_splitter(self.narrow_content.layout())
            self.desktop_slot.hide()
            self.narrow_scroll_area.show()
            self.main_splitter.setSizes([340, 300])
        else:
            self._place_splitter(self.desktop_slot.layout())
            self.narrow_scroll_area.hide()
            self.desktop_slot.show()
            total = max(1, self.width() - 64)
            self.main_splitter.setSizes([round(total * 0.62), round(total * 0.38)])

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the splitter orientation synchronized with window width."""

        self.reflow()
        super().resizeEvent(event)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        root.addWidget(self._top_bar())
        root.addWidget(self.banner)
        desktop_layout = QVBoxLayout(self.desktop_slot)
        desktop_layout.setContentsMargins(0, 0, 0, 0)
        desktop_layout.setSpacing(0)
        self.main_splitter.addWidget(self.transcript_view)
        self.main_splitter.addWidget(self.discussion_panel)
        desktop_layout.addWidget(self.main_splitter)
        root.addWidget(self.desktop_slot, 1)
        narrow_layout = QVBoxLayout(self.narrow_content)
        narrow_layout.setContentsMargins(0, 0, 0, 0)
        narrow_layout.setSpacing(0)
        self.narrow_scroll_area.setWidget(self.narrow_content)
        root.addWidget(self.narrow_scroll_area, 1)
        root.addWidget(self.status_strip)

    def _top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setProperty("flowlensRole", "rail")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        for label in (
            self.product_label,
            self.mode_label,
            self.recording_state,
            self.elapsed_timer,
        ):
            layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(self.pause_resume_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.always_on_top_toggle)
        return frame

    def _configure(self) -> None:
        self.setMinimumSize(900, 600)
        self.setProperty("flowlensRole", "canvas")
        self.product_label.setProperty("flowlensRole", "metric")
        self.elapsed_timer.setProperty("flowlensRole", "timer")
        self.banner.setMinimumHeight(28)
        self.banner.setWordWrap(False)
        self.banner.setProperty("flowlensRole", "helper")
        self.stop_button.set_ui_state("error", "Stop and finalize the session")
        self.always_on_top_toggle.setMinimumSize(44, 44)
        self.always_on_top_toggle.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.narrow_scroll_area.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.narrow_scroll_area.setWidgetResizable(True)
        self.narrow_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.narrow_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

    def _connect(self) -> None:
        self.pause_resume_button.clicked.connect(self._request_pause_or_resume)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.always_on_top_toggle.toggled.connect(self.always_on_top_changed.emit)

    def _request_pause_or_resume(self) -> None:
        if self._current_state is SessionState.PAUSED:
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def _render_banner(self, snapshot: ControllerSnapshot) -> None:
        text = (
            snapshot.fatal_error
            or snapshot.issue
            or (
                "Finalization is taking longer than expected"
                if snapshot.slow_finalization_visible
                else ""
            )
        )
        self.banner.setText(text)
        self.banner.setAccessibleDescription(text or "No current live-session issue")
        self.banner.setProperty("uiState", "error" if text else "default")

    def _render_transcript(self, snapshot: ControllerSnapshot) -> None:
        for record in snapshot.transcript:
            if record.segment_id in self._known_segments:
                self._validate_known_segment(record)
                continue
            self.transcript_view.model.commit(record)
            self._known_segments.add(record.segment_id)
        active_sources = {
            partial.source for partial in snapshot.partials if partial.text
        }
        for partial in snapshot.partials:
            self.transcript_view.model.set_partial(partial.source, partial)
        for source in (AudioSource.ME, AudioSource.OTHERS):
            if source not in active_sources:
                self.transcript_view.model.clear_partial(source)
        self.transcript_view.sync_partials()

    def _validate_known_segment(self, record: TranscriptRecord) -> None:
        for existing in self.transcript_view.model.records():
            if existing.segment_id == record.segment_id:
                if existing != record:
                    raise ImmutableTranscriptError(
                        "committed transcript cannot be replaced"
                    )
                return
        raise ImmutableTranscriptError("known transcript segment is missing")

    def _place_splitter(self, target_layout: QLayout | None) -> None:
        if target_layout is None:
            raise RuntimeError("target layout is unavailable")
        if self.main_splitter.parentWidget() is target_layout.parentWidget():
            return
        self.main_splitter.setParent(None)
        target_layout.addWidget(self.main_splitter)

    def _snapshot_mode(self, snapshot: ControllerSnapshot) -> SessionMode:
        if snapshot.preflight is not None:
            return snapshot.preflight.selection.mode
        if snapshot.discussion_state is not None:
            return snapshot.discussion_state.mode
        return SessionMode.MEETING

    @staticmethod
    def _format_saved(value: datetime | None) -> str:
        if value is None:
            return "Not saved yet"
        return value.strftime("%H:%M:%S")
