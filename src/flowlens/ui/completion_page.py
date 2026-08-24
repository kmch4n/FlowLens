"""Completion screen for a finalized local session."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from flowlens.controller.models import CompletionSummary
from flowlens.ui.widgets import StatefulButton

__all__ = ["CompletionPage", "CompletionSummary"]


class CompletionPage(QWidget):
    """Render the MVP completion summary and three permitted actions."""

    open_folder_requested = Signal()
    start_another_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.duration_value = QLabel()
        self.transcript_count_value = QLabel()
        self.path_value = QLabel()
        self.open_folder_button = StatefulButton("Open folder")
        self.start_another_button = StatefulButton("Start another session")
        self.close_button = StatefulButton("Close")
        self._build_layout()
        self._connect()

    def render(self, summary: CompletionSummary) -> None:  # type: ignore[override]
        """Render one complete session summary."""

        self.duration_value.setText(_format_duration(summary.duration_ms))
        self.transcript_count_value.setText(str(summary.transcript_count))
        self.path_value.setText(str(summary.save_path))
        self.path_value.setAccessibleDescription(
            f"Session saved to {summary.save_path}"
        )

    def action_labels(self) -> list[str]:
        """Return the visible MVP completion action labels."""

        return [
            self.open_folder_button.text(),
            self.start_another_button.text(),
            self.close_button.text(),
        ]

    def _build_layout(self) -> None:
        self.setProperty("flowlensRole", "canvas")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)
        title = QLabel("Session complete")
        title.setProperty("flowlensRole", "metric")
        layout.addWidget(title)
        summary_frame = QFrame()
        summary_frame.setProperty("flowlensRole", "workArea")
        grid = QGridLayout(summary_frame)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)
        self._add_summary_row(grid, 0, "Duration", self.duration_value)
        self._add_summary_row(
            grid, 1, "Transcript entries", self.transcript_count_value
        )
        self._add_summary_row(grid, 2, "Save path", self.path_value)
        layout.addWidget(summary_frame)
        layout.addStretch(1)
        for button in (
            self.open_folder_button,
            self.start_another_button,
            self.close_button,
        ):
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignRight)

    def _connect(self) -> None:
        self.open_folder_button.clicked.connect(self.open_folder_requested.emit)
        self.start_another_button.clicked.connect(self.start_another_requested.emit)
        self.close_button.clicked.connect(self.close_requested.emit)

    @staticmethod
    def _add_summary_row(
        grid: QGridLayout,
        row: int,
        label_text: str,
        value_label: QLabel,
    ) -> None:
        label = QLabel(label_text)
        label.setProperty("flowlensTone", "muted")
        value_label.setProperty("flowlensRole", "metric")
        value_label.setWordWrap(True)
        grid.addWidget(label, row, 0)
        grid.addWidget(value_label, row, 1)


def _format_duration(duration_ms: int) -> str:
    total_seconds = duration_ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"
