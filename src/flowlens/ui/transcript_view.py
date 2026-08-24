"""Transcript view with explicit auto-scroll control."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QListView,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from flowlens.domain.enums import AudioSource
from flowlens.ui.transcript_model import TranscriptListModel
from flowlens.ui.widgets import StatefulButton


class TranscriptView(QFrame):
    """Render committed transcript rows and source-specific ephemeral partial rows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = TranscriptListModel()
        self.auto_scroll_enabled = True
        self.list_view = QListView()
        self.partial_labels = {
            AudioSource.ME: QLabel(),
            AudioSource.OTHERS: QLabel(),
        }
        self.partial_label = self.partial_labels[AudioSource.OTHERS]
        self.return_to_latest_button = StatefulButton("Return to latest")
        self._programmatic_scroll = False

        self._build_layout()
        self._configure()
        self._connect()

    def return_to_latest(self) -> None:
        """Restore automatic scrolling and move to the newest committed row."""

        self.auto_scroll_enabled = True
        self.return_to_latest_button.hide()
        self._scroll_to_maximum()
        self._queue_scroll_to_maximum()

    def source_labels(self) -> list[str]:
        """Return visible source labels in committed-row order plus partial."""

        labels = [
            self.model.source_label(self.model.row(index).source)
            for index in range(self.model.rowCount())
        ]
        for source in (AudioSource.ME, AudioSource.OTHERS):
            partial = self.model.partial(source)
            if partial is not None:
                labels.append(self.model.source_label(source))
        return labels

    def rendered_text(self) -> str:
        """Return all user-visible transcript text without timestamp fields."""

        rows = [
            str(
                self.model.data(
                    self.model.index(index, 0), int(Qt.ItemDataRole.DisplayRole)
                )
            )
            for index in range(self.model.rowCount())
        ]
        for label in self.partial_labels.values():
            if label.text() and label.isVisibleTo(self):
                rows.append(label.text())
        return "\n".join(rows)

    def scrollbar(self) -> QScrollBar:
        """Return the committed-list vertical scrollbar."""

        return self.list_view.verticalScrollBar()

    def sync_partials(self) -> None:
        """Refresh the ephemeral partial label from the current model state."""

        for source, label in self.partial_labels.items():
            partial = self.model.partial(source)
            if partial is not None:
                source_label = self.model.source_label(source)
                label.setText(f"{source_label} · Partial · {partial.text}")
                label.setAccessibleDescription(
                    f"Partial transcript from {source.value}"
                )
                label.show()
            else:
                label.clear()
                label.setAccessibleDescription(
                    f"No partial transcript from {source.value}"
                )
                label.hide()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        title = QLabel("Transcript")
        title.setProperty("flowlensRole", "metric")
        layout.addWidget(title)
        layout.addWidget(self.list_view, 1)
        for label in self.partial_labels.values():
            layout.addWidget(label)
        layout.addWidget(
            self.return_to_latest_button,
            alignment=Qt.AlignmentFlag.AlignRight,
        )

    def _configure(self) -> None:
        self.setProperty("flowlensRole", "workAreaPrimary")
        self.setMinimumSize(360, 240)
        self.list_view.setModel(self.model)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setWordWrap(True)
        self.list_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.list_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for label in self.partial_labels.values():
            label.setProperty("flowlensTone", "muted")
            label.setWordWrap(True)
            label.hide()
        self.return_to_latest_button.hide()

    def _connect(self) -> None:
        self.model.modelReset.connect(self._on_rows_changed)
        self.model.rowsInserted.connect(self._on_rows_changed)
        self.model.partials_changed.connect(self.sync_partials)
        self.return_to_latest_button.clicked.connect(self.return_to_latest)
        self.list_view.verticalScrollBar().valueChanged.connect(self._on_scroll_value)
        self.list_view.verticalScrollBar().rangeChanged.connect(self._on_scroll_range)

    def _on_rows_changed(self, *_: object) -> None:
        self.sync_partials()
        if self.auto_scroll_enabled:
            self._queue_scroll_to_maximum()

    def _on_scroll_range(self, _: int, maximum: int) -> None:
        if self.auto_scroll_enabled:
            self._queue_scroll_to_maximum()
        elif self.scrollbar().value() < maximum:
            self.return_to_latest_button.show()

    def _on_scroll_value(self, value: int) -> None:
        if self._programmatic_scroll:
            return
        maximum = self.scrollbar().maximum()
        if value < maximum:
            self.auto_scroll_enabled = False
            self.return_to_latest_button.show()
        elif self.auto_scroll_enabled:
            self.return_to_latest_button.hide()

    def _scroll_to_maximum(self) -> None:
        self._programmatic_scroll = True
        try:
            self.scrollbar().setValue(self.scrollbar().maximum())
        finally:
            self._programmatic_scroll = False
        if self.auto_scroll_enabled:
            self.return_to_latest_button.hide()

    def _queue_scroll_to_maximum(self) -> None:
        QTimer.singleShot(0, self._scroll_to_maximum_if_enabled)

    def _scroll_to_maximum_if_enabled(self) -> None:
        if not self.auto_scroll_enabled:
            return
        try:
            self._scroll_to_maximum()
        except RuntimeError:
            return
