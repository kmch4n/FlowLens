"""Discussion-state panel for the live workbench."""

from typing import ClassVar

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import SessionMode

DiscussionLabels = tuple[str, str, str, str]


def labels_for(mode: SessionMode) -> DiscussionLabels:
    """Return fixed section labels for one session mode."""

    return {
        SessionMode.MEETING: (
            "Current focus",
            "Key points",
            "Decisions / confirmations",
            "Unresolved / next actions",
        ),
        SessionMode.INTERVIEW: (
            "Current question / topic",
            "Answer highlights",
            "Confirmed content",
            "Follow-ups / points to clarify",
        ),
        SessionMode.GENERAL: (
            "Current topic",
            "Key points",
            "Confirmed items",
            "Items to revisit",
        ),
    }[mode]


class DiscussionPanel(QWidget):
    """Render a conservative complete discussion-state snapshot."""

    _EMPTY_EXPLANATIONS: ClassVar[dict[SessionMode, DiscussionLabels]] = {
        SessionMode.MEETING: (
            "No current meeting focus has been confirmed yet.",
            "Key points will appear after committed speech is analyzed.",
            "Confirmed decisions remain visible here.",
            "Open next actions remain visible here.",
        ),
        SessionMode.INTERVIEW: (
            "No current question has been confirmed yet.",
            "Answer highlights will appear after committed speech is analyzed.",
            "Confirmed content remains visible here.",
            "Follow-ups and clarification points remain visible here.",
        ),
        SessionMode.GENERAL: (
            "No current topic has been confirmed yet.",
            "Key points will appear after committed speech is analyzed.",
            "Confirmed items remain visible here.",
            "Items to revisit remain visible here.",
        ),
    }

    def __init__(
        self, parent: QWidget | None = None, reduced_motion: bool = False
    ) -> None:
        super().__init__(parent)
        self._reduced_motion = reduced_motion
        self._title_labels: list[QLabel] = []
        self._body_labels: list[QLabel] = []
        self._animations: list[QPropertyAnimation] = []
        self._last_texts: tuple[str, ...] = ()
        self._build_layout()

    def render(  # type: ignore[override]
        self, state: DiscussionState, labels: DiscussionLabels
    ) -> None:
        """Render one complete state with caller-provided labels."""

        if not isinstance(state, DiscussionState):
            raise TypeError("state must be a DiscussionState")
        if type(labels) is not tuple or len(labels) != 4:
            raise ValueError("labels must contain exactly four section labels")
        values = (
            state.current_focus,
            "\n".join(state.key_points),
            "\n".join(state.confirmed_outcomes),
            "\n".join(state.follow_up_items),
        )
        empty = self._EMPTY_EXPLANATIONS[state.mode]
        rendered = tuple(
            value if value else empty[index] for index, value in enumerate(values)
        )
        for index, title in enumerate(labels):
            self._title_labels[index].setText(title)
            self._body_labels[index].setText(rendered[index])
            self._body_labels[index].setAccessibleDescription(rendered[index])
        if self._last_texts and self._last_texts != rendered:
            self._animate_bodies()
        self._last_texts = rendered

    def section_titles(self) -> tuple[str, ...]:
        """Return section titles in visual order."""

        return tuple(label.text() for label in self._title_labels)

    def empty_explanations(self) -> tuple[str, ...]:
        """Return the currently visible section body text."""

        return tuple(label.text() for label in self._body_labels)

    def _build_layout(self) -> None:
        self.setProperty("flowlensRole", "workArea")
        self.setMinimumSize(280, 240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        heading = QLabel("Discussion state")
        heading.setProperty("flowlensRole", "metric")
        layout.addWidget(heading)
        for _ in range(4):
            layout.addWidget(self._separator())
            title = QLabel()
            title.setProperty("flowlensRole", "metric")
            body = QLabel()
            body.setWordWrap(True)
            body.setProperty("flowlensTone", "muted")
            self._title_labels.append(title)
            self._body_labels.append(body)
            layout.addWidget(title)
            layout.addWidget(body)
        layout.addStretch(1)

    def _animate_bodies(self) -> None:
        duration = 0 if self._reduced_motion else 120
        self._animations.clear()
        for label in self._body_labels:
            effect = QGraphicsOpacityEffect(label)
            label.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", label)
            animation.setDuration(duration)
            animation.setStartValue(0.72)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.finished.connect(
                lambda target=label: target.setGraphicsEffect(None)
            )
            self._animations.append(animation)
            animation.start()

    @staticmethod
    def _separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        return separator
