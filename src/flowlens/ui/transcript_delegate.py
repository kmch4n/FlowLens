"""Readable conversation rows with separate source and speech hierarchy."""

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from flowlens.domain.enums import AudioSource
from flowlens.ui.design import DesignTokens
from flowlens.ui.transcript_model import TranscriptListModel


class TranscriptDelegate(QStyledItemDelegate):
    """Keep speaker markers quiet and wrap the actual conversation naturally."""

    def __init__(self, view: QListView) -> None:
        super().__init__(view)
        self._view = view
        self._tokens = DesignTokens.approved()

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        """Recalculate row height for the current reading column."""
        width = option.rect.width() or self._view.viewport().width()
        text = str(index.data(TranscriptListModel.text_role) or "")
        bounds = QFontMetrics(option.font).boundingRect(
            QRect(0, 0, max(40, width - 40), 100000),
            int(Qt.TextFlag.TextWordWrap),
            text,
        )
        return QSize(width, max(68, bounds.height() + 52))

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint source labels independently from the full accessible model text."""
        painter.save()
        painter.setClipRect(option.rect)
        tokens = self._tokens
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(tokens.rule))
        source = index.data(TranscriptListModel.source_role)
        marker = tokens.focus if source == AudioSource.ME else tokens.accent
        painter.fillRect(
            QRect(option.rect.x() + 4, option.rect.y() + 16, 3, 16), QColor(marker)
        )
        label_font = QFont(option.font)
        label_font.setPixelSize(12)
        label_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(label_font)
        painter.setPen(QColor(tokens.muted_text))
        painter.drawText(
            option.rect.adjusted(20, 10, -20, 0),
            int(Qt.AlignmentFlag.AlignTop),
            "ME" if source == AudioSource.ME else "OTHERS",
        )
        painter.setFont(option.font)
        painter.setPen(QColor(tokens.primary_text))
        painter.drawText(
            option.rect.adjusted(20, 34, -20, -12),
            int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop),
            str(index.data(TranscriptListModel.text_role) or ""),
        )
        if option.state & QStyle.StateFlag.State_HasFocus:
            painter.setPen(QColor(tokens.focus))
            painter.drawRect(option.rect.adjusted(1, 1, -2, -2))
        painter.restore()
