"""Conversation rows wrap without clipping when the reading area narrows."""

from PySide6.QtWidgets import QStyleOptionViewItem
from pytestqt.qtbot import QtBot

from flowlens.ui.transcript_view import TranscriptView
from tests.ui.test_transcript import make_record


def test_reading_row_expands_for_narrow_columns(qtbot: QtBot) -> None:
    from flowlens.ui.transcript_delegate import TranscriptDelegate

    view = TranscriptView()
    qtbot.addWidget(view)
    view.model.commit(
        make_record(
            sequence=1,
            start_ms=0,
            text="議論を整理しながら話した内容を正確に確認します。" * 8,
        )
    )
    delegate = view.list_view.itemDelegate()
    assert isinstance(delegate, TranscriptDelegate)
    option = QStyleOptionViewItem()
    option.rect.setWidth(700)
    wide = delegate.sizeHint(option, view.model.index(0, 0))
    option.rect.setWidth(180)
    narrow = delegate.sizeHint(option, view.model.index(0, 0))
    assert narrow.height() > wide.height()
