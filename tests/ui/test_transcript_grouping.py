"""Conversation paragraphs keep adjacent speech readable without changing raw logs."""

from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from flowlens.domain.enums import AudioSource
from flowlens.ui.transcript_view import TranscriptView
from tests.ui.test_transcript import make_record


def _visible_lines(view: TranscriptView) -> list[str]:
    model = view.list_view.model()
    assert model is not None
    return [
        str(model.data(model.index(index, 0), int(Qt.ItemDataRole.DisplayRole)))
        for index in range(model.rowCount())
    ]


def test_adjacent_me_fragments_form_one_visible_paragraph(qtbot: QtBot) -> None:
    view = TranscriptView()
    qtbot.addWidget(view)
    view.model.commit(make_record(sequence=1, text="今日は", start_ms=0))
    view.model.commit(make_record(sequence=2, text="予定を確認します", start_ms=800))

    assert _visible_lines(view) == ["● ME · 今日は予定を確認します"]
    assert [record.text for record in view.model.records()] == [
        "今日は",
        "予定を確認します",
    ]


def test_others_and_interrupted_turns_are_not_misattributed(qtbot: QtBot) -> None:
    view = TranscriptView()
    qtbot.addWidget(view)
    view.model.commit(make_record(sequence=1, text="私の前半", start_ms=0))
    view.model.commit(
        make_record(
            sequence=2,
            source=AudioSource.OTHERS,
            text="相手の発言",
            start_ms=800,
        )
    )
    view.model.commit(make_record(sequence=3, text="私の後半", start_ms=1600))
    view.model.commit(
        make_record(
            sequence=4,
            source=AudioSource.OTHERS,
            text="別の発言",
            start_ms=2400,
        )
    )

    assert _visible_lines(view) == [
        "● ME · 私の前半",
        "■ OTHERS · 相手の発言",
        "● ME · 私の後半",
        "■ OTHERS · 別の発言",
    ]


def test_sentence_end_and_long_pause_start_new_paragraphs(qtbot: QtBot) -> None:
    view = TranscriptView()
    qtbot.addWidget(view)
    view.model.commit(make_record(sequence=1, text="最初の文。", start_ms=0))
    view.model.commit(make_record(sequence=2, text="次の話題", start_ms=800))
    view.model.commit(make_record(sequence=3, text="時間を空けた話題", start_ms=5000))

    assert _visible_lines(view) == [
        "● ME · 最初の文。",
        "● ME · 次の話題",
        "● ME · 時間を空けた話題",
    ]
