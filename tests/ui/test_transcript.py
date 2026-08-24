import dataclasses
from datetime import datetime

import pytest
from pytestqt.qtbot import QtBot

from flowlens.asr.types import PartialTranscript
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import TranscriptRecord
from flowlens.ui.transcript_model import (
    ImmutableTranscriptError,
    TranscriptListModel,
)
from flowlens.ui.transcript_view import TranscriptView


def make_record(
    *,
    sequence: int,
    source: AudioSource = AudioSource.ME,
    text: str = "確定",
    start_ms: int = 0,
) -> TranscriptRecord:
    return TranscriptRecord(
        1,
        f"01J0000000000000000000{sequence:04d}",
        sequence,
        source,
        text,
        start_ms,
        start_ms + 800,
        sequence * 16_000,
        sequence * 16_000 + 12_800,
        datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
    )


def make_partial(
    text: str,
    *,
    source: AudioSource = AudioSource.ME,
    start_ms: int = 0,
) -> PartialTranscript:
    return PartialTranscript(
        source,
        text,
        start_ms,
        start_ms + 500,
        0,
        8_000,
    )


def test_commits_sort_by_start_then_me_before_others() -> None:
    model = TranscriptListModel()
    model.commit(make_record(sequence=2, source=AudioSource.OTHERS, start_ms=1000))
    model.commit(make_record(sequence=1, source=AudioSource.ME, start_ms=1000))

    assert [model.row(index).source for index in range(model.rowCount())] == [
        AudioSource.ME,
        AudioSource.OTHERS,
    ]


def test_sequence_breaks_remaining_ties_deterministically() -> None:
    model = TranscriptListModel()
    model.commit(make_record(sequence=2, source=AudioSource.ME, start_ms=1000))
    model.commit(make_record(sequence=1, source=AudioSource.ME, start_ms=1000))

    assert [model.row(index).sequence for index in range(model.rowCount())] == [1, 2]


def test_partial_is_ephemeral_and_commit_is_immutable() -> None:
    model = TranscriptListModel()
    model.set_partial(AudioSource.ME, make_partial("途中", start_ms=0))
    partial = model.partial(AudioSource.ME)
    assert partial is not None
    assert partial.text == "途中"
    record = make_record(sequence=1, source=AudioSource.ME, text="確定", start_ms=0)

    model.commit(record)

    assert model.partial(AudioSource.ME) is None
    with pytest.raises(ImmutableTranscriptError):
        model.commit(dataclasses.replace(record, text="改変"))


def test_view_renders_source_shape_and_partial_without_timestamps(qtbot: QtBot) -> None:
    view = TranscriptView()
    qtbot.addWidget(view)
    view.model.commit(make_record(sequence=1, source=AudioSource.ME, text="本文"))
    view.model.set_partial(
        AudioSource.OTHERS, make_partial("途中", source=AudioSource.OTHERS)
    )

    assert view.source_labels() == ["● ME", "■ OTHERS"]
    assert view.partial_label.text() == "■ OTHERS · Partial · 途中"
    assert (
        view.partial_label.accessibleDescription() == "Partial transcript from OTHERS"
    )
    assert "12:" not in view.rendered_text()


def test_manual_scroll_disables_auto_scroll_until_return_to_latest(
    qtbot: QtBot,
) -> None:
    view = TranscriptView()
    qtbot.addWidget(view)
    view.resize(420, 180)
    view.show()
    for sequence in range(1, 28):
        view.model.commit(make_record(sequence=sequence, start_ms=sequence * 1000))
    qtbot.waitUntil(lambda: view.scrollbar().maximum() > 0, timeout=1000)
    view.return_to_latest()
    latest = view.scrollbar().maximum()
    assert view.auto_scroll_enabled is True

    view.scrollbar().setValue(max(0, latest - 10))
    assert view.auto_scroll_enabled is False
    assert view.return_to_latest_button.isVisibleTo(view) is True
    view.model.commit(make_record(sequence=28, start_ms=28_000))
    assert view.scrollbar().value() < view.scrollbar().maximum()

    view.return_to_latest()

    assert view.auto_scroll_enabled is True
    assert view.scrollbar().value() == view.scrollbar().maximum()
