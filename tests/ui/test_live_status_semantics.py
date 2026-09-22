"""Normal session controls and pauses are not failure states."""

from pytestqt.qtbot import QtBot

from flowlens.ui.live_page import LivePage
from flowlens.ui.status_strip import StatusSnapshot, StatusStrip


def test_stop_is_an_action_not_an_error(qtbot: QtBot) -> None:
    page = LivePage()
    qtbot.addWidget(page)
    assert page.stop_button.text() == "Stop"
    assert page.stop_button.property("uiState") != "error"


def test_manual_pause_is_neutral_not_an_error(qtbot: QtBot) -> None:
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.render(StatusSnapshot(0, 0, "Running", 0, "Paused", "Not saved yet"))
    assert strip.analysis_status.property("uiState") == "default"
    assert "Error" not in strip.analysis_status.text()
