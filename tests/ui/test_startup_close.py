"""Closing during startup keeps a visible, responsive shutdown surface."""

from pytestqt.qtbot import QtBot

from flowlens.ui.main_window import MainWindow


def test_close_waits_visibly_until_startup_finishes(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_startup_busy(True)
    window.show()
    window.close()
    assert window.isVisible()
    assert window.startup_close_pending
    assert "Closing" in window.preflight_page.readiness_summary.text()
    window.set_startup_busy(False)
    qtbot.waitUntil(lambda: not window.isVisible())
