from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from pytestqt.qtbot import QtBot

from flowlens.ui.completion_page import CompletionPage, CompletionSummary
from flowlens.ui.dialogs import SlowFinalizationDialog, StopConfirmationDialog


def test_stop_confirmation_dialog_emits_only_explicit_choices(qtbot: QtBot) -> None:
    dialog = StopConfirmationDialog()
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.text() == "Stop this session?"
    assert dialog.confirm_button.text() == "Stop and finalize"
    assert dialog.cancel_button.text() == "Keep recording"
    with qtbot.waitSignal(dialog.stop_confirmed, timeout=500):
        dialog.confirm_button.click()
    assert dialog.isVisible() is False

    dialog.show()
    with qtbot.waitSignal(dialog.keep_recording_requested, timeout=500):
        dialog.cancel_button.click()
    assert dialog.isVisible() is False


def test_slow_finalization_dialog_never_auto_selects_force_close(
    qtbot: QtBot,
) -> None:
    dialog = SlowFinalizationDialog()
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.text() == "Finalization is taking longer than expected"
    assert dialog.keep_waiting_button.isDefault() is True
    assert dialog.force_close_button.isDefault() is False
    with qtbot.waitSignal(dialog.keep_waiting_requested, timeout=500):
        dialog.keep_waiting_button.click()
    assert dialog.isVisible() is False

    dialog.show()
    with qtbot.waitSignal(dialog.force_close_requested, timeout=500):
        dialog.force_close_button.click()
    assert dialog.isVisible() is False


def test_dialog_escape_and_window_close_emit_default_safe_choice(
    qtbot: QtBot,
) -> None:
    stop_dialog = StopConfirmationDialog()
    qtbot.addWidget(stop_dialog)
    stop_dialog.show()
    with qtbot.waitSignal(stop_dialog.keep_recording_requested, timeout=500):
        QTest.keyClick(stop_dialog, Qt.Key.Key_Escape)
    assert stop_dialog.isVisible() is False

    slow_dialog = SlowFinalizationDialog()
    qtbot.addWidget(slow_dialog)
    slow_dialog.show()
    with qtbot.waitSignal(slow_dialog.keep_waiting_requested, timeout=500):
        QTest.keyClick(slow_dialog, Qt.Key.Key_Escape)
    assert slow_dialog.isVisible() is False

    closing_dialog = StopConfirmationDialog()
    qtbot.addWidget(closing_dialog)
    closing_dialog.show()
    with qtbot.waitSignal(closing_dialog.keep_recording_requested, timeout=500):
        closing_dialog.close()
    assert closing_dialog.isVisible() is False


def test_completion_contains_only_mvp_actions(qtbot: QtBot, tmp_path: Path) -> None:
    page = CompletionPage()
    qtbot.addWidget(page)
    page.render(CompletionSummary(1_800_000, 42, tmp_path.resolve()))

    assert page.duration_value.text() == "30:00"
    assert page.transcript_count_value.text() == "42"
    assert page.path_value.text() == str(tmp_path.resolve())
    assert page.action_labels() == ["Open folder", "Start another session", "Close"]
    assert not hasattr(page, "play_button")
    assert not hasattr(page, "search_box")


def test_completion_actions_emit_exact_signals(qtbot: QtBot, tmp_path: Path) -> None:
    page = CompletionPage()
    qtbot.addWidget(page)
    page.render(CompletionSummary(30_000, 2, tmp_path.resolve()))

    with qtbot.waitSignal(page.open_folder_requested, timeout=500):
        page.open_folder_button.click()
    with qtbot.waitSignal(page.start_another_requested, timeout=500):
        page.start_another_button.click()
    with qtbot.waitSignal(page.close_requested, timeout=500):
        page.close_button.click()
