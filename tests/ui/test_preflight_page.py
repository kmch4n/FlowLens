from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from pytestqt.qtbot import QtBot

from flowlens.controller.models import (
    BlockingIssue,
    DeviceOption,
    ModelCheck,
    PreflightReport,
    PreflightSelection,
    StorageCheck,
)
from flowlens.domain.enums import SessionMode
from flowlens.ui.preflight_page import PreflightPage


def ready_report() -> PreflightReport:
    """Return a complete, startable report for the preflight surface."""

    destination = Path("C:/FlowLens/sessions")
    return PreflightReport(
        selection=PreflightSelection(SessionMode.MEETING, "mic-1", "out-1"),
        microphones=(
            DeviceOption("mic-1", "Built-in microphone", False),
            DeviceOption("mic-2", "USB microphone", False),
        ),
        loopbacks=(DeviceOption("out-1", "Speakers", True),),
        mic_level=0.25,
        loopback_level=0.75,
        models=(
            ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
            ModelCheck("qwen3-4b-instruct-2507", None, True, None),
        ),
        storage=StorageCheck(destination, 500 * 1024 * 1024, True, None),
        destination=destination,
        issues=(),
        can_start=True,
    )


def report_with_issue(control_id: str, message: str) -> PreflightReport:
    """Return the startable fixture with one exact blocking issue."""

    report = ready_report()
    return PreflightReport(
        selection=report.selection,
        microphones=report.microphones,
        loopbacks=report.loopbacks,
        mic_level=report.mic_level,
        loopback_level=report.loopback_level,
        models=report.models,
        storage=report.storage,
        destination=report.destination,
        issues=(BlockingIssue(control_id, message),),
        can_start=False,
    )


def test_preflight_renders_every_required_control(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.render(ready_report())

    assert page.meeting_radio.isChecked() is True
    assert page.microphone_combo.count() == 2
    assert page.loopback_combo.count() == 1
    assert page.mic_meter.accessibleName() == "Microphone activity"
    assert page.loopback_meter.accessibleName() == "PC audio activity"
    assert page.model_status.text() == "Local models ready"
    assert page.storage_status.text() == "Storage ready: 500 MB minimum satisfied"
    assert page.destination_summary.text().startswith("Sessions are saved to ")
    assert page.start_button.isEnabled() is True


def test_blocking_reason_is_adjacent_and_start_is_disabled(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.show()
    page.render(report_with_issue("microphone", "Select an available microphone."))

    assert page.microphone_error.text() == "Select an available microphone."
    assert page.microphone_error.isVisibleTo(page) is True
    assert page.start_button.isEnabled() is False


def test_model_blockers_are_all_visible_in_the_adjacent_error_slot(
    qtbot: QtBot,
) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    report = ready_report()
    page.render(
        PreflightReport(
            selection=report.selection,
            microphones=report.microphones,
            loopbacks=report.loopbacks,
            mic_level=report.mic_level,
            loopback_level=report.loopback_level,
            models=report.models,
            storage=report.storage,
            destination=report.destination,
            issues=(
                BlockingIssue("asr_model", "ASR model is missing."),
                BlockingIssue("discussion_model", "Discussion model is invalid."),
            ),
            can_start=False,
        )
    )

    assert "ASR model is missing." in page.model_error.text()
    assert "Discussion model is invalid." in page.model_error.text()


def test_unavailable_saved_device_stays_unselected(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    report = ready_report()
    unavailable = PreflightReport(
        selection=PreflightSelection(SessionMode.GENERAL, "gone-mic", "gone-out"),
        microphones=report.microphones,
        loopbacks=report.loopbacks,
        mic_level=0.0,
        loopback_level=0.0,
        models=report.models,
        storage=report.storage,
        destination=report.destination,
        issues=(
            BlockingIssue("microphone", "Select an available microphone."),
            BlockingIssue(
                "loopback", "Select a loopback-capable Windows output device."
            ),
        ),
        can_start=False,
    )

    page.render(unavailable)

    assert page.microphone_combo.currentIndex() == -1
    assert page.loopback_combo.currentIndex() == -1


def test_device_change_emits_complete_selection(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.render(ready_report())

    with qtbot.waitSignal(page.selection_changed, timeout=500) as blocker:
        page.microphone_combo.setCurrentIndex(1)

    assert blocker.args == [PreflightSelection(SessionMode.MEETING, "mic-2", "out-1")]


def test_ctrl_enter_emits_start_only_when_valid(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.render(ready_report())
    with qtbot.waitSignal(page.start_requested, timeout=500):
        QTest.keyClick(page, Qt.Key.Key_Enter, Qt.KeyboardModifier.ControlModifier)
    page.render(
        report_with_issue("storage", "At least 500 MB of free space is required.")
    )
    with qtbot.assertNotEmitted(page.start_requested):
        QTest.keyClick(page, Qt.Key.Key_Enter, Qt.KeyboardModifier.ControlModifier)


def test_tab_order_reaches_mode_devices_and_start(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)

    assert page.focus_chain() == [
        page.meeting_radio,
        page.interview_radio,
        page.general_radio,
        page.microphone_combo,
        page.loopback_combo,
        page.start_button,
    ]
