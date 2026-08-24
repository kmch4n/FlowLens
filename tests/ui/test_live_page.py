from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from flowlens.controller.models import (
    DeviceOption,
    ModelCheck,
    PreflightReport,
    PreflightSelection,
    StorageCheck,
)
from flowlens.controller.session_controller import ControllerSnapshot, SessionState
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import SessionMode
from flowlens.ui.discussion_panel import DiscussionPanel, labels_for
from flowlens.ui.live_page import LivePage
from flowlens.ui.status_strip import StatusSnapshot, StatusStrip


def empty_state(mode: SessionMode) -> DiscussionState:
    return DiscussionState.initial(
        mode,
        datetime.fromisoformat("2026-08-19T12:00:00+09:00"),
    )


def ready_preflight() -> PreflightReport:
    root = Path("C:/FlowLens/sessions")
    return PreflightReport(
        selection=PreflightSelection(SessionMode.MEETING, "mic-1", "out-1"),
        microphones=(DeviceOption("mic-1", "Microphone", False),),
        loopbacks=(DeviceOption("out-1", "Speakers", True),),
        mic_level=0.3,
        loopback_level=0.2,
        models=(
            ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
            ModelCheck("qwen3-4b-instruct-2507", None, True, None),
        ),
        storage=StorageCheck(root, 500 * 1024 * 1024, True, None),
        destination=root,
        issues=(),
        can_start=True,
    )


def snapshot(
    *,
    mode: SessionMode = SessionMode.MEETING,
    state: SessionState = SessionState.RECORDING,
    issue: str | None = None,
) -> ControllerSnapshot:
    return ControllerSnapshot(
        state=state,
        preflight=replace(
            ready_preflight(),
            selection=PreflightSelection(mode, "mic-1", "out-1"),
        ),
        issue=issue,
        recording_status="Recording",
        transcript=(),
        partials=(),
        discussion_state=empty_state(mode),
        microphone_level=0.4,
        loopback_level=0.5,
        asr_status="Delayed",
        asr_backlog_ms=2100,
        maximum_asr_backlog_ms=3100,
        analysis_status="Paused",
        latest_successful_save_at=datetime.fromisoformat("2026-08-19T12:35:02+09:00"),
        fatal_error=None,
        stop_confirmation_visible=False,
        slow_finalization_visible=False,
    )


@pytest.mark.parametrize(
    ("mode", "labels"),
    [
        (
            SessionMode.MEETING,
            (
                "Current focus",
                "Key points",
                "Decisions / confirmations",
                "Unresolved / next actions",
            ),
        ),
        (
            SessionMode.INTERVIEW,
            (
                "Current question / topic",
                "Answer highlights",
                "Confirmed content",
                "Follow-ups / points to clarify",
            ),
        ),
        (
            SessionMode.GENERAL,
            (
                "Current topic",
                "Key points",
                "Confirmed items",
                "Items to revisit",
            ),
        ),
    ],
)
def test_discussion_sections_have_fixed_order_and_mode_labels(
    qtbot: QtBot,
    mode: SessionMode,
    labels: tuple[str, str, str, str],
) -> None:
    panel = DiscussionPanel()
    qtbot.addWidget(panel)
    panel.render(empty_state(mode), labels_for(mode))

    assert panel.section_titles() == labels
    assert all(text != "" for text in panel.empty_explanations())


def test_live_layout_switches_at_exact_breakpoint(qtbot: QtBot) -> None:
    page = LivePage()
    qtbot.addWidget(page)
    page.show()
    page.resize(1000, 700)
    page.reflow()
    assert page.main_splitter.orientation() is Qt.Orientation.Horizontal
    assert page.main_splitter.sizes()[0] / sum(
        page.main_splitter.sizes()
    ) == pytest.approx(
        0.62,
        abs=0.03,
    )
    page.resize(999, 700)
    page.reflow()
    assert page.main_splitter.orientation() is Qt.Orientation.Vertical
    assert page.main_splitter.indexOf(
        page.transcript_view
    ) < page.main_splitter.indexOf(
        page.discussion_panel,
    )


def test_narrow_layout_is_keyboard_scrollable_without_horizontal_scroll(
    qtbot: QtBot,
) -> None:
    page = LivePage()
    qtbot.addWidget(page)
    page.resize(900, 600)
    page.reflow()

    assert page.narrow_scroll_area.isVisibleTo(page) is True
    assert page.narrow_scroll_area.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert (
        page.narrow_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert page.narrow_scroll_area.widget() is not None
    assert page.discussion_panel.y() > page.transcript_view.y()


def test_statuses_are_separate_not_one_aggregate_string(qtbot: QtBot) -> None:
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.render(
        StatusSnapshot(
            microphone_level=0.2,
            loopback_level=0.1,
            asr="Delayed",
            delay_ms=2100,
            analysis="Paused",
            saved="12:35:02",
        )
    )

    assert strip.microphone_status.text() != strip.asr_status.text()
    assert "Delayed" in strip.asr_status.text()
    assert "Paused" in strip.analysis_status.text()
    assert "12:35:02" in strip.save_status.text()


def test_live_page_renders_top_bar_banner_and_signals(qtbot: QtBot) -> None:
    page = LivePage()
    qtbot.addWidget(page)
    page.render(snapshot(issue="ASR delay is high. Wait for processing."))

    assert page.product_label.text() == "FlowLens"
    assert page.mode_label.text() == "Meeting / discussion"
    assert page.recording_state.text() == "Recording"
    assert page.elapsed_timer.text() == "00:00"
    assert page.banner.minimumHeight() > 0
    assert page.banner.text() == "ASR delay is high. Wait for processing."
    assert page.pause_resume_button.text() == "Pause"
    assert page.stop_button.property("uiState") == "error"

    with qtbot.waitSignal(page.pause_requested, timeout=500):
        page.pause_resume_button.click()
    with qtbot.waitSignal(page.stop_requested, timeout=500):
        page.stop_button.click()
    with qtbot.waitSignal(page.always_on_top_changed, timeout=500) as blocker:
        page.always_on_top_toggle.click()
    assert blocker.args == [True]
