from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QWidget
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
from flowlens.ui.completion_page import CompletionPage, CompletionSummary
from flowlens.ui.design import DesignTokens, build_stylesheet
from flowlens.ui.live_page import LivePage
from flowlens.ui.preflight_page import PreflightPage


def make_preflight_page() -> QWidget:
    root = Path("C:/FlowLens/sessions")
    page = PreflightPage()
    page.render(
        PreflightReport(
            selection=PreflightSelection(SessionMode.MEETING, "mic-1", "out-1"),
            microphones=(DeviceOption("mic-1", "Microphone", False),),
            loopbacks=(DeviceOption("out-1", "Speakers", True),),
            mic_level=0.2,
            loopback_level=0.4,
            models=(
                ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
                ModelCheck("qwen3-4b-instruct-2507", None, True, None),
            ),
            storage=StorageCheck(root, 500 * 1024 * 1024, True, None),
            destination=root,
            issues=(),
            can_start=True,
        )
    )
    return page


def make_live_page() -> QWidget:
    root = Path("C:/FlowLens/sessions")
    page = LivePage()
    now = datetime.fromisoformat("2026-08-19T12:35:02+09:00")
    page.render(
        ControllerSnapshot(
            state=SessionState.RECORDING,
            preflight=PreflightReport(
                selection=PreflightSelection(SessionMode.MEETING, "mic-1", "out-1"),
                microphones=(DeviceOption("mic-1", "Microphone", False),),
                loopbacks=(DeviceOption("out-1", "Speakers", True),),
                mic_level=0.2,
                loopback_level=0.4,
                models=(
                    ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
                    ModelCheck("qwen3-4b-instruct-2507", None, True, None),
                ),
                storage=StorageCheck(root, 500 * 1024 * 1024, True, None),
                destination=root,
                issues=(),
                can_start=True,
            ),
            issue=None,
            recording_status="Recording",
            transcript=(),
            partials=(),
            discussion_state=DiscussionState.initial(SessionMode.MEETING, now),
            microphone_level=0.2,
            loopback_level=0.4,
            asr_status="Running",
            asr_backlog_ms=120,
            maximum_asr_backlog_ms=240,
            analysis_status="Running",
            latest_successful_save_at=now,
            fatal_error=None,
            stop_confirmation_visible=False,
            slow_finalization_visible=False,
        )
    )
    return page


def make_completion_page() -> QWidget:
    page = CompletionPage()
    page.render(CompletionSummary(1_800_000, 42, Path("C:/FlowLens/sessions/one")))
    return page


@pytest.mark.parametrize(
    "page",
    [make_preflight_page, make_live_page, make_completion_page],
)
def test_rendered_page_uses_no_pure_extremes_and_limits_exact_accent_pixels(
    qtbot: QtBot,
    page: Callable[[], QWidget],
) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    app.setStyleSheet(build_stylesheet(DesignTokens.approved(), reduced_motion=False))
    widget = page()
    qtbot.addWidget(widget)
    widget.resize(1280, 800)
    widget.show()

    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = [
        QColor(image.pixel(x, y)).name().upper()
        for y in range(image.height())
        for x in range(image.width())
    ]

    assert "#000000" not in pixels
    assert "#FFFFFF" not in pixels
    assert pixels.count("#D6A13D") / len(pixels) < 0.03


def test_live_page_size_boundaries_remain_operable(qtbot: QtBot) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    app.setStyleSheet(build_stylesheet(DesignTokens.approved(), reduced_motion=True))
    page = make_live_page()
    assert isinstance(page, LivePage)
    qtbot.addWidget(page)

    for width, height in ((1280, 800), (1000, 700), (999, 700), (900, 600)):
        page.resize(width, height)
        page.reflow()
        page.show()
        image = page.grab().toImage()
        assert image.width() == width
        assert image.height() == height
        assert page.minimumWidth() <= 900
        assert page.minimumHeight() <= 600
        assert page.main_splitter.sizes()
        assert page.narrow_scroll_area.horizontalScrollBar().maximum() == 0
