from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QMenu
from pytestqt.qtbot import QtBot

from flowlens.config.model import AppConfig
from flowlens.controller.models import (
    BlockingIssue,
    DeviceOption,
    ModelCheck,
    PreflightReport,
    PreflightSelection,
    StorageCheck,
)
from flowlens.controller.session_controller import ControllerSnapshot, SessionState
from flowlens.domain.enums import SessionMode
from flowlens.ui.main_window import MainWindow
from flowlens.ui.presenter import QtSessionPresenter
from tests.factories import make_discussion_state, make_transcript_record


class FakeAnnouncer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def announce(
        self,
        widget: object,
        message: str,
        assertive: bool = False,
    ) -> None:
        del widget
        self.messages.append((message, assertive))


class FakeConfigStore:
    def __init__(self) -> None:
        self.saved = AppConfig.default()

    def load(self) -> AppConfig:
        return self.saved

    def save(self, config: AppConfig) -> None:
        self.saved = config


class RecordingController:
    def __init__(
        self,
        *,
        state: SessionState = SessionState.PREFLIGHT,
        can_start: bool = True,
    ) -> None:
        self.state = state
        self.can_start = can_start
        self.stop_confirmation_visible = False
        self.slow_finalization_visible = False
        self.tick_count = 0
        self.drain_count = 0
        self.raise_on_tick = False
        self.started_with: list[PreflightSelection] = []
        self.refreshed: list[PreflightSelection] = []
        self.selection = PreflightSelection(SessionMode.MEETING, "mic-1", "out-1")
        self.transcript = (
            (make_transcript_record(1),) if state is SessionState.COMPLETED else ()
        )

    def snapshot(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self.state,
            preflight=ready_report(self.selection, can_start=self.can_start),
            issue=None,
            recording_status={
                SessionState.IDLE: "Idle",
                SessionState.PREFLIGHT: "Ready",
                SessionState.STARTING: "Starting",
                SessionState.RECORDING: "Recording",
                SessionState.PAUSED: "Paused",
                SessionState.STOPPING: "Finalizing",
                SessionState.COMPLETED: "Completed",
                SessionState.ERROR: "Error",
            }[self.state],
            transcript=self.transcript,
            partials=(),
            discussion_state=make_discussion_state(),
            microphone_level=0.2,
            loopback_level=0.3,
            asr_status="Running" if self.state is SessionState.RECORDING else "Idle",
            asr_backlog_ms=0,
            maximum_asr_backlog_ms=0,
            analysis_status="Running"
            if self.state is SessionState.RECORDING
            else "Idle",
            latest_successful_save_at=datetime.fromisoformat(
                "2026-08-19T12:05:00+09:00"
            ),
            fatal_error=None,
            stop_confirmation_visible=self.stop_confirmation_visible,
            slow_finalization_visible=self.slow_finalization_visible,
        )

    def enter_preflight(self) -> None:
        self.state = SessionState.PREFLIGHT
        self.stop_confirmation_visible = False
        self.slow_finalization_visible = False
        self.transcript = ()

    def refresh_preflight(self, selection: PreflightSelection) -> PreflightReport:
        self.refreshed.append(selection)
        self.selection = selection
        return ready_report(selection, can_start=self.can_start)

    def start(self, selection: PreflightSelection) -> None:
        if not self.can_start:
            return
        self.started_with.append(selection)
        self.selection = selection
        self.state = SessionState.RECORDING

    def pause(self) -> None:
        if self.state is SessionState.RECORDING:
            self.state = SessionState.PAUSED

    def resume(self) -> None:
        if self.state is SessionState.PAUSED:
            self.state = SessionState.RECORDING

    def request_stop(self) -> None:
        if self.state in {SessionState.RECORDING, SessionState.PAUSED}:
            self.stop_confirmation_visible = True

    def cancel_stop(self) -> None:
        self.stop_confirmation_visible = False

    def confirm_stop(self) -> None:
        if self.stop_confirmation_visible:
            self.state = SessionState.STOPPING
            self.stop_confirmation_visible = False

    def keep_waiting(self) -> None:
        self.slow_finalization_visible = False

    def force_close(self) -> None:
        self.state = SessionState.ERROR

    def drain_messages(self) -> None:
        self.drain_count += 1

    def tick(self) -> None:
        self.tick_count += 1
        if self.raise_on_tick:
            raise RuntimeError("simulated tick failure")


def ready_report(
    selection: PreflightSelection,
    *,
    can_start: bool,
) -> PreflightReport:
    destination = Path("C:/FlowLens/sessions").resolve(strict=False)
    return PreflightReport(
        selection=selection,
        microphones=(DeviceOption("mic-1", "Microphone", False),),
        loopbacks=(DeviceOption("out-1", "Speakers", True),),
        mic_level=0.2,
        loopback_level=0.3,
        models=(
            ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
            ModelCheck("qwen3-4b-instruct-2507", None, True, None),
        ),
        storage=StorageCheck(destination, 500 * 1024 * 1024, True, None),
        destination=destination,
        issues=()
        if can_start
        else (BlockingIssue("storage", "At least 500 MB is required."),),
        can_start=can_start,
    )


def make_presenter(
    *,
    recording: bool = False,
    completed: bool = False,
    preflight_valid: bool = True,
) -> tuple[QtSessionPresenter, MainWindow, RecordingController]:
    state = SessionState.PREFLIGHT
    if recording:
        state = SessionState.RECORDING
    if completed:
        state = SessionState.COMPLETED
    controller = RecordingController(state=state, can_start=preflight_valid)
    window = MainWindow()
    presenter = QtSessionPresenter(
        controller,
        window,
        FakeAnnouncer(),
        config_store=FakeConfigStore(),
    )
    window.show()
    return presenter, window, controller


def test_pause_feedback_is_rendered_in_same_event_turn(qtbot: QtBot) -> None:
    presenter, window, _controller = make_presenter(recording=True)
    qtbot.addWidget(window)

    started = time.perf_counter()
    QTest.keyClick(window, Qt.Key.Key_Space)

    qtbot.waitUntil(
        lambda: window.live_page.recording_state.text() == "Paused",
        timeout=100,
    )
    assert (time.perf_counter() - started) < 0.1
    assert presenter.render_count >= 2


def test_space_does_nothing_when_text_input_has_focus(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    qtbot.addWidget(window)
    line_edit = QLineEdit(window)
    line_edit.show()
    line_edit.setFocus()
    qtbot.waitUntil(lambda: line_edit.hasFocus(), timeout=500)

    QTest.keyClick(line_edit, Qt.Key.Key_Space)

    assert controller.state is SessionState.RECORDING
    assert presenter.render_count >= 1


def test_space_does_nothing_when_menu_has_focus(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    qtbot.addWidget(window)
    menu = QMenu(window)
    menu.addAction("Keep menu open")
    menu.popup(window.mapToGlobal(window.rect().center()))
    qtbot.waitUntil(menu.isVisible, timeout=500)

    QTest.keyClick(menu, Qt.Key.Key_Space)
    menu.close()

    assert controller.state is SessionState.RECORDING
    assert presenter.render_count >= 1


def test_global_shortcuts_match_spec(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    qtbot.addWidget(window)

    QTest.keyClick(window, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier)
    assert window.is_always_on_top() is True

    QTest.keyClick(
        window,
        Qt.Key.Key_S,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert window.stop_dialog.isVisible() is True
    assert controller.stop_confirmation_visible is True
    assert presenter.render_count >= 2


def test_global_shortcuts_ignore_auto_repeat(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    qtbot.addWidget(window)
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
        " ",
        True,
        1,
    )

    window.eventFilter(window, event)

    assert controller.state is SessionState.RECORDING
    assert event.isAccepted() is True
    assert presenter.render_count >= 1


def test_ctrl_enter_starts_only_valid_preflight(qtbot: QtBot) -> None:
    valid_presenter, valid_window, valid_controller = make_presenter(
        preflight_valid=True
    )
    qtbot.addWidget(valid_window)

    QTest.keyClick(
        valid_window,
        Qt.Key.Key_Return,
        Qt.KeyboardModifier.ControlModifier,
    )
    assert valid_controller.started_with == [valid_controller.selection]
    assert valid_presenter.render_count >= 2

    invalid_presenter, invalid_window, invalid_controller = make_presenter(
        preflight_valid=False
    )
    qtbot.addWidget(invalid_window)
    QTest.keyClick(
        invalid_window,
        Qt.Key.Key_Return,
        Qt.KeyboardModifier.ControlModifier,
    )
    assert invalid_controller.started_with == []
    assert invalid_presenter.render_count >= 1


def test_timer_drains_ticks_and_renders_only_changed_snapshots(
    qtbot: QtBot,
) -> None:
    presenter, window, controller = make_presenter(recording=True)
    qtbot.addWidget(window)
    initial_renders = presenter.render_count

    presenter.on_timer()
    presenter.on_timer()
    controller.state = SessionState.PAUSED
    presenter.on_timer()

    assert controller.drain_count == 3
    assert controller.tick_count == 3
    assert presenter.render_count == initial_renders + 1
    assert window.live_page.recording_state.text() == "Paused"


def test_timer_exceptions_do_not_break_later_ticks(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    qtbot.addWidget(window)

    controller.raise_on_tick = True
    presenter.on_timer()
    controller.raise_on_tick = False
    presenter.on_timer()

    assert controller.tick_count == 2
    assert presenter.last_timer_error == "simulated tick failure"


def test_start_another_returns_to_fresh_preflight(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(completed=True)
    qtbot.addWidget(window)
    window.live_page.render(
        replace(controller.snapshot(), transcript=(make_transcript_record(1),))
    )
    assert window.live_page.transcript_view.model.rowCount() == 1

    QTest.mouseClick(
        window.completion_page.start_another_button,
        Qt.MouseButton.LeftButton,
    )

    assert controller.state is SessionState.PREFLIGHT
    assert window.current_page() is window.preflight_page
    assert window.live_page.transcript_view.model.rowCount() == 0
    assert presenter.render_count >= 2
