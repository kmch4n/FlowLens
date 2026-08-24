from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QRect, QSize
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from pytestqt.qtbot import QtBot

from flowlens.config.model import AppConfig, DevicePreferences, WindowPreferences
from flowlens.controller.models import (
    BlockingIssue,
    CompletionSummary,
    DeviceOption,
    ModelCheck,
    PreflightReport,
    PreflightSelection,
    StorageCheck,
)
from flowlens.controller.session_controller import ControllerSnapshot, SessionState
from flowlens.domain.enums import SessionMode
from flowlens.ui.main_window import MainWindow, clamp_geometry
from flowlens.ui.presenter import QtAccessibilityAnnouncer, QtSessionPresenter
from tests.factories import make_discussion_state


class FakeConfigStore:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = AppConfig.default() if config is None else config
        self.saved = self.config

    def load(self) -> AppConfig:
        return self.config

    def save(self, config: AppConfig) -> None:
        self.saved = config


class FakeAnnouncer:
    def announce(
        self,
        widget: object,
        message: str,
        assertive: bool = False,
    ) -> None:
        del widget, message, assertive


class ConfigurableController:
    def __init__(
        self,
        *,
        state: SessionState = SessionState.PREFLIGHT,
        microphone_ids: tuple[str, ...] = ("mic-1",),
        loopback_ids: tuple[str, ...] = ("out-1",),
    ) -> None:
        self.state = state
        self.request_stop_count = 0
        self.microphone_ids = microphone_ids
        self.loopback_ids = loopback_ids
        self.selection = PreflightSelection(SessionMode.MEETING, None, None)
        self.stop_confirmation_visible = False
        self.slow_finalization_visible = False

    def snapshot(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self.state,
            preflight=self._report(self.selection),
            issue=None,
            recording_status={
                SessionState.PREFLIGHT: "Ready",
                SessionState.RECORDING: "Recording",
                SessionState.PAUSED: "Paused",
                SessionState.STOPPING: "Finalizing",
                SessionState.COMPLETED: "Completed",
                SessionState.IDLE: "Idle",
                SessionState.STARTING: "Starting",
                SessionState.ERROR: "Error",
            }[self.state],
            transcript=(),
            partials=(),
            discussion_state=make_discussion_state(),
            microphone_level=0.1,
            loopback_level=0.2,
            asr_status="Running",
            asr_backlog_ms=0,
            maximum_asr_backlog_ms=0,
            analysis_status="Running",
            latest_successful_save_at=datetime.fromisoformat(
                "2026-08-19T12:05:00+09:00"
            ),
            fatal_error=None,
            stop_confirmation_visible=self.stop_confirmation_visible,
            slow_finalization_visible=self.slow_finalization_visible,
            completion=self._completion_summary(),
        )

    def enter_preflight(self) -> None:
        self.state = SessionState.PREFLIGHT

    def refresh_preflight(self, selection: PreflightSelection) -> PreflightReport:
        self.selection = selection
        return self._report(selection)

    def start(self, selection: PreflightSelection) -> None:
        self.selection = selection
        self.state = SessionState.RECORDING

    def pause(self) -> None:
        self.state = SessionState.PAUSED

    def resume(self) -> None:
        self.state = SessionState.RECORDING

    def request_stop(self) -> None:
        self.request_stop_count += 1
        self.stop_confirmation_visible = True

    def cancel_stop(self) -> None:
        self.stop_confirmation_visible = False

    def confirm_stop(self) -> None:
        self.state = SessionState.STOPPING
        self.stop_confirmation_visible = False

    def keep_waiting(self) -> None:
        return

    def force_close(self) -> None:
        self.state = SessionState.ERROR

    def tick(self) -> None:
        return

    def _report(self, selection: PreflightSelection) -> PreflightReport:
        destination = Path("C:/FlowLens/sessions").resolve(strict=False)
        issues: tuple[BlockingIssue, ...] = ()
        if selection.microphone_id not in self.microphone_ids:
            issues += (BlockingIssue("microphone", "Select an available microphone."),)
        if selection.loopback_output_id not in self.loopback_ids:
            issues += (
                BlockingIssue(
                    "loopback",
                    "Select a loopback-capable Windows output device.",
                ),
            )
        return PreflightReport(
            selection=selection,
            microphones=tuple(
                DeviceOption(device_id, f"Microphone {device_id}", False)
                for device_id in self.microphone_ids
            ),
            loopbacks=tuple(
                DeviceOption(device_id, f"Output {device_id}", True)
                for device_id in self.loopback_ids
            ),
            mic_level=0.1,
            loopback_level=0.2,
            models=(
                ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
                ModelCheck("qwen3-4b-instruct-2507", None, True, None),
            ),
            storage=StorageCheck(destination, 500 * 1024 * 1024, True, None),
            destination=destination,
            issues=issues,
            can_start=not issues,
        )

    def _completion_summary(self) -> CompletionSummary | None:
        if self.state is not SessionState.COMPLETED:
            return None
        return CompletionSummary(
            30_000,
            0,
            Path("C:/FlowLens/sessions/session-1").resolve(strict=False),
        )


def test_saved_geometry_is_clamped_to_available_screens() -> None:
    screens = [QRect(0, 0, 1920, 1080), QRect(1920, 0, 1920, 1080)]
    saved = WindowPreferences(-9000, 9000, 4000, 3000, False, False)

    actual = clamp_geometry(saved, screens, minimum=QSize(900, 600))

    assert any(screen.contains(actual.center()) for screen in screens)
    assert actual.width() >= 900
    assert actual.height() >= 600


def test_active_close_uses_stop_confirmation(qtbot: QtBot) -> None:
    controller = ConfigurableController(state=SessionState.RECORDING)
    window = MainWindow()
    presenter = QtSessionPresenter(controller, window, FakeAnnouncer())
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert window.isVisible() is True
    assert window.stop_dialog.isVisible() is True
    assert controller.state is SessionState.RECORDING
    assert presenter.render_count >= 1


def test_starting_close_keeps_window_open_with_progress_banner(
    qtbot: QtBot,
) -> None:
    announcer = FakeStatusAnnouncer()
    controller = ConfigurableController(state=SessionState.STARTING)
    window = MainWindow()
    presenter = QtSessionPresenter(controller, window, announcer)
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert window.isVisible() is True
    assert window.live_page.banner.text() == "Session startup is still in progress."
    assert window.stop_dialog.isVisible() is False
    assert window.slow_finalization_dialog.isVisible() is False
    assert controller.request_stop_count == 0
    assert presenter.timer.isActive() is True
    assert ("Session startup is still in progress.", False) in announcer.messages


def test_stopping_close_before_slow_threshold_keeps_finalizing_without_dialog(
    qtbot: QtBot,
) -> None:
    announcer = FakeStatusAnnouncer()
    controller = ConfigurableController(state=SessionState.STOPPING)
    window = MainWindow()
    presenter = QtSessionPresenter(controller, window, announcer)
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert window.isVisible() is True
    assert window.live_page.banner.text() == "Finalization is in progress."
    assert window.stop_dialog.isVisible() is False
    assert window.slow_finalization_dialog.isVisible() is False
    assert controller.request_stop_count == 0
    assert presenter.timer.isActive() is True
    assert ("Finalization is in progress.", False) in announcer.messages


def test_stopping_close_with_slow_flag_shows_existing_slow_dialog(
    qtbot: QtBot,
) -> None:
    controller = ConfigurableController(state=SessionState.STOPPING)
    window = MainWindow()
    presenter = QtSessionPresenter(controller, window, FakeAnnouncer())
    qtbot.addWidget(window)
    window.show()
    controller.slow_finalization_visible = True
    presenter.render_current_snapshot(force=True)

    window.close()

    assert window.isVisible() is True
    assert window.slow_finalization_dialog.isVisible() is True
    assert window.stop_dialog.isVisible() is False
    assert controller.request_stop_count == 0


def test_completion_close_saves_preferences_and_closes(qtbot: QtBot) -> None:
    store = FakeConfigStore()
    controller = ConfigurableController(state=SessionState.PREFLIGHT)
    window = MainWindow()
    presenter = QtSessionPresenter(
        controller,
        window,
        FakeAnnouncer(),
        config_store=store,
    )
    qtbot.addWidget(window)
    window.show()
    controller.state = SessionState.PREFLIGHT
    presenter.render_current_snapshot(force=True)

    window.close()

    assert window.isVisible() is False
    assert store.saved.to_dict()["window"] == window.window_preferences().to_dict()


def test_presenter_restores_only_still_available_saved_devices(
    qtbot: QtBot,
) -> None:
    store = FakeConfigStore(
        AppConfig(
            schema_version=1,
            window=AppConfig.default().window,
            devices=DevicePreferences("missing-mic", "out-1"),
            last_mode=SessionMode.INTERVIEW,
        )
    )
    controller = ConfigurableController(
        microphone_ids=("mic-1",),
        loopback_ids=("out-1",),
    )
    window = MainWindow()

    QtSessionPresenter(controller, window, FakeAnnouncer(), config_store=store)
    qtbot.addWidget(window)

    assert controller.selection == PreflightSelection(
        SessionMode.INTERVIEW,
        None,
        "out-1",
    )
    assert window.preflight_page.microphone_combo.currentIndex() == -1
    assert window.preflight_page.loopback_combo.currentData() == "out-1"


def test_presenter_persists_only_approved_non_session_preferences(
    qtbot: QtBot,
) -> None:
    store = FakeConfigStore()
    controller = ConfigurableController()
    window = MainWindow()
    presenter = QtSessionPresenter(
        controller,
        window,
        FakeAnnouncer(),
        config_store=store,
    )
    qtbot.addWidget(window)
    window.setGeometry(100, 100, 1280, 800)
    window.set_always_on_top(True)

    presenter.on_selection_changed(
        PreflightSelection(SessionMode.INTERVIEW, "mic-1", "out-1")
    )
    presenter.save_preferences()

    saved = store.saved.to_dict()
    assert set(saved) == {"schema_version", "window", "devices", "last_mode"}
    assert saved["last_mode"] == "INTERVIEW"
    assert "transcript" not in json.dumps(saved, ensure_ascii=False).lower()
    assert "prompt" not in json.dumps(saved, ensure_ascii=False).lower()
    assert "credential" not in json.dumps(saved, ensure_ascii=False).lower()


def test_accessibility_announcer_sets_actual_qt_politeness(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    captured: list[QAccessibleAnnouncementEvent] = []

    def capture(event: QAccessibleAnnouncementEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(QAccessible, "updateAccessibility", capture)
    announcer = QtAccessibilityAnnouncer()

    announcer.announce(window, "Ready")
    announcer.announce(window, "Fatal storage error", assertive=True)

    assert len(captured) == 2
    assert captured[0].message() == "Ready"
    assert captured[0].politeness() is QAccessible.AnnouncementPoliteness.Polite
    assert captured[1].message() == "Fatal storage error"
    assert captured[1].politeness() is QAccessible.AnnouncementPoliteness.Assertive


def test_presenter_announces_status_changes_and_fatal_errors(
    qtbot: QtBot,
) -> None:
    announcer = FakeStatusAnnouncer()
    controller = ConfigurableController(state=SessionState.RECORDING)
    window = MainWindow()
    presenter = QtSessionPresenter(controller, window, announcer)
    qtbot.addWidget(window)

    controller.state = SessionState.PAUSED
    presenter.render_current_snapshot()
    fatal = replace(controller.snapshot(), fatal_error="Session storage is unsafe.")
    presenter.render_snapshot(fatal)

    assert ("Paused", False) in announcer.messages
    assert ("Session storage is unsafe.", True) in announcer.messages


class FakeStatusAnnouncer:
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
