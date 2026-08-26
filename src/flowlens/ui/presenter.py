from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QRect, QTimer
from PySide6.QtGui import (
    QAccessible,
    QAccessibleAnnouncementEvent,
)
from PySide6.QtWidgets import QApplication, QWidget

from flowlens.adapters.windows_shell import WindowsFolderOpener
from flowlens.config.model import AppConfig, DevicePreferences
from flowlens.config.store import ConfigStore
from flowlens.controller.models import PreflightReport, PreflightSelection
from flowlens.controller.ports import FolderOpener
from flowlens.controller.session_controller import ControllerSnapshot, SessionState
from flowlens.domain.enums import SessionMode
from flowlens.ui.main_window import MainWindow


class _Controller(Protocol):
    state: SessionState

    def snapshot(self) -> ControllerSnapshot: ...

    def enter_preflight(self) -> None: ...

    def refresh_preflight(self, selection: PreflightSelection) -> PreflightReport: ...

    def start(self, selection: PreflightSelection) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def request_stop(self) -> None: ...

    def cancel_stop(self) -> None: ...

    def confirm_stop(self) -> None: ...

    def keep_waiting(self) -> None: ...

    def force_close(self) -> None: ...

    def tick(self) -> None: ...


class _ConfigStore(Protocol):
    def load(self) -> AppConfig: ...

    def save(self, config: AppConfig) -> None: ...


class _Announcer(Protocol):
    def announce(
        self,
        widget: object,
        message: str,
        assertive: bool = False,
    ) -> None: ...


class QtAccessibilityAnnouncer:
    """Send Qt accessibility announcement events with actual Qt politeness."""

    def announce(
        self,
        widget: object,
        message: str,
        assertive: bool = False,
    ) -> None:
        """Announce one status change through QAccessible."""

        if not isinstance(widget, QWidget):
            return
        event = QAccessibleAnnouncementEvent(widget, message)
        politeness = (
            QAccessible.AnnouncementPoliteness.Assertive
            if assertive
            else QAccessible.AnnouncementPoliteness.Polite
        )
        event.setPoliteness(politeness)
        QAccessible.updateAccessibility(event)


class QtSessionPresenter:
    """Bind the controller snapshot contract to the Qt shell."""

    def __init__(
        self,
        controller: _Controller,
        window: MainWindow,
        announcer: _Announcer,
        *,
        config_store: _ConfigStore | None = None,
        folder_opener: FolderOpener | None = None,
    ) -> None:
        self.controller = controller
        self.window = window
        self.announcer = announcer
        self.config_store = (
            config_store if config_store is not None else _default_store()
        )
        self.folder_opener = (
            folder_opener if folder_opener is not None else WindowsFolderOpener()
        )
        self.timer = QTimer(window)
        self.timer.setInterval(50)
        self.render_count = 0
        self.last_timer_error: str | None = None
        self._in_timer = False
        self._last_snapshot: ControllerSnapshot | None = None
        self._last_render_key: tuple[ControllerSnapshot, str | None] | None = None
        self._close_block_message: str | None = None
        self._selection = PreflightSelection(SessionMode.MEETING, None, None)
        self._completion_path: Path | None = None

        self._connect_signals()
        self._restore_preferences()
        self.render_current_snapshot(force=True)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()
        self.window.destroyed.connect(self._stop_timer)

    def on_timer(self) -> None:
        """Drain worker-facing updates, tick the controller, then render changes."""

        if self._in_timer:
            return
        self._in_timer = True
        try:
            drain = getattr(self.controller, "drain_messages", None)
            if callable(drain):
                drain()
            self.controller.tick()
            self.render_current_snapshot()
        except Exception as error:
            self.last_timer_error = str(error)
        finally:
            self._in_timer = False

    def render_current_snapshot(self, *, force: bool = False) -> None:
        """Render the controller's current authoritative snapshot."""

        self.render_snapshot(self.controller.snapshot(), force=force)

    def render_snapshot(
        self,
        snapshot: ControllerSnapshot,
        *,
        force: bool = False,
    ) -> None:
        """Render one explicit snapshot if it changed."""

        if not isinstance(snapshot, ControllerSnapshot):
            raise TypeError("snapshot must be a ControllerSnapshot")
        if snapshot.state not in {SessionState.STARTING, SessionState.STOPPING}:
            self._close_block_message = None
        key = (snapshot, self._close_block_message)
        previous = self._last_snapshot
        if not force and self._last_render_key == key:
            return
        display_snapshot = self._snapshot_with_shell_message(snapshot)
        self._render_snapshot(display_snapshot)
        self._announce_snapshot_changes(previous, display_snapshot)
        self._last_snapshot = snapshot
        self._last_render_key = key
        self.render_count += 1

    def on_selection_changed(self, selection: PreflightSelection) -> None:
        """Refresh preflight from a complete UI selection."""

        if not isinstance(selection, PreflightSelection):
            raise TypeError("selection must be a PreflightSelection")
        self._selection = selection
        if self.controller.snapshot().state is SessionState.PREFLIGHT:
            try:
                self.controller.refresh_preflight(selection)
            except Exception:
                return
            self.render_current_snapshot()

    def save_preferences(self) -> None:
        """Persist exactly the approved non-session preferences schema."""

        self.config_store.save(
            AppConfig(
                schema_version=1,
                window=self.window.window_preferences(),
                devices=DevicePreferences(
                    microphone_id=self._selection.microphone_id or "",
                    loopback_output_id=self._selection.loopback_output_id or "",
                ),
                last_mode=self._selection.mode,
            )
        )

    def _connect_signals(self) -> None:
        self.window.selection_changed.connect(self.on_selection_changed)
        self.window.start_requested.connect(self._start_requested)
        self.window.pause_resume_requested.connect(self._pause_or_resume_requested)
        self.window.stop_requested.connect(self._request_stop)
        self.window.stop_confirmed_requested.connect(self._confirm_stop)
        self.window.stop_cancel_requested.connect(self._cancel_stop)
        self.window.active_close_requested.connect(self._active_close_requested)
        self.window.orderly_close_requested.connect(self._orderly_close_requested)
        self.window.always_on_top_changed.connect(self._always_on_top_changed)
        self.window.keep_waiting_requested.connect(self._keep_waiting)
        self.window.force_close_requested.connect(self._force_close)
        self.window.completion_page.open_folder_requested.connect(self._open_folder)
        self.window.completion_page.start_another_requested.connect(self._start_another)

    def _restore_preferences(self) -> None:
        config = self.config_store.load()
        self._selection = PreflightSelection(
            config.last_mode,
            config.devices.microphone_id or None,
            config.devices.loopback_output_id or None,
        )
        self.window.apply_window_preferences(config.window, _available_geometries())
        if self._controller_state() is SessionState.IDLE:
            self.controller.enter_preflight()
        if self._controller_state() is SessionState.PREFLIGHT:
            self._refresh_preflight_with_available_devices(self._selection)

    def _refresh_preflight_with_available_devices(
        self,
        selection: PreflightSelection,
    ) -> None:
        report = self.controller.refresh_preflight(selection)
        microphone_ids = {device.id for device in report.microphones}
        loopback_ids = {
            device.id for device in report.loopbacks if device.loopback_capable
        }
        sanitized = PreflightSelection(
            selection.mode,
            (
                selection.microphone_id
                if selection.microphone_id in microphone_ids
                else None
            ),
            (
                selection.loopback_output_id
                if selection.loopback_output_id in loopback_ids
                else None
            ),
        )
        self._selection = sanitized
        if sanitized != selection:
            self.controller.refresh_preflight(sanitized)

    def _render_snapshot(self, snapshot: ControllerSnapshot) -> None:
        self.window.set_preflight_can_start(
            snapshot.preflight.can_start if snapshot.preflight is not None else False
        )
        self.window.set_active_close_requires_stop(
            snapshot.state
            in {
                SessionState.STARTING,
                SessionState.RECORDING,
                SessionState.PAUSED,
                SessionState.STOPPING,
            }
        )
        if snapshot.preflight is not None:
            self._selection = snapshot.preflight.selection
        if snapshot.state is SessionState.COMPLETED:
            if snapshot.completion is None:
                self._completion_path = None
                self.window.show_live()
                self.window.live_page.render(snapshot)
                return
            self._completion_path = snapshot.completion.save_path
            self.window.show_completion(snapshot.completion)
        elif snapshot.state in {
            SessionState.RECORDING,
            SessionState.PAUSED,
            SessionState.STARTING,
            SessionState.STOPPING,
        }:
            self.window.show_live()
            self.window.live_page.render(snapshot)
        else:
            if snapshot.preflight is not None:
                self.window.preflight_page.render(snapshot.preflight)
            self.window.show_preflight()
        self.window.set_stop_confirmation_visible(snapshot.stop_confirmation_visible)
        self.window.set_slow_finalization_visible(snapshot.slow_finalization_visible)

    def _announce_snapshot_changes(
        self,
        previous: ControllerSnapshot | None,
        snapshot: ControllerSnapshot,
    ) -> None:
        if previous is None:
            return
        if (
            snapshot.recording_status
            and snapshot.recording_status != previous.recording_status
        ):
            self.announcer.announce(self.window, snapshot.recording_status, False)
        if snapshot.issue and snapshot.issue != previous.issue:
            self.announcer.announce(self.window, snapshot.issue, False)
        if snapshot.fatal_error and snapshot.fatal_error != previous.fatal_error:
            self.announcer.announce(self.window, snapshot.fatal_error, True)

    def _start_requested(self) -> None:
        snapshot = self.controller.snapshot()
        if snapshot.state is not SessionState.PREFLIGHT:
            return
        if snapshot.preflight is None or not snapshot.preflight.can_start:
            return
        self.controller.start(self._selection)
        self.render_current_snapshot()

    def _pause_or_resume_requested(self) -> None:
        started_ns = time.perf_counter_ns()
        state = self.controller.snapshot().state
        if state is SessionState.RECORDING:
            self.controller.pause()
        elif state is SessionState.PAUSED:
            self.controller.resume()
        else:
            return
        self.render_current_snapshot()
        recorder = getattr(self.controller, "record_ui_feedback", None)
        if callable(recorder):
            latency_ms = max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
            recorder(latency_ms)

    def _request_stop(self) -> None:
        snapshot = self.controller.snapshot()
        if snapshot.stop_confirmation_visible:
            self.render_current_snapshot()
            return
        if snapshot.state not in {SessionState.RECORDING, SessionState.PAUSED}:
            return
        self.controller.request_stop()
        self.render_current_snapshot()

    def _active_close_requested(self) -> None:
        snapshot = self.controller.snapshot()
        if snapshot.state in {SessionState.RECORDING, SessionState.PAUSED}:
            self._request_stop()
            return
        if snapshot.state is SessionState.STARTING:
            self._show_close_block_message(
                "Session startup is still in progress.",
                assertive=False,
            )
            return
        if snapshot.state is SessionState.STOPPING:
            self._show_close_block_message(
                "Finalization is in progress.",
                assertive=False,
            )

    def _cancel_stop(self) -> None:
        try:
            self.controller.cancel_stop()
        finally:
            self.render_current_snapshot()

    def _confirm_stop(self) -> None:
        self.controller.confirm_stop()
        self.render_current_snapshot()

    def _keep_waiting(self) -> None:
        self.controller.keep_waiting()
        self.render_current_snapshot()

    def _force_close(self) -> None:
        self.controller.force_close()
        self.render_current_snapshot()

    def _always_on_top_changed(self, enabled: bool) -> None:
        self.window.set_always_on_top(enabled)
        self.save_preferences()

    def _open_folder(self) -> None:
        if self._completion_path is not None:
            self.folder_opener.open(self._completion_path)

    def _start_another(self) -> None:
        self.window.reset_live_page()
        self.controller.enter_preflight()
        self._refresh_preflight_with_available_devices(self._selection)
        self.render_current_snapshot(force=True)

    def _controller_state(self) -> SessionState:
        return self.controller.snapshot().state

    def _snapshot_with_shell_message(
        self,
        snapshot: ControllerSnapshot,
    ) -> ControllerSnapshot:
        if self._close_block_message is None:
            return snapshot
        return replace(snapshot, issue=self._close_block_message)

    def _show_close_block_message(self, message: str, *, assertive: bool) -> None:
        self._close_block_message = message
        self.render_current_snapshot(force=True)
        del assertive

    def _orderly_close_requested(self) -> None:
        self._stop_timer()
        self.save_preferences()

    def _stop_timer(self, *_: object) -> None:
        self.timer.stop()


def _default_store() -> ConfigStore:
    local_appdata = os.environ.get("LOCALAPPDATA")
    root = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return ConfigStore(root / "FlowLens" / "config.json")


def _available_geometries() -> tuple[QRect, ...]:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return (QRect(100, 100, 1280, 800),)
    screens = app.screens()
    if not screens:
        return (QRect(100, 100, 1280, 800),)
    return tuple(screen.availableGeometry() for screen in screens)
