"""Startup keeps the event loop responsive and owns one local endpoint."""

import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot


def test_secondary_launch_activates_primary_and_releases_on_close(
    qtbot: QtBot, tmp_path: Path
) -> None:
    from flowlens.ui.single_instance import SingleInstance

    name = f"flowlens-test-{uuid4().hex}"
    first = SingleInstance(name, tmp_path / "instance.lock")
    second = SingleInstance(name, tmp_path / "instance.lock")
    try:
        assert first.acquire()
        with qtbot.waitSignal(first.activation_requested):
            assert not second.acquire()
        first.close()
        assert second.acquire()
    finally:
        first.close()
        second.close()


def test_background_startup_does_not_block_ui(qtbot: QtBot) -> None:
    from flowlens.ui.startup import StartupTask

    entered = Event()
    release = Event()

    def prepare() -> str:
        entered.set()
        assert release.wait(5)
        return "ready"

    task = StartupTask(prepare)
    try:
        task.start()
        qtbot.waitUntil(entered.is_set)
        assert task.isRunning()
        with qtbot.waitSignal(task.completed) as signal:
            release.set()
        assert signal.args == ["ready"]
    finally:
        release.set()
        task.wait()


def test_startup_failure_is_reported(qtbot: QtBot) -> None:
    from flowlens.ui.startup import StartupTask

    def prepare() -> object:
        raise OSError("unavailable")

    task = StartupTask(prepare)
    try:
        with qtbot.waitSignal(task.failed) as signal:
            task.start()
        assert signal.args == ["OSError"]
    finally:
        task.wait()


def test_lock_recovers_after_process_exit(qtbot: QtBot, tmp_path: Path) -> None:
    from flowlens.ui.single_instance import SingleInstance

    name = f"flowlens-test-{uuid4().hex}"
    lock_path = tmp_path / "instance.lock"
    script = (
        "import sys; from pathlib import Path; "
        "from PySide6.QtCore import QCoreApplication; "
        "from flowlens.ui.single_instance import SingleInstance; "
        "app=QCoreApplication([]); "
        "instance=SingleInstance(sys.argv[1],Path(sys.argv[2])); "
        "assert instance.acquire(); print('locked',flush=True); app.exec()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, name, str(lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    recovered = SingleInstance(name, lock_path)
    try:
        qtbot.waitUntil(lock_path.exists, timeout=5000)
        process.kill()
        process.wait(timeout=5)
        assert recovered.acquire()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        recovered.close()


@pytest.mark.parametrize("restore_interview", [False, True])
def test_primary_shows_selected_session_before_recovery_finishes(
    qtbot: QtBot, tmp_path: Path, monkeypatch: MonkeyPatch, restore_interview: bool
) -> None:
    # QApplication.exec/quit owns process-global state; isolate full lifecycle tests.
    if os.environ.get("FLOWLENS_STARTUP_TEST_CHILD") != "1":
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_primary_shows_selected_session_before_recovery_finishes[{restore_interview}]",
                "-q",
            ],
            env={**os.environ, "FLOWLENS_STARTUP_TEST_CHILD": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    from dataclasses import replace

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    import flowlens.app as app_module
    from flowlens.config.model import AppConfig
    from flowlens.config.store import ConfigStore
    from flowlens.domain.enums import SessionMode
    from flowlens.integration.composition import AppOptions
    from flowlens.ui.main_window import MainWindow
    from tests.integration.test_composition import fake_paths
    from tests.ui.test_presenter import RecordingController

    application = QApplication.instance()
    assert isinstance(application, QApplication)
    previous_stylesheet = application.styleSheet()
    controller = RecordingController()
    if restore_interview:
        ConfigStore(tmp_path / "config.json").save(
            replace(AppConfig.default(), last_mode=SessionMode.INTERVIEW)
        )
    entered = Event()
    release = Event()
    observed: list[bool] = []
    windows: list[MainWindow] = []

    def recover(paths: Any) -> None:
        assert (paths.root / "instance.lock").exists()
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(app_module, "_recover_startup_sessions", recover)
    monkeypatch.setattr(
        app_module,
        "build_application",
        lambda *args: SimpleNamespace(controller=SimpleNamespace(session=controller)),
    )
    deadline = time.monotonic() + 5

    def inspect() -> None:
        for widget in application.topLevelWidgets():
            if isinstance(widget, MainWindow) and widget.isVisible():
                if widget not in windows:
                    windows.append(widget)
                if entered.is_set() and not release.is_set():
                    observed.append(
                        (
                            widget.preflight_page.interview_radio.isChecked()
                            if restore_interview
                            else widget.preflight_page.meeting_radio.isChecked()
                        )
                        and not widget.preflight_page.start_button.isEnabled()
                    )
                    release.set()
                if hasattr(widget, "_flowlens_presenter"):
                    application.quit()
        if time.monotonic() > deadline:
            release.set()
            application.quit()

    timer = QTimer()
    timer.timeout.connect(inspect)
    timer.start(10)
    try:
        assert app_module._run_qt(fake_paths(tmp_path), AppOptions()) == 0
        assert observed == [True]
        assert len(controller.refreshed) == 1
        assert any(hasattr(window, "_flowlens_presenter") for window in windows)
        assert not (tmp_path / "instance.lock").exists()
    finally:
        release.set()
        timer.stop()
        for window in windows:
            qtbot.addWidget(window)
            window.close()
        application.setStyleSheet(previous_stylesheet)
