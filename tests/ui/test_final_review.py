"""Real controller/Qt coverage of delayed Audio ACK and ASR presentation."""

from dataclasses import replace
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from flowlens.controller.session_controller import ControllerSnapshot, SessionController
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.ui.main_window import MainWindow
from flowlens.ui.presenter import QtSessionPresenter
from tests.controller.test_session_controller import (
    FakeClock,
    make_controller,
    ready,
    selection,
    stopped_envelope,
    worker_envelope,
    writer_ack,
)
from tests.ui.test_presenter import FakeAnnouncer, FakeConfigStore


def running(
    tmp_path: Path, qtbot: QtBot
) -> tuple[SessionController, FakeClock, MainWindow, QtSessionPresenter]:
    controller, _, clock, _, _ = make_controller(tmp_path, acceptance_enabled=True)
    controller.enter_preflight()
    controller.start(selection())
    controller.handle_message(writer_ack())
    for source in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        controller.handle_message(ready(source))
    window = MainWindow()
    qtbot.addWidget(window)
    presenter = QtSessionPresenter(
        controller, window, FakeAnnouncer(), config_store=FakeConfigStore()
    )
    presenter.timer.stop()
    window.show()
    return controller, clock, window, presenter


def test_delayed_audio_ack_keeps_live_and_slow_dialog_truthful(
    tmp_path: Path, qtbot: QtBot
) -> None:
    controller, clock, window, presenter = running(tmp_path, qtbot)
    controller.request_stop()
    controller.confirm_stop()
    clock.ms = 31_001
    controller.tick()
    presenter.render_current_snapshot()
    assert "Stopping capture" in window.live_page.finalization_progress.text()
    assert "Stopping capture" in window.slow_finalization_dialog.warning_label.text()
    assert "Capture stopped" not in window.slow_finalization_dialog.warning_label.text()
    controller.handle_message(stopped_envelope(ProcessSource.AUDIO))
    presenter.render_current_snapshot()
    assert "Capture stopped" in window.live_page.finalization_progress.text()
    assert "Capture stopped" in window.slow_finalization_dialog.warning_label.text()


def test_asr_delivery_is_measured_only_after_qt_widget_update(
    tmp_path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, clock, window, presenter = running(tmp_path, qtbot)
    controller.handle_message(
        replace(
            worker_envelope(
                ProcessSource.ASR,
                MessageType.TRANSCRIPT_PARTIAL,
                2,
                {
                    "source": "OTHERS",
                    "text": "PRIVATE_SENTINEL",
                    "session_start_ms": 0,
                    "session_end_ms": 100,
                    "source_start_sample": 0,
                    "source_end_sample": 1600,
                },
            ),
            created_monotonic_ms=1100,
        )
    )
    assert dict(getattr(controller.snapshot(), "stage_timings_ms", ())) == {}
    render = window.live_page.render

    def delayed_render(snapshot: ControllerSnapshot) -> None:
        clock.ms = 1175
        render(snapshot)

    monkeypatch.setattr(window.live_page, "render", delayed_render)
    clock.ms = 1120
    presenter.render_current_snapshot()
    assert "PRIVATE_SENTINEL" in window.live_page.transcript_view.rendered_text()
    assert dict(getattr(controller.snapshot(), "stage_timings_ms", ())).get(
        "asr_to_ui_others"
    ) == (75,)
    presenter.render_current_snapshot(force=True)
    assert dict(controller.snapshot().stage_timings_ms)["asr_to_ui_others"] == (75,)


def test_pending_presentation_samples_are_bounded_and_discarded_on_restart(
    tmp_path: Path, qtbot: QtBot
) -> None:
    controller, clock, _, presenter = running(tmp_path, qtbot)
    for index in range(300):
        controller.handle_message(
            worker_envelope(
                ProcessSource.ASR,
                MessageType.TRANSCRIPT_PARTIAL,
                index + 2,
                {
                    "source": "ME",
                    "text": str(index),
                    "session_start_ms": 0,
                    "session_end_ms": 100,
                    "source_start_sample": 0,
                    "source_end_sample": 1600,
                },
            )
        )
    assert len(controller.snapshot().asr_pending_presentations) == 256
    clock.ms = 2000
    controller.handle_worker_exit(ProcessSource.ASR)
    assert controller.snapshot().asr_pending_presentations == ()
    presenter.render_current_snapshot()
    assert controller.snapshot().stage_timings_ms == ()
