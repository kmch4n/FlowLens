"""ASR generation changes preserve session metrics and analysis controls."""

from pathlib import Path

import pytest

from flowlens.domain.enums import MessageType, ProcessSource
from tests.controller.test_session_controller import (
    asr_status,
    ready,
    recording_controller,
    selection,
    sent_to,
    writer_ack,
)


def test_restart_accepts_new_generation_maximum_and_preserves_session_maximum(
    tmp_path: Path,
) -> None:
    controller, _, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 7_000, 7_000))
    controller.handle_worker_exit(ProcessSource.ASR)
    controller.handle_message(ready(ProcessSource.ASR))
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    controller.handle_message(asr_status(3, 0, 0))
    controller.handle_message(asr_status(4, 6_000, 6_000))
    assert controller.snapshot().asr_backlog_ms == 6_000
    assert controller.snapshot().maximum_asr_backlog_ms == 7_000
    controller.handle_message(asr_status(5, 0, 5_000))
    assert controller.snapshot().asr_backlog_ms == 6_000


@pytest.mark.parametrize("paused", [False, True])
def test_restart_reconciles_analysis_pause_once(tmp_path: Path, paused: bool) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 7_000, 7_000))
    if paused:
        controller.pause()
    controller.handle_worker_exit(ProcessSource.ASR)
    controller.handle_message(ready(ProcessSource.ASR))
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    assert not any(
        item.message_type is MessageType.WORKER_RESUME
        for item in sent_to(runtime, ProcessSource.DISCUSSION)
    )
    if paused:
        controller.resume()
    controller.handle_message(asr_status(3, 0, 0))
    controller.handle_message(asr_status(4, 0, 0))
    resumes = [
        item
        for item in sent_to(runtime, ProcessSource.DISCUSSION)
        if item.message_type is MessageType.WORKER_RESUME
    ]
    assert len(resumes) == 1
    assert controller.snapshot().analysis_status == "Running"


def test_new_session_resets_asr_backlog_statistics(tmp_path: Path) -> None:
    controller, _, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 7_000, 7_000))
    controller.handle_worker_exit(ProcessSource.WRITER)
    controller.enter_preflight()
    controller.start(selection())
    assert controller.snapshot().maximum_asr_backlog_ms == 0
    assert controller.snapshot().asr_backlog_ms == 0
    controller.handle_message(writer_ack())
    for source in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        controller.handle_message(ready(source))
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    assert controller.snapshot().asr_status == "Ready"


def test_asr_recovery_does_not_resume_disabled_analysis(tmp_path: Path) -> None:
    controller, runtime, _, _ = recording_controller(tmp_path)
    controller.handle_message(asr_status(2, 7_000, 7_000))
    controller.handle_worker_exit(ProcessSource.DISCUSSION)
    controller.handle_worker_exit(ProcessSource.DISCUSSION)
    controller.handle_worker_exit(ProcessSource.ASR)
    controller.handle_message(ready(ProcessSource.ASR))
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    controller.handle_message(asr_status(3, 0, 0))
    assert controller.snapshot().analysis_status == "Unavailable"
    assert not any(
        item.message_type is MessageType.WORKER_RESUME
        for item in sent_to(runtime, ProcessSource.DISCUSSION)
    )
