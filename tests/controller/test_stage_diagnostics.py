"""Privacy, bounds and lifecycle checks for segmented acceptance diagnostics."""

from dataclasses import replace
from pathlib import Path

import pytest

from flowlens.app import _controller_measurements
from flowlens.controller.session_controller import SessionLaunch
from flowlens.domain.enums import MessageType, ProcessSource
from tests.controller.test_session_controller import (
    asr_status,
    make_controller,
    ready,
    selection,
    worker_envelope,
    writer_ack,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_input_timings_are_bounded_opt_in_and_source_separated(
    tmp_path: Path, enabled: bool
) -> None:
    controller, runtime, _, _, _ = make_controller(tmp_path, acceptance_enabled=enabled)
    controller.enter_preflight()
    controller.start(selection())
    launch = runtime.launches[0]
    assert isinstance(launch, SessionLaunch)
    assert launch.asr_config.acceptance_diagnostics is enabled
    controller.handle_message(writer_ack())
    for source in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        controller.handle_message(ready(source))
    before = len(runtime.sent)
    for index in range(300):
        controller.handle_message(
            worker_envelope(
                ProcessSource.ASR,
                MessageType.ASR_INPUT_TIMING,
                index + 2,
                {"source": "ME", "capture_to_asr_ms": index, "backlog_ms": index + 10},
            )
        )
    controller.handle_message(
        worker_envelope(
            ProcessSource.ASR,
            MessageType.ASR_INPUT_TIMING,
            302,
            {"source": "OTHERS", "capture_to_asr_ms": 25, "backlog_ms": 35},
        )
    )
    values = dict(controller.snapshot().stage_timings_ms)
    assert values == (
        {
            "capture_to_asr_me": tuple(range(44, 300)),
            "backlog_me": tuple(range(54, 310)),
            "capture_to_asr_others": (25,),
            "backlog_others": (35,),
        }
        if enabled
        else {}
    )
    assert len(runtime.sent) == before


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "PRIVATE_DEVICE", "capture_to_asr_ms": 1, "backlog_ms": 2},
        {"source": "ME", "capture_to_asr_ms": -1, "backlog_ms": 2},
        {"source": "ME", "capture_to_asr_ms": True, "backlog_ms": 2},
        {"source": "ME", "capture_to_asr_ms": 1, "backlog_ms": 1.5},
        {"source": "ME", "capture_to_asr_ms": 1, "backlog_ms": 2, "text": "PRIVATE"},
    ],
)
def test_input_timings_reject_content_invalid_values_and_stale_generations(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    controller, _, clock, _, _ = make_controller(tmp_path, acceptance_enabled=True)
    controller.enter_preflight()
    controller.start(selection())
    controller.handle_message(writer_ack())
    for source in (ProcessSource.AUDIO, ProcessSource.ASR, ProcessSource.DISCUSSION):
        controller.handle_message(ready(source))
    timing = worker_envelope(
        ProcessSource.ASR,
        MessageType.ASR_INPUT_TIMING,
        2,
        {"source": "ME", "capture_to_asr_ms": 37, "backlog_ms": 42},
    )
    controller.handle_message(replace(timing, payload=payload))
    assert controller.snapshot().stage_timings_ms == ()
    controller.handle_message(replace(timing, session_id="01J00000000000000000000001"))
    controller.handle_message(timing)
    controller.handle_message(timing)
    clock.ms = 2000
    controller.handle_worker_exit(ProcessSource.ASR)
    controller.handle_message(replace(timing, sequence=3))
    controller.handle_message(ready(ProcessSource.ASR))
    controller.handle_message(asr_status(2, 0, 0, state="READY"))
    controller.handle_message(asr_status(3, 0, 0))
    controller.handle_message(replace(timing, sequence=4))
    controller.handle_message(replace(timing, sequence=4, created_monotonic_ms=2001))
    controller.handle_message(
        worker_envelope(
            ProcessSource.AUDIO,
            MessageType.SOURCE_DISCONNECTED,
            2,
            {"source": "ME", "device_id": "PRIVATE_DEVICE"},
        )
    )
    controller.handle_message(
        replace(
            timing,
            sequence=5,
            created_monotonic_ms=2002,
            payload={"source": "OTHERS", "capture_to_asr_ms": 0, "backlog_ms": 0},
        )
    )
    assert dict(controller.snapshot().stage_timings_ms) == {
        "capture_to_asr_me": (37, 37),
        "backlog_me": (42, 42),
        "capture_to_asr_others": (0,),
        "backlog_others": (0,),
    }
    assert "PRIVATE" not in str(_controller_measurements(controller.snapshot()))
