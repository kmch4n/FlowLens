"""Changing a selection must never repeat slow environmental checks."""

from dataclasses import replace
from pathlib import Path

import pytest

from flowlens.controller.models import BlockingIssue, PreflightSelection
from flowlens.controller.session_controller import StartBlocked
from flowlens.domain.enums import SessionMode
from tests.controller.test_session_controller import make_controller, selection


def test_selection_uses_report_without_rechecking_environment(tmp_path: Path) -> None:
    controller, _, _, _, _ = make_controller(tmp_path)
    controller.enter_preflight()
    first = controller.refresh_preflight(selection())

    def forbidden(*args: object) -> object:
        raise AssertionError("Selection repeated environmental checks")

    controller._preflight_service.evaluate = forbidden  # type: ignore[method-assign, assignment]
    updated = controller.update_preflight_selection(
        PreflightSelection(SessionMode.INTERVIEW, "mic-1", "out-1")
    )
    assert updated.selection.mode is SessionMode.INTERVIEW
    assert updated.models == first.models
    assert updated.can_start
    assert updated.mic_level == updated.loopback_level == 0.0


def test_selection_preserves_environmental_blockers(tmp_path: Path) -> None:
    controller, _, _, _, _ = make_controller(tmp_path)
    controller.enter_preflight()
    initial = controller.refresh_preflight(selection())
    blocker = BlockingIssue("asr_model", "Model checksum failed")
    controller._snapshot = replace(
        controller.snapshot(),
        preflight=replace(initial, issues=(blocker,), can_start=False),
    )
    updated = controller.update_preflight_selection(
        PreflightSelection(SessionMode.GENERAL, "missing", "out-1")
    )
    assert updated.selection.microphone_id is None
    assert {item.control_id for item in updated.issues} == {"microphone", "asr_model"}
    assert blocker in updated.issues
    assert not updated.can_start


def test_cached_selection_never_bypasses_start_validation(tmp_path: Path) -> None:
    controller, runtime, _, _, _ = make_controller(tmp_path)
    controller.enter_preflight()
    initial = controller.refresh_preflight(selection())
    controller.update_preflight_selection(selection())
    failed = replace(
        initial,
        can_start=False,
        issues=(BlockingIssue("storage", "Storage became unavailable"),),
    )
    controller._preflight_service.evaluate = lambda selection: failed  # type: ignore[method-assign]
    with pytest.raises(StartBlocked):
        controller.start(selection())
    assert runtime.launches == []
    assert controller.snapshot().preflight == failed
