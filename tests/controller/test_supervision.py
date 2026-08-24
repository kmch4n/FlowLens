"""Deterministic session-scoped worker supervision tests."""

from collections.abc import Mapping
from typing import cast

import pytest

from flowlens.controller.supervision import (
    RecoveryAction,
    RecoveryReason,
    WorkerSupervisor,
)
from flowlens.domain.enums import ProcessSource

SESSION_ID = "01J00000000000000000000000"
NEXT_SESSION_ID = "01J00000000000000000000001"


def test_worker_recovery_policy_is_once_only_per_session() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)

    assert supervisor.on_exit(ProcessSource.ASR).action is RecoveryAction.RESTART
    assert supervisor.on_exit(ProcessSource.ASR).action is RecoveryAction.SAFE_STOP
    assert supervisor.on_exit(ProcessSource.DISCUSSION).action is RecoveryAction.RESTART
    assert (
        supervisor.on_exit(ProcessSource.DISCUSSION).action
        is RecoveryAction.DISABLE_ANALYSIS
    )


def test_writer_and_audio_have_exact_fail_closed_policies() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)

    assert supervisor.on_exit(ProcessSource.WRITER).action is RecoveryAction.FATAL
    assert supervisor.on_exit(ProcessSource.AUDIO).action is RecoveryAction.SAFE_STOP


def test_same_session_reset_is_rejected_but_new_session_resets_restarts() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)
    assert supervisor.on_exit(ProcessSource.ASR).action is RecoveryAction.RESTART

    with pytest.raises(ValueError, match="new session"):
        supervisor.reset(SESSION_ID)
    supervisor.reset(NEXT_SESSION_ID)

    assert supervisor.on_exit(ProcessSource.ASR).action is RecoveryAction.RESTART


def test_health_poll_is_due_every_250_ms_and_clock_rollback_fails_closed() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)

    assert supervisor.health_poll_due(100) is True
    assert supervisor.health_poll_due(349) is False
    assert supervisor.health_poll_due(350) is True
    with pytest.raises(RuntimeError, match="rollback"):
        supervisor.health_poll_due(349)


def test_simultaneous_exits_are_reported_in_safety_priority_order_once() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)
    healthy: dict[ProcessSource, bool] = {
        source: True for source in ProcessSource if source is not ProcessSource.GUI
    }
    supervisor.observe_health(healthy)
    failed = healthy | {
        ProcessSource.WRITER: False,
        ProcessSource.ASR: False,
        ProcessSource.DISCUSSION: False,
    }

    decisions = supervisor.observe_health(failed)

    assert [decision.worker for decision in decisions] == [
        ProcessSource.WRITER,
        ProcessSource.ASR,
        ProcessSource.DISCUSSION,
    ]
    assert supervisor.observe_health(failed) == ()


def test_hostile_or_incomplete_health_map_fails_closed_without_mutation() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)
    valid: dict[ProcessSource, bool] = {
        source: True for source in ProcessSource if source is not ProcessSource.GUI
    }
    supervisor.observe_health(valid)

    with pytest.raises(ValueError, match="health map"):
        supervisor.observe_health({ProcessSource.AUDIO: True})
    hostile: dict[ProcessSource, object] = valid | {ProcessSource.AUDIO: 1}
    with pytest.raises(ValueError, match="health map"):
        supervisor.observe_health(cast(Mapping[ProcessSource, bool], hostile))

    assert supervisor.observe_health(valid) == ()


def test_gpu_oom_always_disables_analysis_without_consuming_restart() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)

    decision = supervisor.on_gpu_oom()

    assert decision.worker is ProcessSource.DISCUSSION
    assert decision.action is RecoveryAction.DISABLE_ANALYSIS
    assert decision.reason is RecoveryReason.GPU_OOM
    assert supervisor.on_exit(ProcessSource.DISCUSSION).action is RecoveryAction.RESTART


def test_exit_disable_reason_is_explicit() -> None:
    supervisor = WorkerSupervisor(SESSION_ID)
    supervisor.on_exit(ProcessSource.DISCUSSION)

    decision = supervisor.on_exit(ProcessSource.DISCUSSION)

    assert decision.action is RecoveryAction.DISABLE_ANALYSIS
    assert decision.reason is RecoveryReason.EXIT
