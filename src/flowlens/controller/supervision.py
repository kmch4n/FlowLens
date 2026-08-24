"""Session-scoped deterministic worker supervision policies."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from flowlens.domain.enums import ProcessSource

HEALTH_POLL_INTERVAL_MS = 250
_WORKERS = frozenset(
    {
        ProcessSource.AUDIO,
        ProcessSource.ASR,
        ProcessSource.DISCUSSION,
        ProcessSource.WRITER,
    }
)
_SAFETY_ORDER = (
    ProcessSource.WRITER,
    ProcessSource.AUDIO,
    ProcessSource.ASR,
    ProcessSource.DISCUSSION,
)


class RecoveryAction(str, Enum):
    """One exact controller response to a worker failure."""

    FATAL = "FATAL"
    SAFE_STOP = "SAFE_STOP"
    RESTART = "RESTART"
    DISABLE_ANALYSIS = "DISABLE_ANALYSIS"


class RecoveryReason(str, Enum):
    """Explicit origin of one recovery decision."""

    EXIT = "EXIT"
    GPU_OOM = "GPU_OOM"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Immutable worker recovery decision."""

    worker: ProcessSource
    action: RecoveryAction
    reason: RecoveryReason


class WorkerSupervisor:
    """Apply bounded once-only policies within one active session."""

    def __init__(self, session_id: str) -> None:
        self._session_id = _session_id(session_id)
        self._restart_used: set[ProcessSource] = set()
        self._down: set[ProcessSource] = set()
        self._last_health_poll_ms: int | None = None

    def reset(self, session_id: str) -> None:
        """Reset policy state only when a distinct session begins."""

        validated = _session_id(session_id)
        if validated == self._session_id:
            raise ValueError("supervisor reset requires a new session")
        self._session_id = validated
        self._restart_used.clear()
        self._down.clear()
        self._last_health_poll_ms = None

    def on_exit(self, worker: ProcessSource) -> RecoveryDecision:
        """Return the exact next action for one worker exit."""

        if worker not in _WORKERS:
            raise ValueError("worker must name a supervised process")
        if worker is ProcessSource.WRITER:
            return RecoveryDecision(worker, RecoveryAction.FATAL, RecoveryReason.EXIT)
        if worker is ProcessSource.AUDIO:
            return RecoveryDecision(
                worker,
                RecoveryAction.SAFE_STOP,
                RecoveryReason.EXIT,
            )
        if worker not in self._restart_used:
            self._restart_used.add(worker)
            return RecoveryDecision(worker, RecoveryAction.RESTART, RecoveryReason.EXIT)
        if worker is ProcessSource.ASR:
            return RecoveryDecision(
                worker,
                RecoveryAction.SAFE_STOP,
                RecoveryReason.EXIT,
            )
        return RecoveryDecision(
            worker,
            RecoveryAction.DISABLE_ANALYSIS,
            RecoveryReason.EXIT,
        )

    def on_gpu_oom(self) -> RecoveryDecision:
        """Disable Discussion without consuming its one restart allowance."""

        return RecoveryDecision(
            ProcessSource.DISCUSSION,
            RecoveryAction.DISABLE_ANALYSIS,
            RecoveryReason.GPU_OOM,
        )

    def health_poll_due(self, now_ms: int) -> bool:
        """Advance the 250 ms poll clock, rejecting rollback."""

        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("now_ms must be a non-negative exact integer")
        previous = self._last_health_poll_ms
        if previous is not None and now_ms < previous:
            raise RuntimeError("monotonic clock rollback")
        if previous is None or now_ms - previous >= HEALTH_POLL_INTERVAL_MS:
            self._last_health_poll_ms = now_ms
            return True
        return False

    def observe_health(
        self,
        health: Mapping[ProcessSource, bool],
    ) -> tuple[RecoveryDecision, ...]:
        """Return newly observed exits in deterministic safety order."""

        if not isinstance(health, Mapping) or frozenset(health) != _WORKERS:
            raise ValueError("health map must contain exactly all workers")
        if not all(
            isinstance(source, ProcessSource) and type(value) is bool
            for source, value in health.items()
        ):
            raise ValueError("health map must contain exact boolean values")
        newly_down = tuple(
            worker
            for worker in _SAFETY_ORDER
            if not health[worker] and worker not in self._down
        )
        self._down = {worker for worker in _WORKERS if not health[worker]}
        return tuple(self.on_exit(worker) for worker in newly_down)


def _session_id(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("session_id must be a non-empty exact string")
    return value
