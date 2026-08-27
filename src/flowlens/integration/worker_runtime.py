"""Multiprocessing runtime composition for FlowLens worker processes."""

from __future__ import annotations

import multiprocessing
import queue
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from multiprocessing.context import BaseContext
from typing import NoReturn, Protocol, cast

from flowlens.controller.session_controller import SessionLaunch
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    MessageEnvelope,
    WriterForceCloseRequest,
    WriterForceCloseResult,
    WriterShutdown,
)
from flowlens.offline_imports import import_local_module
from flowlens.workers.finalization_gate import WriterFinalizationGate

_PROCESS_ORDER = (
    ProcessSource.WRITER,
    ProcessSource.AUDIO,
    ProcessSource.ASR,
    ProcessSource.DISCUSSION,
)
_START_ORDER = _PROCESS_ORDER
_RESTARTABLE = frozenset({ProcessSource.ASR, ProcessSource.DISCUSSION})
_DEFAULT_POLL_BUDGET = 64
_DEFAULT_JOIN_TIMEOUT_SECONDS = 1.0


class _QueueLike(Protocol):
    def put(
        self,
        item: object,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...

    def put_nowait(self, item: object) -> None: ...

    def get_nowait(self) -> object: ...


class _CloseableQueue(_QueueLike, Protocol):
    def close(self) -> None: ...

    def join_thread(self) -> None: ...


class _EventLike(Protocol):
    def set(self) -> None: ...

    def is_set(self) -> bool: ...


class _ProcessLike(Protocol):
    name: str
    daemon: bool

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def close(self) -> None: ...


class _ContextLike(Protocol):
    def get_start_method(self) -> str: ...

    def Queue(self, maxsize: int = 0) -> _QueueLike: ...

    def Event(self) -> _EventLike: ...

    def Process(
        self,
        *,
        name: str,
        target: Callable[..., object],
        args: tuple[object, ...],
    ) -> _ProcessLike: ...


@dataclass(frozen=True, slots=True)
class WorkerTargetReference:
    """Spawn-picklable late import reference to one worker entry point."""

    module: str
    attribute: str

    def __call__(self, *args: object) -> None:
        target = getattr(import_local_module(self.module), self.attribute)
        if not callable(target):
            raise TypeError(f"{self.module}.{self.attribute} is not callable")
        target(*args)


@dataclass(frozen=True, slots=True)
class WorkerTargets:
    """The four process entry points used by the production runtime."""

    writer: Callable[..., object]
    audio: Callable[..., object]
    asr: Callable[..., object]
    discussion: Callable[..., object]


@dataclass(frozen=True, slots=True)
class AudioQueueBindings:
    """Dedicated audio queues exposed only to Audio, Writer, and ASR."""

    writer_audio_out: _QueueLike
    asr_audio_out: _QueueLike


@dataclass(frozen=True, slots=True)
class WorkerShutdownReport:
    """Result of bounded runtime shutdown without any completion claim."""

    completed: bool
    joined: tuple[ProcessSource, ...]
    terminated: tuple[ProcessSource, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkerRestartReport:
    """Result of replacing one restartable worker process."""

    worker: ProcessSource
    joined: tuple[ProcessSource, ...]
    terminated: tuple[ProcessSource, ...]
    errors: tuple[str, ...] = ()


def production_worker_targets() -> WorkerTargets:
    """Return late-bound local worker targets without importing native runtimes."""

    return WorkerTargets(
        writer=WorkerTargetReference(
            "flowlens.workers.writer",
            "run_writer_worker",
        ),
        audio=WorkerTargetReference(
            "flowlens.audio.worker",
            "run_audio_worker",
        ),
        asr=WorkerTargetReference(
            "flowlens.asr.worker",
            "run_asr_worker",
        ),
        discussion=WorkerTargetReference(
            "flowlens.discussion.worker",
            "run_discussion_worker",
        ),
    )


class MultiprocessingWorkerRuntime:
    """Own process objects and queues for one five-process FlowLens session."""

    def __init__(
        self,
        context: _ContextLike | None = None,
        worker_targets: WorkerTargets | None = None,
        *,
        poll_budget: int = _DEFAULT_POLL_BUDGET,
        join_timeout_seconds: float = _DEFAULT_JOIN_TIMEOUT_SECONDS,
    ) -> None:
        self.context = (
            multiprocessing.get_context("spawn") if context is None else context
        )
        if self.context.get_start_method() != "spawn":
            raise ValueError("FlowLens worker runtime requires spawn context")
        if type(poll_budget) is not int or poll_budget <= 0:
            raise ValueError("poll_budget must be a positive integer")
        if (
            not isinstance(join_timeout_seconds, int | float)
            or isinstance(join_timeout_seconds, bool)
            or float(join_timeout_seconds) <= 0.0
        ):
            raise ValueError("join_timeout_seconds must be positive")
        self._targets = (
            production_worker_targets() if worker_targets is None else worker_targets
        )
        self._poll_budget = poll_budget
        self._join_timeout_seconds = float(join_timeout_seconds)
        self._launch: SessionLaunch | None = None
        self._control_queues: dict[ProcessSource, _QueueLike] = {}
        self._processes: dict[ProcessSource, _ProcessLike] = {}
        self._response_queue: _QueueLike | None = None
        self._writer_audio_queue: _QueueLike | None = None
        self._asr_audio_queue: _QueueLike | None = None
        self._writer_stop_event: _EventLike | None = None
        self._writer_finalization_gate: WriterFinalizationGate | None = None
        self._shutdown_report: WorkerShutdownReport | None = None
        self._restart_cleanup_incomplete = False
        self.restart_reports: list[WorkerRestartReport] = []

    @property
    def control_queues(self) -> Mapping[ProcessSource, _QueueLike]:
        """Return worker control queues for integration diagnostics."""

        return self._control_queues

    @property
    def processes(self) -> Mapping[ProcessSource, _ProcessLike]:
        """Return active process handles keyed by worker source."""

        return self._processes

    @property
    def response_queue(self) -> _QueueLike:
        """Return the one shared worker-to-GUI response queue."""

        return self._require_queue(self._response_queue, "response")

    @property
    def writer_audio_queue(self) -> _QueueLike:
        """Return the bounded Audio-to-Writer queue."""

        return self._require_queue(self._writer_audio_queue, "Writer audio")

    @property
    def asr_audio_queue(self) -> _QueueLike:
        """Return the bounded Audio-to-ASR queue."""

        return self._require_queue(self._asr_audio_queue, "ASR audio")

    @property
    def writer_stop_event(self) -> _EventLike:
        """Return the Writer lifecycle stop event."""

        if self._writer_stop_event is None:
            raise RuntimeError("writer stop event is not initialized")
        return self._writer_stop_event

    @property
    def writer_finalization_gate(self) -> WriterFinalizationGate:
        """Return the Writer gate owned by the currently active runtime session."""

        if self._writer_finalization_gate is None:
            raise RuntimeError("writer finalization gate is unavailable")
        return self._writer_finalization_gate

    @property
    def shutdown_report(self) -> WorkerShutdownReport | None:
        """Return the latest shutdown report, if shutdown has run."""

        return self._shutdown_report

    def start_all(self, launch: object) -> None:
        """Create queues first, then start Writer, Audio, ASR, and Discussion."""

        if not isinstance(launch, SessionLaunch):
            raise TypeError("launch must be a SessionLaunch")
        if self._restart_cleanup_incomplete:
            raise RuntimeError("worker runtime has incomplete restart cleanup")
        if self._processes and not self._ready_for_reuse():
            raise RuntimeError("worker runtime is already started")
        self._discard_previous_session()
        self._launch = launch
        self._shutdown_report = None
        self._writer_finalization_gate = WriterFinalizationGate.create(
            cast(BaseContext, self.context),
        )
        self._create_queues(launch)
        for worker in _START_ORDER:
            process = self._create_process(worker, launch)
            self._processes[worker] = process
            try:
                process.start()
            except Exception as error:
                self.shutdown()
                raise RuntimeError(f"Failed to start {process.name}") from error

    def audio_bindings(self) -> AudioQueueBindings:
        """Expose only the dedicated audio queues used by worker arguments."""

        return AudioQueueBindings(
            writer_audio_out=self.writer_audio_queue,
            asr_audio_out=self.asr_audio_queue,
        )

    def send(
        self,
        target: ProcessSource,
        envelope: MessageEnvelope[object],
    ) -> None:
        """Put one small controller envelope on a worker control queue."""

        if target is ProcessSource.GUI or target not in _PROCESS_ORDER:
            raise ValueError("target must be a worker process")
        if not isinstance(envelope, MessageEnvelope):
            raise TypeError("envelope must be a MessageEnvelope")
        if _contains_dedicated_audio_payload(envelope.payload):
            raise ValueError("PCM and drain fences must use dedicated audio queues")
        try:
            _put_nowait(self._control_queues[target], envelope)
        except Exception as error:
            self._fail_closed(f"{target.value} control queue became unavailable", error)

    def poll(self) -> tuple[MessageEnvelope[object], ...]:
        """Drain at most the fixed per-tick budget without blocking Qt."""

        messages: list[MessageEnvelope[object]] = []
        response_queue = self.response_queue
        for _ in range(self._poll_budget):
            try:
                value = response_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as error:
                self._fail_closed("response queue became unavailable", error)
            if not isinstance(value, MessageEnvelope):
                self._fail_closed(
                    "response queue yielded a non-envelope item",
                    ValueError("invalid response payload"),
                )
            messages.append(value)
        return tuple(messages)

    def restart(self, target: ProcessSource) -> None:
        """Replace an ASR or Discussion child with fresh process handles."""

        if target not in _RESTARTABLE:
            raise ValueError("only ASR and Discussion workers can be restarted")
        launch = self._require_launch()
        old_process = self._processes.get(target)
        if old_process is None:
            raise RuntimeError(f"{target.value} process is not active")
        old_queue = self._control_queues[target]
        joined, terminated, errors = self._stop_process(
            target,
            old_process,
            old_queue,
        )
        if target not in joined:
            self.restart_reports.append(
                WorkerRestartReport(target, joined, terminated, errors)
            )
            raise RuntimeError(f"{target.value} process remained alive after restart")
        errors = errors + self._close_queues((old_queue,))
        fresh_queue = self.context.Queue()
        self._control_queues[target] = fresh_queue
        process = self._create_process(target, launch)
        self._processes[target] = process
        try:
            process.start()
        except Exception as error:
            cleanup_errors = self._cleanup_failed_restart(
                target,
                process,
                fresh_queue,
            )
            self._restart_cleanup_incomplete = bool(cleanup_errors)
            self._processes.pop(target, None)
            self._control_queues.pop(target, None)
            restart_errors = (
                *errors,
                f"{target.value} restart start: {_error_name(error)}",
                *cleanup_errors,
            )
            self.restart_reports.append(
                WorkerRestartReport(target, joined, terminated, restart_errors)
            )
            self._fail_closed(
                f"{target.value} restart failed",
                error,
                additional_errors=restart_errors,
            )
        self.restart_reports.append(
            WorkerRestartReport(
                worker=target,
                joined=joined,
                terminated=terminated,
                errors=errors,
            )
        )

    def health(self) -> Mapping[ProcessSource, bool]:
        """Return an exact health map for all four child workers."""

        health: dict[ProcessSource, bool] = {}
        for worker in _PROCESS_ORDER:
            process = self._processes.get(worker)
            if process is None:
                health[worker] = False
                continue
            try:
                health[worker] = bool(process.is_alive())
            except Exception:
                health[worker] = False
        return health

    def shutdown(self) -> WorkerShutdownReport:
        """Send typed non-completion controls, join, and terminate if needed."""

        if self._shutdown_report is not None:
            return self._shutdown_report
        joined: list[ProcessSource] = []
        terminated: list[ProcessSource] = []
        errors: list[str] = []
        try:
            self.writer_stop_event.set()
        except Exception as error:
            errors.append(f"writer stop event: {_error_name(error)}")
        for worker in (
            ProcessSource.AUDIO,
            ProcessSource.ASR,
            ProcessSource.DISCUSSION,
            ProcessSource.WRITER,
        ):
            process = self._processes.get(worker)
            control_queue = self._control_queues.get(worker)
            if process is None or control_queue is None:
                continue
            worker_joined, worker_terminated, worker_errors = self._stop_process(
                worker,
                process,
                control_queue,
            )
            joined.extend(worker_joined)
            terminated.extend(worker_terminated)
            errors.extend(worker_errors)
        if self._all_processes_stopped():
            errors.extend(
                self._close_queues(
                    (
                        self._response_queue,
                        *self._control_queues.values(),
                        self._writer_audio_queue,
                        self._asr_audio_queue,
                    )
                )
            )
        self._shutdown_report = WorkerShutdownReport(
            completed=False,
            joined=tuple(joined),
            terminated=tuple(terminated),
            errors=tuple(errors),
        )
        return self._shutdown_report

    def request_writer_force_close(
        self,
        request: WriterForceCloseRequest,
        timeout_seconds: float,
    ) -> WriterForceCloseResult | None:
        """Delegate a current-session force-close request to Writer's gate."""

        if type(request) is not WriterForceCloseRequest:
            raise TypeError("request must be an exact WriterForceCloseRequest")
        if (
            not isinstance(timeout_seconds, int | float)
            or isinstance(timeout_seconds, bool)
            or float(timeout_seconds) < 0.0
        ):
            raise ValueError("timeout_seconds must be non-negative")
        launch = self._require_launch()
        if request.event.session_id != launch.session_id:
            self._fail_closed(
                "Writer force-close request does not match the active session",
                ValueError("stale force-close request"),
            )
        self._require_live_writer()
        try:
            return self.writer_finalization_gate.request_force_close(
                request,
                timeout_seconds=float(timeout_seconds),
            )
        except Exception as error:
            self._fail_closed("Writer finalization gate became unavailable", error)

    def writer_force_close_result(self) -> WriterForceCloseResult | None:
        """Read Writer's gate result without relying on a response-queue delivery."""

        try:
            result = self.writer_finalization_gate.result()
        except Exception as error:
            self._fail_closed("Writer finalization gate became unavailable", error)
        if result is not None:
            return result
        self._require_live_writer()
        return None

    def general_queues_contain_bytes(self) -> bool:
        """Return whether any general control queue currently contains bytes."""

        return self.general_queues_contain(bytes)

    def general_queues_contain(self, payload_type: type[object]) -> bool:
        """Return whether any general control queue contains the exact type."""

        for control_queue in self._control_queues.values():
            items = getattr(control_queue, "items", None)
            if isinstance(items, list) and any(
                _contains_type(item, payload_type) for item in items
            ):
                return True
        return False

    def _create_queues(self, launch: SessionLaunch) -> None:
        self._response_queue = self.context.Queue()
        self._control_queues = {
            worker: self.context.Queue() for worker in _PROCESS_ORDER
        }
        self._writer_audio_queue = self.context.Queue(
            maxsize=launch.audio_config.writer_queue_max_frames
        )
        self._asr_audio_queue = self.context.Queue(
            maxsize=launch.audio_config.asr_queue_max_frames
        )
        self._writer_stop_event = self.context.Event()

    def _create_process(
        self,
        worker: ProcessSource,
        launch: SessionLaunch,
    ) -> _ProcessLike:
        args = self._process_args(worker, launch)
        process = self.context.Process(
            name=_process_name(worker),
            target=self._target(worker),
            args=args,
        )
        process.daemon = False
        return process

    def _process_args(
        self,
        worker: ProcessSource,
        launch: SessionLaunch,
    ) -> tuple[object, ...]:
        if worker is ProcessSource.WRITER:
            return (
                self._control_queues[ProcessSource.WRITER],
                self.writer_audio_queue,
                self.response_queue,
                self.writer_stop_event,
                self.writer_finalization_gate,
            )
        if worker is ProcessSource.AUDIO:
            return (
                launch.audio_config,
                self._control_queues[ProcessSource.AUDIO],
                self.response_queue,
                self.writer_audio_queue,
                self.asr_audio_queue,
            )
        if worker is ProcessSource.ASR:
            return (
                launch.asr_config,
                self.asr_audio_queue,
                self._control_queues[ProcessSource.ASR],
                self.response_queue,
            )
        if worker is ProcessSource.DISCUSSION:
            return (
                launch.discussion_config,
                self._control_queues[ProcessSource.DISCUSSION],
                self.response_queue,
            )
        raise ValueError("worker must be a runtime process")

    def _target(self, worker: ProcessSource) -> Callable[..., object]:
        targets = {
            ProcessSource.WRITER: self._targets.writer,
            ProcessSource.AUDIO: self._targets.audio,
            ProcessSource.ASR: self._targets.asr,
            ProcessSource.DISCUSSION: self._targets.discussion,
        }
        return targets[worker]

    def _stop_process(
        self,
        worker: ProcessSource,
        process: _ProcessLike,
        control_queue: _QueueLike,
    ) -> tuple[tuple[ProcessSource, ...], tuple[ProcessSource, ...], tuple[str, ...]]:
        errors: list[str] = []
        try:
            _put_nowait(control_queue, self._shutdown_envelope(worker))
        except Exception as error:
            errors.append(f"{worker.value} control: {_error_name(error)}")
        self._join_process(worker, process, errors)
        alive = self._process_is_alive(worker, process, errors)
        terminated: tuple[ProcessSource, ...] = ()
        if alive:
            try:
                process.terminate()
                terminated = (worker,)
            except Exception as error:
                errors.append(f"{worker.value} terminate: {_error_name(error)}")
            self._join_process(worker, process, errors)
            alive = self._process_is_alive(worker, process, errors)
        joined: tuple[ProcessSource, ...] = ()
        if alive:
            errors.append(f"{worker.value} remains alive after bounded termination")
        else:
            joined = (worker,)
            try:
                process.close()
            except Exception as error:
                errors.append(f"{worker.value} close: {_error_name(error)}")
        return joined, terminated, tuple(errors)

    def _cleanup_failed_restart(
        self,
        worker: ProcessSource,
        process: _ProcessLike,
        control_queue: _QueueLike,
    ) -> tuple[str, ...]:
        """Close a failed fresh child without unstarted lifecycle operations."""

        errors: list[str] = []
        alive = self._process_is_alive(worker, process, errors)
        if alive:
            _, _, stop_errors = self._stop_process(worker, process, control_queue)
            errors.extend(stop_errors)
        else:
            try:
                process.close()
            except Exception as error:
                errors.append(f"{worker.value} restart close: {_error_name(error)}")
        errors.extend(
            f"{worker.value} restart {error}"
            for error in self._close_queues((control_queue,))
        )
        return tuple(errors)

    def _join_process(
        self,
        worker: ProcessSource,
        process: _ProcessLike,
        errors: list[str],
    ) -> None:
        try:
            process.join(timeout=self._join_timeout_seconds)
        except Exception as error:
            errors.append(f"{worker.value} join: {_error_name(error)}")

    @staticmethod
    def _process_is_alive(
        worker: ProcessSource,
        process: _ProcessLike,
        errors: list[str],
    ) -> bool:
        try:
            return bool(process.is_alive())
        except Exception as error:
            errors.append(f"{worker.value} health: {_error_name(error)}")
            return True

    def _shutdown_envelope(self, worker: ProcessSource) -> MessageEnvelope[object]:
        launch = self._require_launch()
        if worker is ProcessSource.WRITER:
            message_type = MessageType.WRITER_SHUTDOWN
            payload: object = WriterShutdown()
        elif worker is ProcessSource.ASR:
            message_type = MessageType.WORKER_STOP
            payload = {"worker": "ASR", "finalize": True}
        elif worker is ProcessSource.DISCUSSION:
            message_type = MessageType.WORKER_STOP
            payload = {"worker": "DISCUSSION", "finalize": True}
        elif worker is ProcessSource.AUDIO:
            message_type = MessageType.WORKER_STOP
            payload = {"worker": "AUDIO"}
        else:
            raise ValueError("worker must be a runtime process")
        return MessageEnvelope(
            schema_version=1,
            session_id=launch.session_id,
            message_type=message_type,
            sequence=1,
            source=ProcessSource.GUI,
            created_monotonic_ms=launch.audio_config.session_started_monotonic_ms,
            payload=payload,
        )

    def _fail_closed(
        self,
        message: str,
        error: Exception,
        *,
        additional_errors: tuple[str, ...] = (),
    ) -> NoReturn:
        try:
            report = self.shutdown()
            if additional_errors:
                self._shutdown_report = replace(
                    report,
                    errors=report.errors + additional_errors,
                )
        finally:
            raise RuntimeError(message) from error

    def _require_launch(self) -> SessionLaunch:
        if self._launch is None:
            raise RuntimeError("worker runtime has no active launch")
        return self._launch

    def _require_live_writer(self) -> None:
        writer = self._processes.get(ProcessSource.WRITER)
        if writer is None:
            self._fail_closed(
                "Writer force-close outcome could not be resolved",
                RuntimeError("Writer process is unavailable"),
            )
        errors: list[str] = []
        if not self._process_is_alive(ProcessSource.WRITER, writer, errors):
            self._fail_closed(
                "Writer force-close outcome could not be resolved",
                RuntimeError("Writer process exited"),
            )
        if errors:
            self._fail_closed(
                "Writer force-close outcome could not be resolved",
                RuntimeError(errors[0]),
            )

    def _ready_for_reuse(self) -> bool:
        return self._shutdown_report is not None and self._all_processes_stopped()

    def _discard_previous_session(self) -> None:
        self._launch = None
        self._control_queues = {}
        self._processes = {}
        self._response_queue = None
        self._writer_audio_queue = None
        self._asr_audio_queue = None
        self._writer_stop_event = None
        self._writer_finalization_gate = None
        self.restart_reports = []

    def _all_processes_stopped(self) -> bool:
        for worker, process in self._processes.items():
            errors: list[str] = []
            if self._process_is_alive(worker, process, errors):
                return False
        return True

    @staticmethod
    def _close_queues(
        queues: tuple[_QueueLike | None, ...],
    ) -> tuple[str, ...]:
        errors: list[str] = []
        for queue_value in queues:
            if queue_value is None:
                continue
            for operation_name in ("close", "join_thread"):
                operation = getattr(queue_value, operation_name, None)
                if not callable(operation):
                    continue
                try:
                    operation()
                except Exception as error:
                    errors.append(f"queue {operation_name}: {_error_name(error)}")
        return tuple(errors)

    @staticmethod
    def _require_queue(queue_value: _QueueLike | None, name: str) -> _QueueLike:
        if queue_value is None:
            raise RuntimeError(f"{name} queue is not initialized")
        return queue_value


def _process_name(worker: ProcessSource) -> str:
    names = {
        ProcessSource.WRITER: "FlowLens-Writer",
        ProcessSource.AUDIO: "FlowLens-Audio",
        ProcessSource.ASR: "FlowLens-ASR",
        ProcessSource.DISCUSSION: "FlowLens-Discussion",
    }
    return names[worker]


def _put_nowait(queue_value: _QueueLike, item: object) -> None:
    put_nowait = getattr(queue_value, "put_nowait", None)
    if callable(put_nowait):
        put_nowait(item)
        return
    queue_value.put(item, block=False)


def _contains_dedicated_audio_payload(value: object) -> bool:
    return _contains_payload_type(
        value,
        (AudioDrainFence, bytes, bytearray, memoryview),
        fail_closed=True,
    )


def _contains_type(
    value: object,
    payload_type: type[object],
    active_ids: set[int] | None = None,
) -> bool:
    return _contains_payload_type(
        value,
        (payload_type,),
        active_ids=active_ids,
        fail_closed=False,
    )


def _contains_payload_type(
    value: object,
    payload_types: tuple[type[object], ...],
    *,
    active_ids: set[int] | None = None,
    fail_closed: bool,
) -> bool:
    if isinstance(value, payload_types):
        return True
    active = set() if active_ids is None else active_ids
    if not _is_traversable_payload(value):
        return False
    value_id = id(value)
    if value_id in active:
        return False
    active.add(value_id)
    try:
        if isinstance(value, Mapping):
            return any(
                _contains_payload_type(
                    item,
                    payload_types,
                    active_ids=active,
                    fail_closed=fail_closed,
                )
                for pair in value.items()
                for item in pair
            )
        if isinstance(value, tuple | list | set | frozenset):
            return any(
                _contains_payload_type(
                    item,
                    payload_types,
                    active_ids=active,
                    fail_closed=fail_closed,
                )
                for item in value
            )
        if is_dataclass(value) and not isinstance(value, type):
            return any(
                _contains_payload_type(
                    getattr(value, field.name),
                    payload_types,
                    active_ids=active,
                    fail_closed=fail_closed,
                )
                for field in fields(value)
            )
    except BaseException:
        return fail_closed
    finally:
        active.remove(value_id)
    return False


def _is_traversable_payload(value: object) -> bool:
    return isinstance(value, Mapping | tuple | list | set | frozenset) or (
        is_dataclass(value) and not isinstance(value, type)
    )


def _error_name(error: Exception) -> str:
    return type(error).__name__.strip() or "Exception"


def runtime_from_spawn_context(
    worker_targets: WorkerTargets | None = None,
) -> MultiprocessingWorkerRuntime:
    """Construct the production runtime with multiprocessing spawn context."""

    context = cast(_ContextLike, multiprocessing.get_context("spawn"))
    return MultiprocessingWorkerRuntime(context, worker_targets)
