"""ASR process lifecycle, IPC serialization, and lag-state transitions."""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from flowlens.asr.engine import AsrBatch, AsrEngine
from flowlens.asr.kotoba_whisper import KotobaWhisperDecoder
from flowlens.asr.ports import DecoderPort, SpeechDetectorPort
from flowlens.asr.types import AsrWorkerConfig, PartialTranscript
from flowlens.asr.vad import WebRtcSpeechDetector
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    MessageEnvelope,
    SequenceResult,
    SequenceTracker,
)

MonotonicClock = Callable[[], int]
DecoderFactory = Callable[[Path], DecoderPort]
SpeechDetectorFactory = Callable[[], SpeechDetectorPort]
_ERROR_DETAIL_MAX_CHARS = 512
_DEFAULT_POLL_TIMEOUT_SECONDS = 0.01
_MAX_LIVE_DRAIN_FRAMES = 256
_FENCE_WAIT_TIMEOUT_MS = 30_000


class _QueueIn(Protocol):
    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> object: ...

    def get_nowait(self) -> object: ...


class _ControlOut(Protocol):
    def put(self, item: MessageEnvelope[object]) -> None: ...


class _EnginePort(Protocol):
    def accept(self, frame: AudioFrame) -> None: ...

    def process_ready(self, now_monotonic_ms: int) -> AsrBatch: ...

    def finalize(self, now_monotonic_ms: int) -> AsrBatch: ...

    def backlog_ms(self, now_monotonic_ms: int) -> int: ...


EngineFactory = Callable[
    [AsrWorkerConfig, DecoderPort, SpeechDetectorPort],
    _EnginePort,
]


def _default_engine_factory(
    config: AsrWorkerConfig,
    decoder: DecoderPort,
    speech_detector: SpeechDetectorPort,
) -> _EnginePort:
    return AsrEngine(config, decoder, speech_detector)


class _EnvelopeEmitter:
    """Build contiguous envelopes from the ASR sender namespace."""

    def __init__(
        self,
        session_id: str,
        output: _ControlOut,
        monotonic_ms: MonotonicClock,
    ) -> None:
        self._session_id = session_id
        self._output = output
        self._monotonic_ms = monotonic_ms
        self._sequence = 0

    def emit(self, message_type: MessageType, payload: dict[str, object]) -> None:
        self._sequence += 1
        self._output.put(
            MessageEnvelope(
                schema_version=1,
                session_id=self._session_id,
                message_type=message_type,
                sequence=self._sequence,
                source=ProcessSource.ASR,
                created_monotonic_ms=self._monotonic_ms(),
                payload=payload,
            )
        )


class _LagTracker:
    """Emit only strict hysteresis boundary crossings."""

    def __init__(self, config: AsrWorkerConfig) -> None:
        self._delayed_threshold_ms = config.delayed_threshold_ms
        self._analysis_pause_threshold_ms = config.analysis_pause_threshold_ms
        self.state = "RUNNING"
        self.analysis_paused = False

    def observe(self, backlog_ms: int) -> bool:
        previous = (self.state, self.analysis_paused)
        if backlog_ms > self._analysis_pause_threshold_ms:
            self.state = "DELAYED"
            self.analysis_paused = True
        elif backlog_ms > self._delayed_threshold_ms:
            self.state = "DELAYED"
        elif backlog_ms < self._delayed_threshold_ms:
            self.state = "RUNNING"
            self.analysis_paused = False
        return previous != (self.state, self.analysis_paused)


def _asr_worker_loop(
    config: AsrWorkerConfig,
    audio_in: _QueueIn,
    control_in: _QueueIn,
    control_out: _ControlOut,
    *,
    decoder_factory: DecoderFactory,
    speech_detector_factory: SpeechDetectorFactory,
    engine_factory: EngineFactory = _default_engine_factory,
    monotonic_ms: MonotonicClock,
    poll_timeout_seconds: float = _DEFAULT_POLL_TIMEOUT_SECONDS,
) -> int:
    """Run the hardware/model-independent ASR Worker state machine."""

    emitter = _EnvelopeEmitter(config.session_id, control_out, monotonic_ms)
    try:
        decoder = decoder_factory(config.model_path)
        speech_detector = speech_detector_factory()
        engine = engine_factory(config, decoder, speech_detector)
    except BaseException as exc:
        emitter.emit(
            MessageType.WORKER_ERROR,
            {
                "worker": "ASR",
                "code": "MODEL_LOAD_FAILED",
                "detail": _detail(exc),
            },
        )
        return 1

    sequence_tracker = SequenceTracker()
    state = "READY"
    lag = _LagTracker(config)
    committed_count = 0
    prefer_control = True
    boundary: str | None = None
    fences_needed = 0
    fence_credits = 0
    boundary_started_ms: int | None = None
    resume_pending = False
    emitter.emit(MessageType.WORKER_READY, {"worker": "ASR"})
    _emit_status(emitter, "READY", 0, False)

    def emit_batch(batch: AsrBatch) -> None:
        nonlocal committed_count
        for partial in batch.partials:
            emitter.emit(
                MessageType.TRANSCRIPT_PARTIAL,
                _serialize_partial(partial),
            )
        for record in batch.committed:
            emitter.emit(MessageType.TRANSCRIPT_COMMITTED, record.to_dict())
            committed_count += 1

    def process_engine() -> None:
        now_ms = monotonic_ms()
        observe_lag(engine.backlog_ms(now_ms))
        emit_batch(engine.process_ready(now_ms))
        observe_lag(engine.backlog_ms(monotonic_ms()))

    def observe_lag(backlog_ms: int) -> None:
        if lag.observe(backlog_ms):
            _emit_status(
                emitter,
                lag.state,
                backlog_ms,
                lag.analysis_paused,
            )

    def accept_audio_item(value: object) -> bool:
        nonlocal fence_credits, boundary_started_ms
        if isinstance(value, AudioDrainFence):
            fence_credits += 1
            if boundary_started_ms is None:
                boundary_started_ms = monotonic_ms()
            return True
        if isinstance(value, AudioFrame):
            engine.accept(value)
        return False

    def drain_audio(max_frames: int = _MAX_LIVE_DRAIN_FRAMES) -> None:
        drained = 0
        while drained < max_frames:
            try:
                value = audio_in.get_nowait()
            except queue.Empty:
                return
            drained += 1
            if accept_audio_item(value):
                return

    def begin_boundary(kind: str) -> None:
        nonlocal boundary, fences_needed, boundary_started_ms
        boundary = kind
        fences_needed += 1
        if boundary_started_ms is None:
            boundary_started_ms = monotonic_ms()

    def consume_fence_credits() -> None:
        nonlocal fence_credits, fences_needed
        consumed = min(fence_credits, fences_needed)
        fence_credits -= consumed
        fences_needed -= consumed

    def fence_wait_expired() -> bool:
        if boundary_started_ms is None:
            return False
        waiting = boundary is not None or fence_credits > 0
        return (
            waiting and monotonic_ms() - boundary_started_ms >= _FENCE_WAIT_TIMEOUT_MS
        )

    try:
        while True:
            control: object | None = None
            audio: object | None = None
            audio_enabled = state == "RUNNING" or boundary is not None
            if fence_credits > 0 and boundary is None:
                control = _bounded_get(control_in, poll_timeout_seconds)
            elif audio_enabled:
                if prefer_control:
                    control = _bounded_get(control_in, poll_timeout_seconds)
                    if control is None:
                        audio = _bounded_get(audio_in, poll_timeout_seconds)
                else:
                    audio = _bounded_get(audio_in, poll_timeout_seconds)
                    if audio is None:
                        control = _bounded_get(control_in, poll_timeout_seconds)
                prefer_control = not prefer_control
            else:
                control = _bounded_get(control_in, poll_timeout_seconds)

            command = _target_command(control, config, sequence_tracker)
            if (
                command is MessageType.WORKER_START
                and state == "READY"
                and boundary is None
            ):
                state = "RUNNING"
                _emit_status(emitter, "RUNNING", 0, False)
            elif (
                command is MessageType.WORKER_PAUSE
                and state == "RUNNING"
                and boundary is None
            ):
                begin_boundary("PAUSE")
            elif command is MessageType.WORKER_RESUME and state == "PAUSED":
                state = "RUNNING"
            elif command is MessageType.WORKER_RESUME and boundary == "PAUSE":
                resume_pending = True
            elif command is MessageType.WORKER_STOP and boundary != "STOP":
                begin_boundary("STOP")

            if isinstance(audio, AudioFrame | AudioDrainFence):
                accept_audio_item(audio)
            if (state == "RUNNING" or boundary is not None) and fence_credits == 0:
                drain_audio()
            if state == "RUNNING" or boundary is not None:
                process_engine()

            consume_fence_credits()
            if fence_wait_expired():
                raise RuntimeError("timed out waiting for ASR drain fence boundary")
            if boundary is None or fences_needed > 0:
                continue
            completed_boundary = boundary
            boundary = None
            boundary_started_ms = None
            if completed_boundary == "PAUSE":
                state = "RUNNING" if resume_pending else "PAUSED"
                resume_pending = False
                continue
            emit_batch(engine.finalize(monotonic_ms()))
            emitter.emit(
                MessageType.WORKER_STOPPED,
                {
                    "worker": "ASR",
                    "drained": True,
                    "committed_count": committed_count,
                },
            )
            _emit_status(emitter, "STOPPED", 0, False)
            return 0
    except BaseException as exc:
        emitter.emit(
            MessageType.WORKER_ERROR,
            {
                "worker": "ASR",
                "code": "DECODE_FAILED",
                "detail": _detail(exc),
            },
        )
        return 1


def run_asr_worker(
    config: AsrWorkerConfig,
    audio_in: multiprocessing.Queue[AudioFrame | AudioDrainFence],
    control_in: multiprocessing.Queue[MessageEnvelope[object]],
    control_out: multiprocessing.Queue[MessageEnvelope[object]],
) -> None:
    """Construct production adapters and run the ASR Worker process."""

    def monotonic_ms() -> int:
        return int(time.monotonic() * 1_000)

    exit_code = _asr_worker_loop(
        config,
        audio_in,
        control_in,
        control_out,
        decoder_factory=KotobaWhisperDecoder,
        speech_detector_factory=WebRtcSpeechDetector,
        monotonic_ms=monotonic_ms,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def _bounded_get(source: _QueueIn, timeout_seconds: float) -> object | None:
    try:
        return source.get(timeout=timeout_seconds)
    except queue.Empty:
        return None


def _target_command(
    value: object,
    config: AsrWorkerConfig,
    sequence_tracker: SequenceTracker,
) -> MessageType | None:
    if not isinstance(value, MessageEnvelope):
        return None
    if value.session_id != config.session_id or value.source is not ProcessSource.GUI:
        return None
    expected_payloads: dict[MessageType, dict[str, object]] = {
        MessageType.WORKER_START: {"worker": "ASR"},
        MessageType.WORKER_PAUSE: {"worker": "ASR"},
        MessageType.WORKER_RESUME: {"worker": "ASR"},
        MessageType.WORKER_STOP: {"worker": "ASR", "finalize": True},
    }
    expected = expected_payloads.get(value.message_type)
    if expected is None or value.payload != expected:
        return None
    try:
        result = sequence_tracker.observe(value)
    except ValueError:
        return None
    if result is SequenceResult.DUPLICATE:
        return None
    return value.message_type


def _serialize_partial(partial: PartialTranscript) -> dict[str, object]:
    return {
        "source": partial.source.value,
        "text": partial.text,
        "session_start_ms": partial.session_start_ms,
        "session_end_ms": partial.session_end_ms,
        "source_start_sample": partial.source_start_sample,
        "source_end_sample": partial.source_end_sample,
    }


def _emit_status(
    emitter: _EnvelopeEmitter,
    state: str,
    backlog_ms: int,
    analysis_paused: bool,
) -> None:
    emitter.emit(
        MessageType.ASR_STATUS,
        {
            "state": state,
            "backlog_ms": backlog_ms,
            "analysis_paused": analysis_paused,
        },
    )


def _detail(error: BaseException) -> str:
    return (str(error).strip() or type(error).__name__)[:_ERROR_DETAIL_MAX_CHARS]
