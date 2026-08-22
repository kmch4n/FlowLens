"""Two-source capture worker lifecycle and IPC boundary."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import pyaudiowpatch  # type: ignore[import-untyped]

from flowlens.audio.dispatch import (
    AsrPumpFailed,
    AsrSpoolFull,
    AudioDispatcher,
    WriterQueueFull,
)
from flowlens.audio.normalize import SoxrAudioNormalizer
from flowlens.audio.ports import (
    CaptureBackendPort,
    CaptureStreamPort,
    StreamingNormalizerPort,
)
from flowlens.audio.pyaudiowpatch_backend import PyAudioWPatchBackend
from flowlens.audio.types import (
    AudioFrame,
    AudioWorkerConfig,
    CaptureDevice,
    RawAudioChunk,
)
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    MessageEnvelope,
    SequenceResult,
    SequenceTracker,
)

NormalizerFactory = Callable[[AudioSource, int, int, int], StreamingNormalizerPort]
MonotonicClock = Callable[[], int]
Sleeper = Callable[[float], None]
_ERROR_DETAIL_MAX_CHARS = 512
_ASR_FENCE_TIMEOUT_SECONDS = 10.0


class _ControlIn(Protocol):
    def get_nowait(self) -> object: ...


class _ControlOut(Protocol):
    def put(self, item: MessageEnvelope[object]) -> None: ...


class _WriterAudioOut(Protocol):
    def put_nowait(self, item: AudioWriteCommand | AudioDrainFence) -> None: ...


class _AsrAudioOut(Protocol):
    def put(
        self,
        item: AudioFrame | AudioDrainFence,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...


class _RawAudioQueue(Protocol):
    def put_nowait(self, item: RawAudioChunk) -> None: ...

    def get_nowait(self) -> RawAudioChunk: ...


RawQueueFactory = Callable[[int], _RawAudioQueue]


def _default_raw_queue(maxsize: int) -> _RawAudioQueue:
    return queue.Queue(maxsize=maxsize)


class _FatalWorkerError(RuntimeError):
    """Internal classified failure that crosses the control boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail[:_ERROR_DETAIL_MAX_CHARS]
        super().__init__(self.detail)


class _EnvelopeEmitter:
    """Build AUDIO-local contiguous response envelopes."""

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
                source=ProcessSource.AUDIO,
                created_monotonic_ms=self._monotonic_ms(),
                payload=payload,
            )
        )


def _audio_worker_loop(
    config: AudioWorkerConfig,
    control_in: _ControlIn,
    control_out: _ControlOut,
    writer_audio_out: _WriterAudioOut,
    *,
    backend: CaptureBackendPort,
    normalizer_factory: NormalizerFactory,
    dispatcher: AudioDispatcher,
    monotonic_ms: MonotonicClock,
    sleeper: Sleeper,
    raw_queue_factory: RawQueueFactory = _default_raw_queue,
    asr_fence_timeout_seconds: float = _ASR_FENCE_TIMEOUT_SECONDS,
) -> None:
    """Run the hardware-independent Audio Worker state machine."""

    emitter = _EnvelopeEmitter(config.session_id, control_out, monotonic_ms)
    raw_queues = {source: raw_queue_factory(256) for source in AudioSource}
    raw_overflow = threading.Event()
    callback_condition = threading.Condition()
    callbacks_accepting = False
    callbacks_in_flight = 0
    sequence_tracker = SequenceTracker()
    streams: dict[AudioSource, CaptureStreamPort] = {}
    all_streams: list[CaptureStreamPort] = []
    normalizers: dict[AudioSource, StreamingNormalizerPort] = {}
    disconnected: set[AudioSource] = set()
    retry_at_ms: dict[AudioSource, int] = {}
    writer_frames = 0
    asr_frames = 0
    state = "READY"
    normal_stop = False
    pump_stop = threading.Event()
    pump: threading.Thread | None = None

    device_ids = {
        AudioSource.ME: config.microphone_device_id,
        AudioSource.OTHERS: config.loopback_output_device_id,
    }

    def callback_for(source: AudioSource) -> Callable[[RawAudioChunk], None]:
        def callback(chunk: RawAudioChunk) -> None:
            nonlocal callbacks_in_flight
            with callback_condition:
                if not callbacks_accepting:
                    return
                callbacks_in_flight += 1
            try:
                raw_queues[source].put_nowait(chunk)
            except queue.Full:
                raw_overflow.set()
            finally:
                with callback_condition:
                    callbacks_in_flight -= 1
                    if callbacks_in_flight == 0:
                        callback_condition.notify_all()

        return callback

    def accept_callbacks() -> None:
        nonlocal callbacks_accepting
        with callback_condition:
            callbacks_accepting = True

    def reject_callbacks_and_wait() -> None:
        nonlocal callbacks_accepting
        with callback_condition:
            callbacks_accepting = False
            callback_condition.wait_for(lambda: callbacks_in_flight == 0)

    def open_source(source: AudioSource) -> None:
        devices = (
            backend.list_microphones()
            if source is AudioSource.ME
            else backend.list_loopback_outputs()
        )
        device = _find_exact_device(devices, device_ids[source])
        if source not in normalizers:
            normalizers[source] = normalizer_factory(
                source,
                device.sample_rate_hz,
                device.channels,
                config.session_started_monotonic_ms,
            )
        stream = backend.open_stream(source, device_ids[source], callback_for(source))
        streams[source] = stream
        all_streams.append(stream)

    def stop_streams() -> None:
        nonlocal callbacks_accepting
        with callback_condition:
            callbacks_accepting = False
        for source in AudioSource:
            stream = streams.get(source)
            if stream is None:
                continue
            try:
                stream.stop()
            except BaseException:
                pass
        reject_callbacks_and_wait()

    def dispatch_frame(frame: AudioFrame) -> None:
        nonlocal writer_frames, asr_frames
        try:
            dispatcher.dispatch(frame)
        except WriterQueueFull as exc:
            raise _FatalWorkerError("WRITER_QUEUE_FULL", str(exc)) from exc
        except AsrSpoolFull as exc:
            writer_frames += 1
            raise _FatalWorkerError("ASR_QUEUE_STALLED", str(exc)) from exc
        writer_frames += 1
        asr_frames += 1
        emitter.emit(
            MessageType.AUDIO_LEVEL,
            {"source": frame.source.value, "peak_dbfs": _peak_dbfs(frame)},
        )

    def drain_raw() -> None:
        for source in AudioSource:
            raw_queue = raw_queues[source]
            while True:
                try:
                    chunk = raw_queue.get_nowait()
                except queue.Empty:
                    break
                for frame in normalizers[source].push(chunk):
                    dispatch_frame(frame)

    def fence_asr() -> None:
        try:
            completed = dispatcher.enqueue_asr_fence(timeout=asr_fence_timeout_seconds)
        except AsrPumpFailed as exc:
            raise _FatalWorkerError(
                "ASR_QUEUE_STALLED",
                _detail(exc.failure),
            ) from exc
        if not completed:
            raise _FatalWorkerError(
                "ASR_QUEUE_STALLED",
                "timed out submitting ASR drain fence",
            )

    def process_control(envelope: object) -> None:
        nonlocal state, normal_stop
        if not _is_target_control(envelope, config, sequence_tracker):
            return
        assert isinstance(envelope, MessageEnvelope)
        if envelope.message_type is MessageType.WORKER_START and state == "READY":
            accept_callbacks()
            try:
                for source in AudioSource:
                    streams[source].start()
            except BaseException as exc:
                raise _FatalWorkerError("DEVICE_OPEN_FAILED", _detail(exc)) from exc
            state = "RUNNING"
        elif envelope.message_type is MessageType.WORKER_PAUSE and state == "RUNNING":
            stop_streams()
            drain_raw()
            fence_asr()
            state = "PAUSED"
        elif envelope.message_type is MessageType.WORKER_RESUME and state == "PAUSED":
            accept_callbacks()
            try:
                for source in AudioSource:
                    if source not in disconnected:
                        streams[source].start()
            except BaseException as exc:
                raise _FatalWorkerError("DEVICE_OPEN_FAILED", _detail(exc)) from exc
            state = "RUNNING"
        elif envelope.message_type is MessageType.WORKER_STOP:
            normal_stop = True

    def observe_disconnects() -> None:
        if state != "RUNNING":
            return
        now = monotonic_ms()
        for source in AudioSource:
            if source not in disconnected and not streams[source].is_active():
                disconnected.add(source)
                retry_at_ms[source] = now + config.reconnect_interval_ms
                emitter.emit(
                    MessageType.SOURCE_DISCONNECTED,
                    {"source": source.value, "device_id": device_ids[source]},
                )
            if source not in disconnected or now < retry_at_ms[source]:
                continue
            try:
                open_source(source)
                streams[source].start()
            except BaseException:
                retry_at_ms[source] = now + config.reconnect_interval_ms
                continue
            disconnected.remove(source)
            retry_at_ms.pop(source, None)
            emitter.emit(
                MessageType.SOURCE_RECONNECTED,
                {"source": source.value, "device_id": device_ids[source]},
            )

    fatal: _FatalWorkerError | None = None
    try:
        try:
            for source in AudioSource:
                open_source(source)
        except BaseException as exc:
            raise _FatalWorkerError("DEVICE_OPEN_FAILED", _detail(exc)) from exc

        pump = threading.Thread(
            target=dispatcher.run_asr_pump,
            args=(pump_stop,),
            name="flowlens-audio-asr-pump",
        )
        pump.start()
        emitter.emit(MessageType.WORKER_READY, {"worker": "AUDIO"})

        while not normal_stop:
            if raw_overflow.is_set():
                raise _FatalWorkerError(
                    "CAPTURE_QUEUE_FULL",
                    "raw capture queue is full",
                )
            while True:
                try:
                    control = control_in.get_nowait()
                except queue.Empty:
                    break
                process_control(control)
                if normal_stop:
                    break
            if normal_stop:
                break
            drain_raw()
            observe_disconnects()
            sleeper(0.005)

        stop_streams()
        drain_raw()
        for source in AudioSource:
            for frame in normalizers[source].flush():
                dispatch_frame(frame)
        fence_asr()
        pump_stop.set()
        dispatcher.wake_asr_pump()
        if pump is not None:
            pump.join()
        try:
            writer_audio_out.put_nowait(AudioDrainFence())
        except queue.Full as exc:
            raise _FatalWorkerError(
                "WRITER_QUEUE_FULL", "Writer audio queue is full"
            ) from exc
        emitter.emit(
            MessageType.WORKER_STOPPED,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": writer_frames,
                "asr_frames": asr_frames,
            },
        )
    except _FatalWorkerError as exc:
        fatal = exc
        stop_streams()
        dispatcher.abort_asr_pump()
        pump_stop.set()
        dispatcher.wake_asr_pump()
        if pump is not None:
            pump.join()
        emitter.emit(
            MessageType.WORKER_ERROR,
            {"worker": "AUDIO", "code": exc.code, "detail": exc.detail},
        )
    finally:
        reject_callbacks_and_wait()
        if fatal is None and pump is not None and pump.is_alive():
            dispatcher.abort_asr_pump()
            pump_stop.set()
            dispatcher.wake_asr_pump()
            pump.join()
        for stream in all_streams:
            try:
                stream.close()
            except BaseException:
                pass
        try:
            backend.close()
        except BaseException:
            pass


def run_audio_worker(
    config: AudioWorkerConfig,
    control_in: _ControlIn,
    control_out: _ControlOut,
    writer_audio_out: _WriterAudioOut,
    asr_audio_out: _AsrAudioOut,
) -> None:
    """Construct production adapters and run the Audio Worker process."""

    def monotonic_ms() -> int:
        return int(time.monotonic() * 1_000)

    backend = PyAudioWPatchBackend(pyaudiowpatch.PyAudio, monotonic_ms)
    dispatcher = AudioDispatcher(
        writer_audio_out,
        asr_audio_out,
        asr_spool_max_frames=config.asr_spool_max_frames,
    )
    _audio_worker_loop(
        config,
        control_in,
        control_out,
        writer_audio_out,
        backend=backend,
        normalizer_factory=SoxrAudioNormalizer,
        dispatcher=dispatcher,
        monotonic_ms=monotonic_ms,
        sleeper=time.sleep,
    )


def _find_exact_device(
    devices: tuple[CaptureDevice, ...],
    device_id: str,
) -> CaptureDevice:
    try:
        return next(device for device in devices if device.device_id == device_id)
    except StopIteration as exc:
        raise RuntimeError(f"capture device is unavailable: {device_id}") from exc


def _is_target_control(
    value: object,
    config: AudioWorkerConfig,
    sequence_tracker: SequenceTracker,
) -> bool:
    if not isinstance(value, MessageEnvelope):
        return False
    if value.session_id != config.session_id or value.source is not ProcessSource.GUI:
        return False
    if value.message_type not in {
        MessageType.WORKER_START,
        MessageType.WORKER_PAUSE,
        MessageType.WORKER_RESUME,
        MessageType.WORKER_STOP,
    }:
        return False
    if value.payload != {"worker": "AUDIO"}:
        return False
    try:
        result = sequence_tracker.observe(value)
    except ValueError:
        return False
    return result is not SequenceResult.DUPLICATE


def _peak_dbfs(frame: AudioFrame) -> float:
    samples = np.frombuffer(frame.pcm_s16le, dtype="<i2")
    peak = int(np.max(np.abs(samples.astype(np.int32))))
    if peak == 0:
        return -96.0
    return round(20.0 * math.log10(peak / 32_768.0), 1)


def _detail(error: BaseException) -> str:
    return str(error).strip() or type(error).__name__
