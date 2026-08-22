"""Lifecycle and IPC contract tests for the two-source Audio Worker."""

from __future__ import annotations

import multiprocessing
import pickle
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import TypedDict, cast

from flowlens.audio.dispatch import AudioDispatcher
from flowlens.audio.ports import CaptureCallback, CaptureStreamPort
from flowlens.audio.types import (
    AudioFrame,
    AudioWorkerConfig,
    CaptureDevice,
    RawAudioChunk,
)
from flowlens.audio.worker import _audio_worker_loop, run_audio_worker
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    MessageEnvelope,
)

SESSION_ID = "01J00000000000000000000000"
OTHER_SESSION_ID = "01J00000000000000000000001"


class WorkerStoppedPayload(TypedDict):
    """Runtime-validated normal-stop payload shape."""

    worker: str
    drained: bool
    writer_frames: int
    asr_frames: int


class WorkerErrorPayload(TypedDict):
    """Runtime-validated fatal payload shape."""

    worker: str
    code: str
    detail: str


class SourceStatusPayload(TypedDict):
    """Runtime-validated source lifecycle payload shape."""

    source: str
    device_id: str


def _worker_stopped_payload(
    envelope: MessageEnvelope[object],
) -> WorkerStoppedPayload:
    payload = envelope.payload
    assert isinstance(payload, dict)
    assert set(payload) == {"worker", "drained", "writer_frames", "asr_frames"}
    assert payload["worker"] == "AUDIO"
    assert payload["drained"] is True
    assert isinstance(payload["writer_frames"], int)
    assert not isinstance(payload["writer_frames"], bool)
    assert isinstance(payload["asr_frames"], int)
    assert not isinstance(payload["asr_frames"], bool)
    return cast(WorkerStoppedPayload, payload)


def _worker_error_payload(envelope: MessageEnvelope[object]) -> WorkerErrorPayload:
    payload = envelope.payload
    assert isinstance(payload, dict)
    assert set(payload) == {"worker", "code", "detail"}
    assert payload["worker"] == "AUDIO"
    assert isinstance(payload["code"], str)
    assert isinstance(payload["detail"], str)
    return cast(WorkerErrorPayload, payload)


def _source_status_payload(
    envelope: MessageEnvelope[object],
) -> SourceStatusPayload:
    payload = envelope.payload
    assert isinstance(payload, dict)
    assert set(payload) == {"source", "device_id"}
    assert isinstance(payload["source"], str)
    assert isinstance(payload["device_id"], str)
    return cast(SourceStatusPayload, payload)


class FakeClock:
    """Thread-safe manually advanced monotonic millisecond clock."""

    def __init__(self, initial_ms: int = 1_000) -> None:
        self._value = initial_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._value

    def advance(self, milliseconds: int) -> None:
        with self._lock:
            self._value += milliseconds


class FakeStream:
    """Capture stream whose lifecycle and disconnection are observable."""

    def __init__(self) -> None:
        self.active = False
        self.closed = False
        self.disconnected = False
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.stop_error: BaseException | None = None
        self.close_error: BaseException | None = None

    def start(self) -> None:
        self.start_calls += 1
        self.active = True
        self.disconnected = False

    def stop(self) -> None:
        self.stop_calls += 1
        self.active = False
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        self.active = False
        if self.close_error is not None:
            raise self.close_error

    def is_active(self) -> bool:
        return self.active and not self.disconnected


class FakeCaptureBackend:
    """Exact-device backend with directly invokable capture callbacks."""

    def __init__(self) -> None:
        self.callbacks: dict[AudioSource, CaptureCallback] = {}
        self.streams: dict[AudioSource, FakeStream] = {}
        self.opened_ids: list[str] = []
        self.denied_ids: set[str] = set()
        self.close_calls = 0
        self.close_error: BaseException | None = None

    def list_microphones(self) -> tuple[CaptureDevice, ...]:
        return (CaptureDevice("input:3", "Mic", 3, 48_000, 1, False),)

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
        return (
            CaptureDevice(
                "wasapi-output:7",
                "Speakers",
                11,
                48_000,
                1,
                True,
            ),
        )

    def open_stream(
        self,
        source: AudioSource,
        device_id: str,
        callback: CaptureCallback,
    ) -> CaptureStreamPort:
        self.opened_ids.append(device_id)
        if device_id in self.denied_ids:
            raise OSError(f"unavailable: {device_id}")
        stream = FakeStream()
        self.callbacks[source] = callback
        self.streams[source] = stream
        return stream

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

    def stream(self, source: AudioSource) -> FakeStream:
        return self.streams[source]

    def disconnect(self, source: AudioSource) -> None:
        self.stream(source).disconnected = True

    def allow_open(self, device_id: str) -> None:
        self.denied_ids.discard(device_id)

    def emit(self, source: AudioSource, payload: bytes, at_ms: int) -> None:
        self.callbacks[source](RawAudioChunk(source, payload, 48_000, 1, at_ms))

    def emit_me(self, payload: bytes, at_ms: int) -> None:
        self.emit(AudioSource.ME, payload, at_ms)


class FakeNormalizer:
    """One-input-chunk normalizer preserving source offsets and timestamps."""

    def __init__(self, source: AudioSource, session_started_ms: int) -> None:
        self.source = source
        self.session_started_ms = session_started_ms
        self.next_sample = 0
        self.flush_calls = 0
        self.final_frames: tuple[AudioFrame, ...] = ()

    def push(self, chunk: RawAudioChunk) -> tuple[AudioFrame, ...]:
        frame = AudioFrame(
            source=self.source,
            pcm_s16le=chunk.pcm_s16le_interleaved[:640].ljust(640, b"\0"),
            source_start_sample=self.next_sample,
            source_end_sample=self.next_sample + 320,
            session_start_ms=chunk.captured_monotonic_ms - self.session_started_ms,
            captured_monotonic_ms=chunk.captured_monotonic_ms,
        )
        self.next_sample += 320
        return (frame,)

    def flush(self) -> tuple[AudioFrame, ...]:
        self.flush_calls += 1
        return self.final_frames


class InterleavingRawQueue:
    """Pause one callback between acceptance and its raw queue enqueue."""

    def __init__(self) -> None:
        self._items: queue.Queue[RawAudioChunk] = queue.Queue()
        self.put_entered = threading.Event()
        self.release_put = threading.Event()

    def put_nowait(self, item: RawAudioChunk) -> None:
        self.put_entered.set()
        assert self.release_put.wait(timeout=2)
        self._items.put_nowait(item)

    def get_nowait(self) -> RawAudioChunk:
        return self._items.get_nowait()

    def qsize(self) -> int:
        return self._items.qsize()


class ClosedAsrQueue(queue.Queue[AudioFrame]):
    """ASR process queue probe that fails every terminal put."""

    def __init__(self) -> None:
        super().__init__(maxsize=32)
        self.put_called = threading.Event()

    def put(
        self,
        item: AudioFrame,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        del item, block, timeout
        self.put_called.set()
        raise OSError("closed ASR queue: " + "x" * 2_000)


def _standard_raw_queue(maxsize: int) -> queue.Queue[RawAudioChunk]:
    return queue.Queue(maxsize=maxsize)


def _standard_asr_queue() -> queue.Queue[AudioFrame]:
    return queue.Queue(maxsize=32)


@dataclass
class AudioWorkerHarness:
    """In-process worker harness with deterministic fake adapters."""

    backend: FakeCaptureBackend | None = None
    writer_queue_max_frames: int = 32
    asr_spool_max_frames: int = 32
    loopback_output_device_id: str = "wasapi-output:7"
    raw_queue_factory: Callable[
        [int], InterleavingRawQueue | queue.Queue[RawAudioChunk]
    ] = _standard_raw_queue
    asr_queue_factory: Callable[[], queue.Queue[AudioFrame]] = _standard_asr_queue

    def __post_init__(self) -> None:
        self.backend = self.backend or FakeCaptureBackend()
        self.clock = FakeClock()
        self.control_in: queue.Queue[object] = queue.Queue()
        self.control_out: queue.Queue[MessageEnvelope[object]] = queue.Queue()
        self.writer_out: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(
            maxsize=self.writer_queue_max_frames
        )
        self.asr_out = self.asr_queue_factory()
        self.normalizers: dict[AudioSource, FakeNormalizer] = {}
        self.stop_polling = threading.Event()
        self.release_polling = threading.Event()
        self.thread: threading.Thread | None = None
        self.control_sequence = 0
        self.ready: MessageEnvelope[object] | None = None

    @property
    def config(self) -> AudioWorkerConfig:
        return AudioWorkerConfig(
            session_id=SESSION_ID,
            microphone_device_id="input:3",
            loopback_output_device_id=self.loopback_output_device_id,
            session_started_monotonic_ms=1_000,
            writer_queue_max_frames=self.writer_queue_max_frames,
            asr_queue_max_frames=32,
            asr_spool_max_frames=self.asr_spool_max_frames,
        )

    def start(self) -> None:
        backend = cast(FakeCaptureBackend, self.backend)

        def normalizer_factory(
            source: AudioSource,
            input_rate_hz: int,
            input_channels: int,
            session_started_monotonic_ms: int,
        ) -> FakeNormalizer:
            assert input_rate_hz == 48_000
            assert input_channels == 1
            value = FakeNormalizer(source, session_started_monotonic_ms)
            self.normalizers[source] = value
            return value

        dispatcher = AudioDispatcher(
            self.writer_out,
            self.asr_out,
            asr_spool_max_frames=self.asr_spool_max_frames,
        )
        self.dispatcher = dispatcher

        def sleeper(_: float) -> None:
            if self.stop_polling.is_set():
                self.release_polling.wait(timeout=2)
            else:
                time.sleep(0.001)

        self.thread = threading.Thread(
            target=_audio_worker_loop,
            kwargs={
                "config": self.config,
                "control_in": self.control_in,
                "control_out": self.control_out,
                "writer_audio_out": self.writer_out,
                "backend": backend,
                "normalizer_factory": normalizer_factory,
                "dispatcher": dispatcher,
                "monotonic_ms": self.clock,
                "sleeper": sleeper,
                "raw_queue_factory": self.raw_queue_factory,
            },
            name="test-audio-worker",
        )
        self.thread.start()
        self.ready = self.status(MessageType.WORKER_READY)

    def start_recording(self) -> None:
        self.start()
        self.send(MessageType.WORKER_START, {"worker": "AUDIO"})
        self.wait_until(
            lambda: cast(FakeCaptureBackend, self.backend)
            .stream(AudioSource.ME)
            .is_active()
        )

    def send(
        self,
        message_type: MessageType,
        payload: object,
        *,
        session_id: str = SESSION_ID,
        source: ProcessSource = ProcessSource.GUI,
        sequence: int | None = None,
    ) -> None:
        if sequence is None:
            self.control_sequence += 1
            sequence = self.control_sequence
        self.control_in.put_nowait(
            MessageEnvelope(
                1,
                session_id,
                message_type,
                sequence,
                source,
                self.clock(),
                payload,
            )
        )

    def status(
        self,
        message_type: MessageType,
        timeout: float = 2,
    ) -> MessageEnvelope[object]:
        deadline = time.monotonic() + timeout
        deferred: list[MessageEnvelope[object]] = []
        while time.monotonic() < deadline:
            try:
                item = self.control_out.get(timeout=0.02)
            except queue.Empty:
                continue
            if item.message_type is message_type:
                for other in deferred:
                    self.control_out.put_nowait(other)
                return item
            deferred.append(item)
        raise AssertionError(f"missing status {message_type.value}")

    def wait_until(self, predicate: Callable[[], bool], timeout: float = 2) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        raise AssertionError("condition did not become true")

    def poll_once(self) -> None:
        time.sleep(0.02)

    def stop(self) -> MessageEnvelope[object]:
        self.send(MessageType.WORKER_STOP, {"worker": "AUDIO"})
        stopped = self.status(MessageType.WORKER_STOPPED)
        self.join()
        return stopped

    def join(self) -> None:
        self.release_polling.set()
        assert self.thread is not None
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()


def test_pause_stops_streams_and_resume_preserves_source_offsets() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.emit_me(bytes(1_920), at_ms=1_020)
    first = harness.writer_out.get(timeout=1)
    assert isinstance(first, AudioWriteCommand)

    harness.send(MessageType.WORKER_PAUSE, {"worker": "AUDIO"})
    harness.wait_until(lambda: backend.stream(AudioSource.ME).stop_calls == 1)
    assert harness.normalizers[AudioSource.ME].flush_calls == 0
    harness.send(MessageType.WORKER_RESUME, {"worker": "AUDIO"})
    harness.wait_until(lambda: backend.stream(AudioSource.ME).is_active())
    backend.emit_me(bytes(1_920), at_ms=3_020)
    second = harness.writer_out.get(timeout=1)
    assert isinstance(second, AudioWriteCommand)

    assert first.source_end_sample == second.source_start_sample
    assert second.session_start_ms == 2_020
    stopped = harness.stop()
    assert stopped.payload == {
        "worker": "AUDIO",
        "drained": True,
        "writer_frames": 2,
        "asr_frames": 2,
    }


def test_normal_stop_fences_writer_audio_before_reporting_stopped() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.emit_me(bytes(1_920), at_ms=1_020)
    stopped = harness.stop()

    items: list[AudioWriteCommand | AudioDrainFence] = []
    while not harness.writer_out.empty():
        items.append(harness.writer_out.get_nowait())
    assert isinstance(items[0], AudioWriteCommand)
    assert isinstance(items[-1], AudioDrainFence)
    assert sum(isinstance(item, AudioDrainFence) for item in items) == 1
    assert not any(
        isinstance(item, AudioDrainFence) for item in list(harness.asr_out.queue)
    )
    assert _worker_stopped_payload(stopped)["drained"] is True


def test_pause_drains_chunks_already_accepted_by_callbacks() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    harness.stop_polling.set()
    time.sleep(0.02)
    backend.emit_me(bytes(1_920), at_ms=1_020)
    harness.send(MessageType.WORKER_PAUSE, {"worker": "AUDIO"})
    harness.release_polling.set()
    command = harness.writer_out.get(timeout=1)

    assert isinstance(command, AudioWriteCommand)
    harness.wait_until(lambda: backend.stream(AudioSource.ME).stop_calls == 1)
    harness.stop()


def test_flush_frames_are_dispatched_before_the_single_writer_fence() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    final_frame = AudioFrame(
        AudioSource.ME,
        bytes(640),
        0,
        320,
        20,
        1_020,
    )
    harness.normalizers[AudioSource.ME].final_frames = (final_frame,)
    harness.stop()

    items = list(harness.writer_out.queue)
    assert len(items) == 2
    assert isinstance(items[0], AudioWriteCommand)
    assert items[0].source_start_sample == 0
    assert isinstance(items[1], AudioDrainFence)


def test_one_disconnected_source_does_not_stop_other_source() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.disconnect(AudioSource.OTHERS)
    disconnected = harness.status(MessageType.SOURCE_DISCONNECTED)
    backend.emit_me(bytes(1_920), at_ms=1_040)
    command = harness.writer_out.get(timeout=1)

    assert isinstance(command, AudioWriteCommand)
    assert command.source is AudioSource.ME
    assert _source_status_payload(disconnected)["source"] == "OTHERS"
    assert backend.stream(AudioSource.ME).is_active()
    harness.stop()


def test_reconnect_reopens_only_exact_device_at_two_second_boundary() -> None:
    harness = AudioWorkerHarness(loopback_output_device_id="wasapi-output:7")
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.disconnect(AudioSource.OTHERS)
    harness.status(MessageType.SOURCE_DISCONNECTED)
    backend.denied_ids.add("wasapi-output:7")

    harness.clock.advance(1_999)
    harness.poll_once()
    assert backend.opened_ids.count("wasapi-output:7") == 1
    harness.clock.advance(1)
    backend.allow_open("wasapi-output:7")
    reconnected = harness.status(MessageType.SOURCE_RECONNECTED)

    assert backend.opened_ids.count("wasapi-output:7") == 2
    assert set(backend.opened_ids) <= {"input:3", "wasapi-output:7"}
    assert reconnected.payload == {
        "source": "OTHERS",
        "device_id": "wasapi-output:7",
    }
    harness.stop()


def test_failed_reconnect_retries_once_per_exact_interval() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.disconnect(AudioSource.OTHERS)
    harness.status(MessageType.SOURCE_DISCONNECTED)
    backend.denied_ids.add("wasapi-output:7")

    harness.clock.advance(2_000)
    harness.poll_once()
    assert backend.opened_ids.count("wasapi-output:7") == 2
    harness.clock.advance(1_999)
    harness.poll_once()
    assert backend.opened_ids.count("wasapi-output:7") == 2
    harness.clock.advance(1)
    harness.poll_once()
    assert backend.opened_ids.count("wasapi-output:7") == 3
    harness.stop()


def test_writer_queue_full_emits_fatal_and_stops_without_fence() -> None:
    harness = AudioWorkerHarness(writer_queue_max_frames=1)
    harness.start_recording()
    harness.writer_out.put_nowait(AudioDrainFence())
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.emit_me(bytes(1_920), at_ms=1_020)
    error = harness.status(MessageType.WORKER_ERROR)
    harness.join()

    assert _worker_error_payload(error)["code"] == "WRITER_QUEUE_FULL"
    assert not backend.stream(AudioSource.ME).is_active()
    assert not backend.stream(AudioSource.OTHERS).is_active()
    assert list(harness.writer_out.queue) == [AudioDrainFence()]


def test_asr_spool_limit_is_fatal_and_leaves_no_pump_thread() -> None:
    harness = AudioWorkerHarness(asr_spool_max_frames=1)
    harness.asr_out.put_nowait(
        AudioFrame(AudioSource.ME, bytes(640), 9_600, 9_920, 600, 1_600)
    )
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.emit_me(bytes(1_920), at_ms=1_020)
    backend.emit_me(bytes(1_920), at_ms=1_040)
    error = harness.status(MessageType.WORKER_ERROR)
    harness.join()

    assert _worker_error_payload(error)["code"] == "ASR_QUEUE_STALLED"
    assert (
        sum(isinstance(item, AudioWriteCommand) for item in harness.writer_out.queue)
        == 2
    )
    assert not any(
        thread.name == "flowlens-audio-asr-pump" for thread in threading.enumerate()
    )


def test_closed_asr_queue_reports_fatal_without_hanging_normal_stop() -> None:
    harness = AudioWorkerHarness(asr_queue_factory=ClosedAsrQueue)
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    asr = cast(ClosedAsrQueue, harness.asr_out)
    backend.emit_me(bytes(1_920), at_ms=1_020)
    assert asr.put_called.wait(timeout=1)
    harness.send(MessageType.WORKER_STOP, {"worker": "AUDIO"})

    try:
        error = harness.status(MessageType.WORKER_ERROR, timeout=0.25)
    finally:
        if harness.thread is not None and harness.thread.is_alive():
            harness.dispatcher.abort_asr_pump()
            harness.join()

    harness.join()
    payload = _worker_error_payload(error)
    assert payload["code"] == "ASR_QUEUE_STALLED"
    assert payload["detail"].startswith("closed ASR queue")
    assert len(payload["detail"]) <= 512
    assert not any(
        isinstance(item, AudioDrainFence) for item in list(harness.writer_out.queue)
    )
    statuses = list(harness.control_out.queue)
    assert not any(
        status.message_type is MessageType.WORKER_STOPPED for status in statuses
    )
    assert harness.dispatcher.pending_asr_frames == 0
    assert backend.close_calls == 1
    assert not any(
        thread.name == "flowlens-audio-asr-pump" for thread in threading.enumerate()
    )


def test_initial_open_failure_emits_error_without_ready_or_fence() -> None:
    backend = FakeCaptureBackend()
    backend.denied_ids.add("wasapi-output:7")
    harness = AudioWorkerHarness(backend=backend)
    harness.thread = threading.Thread(
        target=_audio_worker_loop,
        kwargs={
            "config": harness.config,
            "control_in": harness.control_in,
            "control_out": harness.control_out,
            "writer_audio_out": harness.writer_out,
            "backend": backend,
            "normalizer_factory": lambda source, rate, channels, started: (
                FakeNormalizer(source, started)
            ),
            "dispatcher": AudioDispatcher(harness.writer_out, harness.asr_out, 32),
            "monotonic_ms": harness.clock,
            "sleeper": lambda _: None,
        },
    )
    harness.thread.start()
    error = harness.status(MessageType.WORKER_ERROR)
    harness.join()

    assert _worker_error_payload(error)["code"] == "DEVICE_OPEN_FAILED"
    assert harness.control_out.empty()
    assert harness.writer_out.empty()
    assert backend.stream(AudioSource.ME).closed
    assert backend.close_calls == 1


def test_fence_queue_full_is_fatal_and_never_reports_drained() -> None:
    harness = AudioWorkerHarness(writer_queue_max_frames=1)
    harness.start_recording()
    existing = AudioWriteCommand(AudioSource.ME, bytes(640), 0, 320, 0, 1_000)
    harness.writer_out.put_nowait(existing)
    harness.send(MessageType.WORKER_STOP, {"worker": "AUDIO"})
    error = harness.status(MessageType.WORKER_ERROR)
    harness.join()

    assert _worker_error_payload(error)["code"] == "WRITER_QUEUE_FULL"
    assert list(harness.writer_out.queue) == [existing]
    assert harness.control_out.empty()


def test_raw_callback_overflow_becomes_fatal_outside_callback() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    harness.stop_polling.set()
    time.sleep(0.02)
    for _ in range(257):
        backend.emit_me(bytes(1_920), at_ms=1_020)
    harness.release_polling.set()
    error = harness.status(MessageType.WORKER_ERROR)
    harness.join()

    assert _worker_error_payload(error)["code"] == "CAPTURE_QUEUE_FULL"


def test_malformed_duplicate_and_wrong_target_controls_are_ignored() -> None:
    harness = AudioWorkerHarness()
    harness.start()
    backend = cast(FakeCaptureBackend, harness.backend)
    harness.control_in.put_nowait("not an envelope")
    harness.send(
        MessageType.WORKER_START,
        {"worker": "AUDIO"},
        session_id=OTHER_SESSION_ID,
    )
    harness.send(
        MessageType.WORKER_START,
        {"worker": "AUDIO"},
        source=ProcessSource.ASR,
    )
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.send(MessageType.WORKER_START, {"worker": "AUDIO"}, sequence=10)
    harness.send(MessageType.WORKER_START, {"worker": "AUDIO"}, sequence=10)
    harness.wait_until(lambda: backend.stream(AudioSource.ME).start_calls == 1)

    assert backend.stream(AudioSource.OTHERS).start_calls == 1
    harness.send(MessageType.WORKER_STOP, {"worker": "AUDIO"}, sequence=11)
    harness.status(MessageType.WORKER_STOPPED)
    harness.join()


def test_status_envelopes_have_audio_local_contiguous_sequence_and_shape() -> None:
    harness = AudioWorkerHarness()
    harness.start()
    assert harness.ready is not None
    ready = harness.ready
    harness.send(MessageType.WORKER_START, {"worker": "AUDIO"})
    backend = cast(FakeCaptureBackend, harness.backend)
    harness.wait_until(lambda: backend.stream(AudioSource.ME).is_active())
    backend.emit_me((32767).to_bytes(2, "little", signed=True) * 960, at_ms=1_020)
    level = harness.status(MessageType.AUDIO_LEVEL)
    harness.send(MessageType.WORKER_STOP, {"worker": "AUDIO"})
    stopped = harness.status(MessageType.WORKER_STOPPED)
    harness.join()

    assert [ready.sequence, level.sequence, stopped.sequence] == [1, 2, 3]
    assert all(
        envelope.session_id == SESSION_ID
        and envelope.source is ProcessSource.AUDIO
        and envelope.schema_version == 1
        for envelope in (ready, level, stopped)
    )
    assert level.payload == {"source": "ME", "peak_dbfs": 0.0}


def test_callback_after_stop_does_not_put_audio_after_fence() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    callback = backend.callbacks[AudioSource.ME]
    harness.stop()
    callback(RawAudioChunk(AudioSource.ME, bytes(1_920), 48_000, 1, 2_000))

    items = list(harness.writer_out.queue)
    assert items == [AudioDrainFence()]


def test_stop_waits_for_accepted_callback_enqueue_before_drain_and_fence() -> None:
    interleaving_queue = InterleavingRawQueue()
    created = 0

    def raw_queue_factory(
        maxsize: int,
    ) -> InterleavingRawQueue | queue.Queue[RawAudioChunk]:
        nonlocal created
        created += 1
        if created == 1:
            return interleaving_queue
        return queue.Queue(maxsize=maxsize)

    harness = AudioWorkerHarness(raw_queue_factory=raw_queue_factory)
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    callback = threading.Thread(
        target=backend.emit_me,
        args=(bytes(1_920), 1_020),
        name="interleaved-capture-callback",
    )
    callback.start()
    assert interleaving_queue.put_entered.wait(timeout=1)

    harness.send(MessageType.WORKER_STOP, {"worker": "AUDIO"})
    time.sleep(0.02)
    interleaving_queue.release_put.set()
    callback.join(timeout=1)
    assert not callback.is_alive()
    stopped = harness.status(MessageType.WORKER_STOPPED)
    harness.join()

    items = list(harness.writer_out.queue)
    assert isinstance(items[0], AudioWriteCommand)
    assert isinstance(items[1], AudioDrainFence)
    assert len(items) == 2
    assert interleaving_queue.qsize() == 0
    assert _worker_stopped_payload(stopped)["drained"] is True


def test_cleanup_failures_attempt_every_resource_and_preserve_fatal_code() -> None:
    harness = AudioWorkerHarness(writer_queue_max_frames=1)
    harness.start_recording()
    backend = cast(FakeCaptureBackend, harness.backend)
    backend.stream(AudioSource.ME).stop_error = OSError("stop failed")
    backend.stream(AudioSource.ME).close_error = OSError("close failed")
    backend.close_error = OSError("backend close failed")
    harness.writer_out.put_nowait(AudioDrainFence())
    backend.emit_me(bytes(1_920), at_ms=1_020)
    error = harness.status(MessageType.WORKER_ERROR)
    harness.join()

    assert _worker_error_payload(error)["code"] == "WRITER_QUEUE_FULL"
    assert backend.stream(AudioSource.ME).close_calls == 1
    assert backend.stream(AudioSource.OTHERS).stop_calls == 1
    assert backend.stream(AudioSource.OTHERS).close_calls == 1
    assert backend.close_calls == 1


def test_public_entrypoint_is_spawn_pickle_safe_without_opening_hardware() -> None:
    assert pickle.loads(pickle.dumps(run_audio_worker)) is run_audio_worker
    assert ForkingPickler.loads(ForkingPickler.dumps(run_audio_worker)) is (
        run_audio_worker
    )
    context = multiprocessing.get_context("spawn")
    assert context.get_start_method() == "spawn"
