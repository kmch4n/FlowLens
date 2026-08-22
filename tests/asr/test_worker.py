"""ASR Worker lifecycle, serialization, and lag-state contract tests."""

from __future__ import annotations

import pickle
import queue
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from flowlens.asr.engine import AsrBatch
from flowlens.asr.ports import DecoderPort
from flowlens.asr.types import AsrWorkerConfig, DecodeHypothesis, PartialTranscript
from flowlens.asr.worker import _asr_worker_loop, run_asr_worker
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import AudioDrainFence, MessageEnvelope, TranscriptRecord

SESSION_ID = "01J00000000000000000000000"
_SYNC = object()


class ObservedControlQueue(queue.Queue[object]):
    """Expose when the worker crosses a test synchronization point."""

    def __init__(self) -> None:
        super().__init__(maxsize=32)
        self.sync_observed = threading.Event()

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        value = super().get(block=block, timeout=timeout)
        if value is _SYNC:
            self.sync_observed.set()
        return value


class ClosedControlQueue:
    """Control queue double that fails immediately instead of blocking."""

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        del block, timeout
        raise OSError("closed control queue")

    def get_nowait(self) -> object:
        raise OSError("closed control queue")


class FakeDecoder:
    """No-op decoder satisfying the injected model boundary."""

    def decode(self, pcm_s16le: bytes) -> DecodeHypothesis:
        del pcm_s16le
        return DecodeHypothesis(())


class FakeSpeechDetector:
    """No-op detector satisfying the injected VAD boundary."""

    def is_speech(self, frame: AudioFrame) -> bool:
        del frame
        return False


class FakeClock:
    """Thread-safe monotonic clock controlled by the test."""

    def __init__(self, initial_ms: int = 1_000) -> None:
        self._value = initial_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._value

    def advance(self, milliseconds: int) -> None:
        with self._lock:
            self._value += milliseconds


class ClockAdvancingAudioQueue(queue.Queue[object]):
    """Advance fake time whenever the worker dequeues an audio item."""

    def __init__(self, clock: FakeClock, step_ms: int) -> None:
        super().__init__(maxsize=32)
        self._clock = clock
        self._step_ms = step_ms

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        value = super().get(block=block, timeout=timeout)
        self._clock.advance(self._step_ms)
        return value


class FakeEngine:
    """Deterministic engine seam that records worker-facing behavior."""

    def __init__(self) -> None:
        self.accepted: list[AudioFrame] = []
        self.finalize_calls = 0
        self.finalized = threading.Event()
        self.final_batch = AsrBatch((), ())
        self.process_batches: deque[AsrBatch] = deque()
        self.process_calls = 0
        self.backlog = 0
        self.fail_on_process_call: int | None = None
        self._condition = threading.Condition()

    def accept(self, frame: AudioFrame) -> None:
        with self._condition:
            self.accepted.append(frame)
            self._condition.notify_all()

    def process_ready(self, now_monotonic_ms: int) -> AsrBatch:
        del now_monotonic_ms
        with self._condition:
            self.process_calls += 1
            self._condition.notify_all()
            if self.fail_on_process_call == self.process_calls:
                raise RuntimeError("decoder exploded")
            if self.process_batches:
                return self.process_batches.popleft()
            return AsrBatch((), ())

    def finalize(self, now_monotonic_ms: int) -> AsrBatch:
        del now_monotonic_ms
        self.finalize_calls += 1
        self.finalized.set()
        return self.final_batch

    def backlog_ms(self, now_monotonic_ms: int) -> int:
        del now_monotonic_ms
        return self.backlog

    def wait_for_process_calls(self, minimum: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: self.process_calls >= minimum,
                timeout=1,
            )

    def wait_for_accepted(self, minimum: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: len(self.accepted) >= minimum,
                timeout=1,
            )


class AsrWorkerHarness:
    """Run the injected worker loop against bounded in-memory queues."""

    def __init__(
        self,
        *,
        decoder_factory: Callable[[Path], DecoderPort] | None = None,
    ) -> None:
        self.clock = FakeClock()
        self.engine = FakeEngine()
        self.audio_in: queue.Queue[object] = queue.Queue(maxsize=32)
        self.control_in = ObservedControlQueue()
        self.control_out: queue.Queue[MessageEnvelope[object]] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.control_sequence = 0
        self.exit_code: int | None = None
        self._decoder_factory = decoder_factory

    @property
    def config(self) -> AsrWorkerConfig:
        return AsrWorkerConfig(
            session_id=SESSION_ID,
            model_path=Path.cwd().resolve(),
        )

    def start(self) -> None:
        decoder_factory = self._decoder_factory

        def run() -> None:
            self.exit_code = _asr_worker_loop(
                config=self.config,
                audio_in=self.audio_in,
                control_in=self.control_in,
                control_out=self.control_out,
                decoder_factory=(
                    (lambda _path: FakeDecoder())
                    if decoder_factory is None
                    else decoder_factory
                ),
                speech_detector_factory=FakeSpeechDetector,
                engine_factory=lambda _config, _decoder, _detector: self.engine,
                monotonic_ms=self.clock,
                poll_timeout_seconds=0.001,
            )

        self.thread = threading.Thread(
            target=run,
            name="test-asr-worker",
        )
        self.thread.start()
        if decoder_factory is not None:
            return
        self.output(MessageType.WORKER_READY)
        ready_status = self.output(MessageType.ASR_STATUS)
        assert ready_status.payload == {
            "state": "READY",
            "backlog_ms": 0,
            "analysis_paused": False,
        }

    def send(self, message_type: MessageType, payload: object) -> None:
        self.control_sequence += 1
        self.control_in.put_nowait(
            MessageEnvelope(
                1,
                SESSION_ID,
                message_type,
                self.control_sequence,
                ProcessSource.GUI,
                self.clock(),
                payload,
            )
        )

    def output(self, message_type: MessageType) -> MessageEnvelope[object]:
        deferred: list[MessageEnvelope[object]] = []
        while True:
            try:
                item = self.control_out.get(timeout=1)
            except queue.Empty as exc:
                raise AssertionError(f"missing output {message_type.value}") from exc
            if item.message_type is message_type:
                for other in deferred:
                    self.control_out.put_nowait(other)
                return item
            deferred.append(item)

    def snapshot(self) -> list[MessageEnvelope[object]]:
        items: list[MessageEnvelope[object]] = []
        while True:
            try:
                items.append(self.control_out.get_nowait())
            except queue.Empty:
                return items

    def join(self) -> None:
        assert self.thread is not None
        self.thread.join(timeout=1)
        assert not self.thread.is_alive()


def _frame(index: int = 0) -> AudioFrame:
    return AudioFrame(
        source=AudioSource.ME,
        pcm_s16le=bytes(640),
        source_start_sample=index * 320,
        source_end_sample=(index + 1) * 320,
        session_start_ms=index * 20,
        captured_monotonic_ms=1_000 + index * 20,
    )


def _record(text: str = "最終発言") -> TranscriptRecord:
    return TranscriptRecord(
        schema_version=1,
        segment_id="01J00000000000000000000001",
        sequence=1,
        source=AudioSource.ME,
        text=text,
        session_start_ms=0,
        session_end_ms=200,
        source_start_sample=0,
        source_end_sample=3_200,
        committed_at=datetime(2026, 8, 22, 1, 2, 3, 456000, tzinfo=UTC),
    )


def _partial(text: str = "確認中") -> PartialTranscript:
    return PartialTranscript(
        source=AudioSource.OTHERS,
        text=text,
        session_start_ms=40,
        session_end_ms=100,
        source_start_sample=640,
        source_end_sample=1_600,
    )


def test_stop_drains_audio_then_finalizes_uncommitted_text() -> None:
    harness = AsrWorkerHarness()
    harness.engine.final_batch = AsrBatch((), (_record(),))
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    running_status = harness.output(MessageType.ASR_STATUS)
    assert running_status.payload == {
        "state": "RUNNING",
        "backlog_ms": 0,
        "analysis_paused": False,
    }
    for index in range(10):
        harness.audio_in.put_nowait(_frame(index))
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    assert not harness.engine.finalized.wait(timeout=0.05)
    harness.audio_in.put_nowait(AudioDrainFence())

    committed = harness.output(MessageType.TRANSCRIPT_COMMITTED)
    stopped = harness.output(MessageType.WORKER_STOPPED)
    status = harness.output(MessageType.ASR_STATUS)
    harness.join()

    assert committed.payload == _record().to_dict()
    assert stopped.payload == {
        "worker": "ASR",
        "drained": True,
        "committed_count": 1,
    }
    assert status.payload == {
        "state": "STOPPED",
        "backlog_ms": 0,
        "analysis_paused": False,
    }
    assert harness.engine.accepted == [_frame(index) for index in range(10)]
    assert harness.engine.finalize_calls == 1


def test_partial_and_commit_payloads_are_exact_and_sequences_are_independent() -> None:
    harness = AsrWorkerHarness()
    harness.engine.process_batches.append(AsrBatch((_partial(),), (_record(),)))
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)
    harness.audio_in.put_nowait(_frame())

    partial = harness.output(MessageType.TRANSCRIPT_PARTIAL)
    committed = harness.output(MessageType.TRANSCRIPT_COMMITTED)
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.audio_in.put_nowait(AudioDrainFence())
    stopped = harness.output(MessageType.WORKER_STOPPED)
    stopped_status = harness.output(MessageType.ASR_STATUS)
    harness.join()

    assert partial.payload == {
        "source": "OTHERS",
        "text": "確認中",
        "session_start_ms": 40,
        "session_end_ms": 100,
        "source_start_sample": 640,
        "source_end_sample": 1_600,
    }
    assert committed.payload == {
        "schema_version": 1,
        "segment_id": "01J00000000000000000000001",
        "sequence": 1,
        "source": "ME",
        "text": "最終発言",
        "session_start_ms": 0,
        "session_end_ms": 200,
        "source_start_sample": 0,
        "source_end_sample": 3_200,
        "committed_at": "2026-08-22T01:02:03.456+00:00",
    }
    assert [partial.sequence, committed.sequence, stopped.sequence] == [4, 5, 6]
    assert stopped_status.sequence == 7
    assert all(
        item.source is ProcessSource.ASR
        for item in (partial, committed, stopped, stopped_status)
    )


def test_delayed_transition_is_strictly_above_two_seconds_and_emitted_once() -> None:
    harness = AsrWorkerHarness()
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)

    harness.engine.backlog = 2_000
    baseline = harness.engine.process_calls
    harness.engine.wait_for_process_calls(baseline + 2)
    assert harness.snapshot() == []

    harness.engine.backlog = 2_001
    delayed = harness.output(MessageType.ASR_STATUS)
    assert delayed.payload == {
        "state": "DELAYED",
        "backlog_ms": 2_001,
        "analysis_paused": False,
    }
    baseline = harness.engine.process_calls
    harness.engine.wait_for_process_calls(baseline + 2)
    assert harness.snapshot() == []
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.audio_in.put_nowait(AudioDrainFence())
    harness.output(MessageType.WORKER_STOPPED)
    harness.join()


def test_analysis_pause_and_resume_thresholds_are_strict_and_transition_only() -> None:
    harness = AsrWorkerHarness()
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)

    harness.engine.backlog = 5_000
    delayed = harness.output(MessageType.ASR_STATUS)
    assert delayed.payload["state"] == "DELAYED"  # type: ignore[index]
    assert delayed.payload["analysis_paused"] is False  # type: ignore[index]
    baseline = harness.engine.process_calls
    harness.engine.wait_for_process_calls(baseline + 2)
    assert harness.snapshot() == []

    harness.engine.backlog = 5_001
    paused = harness.output(MessageType.ASR_STATUS)
    assert paused.payload == {
        "state": "DELAYED",
        "backlog_ms": 5_001,
        "analysis_paused": True,
    }
    harness.engine.backlog = 2_000
    baseline = harness.engine.process_calls
    harness.engine.wait_for_process_calls(baseline + 2)
    assert harness.snapshot() == []

    harness.engine.backlog = 1_999
    resumed = harness.output(MessageType.ASR_STATUS)
    assert resumed.payload == {
        "state": "RUNNING",
        "backlog_ms": 1_999,
        "analysis_paused": False,
    }
    baseline = harness.engine.process_calls
    harness.engine.wait_for_process_calls(baseline + 2)
    assert harness.snapshot() == []
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.audio_in.put_nowait(AudioDrainFence())
    harness.output(MessageType.WORKER_STOPPED)
    harness.join()


def test_pause_drains_existing_audio_and_accepts_nothing_until_resume() -> None:
    harness = AsrWorkerHarness()
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)
    harness.audio_in.put_nowait(_frame(0))
    harness.audio_in.put_nowait(_frame(1))
    harness.send(MessageType.WORKER_PAUSE, {"worker": "ASR"})
    harness.audio_in.put_nowait(AudioDrainFence())
    harness.engine.wait_for_accepted(2)

    baseline = harness.engine.process_calls
    harness.audio_in.put_nowait(_frame(2))
    harness.control_in.put_nowait(
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_RESUME,
            harness.control_sequence,
            ProcessSource.GUI,
            harness.clock(),
            {"worker": "ASR"},
        )
    )
    harness.control_in.sync_observed.clear()
    harness.control_in.put_nowait(_SYNC)
    assert harness.control_in.sync_observed.wait(timeout=1)
    assert harness.engine.process_calls == baseline
    assert harness.engine.accepted == [_frame(0), _frame(1)]

    harness.send(MessageType.WORKER_RESUME, {"worker": "ASR"})
    harness.engine.wait_for_accepted(3)
    assert harness.engine.accepted == [_frame(0), _frame(1), _frame(2)]
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.audio_in.put_nowait(AudioDrainFence())
    harness.output(MessageType.WORKER_STOPPED)
    harness.join()


def test_invalid_misaddressed_duplicate_and_invalid_state_commands_are_ignored() -> (
    None
):
    harness = AsrWorkerHarness()
    harness.start()
    harness.control_in.put_nowait("not an envelope")
    harness.send(MessageType.WORKER_START, {"worker": "AUDIO"})
    harness.control_sequence += 1
    wrong_session = MessageEnvelope(
        1,
        "01J00000000000000000000002",
        MessageType.WORKER_START,
        harness.control_sequence,
        ProcessSource.GUI,
        harness.clock(),
        {"worker": "ASR"},
    )
    harness.control_in.put_nowait(wrong_session)
    harness.control_in.put_nowait(
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_START,
            99,
            ProcessSource.AUDIO,
            harness.clock(),
            {"worker": "ASR"},
        )
    )
    harness.send(MessageType.WORKER_PAUSE, {"worker": "ASR"})
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    running = harness.output(MessageType.ASR_STATUS)
    assert running.payload["state"] == "RUNNING"  # type: ignore[index]
    harness.control_in.put_nowait(
        MessageEnvelope(
            1,
            SESSION_ID,
            MessageType.WORKER_START,
            harness.control_sequence,
            ProcessSource.GUI,
            harness.clock(),
            {"worker": "ASR"},
        )
    )
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR"})
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.audio_in.put_nowait(AudioDrainFence())
    harness.output(MessageType.WORKER_STOPPED)
    harness.join()
    assert harness.engine.finalize_calls == 1


def test_model_load_failure_emits_once_without_ready() -> None:
    attempts = 0

    def raising_factory(_path: Path) -> DecoderPort:
        nonlocal attempts
        attempts += 1
        raise OSError("local model unavailable")

    harness = AsrWorkerHarness(decoder_factory=raising_factory)
    harness.start()
    error = harness.output(MessageType.WORKER_ERROR)
    harness.join()

    assert error.payload == {
        "worker": "ASR",
        "code": "MODEL_LOAD_FAILED",
        "detail": "local model unavailable",
    }
    assert harness.exit_code == 1
    assert attempts == 1
    assert harness.snapshot() == []


def test_decode_failure_preserves_prior_committed_payload_and_exits() -> None:
    harness = AsrWorkerHarness()
    harness.engine.process_batches.append(AsrBatch((), (_record("保存済み"),)))
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)
    harness.audio_in.put_nowait(_frame(0))
    first = harness.output(MessageType.TRANSCRIPT_COMMITTED)
    harness.engine.fail_on_process_call = harness.engine.process_calls + 1
    harness.audio_in.put_nowait(_frame(1))
    error = harness.output(MessageType.WORKER_ERROR)
    harness.join()

    assert first.payload["text"] == "保存済み"  # type: ignore[index]
    assert error.payload == {
        "worker": "ASR",
        "code": "DECODE_FAILED",
        "detail": "decoder exploded",
    }
    assert harness.exit_code == 1
    assert not any(
        item.message_type is MessageType.TRANSCRIPT_COMMITTED
        for item in harness.snapshot()
    )


def test_public_entrypoint_is_spawn_pickle_safe() -> None:
    assert pickle.loads(pickle.dumps(run_asr_worker)) is run_asr_worker


def test_closed_control_queue_reports_error_and_exits_without_hanging() -> None:
    output: queue.Queue[MessageEnvelope[object]] = queue.Queue()
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            _asr_worker_loop(
                AsrWorkerConfig(SESSION_ID, Path.cwd().resolve()),
                queue.Queue(),
                ClosedControlQueue(),
                output,
                decoder_factory=lambda _path: FakeDecoder(),
                speech_detector_factory=FakeSpeechDetector,
                monotonic_ms=FakeClock(),
                poll_timeout_seconds=0.001,
            )
        )
    )
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result == [1]
    items = list(output.queue)
    assert [item.message_type for item in items] == [
        MessageType.WORKER_READY,
        MessageType.ASR_STATUS,
        MessageType.WORKER_ERROR,
    ]
    assert items[-1].payload == {
        "worker": "ASR",
        "code": "DECODE_FAILED",
        "detail": "closed control queue",
    }


def test_missing_stop_fence_times_out_without_finalizing() -> None:
    harness = AsrWorkerHarness()
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)
    baseline = harness.engine.process_calls
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.engine.wait_for_process_calls(baseline + 1)
    harness.clock.advance(30_000)

    error = harness.output(MessageType.WORKER_ERROR)
    harness.join()

    assert harness.engine.finalize_calls == 0
    assert error.payload == {
        "worker": "ASR",
        "code": "DECODE_FAILED",
        "detail": "timed out waiting for ASR drain fence boundary",
    }


def _assert_continuous_items_do_not_extend_stop_deadline(items: list[object]) -> None:
    harness = AsrWorkerHarness()
    harness.audio_in = ClockAdvancingAudioQueue(harness.clock, step_ms=1_001)
    harness.start()
    harness.send(MessageType.WORKER_START, {"worker": "ASR"})
    harness.output(MessageType.ASR_STATUS)
    harness.send(MessageType.WORKER_STOP, {"worker": "ASR", "finalize": True})
    harness.control_in.put_nowait(_SYNC)
    assert harness.control_in.sync_observed.wait(timeout=1)
    for item in items:
        harness.audio_in.put_nowait(item)

    try:
        error = harness.output(MessageType.WORKER_ERROR)
    except BaseException:
        harness.audio_in.put_nowait(AudioDrainFence())
        harness.output(MessageType.WORKER_STOPPED)
        harness.join()
        raise
    harness.join()

    assert harness.engine.finalize_calls == 0
    assert error.payload == {
        "worker": "ASR",
        "code": "DECODE_FAILED",
        "detail": "timed out waiting for ASR drain fence boundary",
    }


def test_continuous_audio_frames_do_not_extend_stop_fence_deadline() -> None:
    _assert_continuous_items_do_not_extend_stop_deadline(
        [_frame(index) for index in range(30)]
    )


def test_continuous_unknown_items_do_not_extend_stop_fence_deadline() -> None:
    _assert_continuous_items_do_not_extend_stop_deadline(
        [object() for _index in range(30)]
    )
