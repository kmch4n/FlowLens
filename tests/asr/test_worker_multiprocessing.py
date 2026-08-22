"""Real multiprocessing queue regressions for ASR drain fences."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from pathlib import Path

from flowlens.asr.types import AsrWorkerConfig
from flowlens.asr.worker import _asr_worker_loop
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import AudioDrainFence, MessageEnvelope
from tests.asr.test_worker import (
    SESSION_ID,
    FakeClock,
    FakeDecoder,
    FakeEngine,
    FakeSpeechDetector,
    _frame,
)


def _run_worker(
    audio: multiprocessing.Queue[AudioFrame | AudioDrainFence],
    control: multiprocessing.Queue[MessageEnvelope[object]],
    output: multiprocessing.Queue[MessageEnvelope[object]],
    engine: FakeEngine,
    clock: FakeClock,
    result: list[int],
    errors: list[BaseException],
) -> None:
    try:
        result.append(
            _asr_worker_loop(
                AsrWorkerConfig(SESSION_ID, Path.cwd().resolve()),
                audio,
                control,
                output,
                decoder_factory=lambda _path: FakeDecoder(),
                speech_detector_factory=FakeSpeechDetector,
                engine_factory=lambda _config, _decoder, _detector: engine,
                monotonic_ms=clock,
                poll_timeout_seconds=0.001,
            )
        )
    except BaseException as exc:
        errors.append(exc)


def test_multiprocessing_queue_fence_preserves_precommand_frames_repeatedly() -> None:
    context = multiprocessing.get_context("spawn")
    for iteration in range(5):
        audio = context.Queue(maxsize=64)
        control = context.Queue(maxsize=8)
        output = context.Queue(maxsize=16)
        engine = FakeEngine()
        clock = FakeClock()
        result: list[int] = []
        errors: list[BaseException] = []
        thread = threading.Thread(
            target=_run_worker,
            args=(audio, control, output, engine, clock, result, errors),
        )
        thread.start()
        try:
            try:
                ready = output.get(timeout=2)
            except queue.Empty as exc:
                raise AssertionError(
                    f"worker produced no READY; errors={errors!r}"
                ) from exc
            assert ready.message_type is MessageType.WORKER_READY
            assert output.get(timeout=2).message_type is MessageType.ASR_STATUS
            control.put(
                MessageEnvelope(
                    1,
                    SESSION_ID,
                    MessageType.WORKER_START,
                    1,
                    ProcessSource.GUI,
                    clock(),
                    {"worker": "ASR"},
                )
            )
            assert output.get(timeout=1).payload["state"] == "RUNNING"
            for index in range(20):
                audio.put(_frame(index))
            boundary = (
                MessageType.WORKER_STOP
                if iteration % 2 == 0
                else MessageType.WORKER_PAUSE
            )
            payload: dict[str, object] = {"worker": "ASR"}
            if boundary is MessageType.WORKER_STOP:
                payload["finalize"] = True
            control.put(
                MessageEnvelope(
                    1,
                    SESSION_ID,
                    boundary,
                    2,
                    ProcessSource.GUI,
                    clock(),
                    payload,
                )
            )
            audio.put(AudioDrainFence())
            if boundary is MessageType.WORKER_PAUSE:
                control.put(
                    MessageEnvelope(
                        1,
                        SESSION_ID,
                        MessageType.WORKER_STOP,
                        3,
                        ProcessSource.GUI,
                        clock(),
                        {"worker": "ASR", "finalize": True},
                    )
                )
                audio.put(AudioDrainFence())
            stopped = output.get(timeout=2)
            while stopped.message_type is not MessageType.WORKER_STOPPED:
                stopped = output.get(timeout=2)
            thread.join(timeout=2)

            assert not thread.is_alive()
            assert result == [0]
            assert engine.accepted == [_frame(index) for index in range(20)]
            assert engine.finalize_calls == 1
        finally:
            if thread.is_alive():
                clock.advance(30_000)
                thread.join(timeout=1)
            for process_queue in (audio, control, output):
                process_queue.close()
                process_queue.join_thread()
