"""Priority, backpressure, and pump tests for audio dispatch."""

import queue
import threading
import time

import pytest

from flowlens.audio.dispatch import (
    AsrSpoolFull,
    AudioDispatcher,
    WriterQueueFull,
)
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import AudioDrainFence, AudioWriteCommand


def _frame(index: int = 0, *, captured_ms: int | None = None) -> AudioFrame:
    start = index * 320
    return AudioFrame(
        AudioSource.ME,
        bytes([index % 128]) * 640,
        start,
        start + 320,
        100 + index * 20,
        1_100 + index * 20 if captured_ms is None else captured_ms,
    )


def test_dispatches_writer_command_before_same_asr_frame() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=3_000)
    frame = _frame()

    dispatcher.dispatch(frame)

    command = writer.get_nowait()
    assert isinstance(command, AudioWriteCommand)
    assert command == AudioWriteCommand(
        frame.source,
        frame.pcm_s16le,
        frame.source_start_sample,
        frame.source_end_sample,
        frame.session_start_ms,
        frame.captured_monotonic_ms,
    )
    assert dispatcher.pending_asr_frames == 1


def test_full_writer_queue_is_fatal_before_asr_dispatch() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    writer.put_nowait(AudioWriteCommand(AudioSource.ME, bytes(640), 0, 320, 100, 1_100))
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=3_000)

    with pytest.raises(WriterQueueFull):
        dispatcher.dispatch(_frame())

    assert dispatcher.pending_asr_frames == 0
    assert asr.empty()


def test_full_asr_spool_raises_after_writer_accepts_without_silent_drop() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=2)
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=1)
    dispatcher.dispatch(_frame(0))

    with pytest.raises(AsrSpoolFull):
        dispatcher.dispatch(_frame(1))

    assert writer.qsize() == 2
    assert dispatcher.pending_asr_frames == 1
    assert asr.empty()


def test_blocked_asr_consumer_never_delays_writer_dispatch_and_pump_drains() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=10)
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=1)
    asr.put_nowait(_frame(99))
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=10)
    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()

    for index in range(10):
        dispatcher.dispatch(_frame(index))

    assert writer.qsize() == 10
    assert asr.get(timeout=1).source_start_sample == 99 * 320
    submitted = [asr.get(timeout=1) for _ in range(10)]
    assert [frame.source_start_sample for frame in submitted] == [
        index * 320 for index in range(10)
    ]
    assert dispatcher.wait_for_asr_spool_empty(timeout=1)
    stop.set()
    dispatcher.wake_asr_pump()
    pump.join(timeout=1)

    assert not pump.is_alive()


def test_pump_stop_waits_for_accepted_spool_instead_of_dropping() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=1)
    asr.put_nowait(_frame(99))
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=2)
    accepted = _frame()
    dispatcher.dispatch(accepted)
    stop = threading.Event()
    stop.set()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()

    time.sleep(0.15)
    assert pump.is_alive()
    asr.get(timeout=1)
    assert asr.get(timeout=1) is accepted
    pump.join(timeout=1)

    assert not pump.is_alive()


def test_concurrent_dispatch_preserves_same_writer_and_asr_order() -> None:
    first_inside_put = threading.Event()
    release_first = threading.Event()

    class InterleavingWriter:
        def __init__(self) -> None:
            self.items: list[AudioWriteCommand] = []

        def put_nowait(self, item: AudioWriteCommand | AudioDrainFence) -> None:
            assert isinstance(item, AudioWriteCommand)
            self.items.append(item)
            if len(self.items) == 1:
                first_inside_put.set()
                assert release_first.wait(timeout=1)

    writer = InterleavingWriter()
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=2)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=2)
    first = threading.Thread(target=dispatcher.dispatch, args=(_frame(0),))
    second = threading.Thread(target=dispatcher.dispatch, args=(_frame(1),))

    first.start()
    assert first_inside_put.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive() and not second.is_alive()

    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()
    asr_order = [asr.get(timeout=1).source_start_sample for _ in range(2)]
    assert dispatcher.wait_for_asr_spool_empty(timeout=1)
    stop.set()
    dispatcher.wake_asr_pump()
    pump.join(timeout=1)

    writer_order = [item.source_start_sample for item in writer.items]
    assert asr_order == writer_order


def test_backlog_uses_oldest_submitted_or_spooled_frame_and_clamps() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=2)
    asr: queue.Queue[AudioFrame] = queue.Queue(maxsize=2)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=2)
    dispatcher.dispatch(_frame(0, captured_ms=1_100))
    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()
    assert dispatcher.wait_for_asr_spool_empty(timeout=1)
    dispatcher.dispatch(_frame(1, captured_ms=1_120))

    assert dispatcher.asr_backlog_ms(1_200) == 100
    assert dispatcher.asr_backlog_ms(1_000) == 0

    asr.get(timeout=1)
    asr.get(timeout=1)
    assert dispatcher.wait_for_asr_spool_empty(timeout=1)
    stop.set()
    dispatcher.wake_asr_pump()
    pump.join(timeout=1)
