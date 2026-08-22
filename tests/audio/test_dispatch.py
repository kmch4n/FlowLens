"""Priority, backpressure, and pump tests for audio dispatch."""

import queue
import threading
import time

import pytest

from flowlens.audio.dispatch import (
    AsrPumpFailed,
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


def _require_frame(item: AudioFrame | AudioDrainFence) -> AudioFrame:
    assert isinstance(item, AudioFrame)
    return item


def test_dispatches_writer_command_before_same_asr_frame() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=1)
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
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=3_000)

    with pytest.raises(WriterQueueFull):
        dispatcher.dispatch(_frame())

    assert dispatcher.pending_asr_frames == 0
    assert asr.empty()


def test_full_asr_spool_raises_after_writer_accepts_without_silent_drop() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=2)
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=1)
    dispatcher.dispatch(_frame(0))

    with pytest.raises(AsrSpoolFull):
        dispatcher.dispatch(_frame(1))

    assert writer.qsize() == 2
    assert dispatcher.pending_asr_frames == 1
    assert asr.empty()


def test_blocked_asr_consumer_never_delays_writer_dispatch_and_pump_drains() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=10)
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=1)
    asr.put_nowait(_frame(99))
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=10)
    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()

    for index in range(10):
        dispatcher.dispatch(_frame(index))

    assert writer.qsize() == 10
    assert _require_frame(asr.get(timeout=1)).source_start_sample == 99 * 320
    submitted = [_require_frame(asr.get(timeout=1)) for _ in range(10)]
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
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=1)
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


def test_asr_fence_is_pumped_after_every_preceding_frame() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=2)
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=3)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=2)
    dispatcher.dispatch(_frame(0))
    dispatcher.dispatch(_frame(1))
    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()

    assert dispatcher.enqueue_asr_fence(timeout=1)
    items = [asr.get(timeout=1) for _ in range(3)]
    stop.set()
    dispatcher.wake_asr_pump()
    pump.join(timeout=1)

    assert items == [_frame(0), _frame(1), AudioDrainFence()]
    assert not pump.is_alive()


def test_asr_fence_submission_times_out_on_full_queue_without_hanging() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=1)
    asr.put_nowait(_frame(99))
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=1)
    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()

    assert dispatcher.enqueue_asr_fence(timeout=0.01) is False
    dispatcher.abort_asr_pump()
    stop.set()
    dispatcher.wake_asr_pump()
    pump.join(timeout=1)

    assert not pump.is_alive()
    assert dispatcher.pending_asr_frames == 0


def test_abort_waits_for_in_flight_put_and_prevents_late_submission() -> None:
    class InterleavingAsrOut:
        def __init__(self) -> None:
            self.put_entered = threading.Event()
            self.release_put = threading.Event()
            self.items: list[AudioFrame | AudioDrainFence] = []

        def put(
            self,
            item: AudioFrame | AudioDrainFence,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            del block, timeout
            self.put_entered.set()
            assert self.release_put.wait(timeout=2)
            self.items.append(item)

    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    asr = InterleavingAsrOut()
    dispatcher = AudioDispatcher(
        asr_audio_out=asr,
        writer_audio_out=writer,
        asr_spool_max_frames=1,
    )
    frame = _frame()
    dispatcher.dispatch(frame)
    stop = threading.Event()
    pump_errors: list[BaseException] = []

    def pump_target() -> None:
        try:
            dispatcher.run_asr_pump(stop)
        except BaseException as exc:
            pump_errors.append(exc)

    pump = threading.Thread(target=pump_target)
    pump.start()
    assert asr.put_entered.wait(timeout=1)
    abort_done = threading.Event()

    def abort_target() -> None:
        dispatcher.abort_asr_pump()
        abort_done.set()

    abort = threading.Thread(target=abort_target)
    abort.start()
    assert not abort_done.wait(timeout=0.05)
    asr.release_put.set()
    abort.join(timeout=1)
    pump.join(timeout=1)

    assert not abort.is_alive()
    assert not pump.is_alive()
    assert abort_done.is_set()
    assert pump_errors == []
    assert asr.items == [frame]
    assert dispatcher.pending_asr_frames == 0


def test_terminal_asr_put_failure_wakes_drain_waiter_with_typed_error() -> None:
    class ClosedAsrOut:
        def __init__(self) -> None:
            self.put_called = threading.Event()

        def put(
            self,
            item: AudioFrame | AudioDrainFence,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            del item, block, timeout
            self.put_called.set()
            raise EOFError("ASR queue is closed")

    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=1)
    asr = ClosedAsrOut()
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=1)
    dispatcher.dispatch(_frame())
    stop = threading.Event()
    pump = threading.Thread(target=dispatcher.run_asr_pump, args=(stop,))
    pump.start()
    assert asr.put_called.wait(timeout=1)

    with pytest.raises(AsrPumpFailed, match="ASR queue is closed") as failure:
        dispatcher.wait_for_asr_spool_empty(timeout=1)

    assert isinstance(failure.value.failure, EOFError)
    pump.join(timeout=1)
    assert not pump.is_alive()
    assert dispatcher.pending_asr_frames == 1
    dispatcher.abort_asr_pump()
    assert dispatcher.pending_asr_frames == 0


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
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=2)
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
    asr_order = [
        _require_frame(asr.get(timeout=1)).source_start_sample for _ in range(2)
    ]
    assert dispatcher.wait_for_asr_spool_empty(timeout=1)
    stop.set()
    dispatcher.wake_asr_pump()
    pump.join(timeout=1)

    writer_order = [item.source_start_sample for item in writer.items]
    assert asr_order == writer_order


def test_backlog_uses_oldest_submitted_or_spooled_frame_and_clamps() -> None:
    writer: queue.Queue[AudioWriteCommand | AudioDrainFence] = queue.Queue(maxsize=2)
    asr: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(maxsize=2)
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
