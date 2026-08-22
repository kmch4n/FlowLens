"""Priority-safe fan-out from canonical audio to Writer and ASR paths."""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Protocol

from flowlens.audio.types import AudioFrame
from flowlens.domain._validation import require_non_negative_int
from flowlens.domain.messages import AudioDrainFence, AudioWriteCommand


class WriterQueueFull(RuntimeError):
    """Raised when canonical audio cannot be handed to the Writer."""


class AsrSpoolFull(RuntimeError):
    """Raised when the bounded in-process ASR spool is exhausted."""


class AsrPumpFailed(RuntimeError):
    """Raised when the ASR queue pump terminates before draining its spool."""

    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        detail = str(failure).strip() or type(failure).__name__
        super().__init__(f"ASR pump failed: {detail}")


class _WriterAudioOut(Protocol):
    def put_nowait(self, item: AudioWriteCommand | AudioDrainFence) -> None: ...


class _AsrAudioOut(Protocol):
    def put(
        self,
        item: AudioFrame | AudioDrainFence,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...


class _StopEvent(Protocol):
    def is_set(self) -> bool: ...


class AudioDispatcher:
    """Persist every frame first, then asynchronously feed the ASR path."""

    def __init__(
        self,
        writer_audio_out: _WriterAudioOut,
        asr_audio_out: _AsrAudioOut,
        asr_spool_max_frames: int,
    ) -> None:
        if (
            not isinstance(asr_spool_max_frames, int)
            or isinstance(asr_spool_max_frames, bool)
            or asr_spool_max_frames <= 0
        ):
            raise ValueError("asr_spool_max_frames must be a positive integer")
        self._writer_audio_out = writer_audio_out
        self._asr_audio_out = asr_audio_out
        self._asr_spool_max_frames = asr_spool_max_frames
        self._spool: deque[AudioFrame | AudioDrainFence] = deque()
        self._dispatch_lock = threading.Lock()
        self._condition = threading.Condition()
        self._last_submitted_frame: AudioFrame | None = None
        self._aborted = False
        self._pump_in_flight = False
        self._pump_failure: Exception | None = None

    def dispatch(self, frame: AudioFrame) -> None:
        """Send one frame to Writer before accepting it into the ASR spool."""

        if not isinstance(frame, AudioFrame):
            raise ValueError("frame must be an AudioFrame")
        command = AudioWriteCommand(
            source=frame.source,
            pcm_s16le=frame.pcm_s16le,
            source_start_sample=frame.source_start_sample,
            source_end_sample=frame.source_end_sample,
            session_start_ms=frame.session_start_ms,
            captured_monotonic_ms=frame.captured_monotonic_ms,
        )
        with self._dispatch_lock:
            try:
                self._writer_audio_out.put_nowait(command)
            except queue.Full as exc:
                raise WriterQueueFull("Writer audio queue is full") from exc

            with self._condition:
                if len(self._spool) >= self._asr_spool_max_frames:
                    raise AsrSpoolFull("ASR audio spool is full")
                self._spool.append(frame)
                self._condition.notify()

    def run_asr_pump(self, stop_event: _StopEvent) -> None:
        """Drain accepted frames to ASR without blocking capture dispatch."""

        while True:
            with self._condition:
                if self._aborted:
                    return
                while not self._spool:
                    if stop_event.is_set():
                        return
                    self._condition.wait(timeout=0.1)
                    if self._aborted:
                        return
                item = self._spool[0]
                self._pump_in_flight = True
            try:
                self._asr_audio_out.put(item, timeout=0.1)
            except queue.Full:
                with self._condition:
                    self._pump_in_flight = False
                    self._condition.notify_all()
                continue
            except Exception as exc:
                with self._condition:
                    if self._pump_failure is None:
                        self._pump_failure = exc
                    self._pump_in_flight = False
                    self._condition.notify_all()
                return
            with self._condition:
                try:
                    if not self._spool or self._spool[0] is not item:
                        raise RuntimeError("ASR spool order changed while submitting")
                    self._spool.popleft()
                    if isinstance(item, AudioFrame):
                        self._last_submitted_frame = item
                finally:
                    self._pump_in_flight = False
                    self._condition.notify_all()

    def asr_backlog_ms(self, now_monotonic_ms: int) -> int:
        """Estimate backlog age from the spool and last ASR submission."""

        now = require_non_negative_int(now_monotonic_ms, "now_monotonic_ms")
        with self._condition:
            candidates: list[AudioFrame] = []
            if self._last_submitted_frame is not None:
                candidates.append(self._last_submitted_frame)
            if self._spool:
                candidates.extend(
                    item for item in self._spool if isinstance(item, AudioFrame)
                )
            if not candidates:
                return 0
            oldest_ms = min(frame.captured_monotonic_ms for frame in candidates)
        return max(0, now - oldest_ms)

    @property
    def pending_asr_frames(self) -> int:
        """Return the number of frames still held in the in-process spool."""

        with self._condition:
            return sum(isinstance(item, AudioFrame) for item in self._spool)

    def enqueue_asr_fence(self, timeout: float | None = None) -> bool:
        """Submit an ordered ASR boundary after all preceding frames."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        with self._dispatch_lock:
            with self._condition:
                if self._aborted:
                    raise AsrPumpFailed(RuntimeError("ASR pump is aborted"))
                if self._pump_failure is not None:
                    raise AsrPumpFailed(self._pump_failure)
                self._spool.append(AudioDrainFence())
                self._condition.notify_all()
                completed = self._condition.wait_for(
                    lambda: not self._spool or self._pump_failure is not None,
                    timeout=timeout,
                )
                if self._pump_failure is not None:
                    raise AsrPumpFailed(self._pump_failure)
                return completed and not self._spool

    def wait_for_asr_spool_empty(self, timeout: float | None = None) -> bool:
        """Wait until every accepted frame has reached the ASR process queue."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        with self._condition:
            completed = self._condition.wait_for(
                lambda: not self._spool or self._pump_failure is not None,
                timeout=timeout,
            )
            if self._pump_failure is not None:
                raise AsrPumpFailed(self._pump_failure)
            return completed and not self._spool

    def wake_asr_pump(self) -> None:
        """Wake a pump waiting on an empty spool after its stop event changes."""

        with self._condition:
            self._condition.notify_all()

    def abort_asr_pump(self) -> None:
        """Discard pending ASR work only after a fatal worker shutdown."""

        with self._condition:
            self._aborted = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: not self._pump_in_flight)
            self._spool.clear()
            self._condition.notify_all()
