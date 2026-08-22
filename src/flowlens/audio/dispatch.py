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


class _WriterAudioOut(Protocol):
    def put_nowait(self, item: AudioWriteCommand | AudioDrainFence) -> None: ...


class _AsrAudioOut(Protocol):
    def put(
        self,
        item: AudioFrame,
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
        self._spool: deque[AudioFrame] = deque()
        self._dispatch_lock = threading.Lock()
        self._condition = threading.Condition()
        self._last_submitted_frame: AudioFrame | None = None

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
                while not self._spool:
                    if stop_event.is_set():
                        return
                    self._condition.wait(timeout=0.1)
                frame = self._spool[0]
            try:
                self._asr_audio_out.put(frame, timeout=0.1)
            except queue.Full:
                continue
            with self._condition:
                if not self._spool or self._spool[0] is not frame:
                    raise RuntimeError("ASR spool order changed while submitting")
                self._spool.popleft()
                self._last_submitted_frame = frame
                self._condition.notify_all()

    def asr_backlog_ms(self, now_monotonic_ms: int) -> int:
        """Estimate backlog age from the spool and last ASR submission."""

        now = require_non_negative_int(now_monotonic_ms, "now_monotonic_ms")
        with self._condition:
            candidates: list[AudioFrame] = []
            if self._last_submitted_frame is not None:
                candidates.append(self._last_submitted_frame)
            if self._spool:
                candidates.append(self._spool[0])
            if not candidates:
                return 0
            oldest_ms = min(frame.captured_monotonic_ms for frame in candidates)
        return max(0, now - oldest_ms)

    @property
    def pending_asr_frames(self) -> int:
        """Return the number of frames still held in the in-process spool."""

        with self._condition:
            return len(self._spool)

    def wait_for_asr_spool_empty(self, timeout: float | None = None) -> bool:
        """Wait until every accepted frame has reached the ASR process queue."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        with self._condition:
            return self._condition.wait_for(lambda: not self._spool, timeout=timeout)

    def wake_asr_pump(self) -> None:
        """Wake a pump waiting on an empty spool after its stop event changes."""

        with self._condition:
            self._condition.notify_all()
