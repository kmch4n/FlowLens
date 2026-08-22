"""Hardware-free integration proof for the canonical Audio-to-ASR path."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pytest

from flowlens.asr.engine import AsrEngine
from flowlens.asr.types import AsrWorkerConfig, DecodedToken, DecodeHypothesis
from flowlens.audio.dispatch import AudioDispatcher
from flowlens.audio.normalize import SoxrAudioNormalizer
from flowlens.audio.ports import CaptureCallback
from flowlens.audio.types import AudioFrame, CaptureDevice, RawAudioChunk
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    TranscriptRecord,
)

_SESSION_STARTED_MS = 10_000
_SESSION_ID = "01J00000000000000000000000"
_NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)
_NATIVE_RATE_HZ = 16_000
_CHUNK_DURATION_MS_PATTERN = (7, 32, 61, 4)
_SAMPLE_VALUES = {AudioSource.ME: 1_000, AudioSource.OTHERS: 2_000}


class _FenceQueue(Protocol):
    def put(
        self,
        item: AudioDrainFence,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...


class PerSourceDecoder:
    """Return one literal Japanese hypothesis for each source-coded input."""

    def __init__(self, me_text: str, others_text: str) -> None:
        self._text = {AudioSource.ME: me_text, AudioSource.OTHERS: others_text}

    def decode(self, pcm_s16le: bytes) -> DecodeHypothesis:
        """Identify the source marker and cover every supplied frame."""

        middle_offset = len(pcm_s16le) // 4 * 2
        source_sample = int.from_bytes(
            pcm_s16le[middle_offset : middle_offset + 2],
            "little",
            signed=True,
        )
        source = next(
            source
            for source, sample_value in _SAMPLE_VALUES.items()
            if sample_value == source_sample
        )
        duration_ms = len(pcm_s16le) // 2 * 1_000 // _NATIVE_RATE_HZ
        return DecodeHypothesis((DecodedToken(self._text[source], 0, duration_ms),))


class _AlwaysSpeechDetector:
    """Classify every canonical frame as speech without hardware."""

    def is_speech(self, frame: AudioFrame) -> bool:
        del frame
        return True


class _FakeCaptureStream:
    """Deliver native chunks through the public capture callback contract."""

    def __init__(self, callback: CaptureCallback) -> None:
        self._callback = callback
        self._active = False

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def close(self) -> None:
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def emit(self, chunk: RawAudioChunk) -> None:
        if not self._active:
            raise RuntimeError("capture stream is not active")
        self._callback(chunk)


class _FakeCaptureBackend:
    """Expose exact mono/stereo devices and controllable fake streams."""

    def __init__(self, others_rate_hz: int) -> None:
        self._devices = {
            AudioSource.ME: CaptureDevice(
                "fake-microphone", "Fake Microphone", 1, _NATIVE_RATE_HZ, 1, False
            ),
            AudioSource.OTHERS: CaptureDevice(
                "fake-loopback", "Fake Loopback", 2, others_rate_hz, 2, True
            ),
        }
        self._streams: dict[AudioSource, _FakeCaptureStream] = {}

    def list_microphones(self) -> tuple[CaptureDevice, ...]:
        return (self._devices[AudioSource.ME],)

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
        return (self._devices[AudioSource.OTHERS],)

    def open_stream(
        self,
        source: AudioSource,
        device_id: str,
        callback: CaptureCallback,
    ) -> _FakeCaptureStream:
        device = self._devices[source]
        if device_id != device.device_id:
            raise RuntimeError(f"capture device is unavailable: {device_id}")
        stream = _FakeCaptureStream(callback)
        self._streams[source] = stream
        return stream

    def emit(self, chunk: RawAudioChunk) -> None:
        self._streams[chunk.source].emit(chunk)

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()


class HardwareFreePipeline:
    """Connect production Audio stages to ASR through bounded real queues."""

    def __init__(
        self,
        decoder: PerSourceDecoder,
        *,
        others_rate_hz: int = _NATIVE_RATE_HZ,
        others_stereo_samples: tuple[int, int] = (2_000, 2_000),
    ) -> None:
        self._writer_queue: queue.Queue[AudioWriteCommand | AudioDrainFence] = (
            queue.Queue(maxsize=128)
        )
        self._asr_queue: queue.Queue[AudioFrame | AudioDrainFence] = queue.Queue(
            maxsize=1
        )
        self._dispatcher = AudioDispatcher(
            self._writer_queue, self._asr_queue, asr_spool_max_frames=256
        )
        self._engine = AsrEngine(
            AsrWorkerConfig(
                session_id=_SESSION_ID, model_path=Path("C:/models/kotoba")
            ),
            decoder,
            _AlwaysSpeechDetector(),
            segment_id_factory=self._next_segment_id,
            now=lambda: _NOW,
        )
        self._normalizers = {
            AudioSource.ME: SoxrAudioNormalizer(
                AudioSource.ME, _NATIVE_RATE_HZ, 1, _SESSION_STARTED_MS
            ),
            AudioSource.OTHERS: SoxrAudioNormalizer(
                AudioSource.OTHERS, others_rate_hz, 2, _SESSION_STARTED_MS
            ),
        }
        self._native_rate_hz = {
            AudioSource.ME: _NATIVE_RATE_HZ,
            AudioSource.OTHERS: others_rate_hz,
        }
        self._native_samples = {
            AudioSource.ME: (_SAMPLE_VALUES[AudioSource.ME],),
            AudioSource.OTHERS: others_stereo_samples,
        }
        self._backend = _FakeCaptureBackend(others_rate_hz)
        self._streams = {
            source: self._backend.open_stream(
                source,
                "fake-microphone" if source is AudioSource.ME else "fake-loopback",
                self._accept_native_chunk,
            )
            for source in AudioSource
        }
        for stream in self._streams.values():
            stream.start()

        self._pump_stop = threading.Event()
        self._consumer_gate = threading.Event()
        self._closing = threading.Event()
        self._asr_condition = threading.Condition()
        self._shutdown_asr_on_fence = False
        self._fence_count = 0
        self._segment_sequence = 0
        self._consumer_errors: list[BaseException] = []
        self.writer_items: list[AudioWriteCommand | AudioDrainFence] = []
        self.asr_items: list[AudioFrame | AudioDrainFence] = []
        self.records: tuple[TranscriptRecord, ...] = ()
        self.flushed_frames: list[AudioFrame] = []
        self.max_pending_asr_frames = 0
        self._pump = threading.Thread(
            target=self._dispatcher.run_asr_pump,
            args=(self._pump_stop,),
            name="test-audio-asr-pump",
        )
        self._writer_consumer = threading.Thread(
            target=self._consume_writer, name="test-writer-consumer"
        )
        self._asr_consumer = threading.Thread(
            target=self._consume_asr, name="test-asr-consumer"
        )
        self._pump.start()
        self._writer_consumer.start()
        self._asr_consumer.start()
        self._stopped = False

    def emit_native_mono(
        self, source: AudioSource, *, start_ms: int, duration_ms: int
    ) -> None:
        if source is not AudioSource.ME:
            raise ValueError("mono fake input is reserved for ME")
        self._emit_native(source, start_ms, duration_ms, channels=1)

    def emit_native_stereo(
        self, source: AudioSource, *, start_ms: int, duration_ms: int
    ) -> None:
        if source is not AudioSource.OTHERS:
            raise ValueError("stereo fake input is reserved for OTHERS")
        self._emit_native(source, start_ms, duration_ms, channels=2)

    def pause(self) -> None:
        """Put an ordered ASR fence after all pre-pause canonical frames."""

        self._consumer_gate.set()
        self._submit_fence()

    def stop_and_finalize(self) -> None:
        """Flush normalizers, traverse the stop fence, and finalize ASR once."""

        if self._stopped:
            return
        for source in AudioSource:
            flushed = self._normalizers[source].flush()
            self.flushed_frames.extend(flushed)
            for frame in flushed:
                self._dispatcher.dispatch(frame)
        self._consumer_gate.set()
        with self._asr_condition:
            self._shutdown_asr_on_fence = True
        self._submit_fence()
        self._pump_stop.set()
        self._dispatcher.wake_asr_pump()
        self._pump.join(timeout=2)
        self._writer_queue.put(AudioDrainFence(), timeout=1)
        self._writer_consumer.join(timeout=2)
        self._asr_consumer.join(timeout=2)
        self.records = self._engine.finalize(20_000).committed
        self._backend.close()
        self._raise_consumer_errors()
        if self.threads_alive:
            raise RuntimeError("pipeline threads did not terminate")
        if self.residual_item_count:
            raise RuntimeError("pipeline queues did not drain")
        self._stopped = True

    def close(self) -> None:
        """Release every test thread even if a test assertion interrupts use."""

        if self._stopped:
            return
        self._closing.set()
        self._consumer_gate.set()
        self._dispatcher.abort_asr_pump()
        self._pump_stop.set()
        self._dispatcher.wake_asr_pump()
        self._pump.join(timeout=2)
        self._put_cleanup_fence(self._writer_queue)
        self._put_cleanup_fence(self._asr_queue)
        self._writer_consumer.join(timeout=2)
        self._asr_consumer.join(timeout=2)
        self._backend.close()
        self._stopped = True

    @property
    def writer_commands(self) -> tuple[AudioWriteCommand, ...]:
        return tuple(
            item for item in self.writer_items if isinstance(item, AudioWriteCommand)
        )

    @property
    def asr_frames(self) -> tuple[AudioFrame, ...]:
        return tuple(item for item in self.asr_items if isinstance(item, AudioFrame))

    @property
    def threads_alive(self) -> bool:
        return any(
            thread.is_alive()
            for thread in (self._pump, self._writer_consumer, self._asr_consumer)
        )

    @property
    def residual_item_count(self) -> int:
        return (
            self._writer_queue.qsize()
            + self._asr_queue.qsize()
            + self._dispatcher.pending_asr_frames
        )

    def writer_pcm_duration_ms(self, source: AudioSource) -> int:
        sample_count = sum(
            len(command.pcm_s16le) // 2
            for command in self.writer_commands
            if command.source is source
        )
        return sample_count * 1_000 // _NATIVE_RATE_HZ

    def _emit_native(
        self,
        source: AudioSource,
        start_ms: int,
        duration_ms: int,
        *,
        channels: int,
    ) -> None:
        input_rate_hz = self._native_rate_hz[source]
        emitted_ms = 0
        chunk_index = 0
        source_samples = self._native_samples[source]
        if len(source_samples) != channels:
            raise RuntimeError("fake native channel shape does not match")
        native_frame = b"".join(
            sample.to_bytes(2, "little", signed=True) for sample in source_samples
        )
        while emitted_ms < duration_ms:
            chunk_duration_ms = min(
                _CHUNK_DURATION_MS_PATTERN[
                    chunk_index % len(_CHUNK_DURATION_MS_PATTERN)
                ],
                duration_ms - emitted_ms,
            )
            chunk_samples = chunk_duration_ms * input_rate_hz // 1_000
            self._backend.emit(
                RawAudioChunk(
                    source=source,
                    pcm_s16le_interleaved=native_frame * chunk_samples,
                    sample_rate_hz=input_rate_hz,
                    channels=channels,
                    captured_monotonic_ms=(_SESSION_STARTED_MS + start_ms + emitted_ms),
                )
            )
            emitted_ms += chunk_duration_ms
            chunk_index += 1
        self.max_pending_asr_frames = max(
            self.max_pending_asr_frames, self._dispatcher.pending_asr_frames
        )

    def _accept_native_chunk(self, chunk: RawAudioChunk) -> None:
        for frame in self._normalizers[chunk.source].push(chunk):
            self._dispatcher.dispatch(frame)

    def _consume_writer(self) -> None:
        try:
            while True:
                try:
                    item = self._writer_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closing.is_set():
                        return
                    continue
                self.writer_items.append(item)
                if isinstance(item, AudioDrainFence):
                    return
        except BaseException as exc:
            self._consumer_errors.append(exc)

    def _consume_asr(self) -> None:
        try:
            self._consumer_gate.wait()
            while True:
                try:
                    item = self._asr_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closing.is_set():
                        return
                    continue
                self.asr_items.append(item)
                if isinstance(item, AudioFrame):
                    self._engine.accept(item)
                    continue
                with self._asr_condition:
                    self._fence_count += 1
                    should_stop = self._shutdown_asr_on_fence or self._closing.is_set()
                    self._asr_condition.notify_all()
                if should_stop:
                    return
        except BaseException as exc:
            self._consumer_errors.append(exc)
            with self._asr_condition:
                self._asr_condition.notify_all()

    def _submit_fence(self) -> None:
        with self._asr_condition:
            expected_count = self._fence_count + 1
        if not self._dispatcher.enqueue_asr_fence(timeout=2):
            raise RuntimeError("ASR fence did not leave the dispatcher spool")
        with self._asr_condition:
            consumed = self._asr_condition.wait_for(
                lambda: self._fence_count >= expected_count
                or bool(self._consumer_errors),
                timeout=2,
            )
        if not consumed:
            raise RuntimeError("ASR consumer did not observe the ordered fence")
        self._raise_consumer_errors()

    def _next_segment_id(self) -> str:
        self._segment_sequence += 1
        return f"01J{'0' * 22}{self._segment_sequence}"

    def _raise_consumer_errors(self) -> None:
        if self._consumer_errors:
            raise RuntimeError("pipeline consumer failed") from self._consumer_errors[0]

    @staticmethod
    def _put_cleanup_fence(target: _FenceQueue) -> None:
        try:
            target.put(AudioDrainFence(), timeout=1)
        except queue.Full:
            pass


class _PipelineFactory(Protocol):
    def __call__(
        self,
        decoder: PerSourceDecoder,
        *,
        others_rate_hz: int = _NATIVE_RATE_HZ,
        others_stereo_samples: tuple[int, int] = (2_000, 2_000),
    ) -> HardwareFreePipeline: ...


@pytest.fixture
def pipeline_factory() -> Iterator[_PipelineFactory]:
    """Guarantee deterministic test-thread cleanup on every assertion path."""

    pipelines: list[HardwareFreePipeline] = []

    def create(
        decoder: PerSourceDecoder,
        *,
        others_rate_hz: int = _NATIVE_RATE_HZ,
        others_stereo_samples: tuple[int, int] = (2_000, 2_000),
    ) -> HardwareFreePipeline:
        pipeline = HardwareFreePipeline(
            decoder,
            others_rate_hz=others_rate_hz,
            others_stereo_samples=others_stereo_samples,
        )
        pipelines.append(pipeline)
        return pipeline

    yield create
    for pipeline in reversed(pipelines):
        pipeline.close()


@pytest.mark.parametrize(
    "insertion_order",
    [
        (AudioSource.OTHERS, AudioSource.ME),
        (AudioSource.ME, AudioSource.OTHERS),
    ],
)
def test_overlapping_me_and_others_frames_preserve_audio_and_transcript_order(
    pipeline_factory: _PipelineFactory,
    insertion_order: tuple[AudioSource, AudioSource],
) -> None:
    pipeline = pipeline_factory(
        PerSourceDecoder(me_text="確認します。", others_text="承知しました。")
    )
    emitters: dict[AudioSource, Callable[[], None]] = {
        AudioSource.ME: lambda: pipeline.emit_native_mono(
            AudioSource.ME, start_ms=100, duration_ms=700
        ),
        AudioSource.OTHERS: lambda: pipeline.emit_native_stereo(
            AudioSource.OTHERS, start_ms=120, duration_ms=700
        ),
    }
    for source in insertion_order:
        emitters[source]()
    pipeline.stop_and_finalize()

    assert pipeline.writer_pcm_duration_ms(AudioSource.ME) == 700
    assert pipeline.writer_pcm_duration_ms(AudioSource.OTHERS) == 700
    assert [
        (
            record.source.value,
            record.text,
            record.sequence,
            record.session_start_ms,
            record.session_end_ms,
            record.source_start_sample,
            record.source_end_sample,
        )
        for record in pipeline.records
    ] == [
        ("ME", "確認します。", 1, 100, 800, 0, 11_200),
        ("OTHERS", "承知しました。", 2, 120, 820, 0, 11_200),
    ]


def test_partial_native_chunks_preserve_identical_writer_and_asr_frames(
    pipeline_factory: _PipelineFactory,
) -> None:
    pipeline = pipeline_factory(PerSourceDecoder("自分", "相手"))
    pipeline.emit_native_mono(AudioSource.ME, start_ms=0, duration_ms=700)
    pipeline.emit_native_stereo(AudioSource.OTHERS, start_ms=0, duration_ms=700)
    assert pipeline.max_pending_asr_frames > 0

    pipeline.stop_and_finalize()

    writer_projection = [
        (
            item.source,
            item.source_start_sample,
            item.source_end_sample,
            item.session_start_ms,
            item.pcm_s16le,
        )
        for item in pipeline.writer_commands
    ]
    asr_projection = [
        (
            item.source,
            item.source_start_sample,
            item.source_end_sample,
            item.session_start_ms,
            item.pcm_s16le,
        )
        for item in pipeline.asr_frames
    ]
    assert writer_projection == asr_projection
    assert len(pipeline.writer_commands) == 70
    assert len(pipeline.asr_frames) == 70
    assert pipeline.asr_items[-1] == AudioDrainFence()
    assert pipeline.writer_items[-1] == AudioDrainFence()
    assert pipeline.residual_item_count == 0
    assert not pipeline.threads_alive


def test_pause_fence_precedes_resumed_audio_and_preserves_session_gap(
    pipeline_factory: _PipelineFactory,
) -> None:
    pipeline = pipeline_factory(PerSourceDecoder("再開します。", "了解です。"))
    pipeline.emit_native_mono(AudioSource.ME, start_ms=100, duration_ms=340)
    pipeline.emit_native_stereo(AudioSource.OTHERS, start_ms=120, duration_ms=340)
    pipeline.pause()
    pipeline.emit_native_mono(AudioSource.ME, start_ms=2_440, duration_ms=360)
    pipeline.emit_native_stereo(AudioSource.OTHERS, start_ms=2_460, duration_ms=360)
    pipeline.stop_and_finalize()

    fence_indexes = [
        index
        for index, item in enumerate(pipeline.asr_items)
        if isinstance(item, AudioDrainFence)
    ]
    assert fence_indexes == [34, 71]
    before_pause = pipeline.asr_items[: fence_indexes[0]]
    after_pause = pipeline.asr_items[fence_indexes[0] + 1 : fence_indexes[1]]
    assert all(
        isinstance(item, AudioFrame) and item.source_end_sample <= 5_440
        for item in before_pause
    )
    assert all(
        isinstance(item, AudioFrame) and item.source_start_sample >= 5_440
        for item in after_pause
    )
    assert [
        (
            record.source,
            record.sequence,
            record.session_start_ms,
            record.session_end_ms,
            record.source_start_sample,
            record.source_end_sample,
        )
        for record in pipeline.records
    ] == [
        (AudioSource.ME, 1, 100, 2_800, 0, 11_200),
        (AudioSource.OTHERS, 2, 120, 2_820, 0, 11_200),
    ]
    assert pipeline.writer_pcm_duration_ms(AudioSource.ME) == 700
    assert pipeline.writer_pcm_duration_ms(AudioSource.OTHERS) == 700
    assert pipeline.residual_item_count == 0
    assert not pipeline.threads_alive


def test_resampler_flush_frame_reaches_writer_and_asr_identically(
    pipeline_factory: _PipelineFactory,
) -> None:
    pipeline = pipeline_factory(
        PerSourceDecoder("未使用", "48 kHz 確認"),
        others_rate_hz=48_000,
        others_stereo_samples=(1_000, 3_000),
    )
    pipeline.emit_native_stereo(AudioSource.OTHERS, start_ms=120, duration_ms=40)
    pipeline.stop_and_finalize()

    assert len(pipeline.flushed_frames) == 1
    flushed = pipeline.flushed_frames[0]
    assert (
        flushed.source,
        flushed.source_start_sample,
        flushed.source_end_sample,
        flushed.session_start_ms,
        flushed.captured_monotonic_ms,
    ) == (AudioSource.OTHERS, 320, 640, 140, 10_140)
    assert len(pipeline.writer_commands) == 2
    assert len(pipeline.asr_frames) == 2
    writer_flush = pipeline.writer_commands[-1]
    asr_flush = pipeline.asr_frames[-1]
    assert (
        writer_flush.source,
        writer_flush.source_start_sample,
        writer_flush.source_end_sample,
        writer_flush.session_start_ms,
        writer_flush.captured_monotonic_ms,
        writer_flush.pcm_s16le,
    ) == (
        flushed.source,
        flushed.source_start_sample,
        flushed.source_end_sample,
        flushed.session_start_ms,
        flushed.captured_monotonic_ms,
        flushed.pcm_s16le,
    )
    assert asr_flush == flushed
    samples = [
        int.from_bytes(flushed.pcm_s16le[index : index + 2], "little", signed=True)
        for index in range(0, len(flushed.pcm_s16le), 2)
    ]
    assert sorted(samples)[len(samples) // 2] == 2_000
    assert pipeline.writer_pcm_duration_ms(AudioSource.OTHERS) == 40
    assert [
        (
            record.source,
            record.text,
            record.sequence,
            record.session_start_ms,
            record.session_end_ms,
            record.source_start_sample,
            record.source_end_sample,
        )
        for record in pipeline.records
    ] == [(AudioSource.OTHERS, "48 kHz 確認", 1, 120, 160, 0, 640)]
