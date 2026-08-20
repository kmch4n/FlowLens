# Audio Capture and Live Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the selected microphone and WASAPI loopback concurrently, preserve canonical audio for Writer and ASR, and produce immutable Japanese partial/committed transcripts with the fixed Kotoba-Whisper model.

**Architecture:** Keep hardware and CUDA behind ports. The Audio Worker opens two PyAudioWPatch streams, performs stateful mono/16 kHz conversion outside callbacks, emits identical 20 ms frames to independent Writer and ASR paths, and treats Writer pressure as fatal. The ASR Worker owns one Kotoba-Whisper adapter and two source states, schedules the oldest pending speech first, applies deterministic VAD/partial/commit rules, and emits shared domain records through the IPC envelope.

**Tech Stack:** Python 3.12, `multiprocessing.Queue`, PyAudioWPatch/WASAPI, NumPy, python-soxr streaming resampling, WebRTC VAD, faster-whisper/CTranslate2 CUDA, pytest, Black, Ruff, mypy.

**Spec:** `docs/mvp-spec.md` sections 13, 15-19, 21, 24-27.

## Global Constraints

- Target Windows 10 Pro 64-bit and the designated NVIDIA GeForce RTX 4060 PC; no cross-platform capture backend is required.
- Runtime is Python 3.12 and must make no network request after model installation.
- Use `kotoba-tech/kotoba-whisper-v2.0-faster` from a validated local path with CUDA, `float16`, Japanese transcription, beam size 1, temperature 0, and `condition_on_previous_text=False`.
- `ME` is the selected microphone; `OTHERS` is the selected output device's WASAPI loopback. Never silently substitute another device.
- Canonical audio is mono, signed 16-bit little-endian PCM, 16,000 Hz, in 20 ms / 320-sample / 640-byte frames.
- Audio persistence has priority over ASR; committed transcription has priority over partial transcription; discussion work must not block either path.
- Pause stops capture and dispatch and inserts no synthetic silence. Source sample offsets remain contiguous across pauses; session monotonic timestamps retain the pause gap.
- Writer queue exhaustion is fatal. ASR backlog is reported and must not remove or delay already-dispatched Writer audio.
- Partial transcripts are ephemeral. Committed text is immutable and uses the exact `TranscriptRecord` schema from the foundation plan.
- Keep production files below approximately 700 lines, use four spaces, double quotes, complete type annotations, and Google-style docstrings.
- The foundation plan owns and pins runtime dependencies in `requirements.txt`; this plan consumes PyAudioWPatch, NumPy, python-soxr, WebRTC VAD, faster-whisper, pytest, Black, Ruff, and mypy without changing dependency files.
- Do not perform Git staging, commits, or pushes while executing this plan unless the user gives a separate explicit instruction.

## Shared Interfaces Consumed From the Foundation Plan

The executor must read the foundation plan and `docs/mvp-spec.md` before Task 1. Import these definitions; do not duplicate them:

```python
from flowlens.domain.enums import AudioSource, MessageType, WorkerName
from flowlens.domain.ids import new_ulid
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    MessageEnvelope,
    TranscriptRecord,
)
```

Exact constructors used by this plan:

```python
AudioWriteCommand(
    source: AudioSource,
    pcm_s16le: bytes,
    source_start_sample: int,
    source_end_sample: int,
    session_start_ms: int,
    captured_monotonic_ms: int,
)

TranscriptRecord(
    schema_version: int,
    segment_id: str,
    sequence: int,
    source: AudioSource,
    text: str,
    session_start_ms: int,
    session_end_ms: int,
    source_start_sample: int,
    source_end_sample: int,
    committed_at: datetime,
)
```

Every control/status `MessageEnvelope` has the exact section 21 fields: `schema_version`, `session_id`, `message_type`, sender-local `sequence`, `source`, `created_monotonic_ms`, and a JSON-serializable `payload` dictionary. The envelope sequence is not the transcript `TranscriptRecord.sequence`.

`AudioDrainFence` is the foundation-owned, empty immutable marker for the
dedicated Audio-to-Writer queue. It is not a general `MessageEnvelope`, never
enters a control queue, and is not sent to the ASR audio queue unless a separate
future contract explicitly introduces an ASR fence.

## File Map

- `src/flowlens/audio/types.py`: capture devices, raw chunks, canonical `AudioFrame`, and `AudioWorkerConfig`.
- `src/flowlens/audio/ports.py`: capture stream/backend and streaming normalizer protocols.
- `src/flowlens/audio/normalize.py`: stereo downmix, stateful SoXR conversion, 20 ms framing, and sample/timestamp accounting.
- `src/flowlens/audio/pyaudiowpatch_backend.py`: PyAudioWPatch device enumeration, WASAPI loopback resolution, and minimal callbacks.
- `src/flowlens/audio/dispatch.py`: independent Writer/ASR fan-out, backlog measurement, and fatal queue policy.
- `src/flowlens/audio/worker.py`: command loop, two-source lifecycle, reconnection, pause/resume, and drain.
- `src/flowlens/asr/types.py`: decoded token/hypothesis, partial/commit candidates, and `AsrWorkerConfig`.
- `src/flowlens/asr/ports.py`: speech detector, clock, and decoder protocols.
- `src/flowlens/asr/vad.py`: 20 ms WebRTC VAD adapter and utterance boundary state.
- `src/flowlens/asr/kotoba_whisper.py`: offline-only faster-whisper adapter.
- `src/flowlens/asr/commit.py`: stable-prefix, text filtering, split boundary, chronological buffering, and sequence assignment.
- `src/flowlens/asr/engine.py`: per-source state and oldest-pending decode scheduler.
- `src/flowlens/asr/worker.py`: ASR process entrypoint and IPC emission.
- `tests/audio/`: deterministic unit/contract tests with fake devices and queues.
- `tests/asr/`: deterministic unit/contract tests with fake VAD, clock, and decoder.
- `tests/integration/test_audio_asr_pipeline.py`: hardware-free end-to-end frame-to-record test.
- `src/flowlens/smoke/audio.py`: designated-PC dual-source capture/WAV smoke entrypoint.
- `src/flowlens/smoke/asr.py`: designated-PC fixed-model transcription smoke entrypoint.
- `scripts/smoke_audio.ps1`: designated-PC one-minute dual-source/WAV inspection.
- `scripts/smoke_asr.ps1`: designated-PC two-minute fixed-model/overlap inspection.

---

### Task 1: Define capture-local types and ports

**Files:**
- Create: `src/flowlens/audio/__init__.py`
- Create: `src/flowlens/audio/types.py`
- Create: `src/flowlens/audio/ports.py`
- Create: `tests/audio/test_types.py`

**Interfaces:**
- Consumes: `flowlens.domain.enums.AudioSource`.
- Produces: `CaptureDevice`, `RawAudioChunk`, `AudioFrame`, `AudioWorkerConfig`, `CaptureCallback`, `CaptureStreamPort`, `CaptureBackendPort`, and `StreamingNormalizerPort` with the signatures below.

- [ ] **Step 1: Write the failing frame-invariant test**

```python
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource


def test_audio_frame_requires_exactly_twenty_milliseconds() -> None:
    frame = AudioFrame(
        source=AudioSource.ME,
        pcm_s16le=bytes(640),
        source_start_sample=320,
        source_end_sample=640,
        session_start_ms=20,
        captured_monotonic_ms=1_020,
    )
    assert frame.duration_ms == 20
    assert frame.queue_age_ms(now_monotonic_ms=1_075) == 55


def test_audio_frame_rejects_noncanonical_payload() -> None:
    try:
        AudioFrame(
            source=AudioSource.OTHERS,
            pcm_s16le=bytes(638),
            source_start_sample=0,
            source_end_sample=319,
            session_start_ms=0,
            captured_monotonic_ms=1_000,
        )
    except ValueError as exc:
        assert str(exc) == "AudioFrame must contain 320 mono int16 samples"
    else:
        raise AssertionError("invalid frame was accepted")
```

- [ ] **Step 2: Run the test and confirm collection fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_types.py -v`

Expected: FAIL because `flowlens.audio.types` does not exist.

- [ ] **Step 3: Implement the exact dataclasses and validation**

```python
# src/flowlens/audio/types.py
from dataclasses import dataclass

from flowlens.domain.enums import AudioSource

CANONICAL_RATE_HZ = 16_000
FRAME_SAMPLES = 320
FRAME_DURATION_MS = 20
FRAME_BYTES = 640


@dataclass(frozen=True, slots=True)
class CaptureDevice:
    device_id: str
    display_name: str
    input_device_index: int
    sample_rate_hz: int
    channels: int
    is_loopback: bool


@dataclass(frozen=True, slots=True)
class RawAudioChunk:
    source: AudioSource
    pcm_s16le_interleaved: bytes
    sample_rate_hz: int
    channels: int
    captured_monotonic_ms: int


@dataclass(frozen=True, slots=True)
class AudioFrame:
    source: AudioSource
    pcm_s16le: bytes
    source_start_sample: int
    source_end_sample: int
    session_start_ms: int
    captured_monotonic_ms: int

    def __post_init__(self) -> None:
        if len(self.pcm_s16le) != FRAME_BYTES or (
            self.source_end_sample - self.source_start_sample != FRAME_SAMPLES
        ):
            raise ValueError("AudioFrame must contain 320 mono int16 samples")

    @property
    def duration_ms(self) -> int:
        return FRAME_DURATION_MS

    def queue_age_ms(self, now_monotonic_ms: int) -> int:
        return max(0, now_monotonic_ms - self.captured_monotonic_ms)


@dataclass(frozen=True, slots=True)
class AudioWorkerConfig:
    session_id: str
    microphone_device_id: str
    loopback_output_device_id: str
    session_started_monotonic_ms: int
    writer_queue_max_frames: int
    asr_queue_max_frames: int
    asr_spool_max_frames: int = 3_000
    reconnect_interval_ms: int = 2_000
```

```python
# src/flowlens/audio/ports.py
from collections.abc import Callable
from typing import Protocol

from flowlens.audio.types import AudioFrame, CaptureDevice, RawAudioChunk
from flowlens.domain.enums import AudioSource

CaptureCallback = Callable[[RawAudioChunk], None]


class CaptureStreamPort(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def is_active(self) -> bool: ...


class CaptureBackendPort(Protocol):
    def list_microphones(self) -> tuple[CaptureDevice, ...]: ...
    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]: ...
    def open_stream(
        self,
        source: AudioSource,
        device_id: str,
        callback: CaptureCallback,
    ) -> CaptureStreamPort: ...
    def close(self) -> None: ...


class StreamingNormalizerPort(Protocol):
    def push(self, chunk: RawAudioChunk) -> tuple[AudioFrame, ...]: ...
    def flush(self) -> tuple[AudioFrame, ...]: ...
```

- [ ] **Step 4: Run focused checks**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_types.py -v`

Expected: 2 passed.

Run: `.\.venv\Scripts\python.exe -m mypy src/flowlens/audio/types.py src/flowlens/audio/ports.py`

Expected: success with no issues.

---

### Task 2: Normalize native PCM into canonical frames with streaming state

**Files:**
- Create: `src/flowlens/audio/normalize.py`
- Create: `tests/audio/test_normalize.py`

**Interfaces:**
- Consumes: `RawAudioChunk`, `AudioFrame`, and `soxr.ResampleStream`.
- Produces: `SoxrAudioNormalizer(source, input_rate_hz, input_channels, session_started_monotonic_ms)` implementing `push()` and `flush()`.

- [ ] **Step 1: Write failing tests for downmix, framing, accounting, and pause gaps**

```python
import numpy as np

from flowlens.audio.normalize import SoxrAudioNormalizer
from flowlens.audio.types import RawAudioChunk
from flowlens.domain.enums import AudioSource


def _stereo_constant(left: int, right: int, frames: int) -> bytes:
    values = np.column_stack(
        (
            np.full(frames, left, dtype=np.int16),
            np.full(frames, right, dtype=np.int16),
        )
    )
    return values.astype("<i2", copy=False).tobytes()


def test_normalizer_downmixes_and_emits_exact_frames() -> None:
    normalizer = SoxrAudioNormalizer(
        source=AudioSource.OTHERS,
        input_rate_hz=48_000,
        input_channels=2,
        session_started_monotonic_ms=1_000,
    )
    output = normalizer.push(
        RawAudioChunk(
            source=AudioSource.OTHERS,
            pcm_s16le_interleaved=_stereo_constant(2_000, 6_000, 1_920),
            sample_rate_hz=48_000,
            channels=2,
            captured_monotonic_ms=1_100,
        )
    ) + normalizer.flush()
    assert len(output) == 2
    assert all(len(frame.pcm_s16le) == 640 for frame in output)
    assert output[0].source_start_sample == 0
    assert output[1].source_end_sample == 640
    assert output[0].session_start_ms == 100
    samples = np.frombuffer(output[1].pcm_s16le, dtype="<i2")
    assert 3_950 <= int(np.median(samples)) <= 4_050


def test_normalizer_keeps_sample_offsets_contiguous_across_capture_gap() -> None:
    normalizer = SoxrAudioNormalizer(
        source=AudioSource.ME,
        input_rate_hz=16_000,
        input_channels=1,
        session_started_monotonic_ms=1_000,
    )
    first = normalizer.push(
        RawAudioChunk(AudioSource.ME, bytes(640), 16_000, 1, 1_020)
    )
    second = normalizer.push(
        RawAudioChunk(AudioSource.ME, bytes(640), 16_000, 1, 3_020)
    )
    assert first[0].source_end_sample == second[0].source_start_sample
    assert second[0].session_start_ms == 2_020
```

- [ ] **Step 2: Run the tests and confirm the normalizer is absent**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_normalize.py -v`

Expected: FAIL importing `SoxrAudioNormalizer`.

- [ ] **Step 3: Implement stateful conversion, never per-chunk stateless resampling**

Implement `SoxrAudioNormalizer` with these exact private fields and methods:

```python
class SoxrAudioNormalizer:
    def __init__(
        self,
        source: AudioSource,
        input_rate_hz: int,
        input_channels: int,
        session_started_monotonic_ms: int,
    ) -> None: ...

    def push(self, chunk: RawAudioChunk) -> tuple[AudioFrame, ...]: ...
    def flush(self) -> tuple[AudioFrame, ...]: ...
    def _decode_mono_float32(self, chunk: RawAudioChunk) -> np.ndarray: ...
    def _append_resampled(
        self, mono: np.ndarray, captured_monotonic_ms: int, last: bool
    ) -> tuple[AudioFrame, ...]: ...
```

Use `soxr.ResampleStream(input_rate_hz, 16_000, 1, dtype="float32", quality="HQ")`. Decode little-endian int16 to `float32 / 32768.0`, reshape by channels, and average channels in `float32`. Keep resampler output plus input timeline spans in FIFOs; each span records its first-sample monotonic time and its duration-derived output-sample count. Consume those spans as 320-sample frames are emitted so consecutive frames from one chunk advance 20 ms, while a post-pause chunk retains its later monotonic time. Emit only complete frames. Clip to `[-1.0, 32767.0 / 32768.0]`, multiply by 32768, round, and encode as `<i2`. `flush()` calls `resample_chunk(empty_float32, last=True)` and deliberately discards a final fragment shorter than 320 samples; it must not pad silence.

When `input_rate_hz == 16_000`, bypass SoXR but use the same FIFO/framer; this keeps native-rate live output immediate and preserves identical accounting.

For every emitted frame:

```python
AudioFrame(
    source=self._source,
    pcm_s16le=pcm,
    source_start_sample=self._next_source_sample,
    source_end_sample=self._next_source_sample + 320,
    session_start_ms=first_input_monotonic_ms - self._session_started_monotonic_ms,
    captured_monotonic_ms=first_input_monotonic_ms,
)
```

Advance `_next_source_sample` only when a frame is emitted. Do not derive `session_start_ms` from source samples because pauses must remain visible in the session clock.

- [ ] **Step 4: Run focused and numerical tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_normalize.py -v`

Expected: 2 passed.

Run: `.\.venv\Scripts\python.exe -m ruff check src/flowlens/audio/normalize.py tests/audio/test_normalize.py`

Expected: all checks passed.

---

### Task 3: Implement the PyAudioWPatch/WASAPI adapter behind a fakeable seam

**Files:**
- Create: `src/flowlens/audio/pyaudiowpatch_backend.py`
- Create: `tests/audio/test_pyaudiowpatch_backend.py`

**Interfaces:**
- Consumes: `CaptureBackendPort`, `CaptureDevice`, and injectable `PyAudioFactory = Callable[[], PyAudioApi]`.
- Produces: `PyAudioWPatchBackend(py_audio_factory, monotonic_ms)` and `PyAudioCaptureStream`.

- [ ] **Step 1: Write fake-API tests for selected device identity and loopback resolution**

```python
from flowlens.audio.pyaudiowpatch_backend import PyAudioWPatchBackend
from flowlens.domain.enums import AudioSource


class FakeStream:
    def start_stream(self) -> None: pass
    def stop_stream(self) -> None: pass
    def close(self) -> None: pass
    def is_active(self) -> bool: return True


class FakePyAudio:
    def __init__(self) -> None:
        self.open_kwargs: dict[str, object] = {}
        self.devices = (
            {"index": 3, "name": "USB Mic", "maxInputChannels": 1,
             "defaultSampleRate": 48_000.0, "isLoopbackDevice": False},
            {"index": 7, "name": "Speakers", "maxInputChannels": 0,
             "defaultSampleRate": 48_000.0, "isLoopbackDevice": False},
        )
        self.loopback = {"index": 11, "name": "Speakers [Loopback]",
                         "maxInputChannels": 2, "defaultSampleRate": 48_000.0,
                         "isLoopbackDevice": True}

    def get_device_info_generator(self): return iter(self.devices)
    def get_wasapi_loopback_analogue_by_index(self, index: int):
        assert index == 7
        return self.loopback
    def open(self, **kwargs):
        self.open_kwargs = kwargs
        return FakeStream()
    def terminate(self) -> None: pass


def test_loopback_selection_opens_analogue_not_output_device() -> None:
    api = FakePyAudio()
    backend = PyAudioWPatchBackend(lambda: api, monotonic_ms=lambda: 9_000)
    devices = backend.list_loopback_outputs()
    assert devices[0].device_id == "wasapi-output:7"
    assert devices[0].input_device_index == 11
    chunks = []
    backend.open_stream(AudioSource.OTHERS, "wasapi-output:7", chunks.append)
    assert api.open_kwargs["input_device_index"] == 11
    assert api.open_kwargs["format"] == backend.pa_int16
    assert api.open_kwargs["stream_callback"] is not None
```

- [ ] **Step 2: Run the adapter test and confirm it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_pyaudiowpatch_backend.py -v`

Expected: FAIL importing `PyAudioWPatchBackend`.

- [ ] **Step 3: Implement enumeration and callback behavior**

Define a private `PyAudioApi` protocol containing only the methods exercised above plus `get_wasapi_loopback_analogue_by_index`. Device IDs are stable within the designated machine configuration:

- microphone: `input:<device index>`
- selected output: `wasapi-output:<output device index>`

`list_microphones()` includes non-loopback devices with `maxInputChannels > 0`. `list_loopback_outputs()` starts from non-loopback output devices with `maxInputChannels == 0`, resolves each by `get_wasapi_loopback_analogue_by_index`, and returns the output display name while storing the loopback input index. A missing analogue omits that output from the selectable list; `open_stream()` with an unknown exact ID raises `DeviceUnavailableError(device_id)` and never falls back.

Open with native `defaultSampleRate`, native input channel count, `paInt16`, `input=True`, `start=False`, `frames_per_buffer=round(device.sample_rate_hz * 0.020)`, and a callback that performs only:

```python
callback(
    RawAudioChunk(
        source=source,
        pcm_s16le_interleaved=in_data,
        sample_rate_hz=device.sample_rate_hz,
        channels=device.channels,
        captured_monotonic_ms=(
            monotonic_ms()
            - round(frame_count * 1_000 / device.sample_rate_hz)
        ),
    )
)
return (None, pa_continue)
```

No NumPy conversion, resampling, queue wait, logging, or IPC operation is allowed inside the PortAudio callback.

- [ ] **Step 4: Run fake-adapter tests and type checks**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_pyaudiowpatch_backend.py -v`

Expected: tests pass without installed audio hardware.

Run: `.\.venv\Scripts\python.exe -m mypy src/flowlens/audio/pyaudiowpatch_backend.py`

Expected: success with no issues.

---

### Task 4: Fan canonical frames into independent Writer and ASR paths

**Files:**
- Create: `src/flowlens/audio/dispatch.py`
- Create: `tests/audio/test_dispatch.py`

**Interfaces:**
- Consumes: `AudioFrame`, shared `AudioWriteCommand`, and queue-like objects exposing `put_nowait()`.
- Produces: `AudioDispatcher.dispatch(frame)`, `AudioDispatcher.asr_backlog_ms(now_monotonic_ms)`, `WriterQueueFull`, and `AsrSpoolFull`.

- [ ] **Step 1: Write failing priority and identity tests**

```python
from queue import Queue
from typing import cast

from flowlens.audio.dispatch import AudioDispatcher, WriterQueueFull
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import AudioDrainFence, AudioWriteCommand


def make_frame() -> AudioFrame:
    return AudioFrame(AudioSource.ME, bytes(640), 0, 320, 100, 1_100)


def test_dispatches_writer_command_before_asr_frame() -> None:
    writer: Queue[AudioWriteCommand | AudioDrainFence] = Queue(maxsize=1)
    asr: Queue[AudioFrame] = Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=3_000)
    dispatcher.dispatch(make_frame())
    command = cast(AudioWriteCommand, writer.get_nowait())
    assert command.pcm_s16le == asr.get_nowait().pcm_s16le
    assert command.source_start_sample == 0


def test_full_writer_queue_is_fatal_before_asr_dispatch() -> None:
    writer: Queue[AudioWriteCommand | AudioDrainFence] = Queue(maxsize=1)
    writer.put_nowait(AudioWriteCommand(AudioSource.ME, bytes(640), 0, 320, 100, 1_100))
    asr: Queue[AudioFrame] = Queue(maxsize=1)
    dispatcher = AudioDispatcher(writer, asr, asr_spool_max_frames=3_000)
    try:
        dispatcher.dispatch(make_frame())
    except WriterQueueFull:
        pass
    else:
        raise AssertionError("writer exhaustion was not fatal")
    assert asr.empty()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_dispatch.py -v`

Expected: FAIL importing `AudioDispatcher`.

- [ ] **Step 3: Implement priority-safe dispatch**

`AudioDispatcher.dispatch()` first constructs the exact foundation `AudioWriteCommand` from the frame and calls `writer_audio_out.put_nowait(command)`. Convert `queue.Full` into `WriterQueueFull`. Only after Writer accepts the command, append the same immutable `AudioFrame` to an internal ASR `deque` protected by a `Condition`.

A dedicated `run_asr_pump(stop_event)` method waits on that deque and performs blocking `asr_audio_out.put(frame, timeout=0.1)`. This prevents a full ASR process queue from delaying Writer dispatch. Cap the internal ASR spool at `asr_spool_max_frames=3_000` (60 seconds across both sources). If the cap is reached, raise `AsrSpoolFull`; the worker safely stops instead of discarding transcript audio. `asr_backlog_ms()` reports `max(0, now - oldest_frame.captured_monotonic_ms)` across the spool and multiprocessing queue's last submitted frame.

- [ ] **Step 4: Run dispatch tests including a blocked ASR consumer**

Add a test whose ASR queue is full, start `run_asr_pump()` in a thread, dispatch ten Writer frames, and assert all ten Writer commands arrive before releasing the ASR queue. Then stop and join the pump.

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_dispatch.py -v`

Expected: all tests pass and no dispatch thread remains alive.

---

### Task 5: Build the two-source Audio Worker lifecycle and reconnection loop

**Files:**
- Create: `src/flowlens/audio/worker.py`
- Create: `tests/audio/test_worker.py`

**Interfaces:**
- Consumes: `MessageEnvelope`, `AudioWriteCommand`, shared `AudioDrainFence`, `AudioFrame`, `CaptureBackendPort`, `SoxrAudioNormalizer`, and `AudioDispatcher`.
- Produces:

```python
def run_audio_worker(
    config: AudioWorkerConfig,
    control_in: "multiprocessing.Queue[MessageEnvelope]",
    control_out: "multiprocessing.Queue[MessageEnvelope]",
    writer_audio_out: "multiprocessing.Queue[AudioWriteCommand | AudioDrainFence]",
    asr_audio_out: "multiprocessing.Queue[AudioFrame]",
) -> None: ...
```

Controller commands use shared message types `WORKER_START`, `WORKER_PAUSE`, `WORKER_RESUME`, and `WORKER_STOP`, each with payload `{"worker": "AUDIO"}`. Outputs and payloads are:

```text
WORKER_READY         {"worker": "AUDIO"}
AUDIO_LEVEL          {"source": "ME"|"OTHERS", "peak_dbfs": float}
SOURCE_DISCONNECTED  {"source": "ME"|"OTHERS", "device_id": str}
SOURCE_RECONNECTED   {"source": "ME"|"OTHERS", "device_id": str}
WORKER_STOPPED       {"worker": "AUDIO", "drained": true, "writer_frames": int, "asr_frames": int}
WORKER_ERROR         {"worker": "AUDIO", "code": "CAPTURE_QUEUE_FULL"|"WRITER_QUEUE_FULL"|"ASR_QUEUE_STALLED"|"DEVICE_OPEN_FAILED", "detail": str}
```

- [ ] **Step 1: Write a failing lifecycle test with a fake backend**

```python
def test_pause_stops_streams_and_resume_preserves_source_offsets() -> None:
    backend = FakeCaptureBackend()
    harness = AudioWorkerHarness(backend=backend)
    harness.start()
    harness.send("WORKER_START", {"worker": "AUDIO"})
    backend.emit_me(bytes(1_920), at_ms=1_020)
    first = harness.writer_command()
    harness.send("WORKER_PAUSE", {"worker": "AUDIO"})
    assert backend.stream(AudioSource.ME).stop_calls == 1
    harness.send("WORKER_RESUME", {"worker": "AUDIO"})
    backend.emit_me(bytes(1_920), at_ms=3_020)
    second = harness.writer_command()
    assert first.source_end_sample == second.source_start_sample
    assert second.session_start_ms == 2_020
    harness.send("WORKER_STOP", {"worker": "AUDIO"})
    stopped = harness.status("WORKER_STOPPED")
    assert stopped.payload["drained"] is True
    assert stopped.payload["writer_frames"] == 2


def test_normal_stop_fences_writer_audio_before_reporting_stopped() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    harness.backend.emit_me(bytes(1_920), at_ms=1_020)
    harness.send("WORKER_STOP", {"worker": "AUDIO"})
    writer_items = harness.drain_writer_items()
    assert all(isinstance(item, AudioWriteCommand) for item in writer_items[:-1])
    assert isinstance(writer_items[-1], AudioDrainFence)
    assert sum(isinstance(item, AudioDrainFence) for item in writer_items) == 1
    assert not any(isinstance(item, AudioDrainFence) for item in harness.asr_items())
    assert harness.timeline.index("writer:AudioDrainFence") < harness.timeline.index(
        "status:WORKER_STOPPED"
    )
```

Put the reusable `FakeCaptureBackend` and `AudioWorkerHarness` in the same test file with fully typed methods. Inject backend, normalizer factory, monotonic clock, and sleeper through a private `_audio_worker_loop(...)`; the public process entrypoint constructs production adapters.

- [ ] **Step 2: Run the worker test and confirm it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_worker.py::test_pause_stops_streams_and_resume_preserves_source_offsets -v`

Expected: FAIL because the worker loop is absent.

- [ ] **Step 3: Implement command handling and drain ordering**

Use a bounded `queue.Queue[RawAudioChunk](maxsize=256)` per callback. Callback code only calls `put_nowait`; overflow sets a fatal event and returns. During initialization, resolve and open both exact configured devices with streams stopped; emit `WORKER_READY` only after both opens succeed. `WORKER_START` starts both streams together. The worker loop drains raw chunks, calls the matching source normalizer, calculates peak dBFS from canonical samples, and dispatches each frame. On pause, stop both streams and finish already-captured callback chunks before acknowledging; do not call normalizer `flush()`. On resume, restart the same stream objects, or reopen only the exact configured ID if a stream disconnected.

On stop, execute this order:

1. stop both streams so no callback can add audio;
2. drain both raw queues;
3. call each normalizer's `flush()` and dispatch every remaining complete frame, putting every final `AudioWriteCommand` on the Writer queue;
4. wait for the ASR pump spool to empty;
5. put exactly one shared `AudioDrainFence()` on the Writer queue after the last Writer audio command;
6. emit `WORKER_STOPPED` with `worker="AUDIO"` and `drained=true`, then close streams/backend.

The Audio Worker must never put an `AudioWriteCommand` after the fence.
Same-producer FIFO is the ordering proof consumed by the Writer Worker; queue
emptiness is not. `WORKER_STOPPED/drained=true` therefore means the fence was
already enqueued after all Writer audio. The fence is Writer-queue-only and is
neither wrapped in `MessageEnvelope` nor copied to `asr_audio_out`.

- [ ] **Step 4: Add disconnection/reconnection and fatal queue tests**

Add exact tests:

```python
def test_one_disconnected_source_does_not_stop_other_source() -> None:
    harness = AudioWorkerHarness()
    harness.start_recording()
    harness.backend.disconnect(AudioSource.OTHERS)
    harness.poll_once()
    harness.backend.emit_me(bytes(1_920), at_ms=1_040)
    assert harness.writer_command().source is AudioSource.ME
    assert harness.status("SOURCE_DISCONNECTED").payload["source"] == "OTHERS"
    assert harness.backend.stream(AudioSource.ME).is_active()


def test_reconnect_reopens_only_exact_configured_device_after_two_seconds() -> None:
    harness = AudioWorkerHarness(loopback_output_device_id="wasapi-output:7")
    harness.start_recording()
    harness.backend.disconnect(AudioSource.OTHERS)
    harness.poll_once()
    harness.clock.advance(1_999)
    harness.poll_once()
    assert harness.backend.opened_ids.count("wasapi-output:7") == 1
    harness.clock.advance(1)
    harness.backend.allow_open("wasapi-output:7")
    harness.poll_once()
    assert harness.backend.opened_ids.count("wasapi-output:7") == 2
    assert set(harness.backend.opened_ids) <= {"input:3", "wasapi-output:7"}


def test_writer_queue_full_emits_fatal_error_and_stops_both_streams() -> None:
    harness = AudioWorkerHarness(writer_queue_max_frames=1)
    harness.start_recording()
    harness.fill_writer_queue()
    harness.backend.emit_me(bytes(1_920), at_ms=1_020)
    harness.poll_once()
    error = harness.status("WORKER_ERROR")
    assert error.payload["code"] == "WRITER_QUEUE_FULL"
    assert not harness.backend.stream(AudioSource.ME).is_active()
    assert not harness.backend.stream(AudioSource.OTHERS).is_active()


def test_asr_spool_limit_stops_instead_of_dropping_audio() -> None:
    harness = AudioWorkerHarness(asr_spool_max_frames=1)
    harness.start_recording()
    harness.block_asr_consumer()
    harness.backend.emit_me(bytes(3_840), at_ms=1_020)
    harness.poll_until_stopped()
    assert harness.writer_command_count() == 2
    assert harness.status("WORKER_ERROR").payload["code"] == "ASR_QUEUE_STALLED"
```

The fake clock advances exactly 1,999 ms and 2,000 ms to verify the reconnect boundary. A disconnected source emits `SOURCE_DISCONNECTED` once, retries every `config.reconnect_interval_ms`, and emits `SOURCE_RECONNECTED` once after success. It never selects a different ID. Continue capturing the healthy source throughout.

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_worker.py -v`

Expected: all Audio Worker lifecycle tests pass.

---

### Task 6: Define ASR-local types, ports, VAD, and utterance boundaries

**Files:**
- Create: `src/flowlens/asr/__init__.py`
- Create: `src/flowlens/asr/types.py`
- Create: `src/flowlens/asr/ports.py`
- Create: `src/flowlens/asr/vad.py`
- Create: `tests/asr/test_vad.py`

**Interfaces:**
- Consumes: `AudioFrame` and `AudioSource`.
- Produces: `DecodedToken`, `DecodeHypothesis`, `PartialTranscript`, `CommitCandidate`, `AsrWorkerConfig`, `SpeechDetectorPort`, `DecoderPort`, `WebRtcSpeechDetector`, and `UtteranceBoundaryTracker`.

- [ ] **Step 1: Write failing 450 ms and 12-second boundary tests**

```python
from flowlens.asr.vad import UtteranceBoundaryTracker


def test_utterance_ends_after_twenty_three_silent_frames() -> None:
    tracker = UtteranceBoundaryTracker(silence_end_ms=450, max_utterance_ms=12_000)
    assert tracker.observe(is_speech=True) == "ACTIVE"
    for _ in range(22):
        assert tracker.observe(is_speech=False) == "ACTIVE"
    assert tracker.observe(is_speech=False) == "END"


def test_continuous_speech_forces_split_at_twelve_seconds() -> None:
    tracker = UtteranceBoundaryTracker(silence_end_ms=450, max_utterance_ms=12_000)
    for _ in range(599):
        assert tracker.observe(is_speech=True) == "ACTIVE"
    assert tracker.observe(is_speech=True) == "HARD_SPLIT"
```

- [ ] **Step 2: Run and confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_vad.py -v`

Expected: FAIL importing `UtteranceBoundaryTracker`.

- [ ] **Step 3: Implement exact types and protocols**

```python
# src/flowlens/asr/types.py
@dataclass(frozen=True, slots=True)
class DecodedToken:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class DecodeHypothesis:
    tokens: tuple[DecodedToken, ...]

    @property
    def text(self) -> str:
        return "".join(token.text for token in self.tokens).strip()


@dataclass(frozen=True, slots=True)
class AsrWorkerConfig:
    session_id: str
    model_path: Path
    partial_interval_ms: int = 500
    silence_end_ms: int = 450
    stable_age_ms: int = 1_200
    max_utterance_ms: int = 12_000
    delayed_threshold_ms: int = 2_000
    analysis_pause_threshold_ms: int = 5_000
```

`PartialTranscript` and `CommitCandidate` both contain `source`, `text`, `session_start_ms`, `session_end_ms`, `source_start_sample`, and `source_end_sample`; `CommitCandidate` additionally contains `committed_at: datetime`.

```python
# src/flowlens/asr/ports.py
class SpeechDetectorPort(Protocol):
    def is_speech(self, frame: AudioFrame) -> bool: ...


class DecoderPort(Protocol):
    def decode(self, pcm_s16le: bytes) -> DecodeHypothesis: ...
```

`WebRtcSpeechDetector(mode=2)` calls `webrtcvad.Vad(2).is_speech(frame.pcm_s16le, 16_000)` and rejects non-640-byte frames. `UtteranceBoundaryTracker.observe()` counts 20 ms frames, treats 23 consecutive silent frames as the first boundary at or above 450 ms, resets after `END`/`HARD_SPLIT`, and exposes `active_duration_ms`.

- [ ] **Step 4: Run VAD tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_vad.py -v`

Expected: 2 passed.

---

### Task 7: Wrap the fixed Kotoba-Whisper model with offline-only arguments

**Files:**
- Create: `src/flowlens/asr/kotoba_whisper.py`
- Create: `tests/asr/test_kotoba_whisper.py`

**Interfaces:**
- Consumes: injectable `WhisperModelFactory`, validated local `Path`, NumPy, and faster-whisper segment/word iterables.
- Produces: `KotobaWhisperDecoder(model_path, model_factory)` implementing `DecoderPort`.

- [ ] **Step 1: Write a failing constructor/argument contract test**

```python
from pathlib import Path

from flowlens.asr.kotoba_whisper import KotobaWhisperDecoder


class FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        word = type("Word", (), {"word": "方針", "start": 0.1, "end": 0.4})()
        segment = type("Segment", (), {"words": [word]})()
        return iter([segment]), object()


def test_decoder_uses_fixed_offline_kotoba_settings(tmp_path: Path) -> None:
    model_dir = tmp_path / "kotoba-whisper-v2.0-faster"
    model_dir.mkdir()
    fake = FakeModel()
    constructor: dict[str, object] = {}

    def factory(path: str, **kwargs):
        constructor.update({"path": path, **kwargs})
        return fake

    decoder = KotobaWhisperDecoder(model_dir, model_factory=factory)
    result = decoder.decode(bytes(16_000 * 2))
    assert constructor == {
        "path": str(model_dir),
        "device": "cuda",
        "compute_type": "float16",
        "local_files_only": True,
    }
    assert fake.kwargs == {
        "language": "ja",
        "task": "transcribe",
        "beam_size": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": False,
    }
    assert result.text == "方針"
```

- [ ] **Step 2: Run and confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_kotoba_whisper.py -v`

Expected: FAIL importing `KotobaWhisperDecoder`.

- [ ] **Step 3: Implement local-path validation and token conversion**

Raise `ModelPathError` unless `model_path.is_dir()` before constructing faster-whisper. Pass the exact constructor and transcription arguments asserted above. Convert int16 PCM to `float32 / 32768.0` before `transcribe`. Fully consume the segment generator inside `decode()`. Convert each non-`None` faster-whisper word to `DecodedToken(text=word.word, start_ms=round(word.start * 1000), end_ms=round(word.end * 1000))`. If word timestamps are absent for a non-empty segment, create one token from the segment text and segment start/end. Never pass a repository ID or enable download fallback.

- [ ] **Step 4: Run adapter tests and offline guard test**

Add a test that passes a missing path and asserts the factory is never called.

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_kotoba_whisper.py -v`

Expected: all tests pass without CUDA or a model installation.

---

### Task 8: Implement stable-prefix commits, filtering, splitting, and chronological sequencing

**Files:**
- Create: `src/flowlens/asr/commit.py`
- Create: `tests/asr/test_commit.py`

**Interfaces:**
- Consumes: `DecodeHypothesis`, `CommitCandidate`, `TranscriptRecord`, `new_ulid()`.
- Produces: `StablePrefixTracker.observe(hypothesis, decoded_at_ms, final)`, `is_transcript_content(text)`, `choose_split_ms(hypothesis)`, and `ChronologicalCommitBuffer`.

- [ ] **Step 1: Write failing stable-prefix and end-of-utterance tests**

```python
from flowlens.asr.commit import StablePrefixTracker
from flowlens.asr.types import DecodedToken, DecodeHypothesis


def hypothesis(*texts: str) -> DecodeHypothesis:
    return DecodeHypothesis(tuple(
        DecodedToken(text, index * 200, (index + 1) * 200)
        for index, text in enumerate(texts)
    ))


def test_prefix_commits_only_after_two_matches_and_twelve_hundred_ms_age() -> None:
    tracker = StablePrefixTracker(stable_age_ms=1_200)
    assert tracker.observe(hypothesis("今回", "は", "方針"), 0, final=False) == ()
    assert tracker.observe(hypothesis("今回", "は", "方針"), 500, final=False) == ()
    assert tracker.observe(hypothesis("今回", "は", "方針"), 1_199, final=False) == ()
    committed = tracker.observe(hypothesis("今回", "は", "方針"), 1_200, final=False)
    assert tuple(token.text for token in committed) == ("今回", "は", "方針")


def test_final_decode_commits_remaining_text_once() -> None:
    tracker = StablePrefixTracker(stable_age_ms=1_200)
    tracker.observe(hypothesis("確認", "します"), 0, final=False)
    first = tracker.observe(hypothesis("確認", "します"), 450, final=True)
    second = tracker.observe(hypothesis("確認", "します"), 900, final=True)
    assert "".join(token.text for token in first) == "確認します"
    assert second == ()
```

- [ ] **Step 2: Run and confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_commit.py -v`

Expected: FAIL importing commit logic.

- [ ] **Step 3: Implement exact stable-prefix semantics and content filter**

Track uncommitted token identities `(text, start_ms, end_ms)`, first-seen decode time, consecutive-decode count, and committed token count. A non-final call may return the longest unchanged prefix only when every returned token has appeared in at least two consecutive decodes and `decoded_at_ms - first_seen_ms >= stable_age_ms`. A final call returns all not-yet-committed tokens from the final hypothesis. Already-returned tokens are never returned again.

`is_transcript_content(text)` strips whitespace and returns false for empty text and for a full string matching this exact case-insensitive pattern:

```python
r"^(?:\[(?:music|applause|laughter|silence|noise|音楽|拍手|笑い|無音|雑音)\]|\((?:music|applause|laughter|silence|noise|音楽|拍手|笑い|無音|雑音)\))$"
```

It returns true for fillers including `えー`, `えっと`, and `あの`.

- [ ] **Step 4: Add split and merge-order tests**

```python
def test_split_prefers_latest_punctuation_between_ten_and_twelve_seconds() -> None:
    decoded = DecodeHypothesis((
        DecodedToken("前半。", 0, 10_400),
        DecodedToken("続き", 10_400, 11_800),
    ))
    assert choose_split_ms(decoded) == 10_400


def test_split_falls_back_to_twelve_seconds_without_punctuation() -> None:
    decoded = DecodeHypothesis((DecodedToken("継続発言", 0, 11_900),))
    assert choose_split_ms(decoded) == 12_000


def test_equal_start_time_orders_me_before_others() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.ME, 100)
    buffer.set_frontier(AudioSource.OTHERS, 100)
    buffer.push(candidate(AudioSource.OTHERS, "他者", start_ms=100))
    buffer.push(candidate(AudioSource.ME, "自分", start_ms=100))
    buffer.set_frontier(AudioSource.ME, None)
    assert [record.text for record in buffer.release_ready()] == ["自分"]
    buffer.set_frontier(AudioSource.OTHERS, None)
    assert [record.text for record in buffer.release_ready()] == ["他者"]


def test_later_finishing_older_utterance_blocks_newer_commit() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.ME, 100)
    buffer.push(candidate(AudioSource.OTHERS, "新しい", start_ms=200))
    assert buffer.release_ready() == ()
    buffer.push(candidate(AudioSource.ME, "古い", start_ms=100))
    buffer.set_frontier(AudioSource.ME, None)
    assert [record.text for record in buffer.release_ready()] == ["古い", "新しい"]
```

Define typed `candidate()` and `make_commit_buffer()` fixtures directly above these tests. Use deterministic IDs `SEG-001`, `SEG-002` and a fixed timezone-aware datetime so assertions also cover every `TranscriptRecord` field.

`choose_split_ms()` selects the latest token end from 10,000 through 12,000 ms whose text ends in `。`, `！`, `？`, `、`, `.`, `!`, or `?`; otherwise it returns 12,000. Round down to a 20 ms frame boundary.

`ChronologicalCommitBuffer` maintains a frontier key per source: `(earliest_uncommitted_session_start_ms, source_rank)` where ME rank is 0 and OTHERS rank is 1. `push(candidate)` stores by that key. `release_ready()` releases only candidates whose key is strictly before every active frontier, then constructs `TranscriptRecord` with consecutive `sequence` values, `schema_version=1`, injected `segment_id_factory: Callable[[], str] = new_ulid`, and injected timezone-aware `now: Callable[[], datetime]`. `finalize()` clears frontiers and releases all candidates. This is what preserves merged start-time order during overlapping speech.

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_commit.py -v`

Expected: all commit, filter, split, and ordering tests pass.

---

### Task 9: Build the two-stream ASR engine and oldest-pending scheduler

**Files:**
- Create: `src/flowlens/asr/engine.py`
- Create: `tests/asr/test_engine.py`

**Interfaces:**
- Consumes: `AudioFrame`, `SpeechDetectorPort`, one shared `DecoderPort`, `StablePrefixTracker`, `ChronologicalCommitBuffer`, and `AsrWorkerConfig`.
- Produces:

```python
class AsrEngine:
    def __init__(
        self,
        config: AsrWorkerConfig,
        decoder: DecoderPort,
        speech_detector: SpeechDetectorPort,
        segment_id_factory: Callable[[], str] = new_ulid,
        now: Callable[[], datetime] | None = None,
    ) -> None: ...

    def accept(self, frame: AudioFrame) -> None: ...
    def process_ready(self, now_monotonic_ms: int) -> AsrBatch: ...
    def finalize(self, now_monotonic_ms: int) -> AsrBatch: ...
    def backlog_ms(self, now_monotonic_ms: int) -> int: ...
```

`AsrBatch(partials: tuple[PartialTranscript, ...], committed: tuple[TranscriptRecord, ...])` is immutable.

- [ ] **Step 1: Write a failing fair-scheduling test**

```python
def test_oldest_pending_source_is_decoded_first_with_one_shared_model() -> None:
    decoder = RecordingDecoder((hypothesis("先"), hypothesis("後")))
    engine = make_engine(decoder=decoder, speech=True)
    engine.accept(frame(AudioSource.OTHERS, session_ms=100, captured_ms=1_100))
    engine.accept(frame(AudioSource.ME, session_ms=80, captured_ms=1_080))
    engine.process_ready(now_monotonic_ms=1_600)
    assert decoder.decoded_sources == [AudioSource.ME, AudioSource.OTHERS]
    assert decoder.max_concurrent_calls == 1
```

The fake decoder receives source through a test-only wrapper around the PCM buffer; production calls remain `decode(pcm_s16le)`.

- [ ] **Step 2: Run and confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_engine.py::test_oldest_pending_source_is_decoded_first_with_one_shared_model -v`

Expected: FAIL because `AsrEngine` is absent.

- [ ] **Step 3: Implement per-source state and decode cadence**

Each source state owns its frame deque, utterance frames, boundary tracker, stable-prefix tracker, last decode monotonic time, and earliest uncommitted start key. `accept()` validates monotonic non-overlapping source sample offsets and appends only to that source. `process_ready()` repeatedly chooses the source whose oldest undecoded frame has the lowest `(captured_monotonic_ms, source_rank)`.

Decode when one of these is true:

- active speech has at least 500 ms since the prior decode;
- the boundary tracker returns `END` after 23 silent frames;
- continuous speech reaches 12,000 ms.

At a 12-second split, decode, call `choose_split_ms`, commit through the chosen boundary, and retain post-boundary frames as the start of the next utterance. At `END`, perform a final decode before resetting. Emit a `PartialTranscript` only when filtered text differs from the last emitted partial. Clear the partial by emitting the same source with empty `text` immediately after its final commit. Filter empty/non-speech-marker commits; retain filler-only commits.

Translate token-relative times to absolute session/source offsets from the first utterance frame. Clamp boundaries to available 20 ms frame boundaries. Update the chronological buffer frontier whenever an utterance begins, commits a prefix, splits, or ends.

- [ ] **Step 4: Add exact behavior tests**

```python
def test_active_speech_decodes_at_five_hundred_ms_cadence() -> None:
    decoder = RecordingDecoder.repeat(hypothesis("発言"))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=24, start_ms=0)
    engine.process_ready(499)
    assert decoder.call_count == 0
    feed(engine, AudioSource.ME, frame_count=1, start_ms=480)
    engine.process_ready(500)
    assert decoder.call_count == 1


def test_silence_end_runs_final_decode_and_commits_remaining_text() -> None:
    decoder = RecordingDecoder((hypothesis("確認します。"),))
    engine = make_engine(decoder=decoder, speech_pattern=[True] * 10 + [False] * 23)
    feed(engine, AudioSource.ME, frame_count=33, start_ms=0)
    batch = engine.process_ready(660)
    assert [record.text for record in batch.committed] == ["確認します。"]


def test_partial_change_is_ephemeral_and_not_a_transcript_record() -> None:
    decoder = RecordingDecoder((hypothesis("確認"), hypothesis("確認中")))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=50, start_ms=0)
    first = engine.process_ready(500)
    second = engine.process_ready(1_000)
    assert [partial.text for partial in first.partials + second.partials] == ["確認", "確認中"]
    assert first.committed + second.committed == ()


def test_filler_only_text_is_committed() -> None:
    engine = final_decode_engine("えっと")
    assert [record.text for record in engine.finalize(1_000).committed] == ["えっと"]


def test_non_speech_marker_is_discarded() -> None:
    engine = final_decode_engine("[音楽]")
    assert engine.finalize(1_000).committed == ()


def test_continuous_speech_splits_no_later_than_twelve_seconds() -> None:
    engine = make_engine(decoder=RecordingDecoder.repeat(hypothesis("継続。")), speech=True)
    feed(engine, AudioSource.ME, frame_count=600, start_ms=0)
    batch = engine.process_ready(12_000)
    assert batch.committed
    assert batch.committed[0].session_end_ms <= 12_000


def test_overlapping_sources_produce_separate_ordered_records() -> None:
    engine = overlapping_final_decode_engine(me_start_ms=100, others_start_ms=120)
    records = engine.finalize(2_000).committed
    assert [(record.source, record.sequence) for record in records] == [
        (AudioSource.ME, 1),
        (AudioSource.OTHERS, 2),
    ]


def test_finalize_commits_both_sources_once() -> None:
    engine = overlapping_final_decode_engine(me_start_ms=100, others_start_ms=120)
    first = engine.finalize(2_000).committed
    second = engine.finalize(2_100).committed
    assert len(first) == 2
    assert second == ()
```

Implement the typed test helpers named above in `tests/asr/test_engine.py`; each helper delegates only through `AsrEngine.accept/process_ready/finalize` and fake port methods, never through production internals.

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_engine.py -v`

Expected: all ASR engine tests pass using only fake VAD/decoder/clock.

---

### Task 10: Implement the ASR Worker IPC contract and lag transitions

**Files:**
- Create: `src/flowlens/asr/worker.py`
- Create: `tests/asr/test_worker.py`

**Interfaces:**
- Consumes: `MessageEnvelope`, `AudioFrame`, `TranscriptRecord`, `KotobaWhisperDecoder`, `WebRtcSpeechDetector`, and `AsrEngine`.
- Produces:

```python
def run_asr_worker(
    config: AsrWorkerConfig,
    audio_in: "multiprocessing.Queue[AudioFrame]",
    control_in: "multiprocessing.Queue[MessageEnvelope]",
    control_out: "multiprocessing.Queue[MessageEnvelope]",
) -> None: ...
```

Commands use shared `WORKER_START`, `WORKER_PAUSE`, `WORKER_RESUME`, and `WORKER_STOP`. Each payload contains `{"worker": "ASR"}`; stop additionally contains `{"finalize": True}`. Outputs are:

```text
WORKER_READY         {"worker": "ASR"}
TRANSCRIPT_PARTIAL   {"source", "text", "session_start_ms", "session_end_ms", "source_start_sample", "source_end_sample"}
TRANSCRIPT_COMMITTED {all TranscriptRecord fields, with "source" and "committed_at" serialized}
ASR_STATUS           {"state": "READY"|"RUNNING"|"DELAYED"|"STOPPED", "backlog_ms": int, "analysis_paused": bool}
WORKER_STOPPED       {"worker": "ASR", "drained": true, "committed_count": int}
WORKER_ERROR         {"worker": "ASR", "code": "MODEL_LOAD_FAILED"|"DECODE_FAILED", "detail": str}
```

- [ ] **Step 1: Write a failing worker contract test**

```python
def test_stop_drains_audio_then_finalizes_uncommitted_text() -> None:
    harness = AsrWorkerHarness(decoder=ScriptedDecoder.final_text("最終発言"))
    harness.start()
    harness.send("WORKER_START", {"worker": "ASR"})
    harness.put_audio(speech_frames(AudioSource.ME, count=10))
    harness.send("WORKER_STOP", {"worker": "ASR", "finalize": True})
    committed = harness.output("TRANSCRIPT_COMMITTED")
    drained = harness.output("WORKER_STOPPED")
    assert committed.payload["text"] == "最終発言"
    assert drained.payload["drained"] is True
    assert drained.payload["committed_count"] == 1
    assert harness.output("ASR_STATUS").payload["state"] == "STOPPED"
```

- [ ] **Step 2: Run and confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_worker.py::test_stop_drains_audio_then_finalizes_uncommitted_text -v`

Expected: FAIL because `run_asr_worker` is absent.

- [ ] **Step 3: Implement model readiness, commands, and envelope emission**

Construct the decoder before `WORKER_READY`; convert any path/CUDA/model-load exception into `WORKER_ERROR/MODEL_LOAD_FAILED` and exit without readiness. The public entrypoint uses the production decoder and VAD; a private `_asr_worker_loop()` accepts factories for tests.

If `decoder.decode()` raises after readiness, emit `WORKER_ERROR/DECODE_FAILED`, leave all previously emitted `TranscriptRecord` objects unchanged, and exit nonzero. The Session Controller owns the specification's single ASR restart and transcript-gap event; this worker must not retry or switch models internally.

While running, alternate short timed reads from `control_in` and `audio_in`, drain all immediately available frames, call `engine.process_ready(monotonic_ms())`, and emit one envelope per partial/record. Serialize `AudioSource` as `ME`/`OTHERS` and `committed_at` as ISO 8601 with timezone. Envelope sequence increments independently for every ASR envelope. Pause stops accepting new audio after already-queued frames drain; resume continues with the same engine. Stop drains `audio_in`, calls `engine.finalize()` exactly once, emits records before `WORKER_STOPPED`, then emits `ASR_STATUS/STOPPED`.

- [ ] **Step 4: Add lag hysteresis and error tests**

Add exact tests:

```python
def test_delayed_transition_is_strictly_above_two_seconds() -> None:
    harness = AsrWorkerHarness()
    harness.set_backlog_ms(2_000)
    harness.poll_once()
    assert harness.outputs("ASR_STATUS", state="DELAYED") == []
    harness.set_backlog_ms(2_001)
    harness.poll_twice()
    statuses = harness.outputs("ASR_STATUS", state="DELAYED")
    assert len(statuses) == 1
    assert statuses[0].payload["backlog_ms"] == 2_001


def test_analysis_pause_transition_is_strictly_above_five_seconds() -> None:
    harness = AsrWorkerHarness()
    harness.set_backlog_ms(5_000)
    harness.poll_once()
    assert harness.outputs("ASR_STATUS", analysis_paused=True) == []
    harness.set_backlog_ms(5_001)
    harness.poll_twice()
    statuses = harness.outputs("ASR_STATUS", analysis_paused=True)
    assert len(statuses) == 1


def test_backlog_below_two_seconds_clears_analysis_paused_once() -> None:
    harness = AsrWorkerHarness()
    harness.set_backlog_ms(5_001)
    harness.poll_once()
    harness.set_backlog_ms(1_999)
    harness.poll_twice()
    resumed = harness.outputs("ASR_STATUS", analysis_paused=False, state="RUNNING")
    assert len(resumed) == 1


def test_model_load_failure_never_emits_ready() -> None:
    harness = AsrWorkerHarness(model_factory=raising_model_factory)
    harness.start()
    assert harness.output("WORKER_ERROR").payload["code"] == "MODEL_LOAD_FAILED"
    assert harness.outputs("WORKER_READY") == []


def test_decode_failure_does_not_mutate_prior_commits() -> None:
    harness = AsrWorkerHarness(decoder=DecoderThatFailsAfter("保存済み"))
    harness.commit_first_utterance()
    before = harness.committed_payloads()
    harness.decode_second_utterance()
    assert harness.output("WORKER_ERROR").payload["code"] == "DECODE_FAILED"
    assert harness.committed_payloads() == before
```

Define the harness and fake factories in the same test file; inject decoder/engine factories into `_asr_worker_loop()` so no test imports or patches faster-whisper.

State thresholds are strict: backlog `> 2_000` changes the state to delayed, backlog `> 5_000` changes `ASR_STATUS.analysis_paused` to true, and only backlog `< 2_000` changes it back to false and returns to running. Exactly 2,000 ms and 5,000 ms do not enter the higher degradation state; exactly 2,000 ms also does not resume an already degraded state. Emit `ASR_STATUS` once per boundary crossing, not on every poll; the Session Controller owns any corresponding persisted operational events.

Run: `.\.venv\Scripts\python.exe -m pytest tests/asr/test_worker.py -v`

Expected: all ASR Worker tests pass.

---

### Task 11: Verify the hardware-free Audio-to-ASR pipeline

**Files:**
- Create: `tests/integration/test_audio_asr_pipeline.py`

**Interfaces:**
- Consumes: fake capture backend, production normalizer/dispatcher/ASR engine, fake decoder, and shared Writer/Transcript types.
- Produces: a deterministic integration proof that Writer bytes and ASR source/timing metadata stay aligned.

- [ ] **Step 1: Write the failing integration test**

```python
def test_overlapping_me_and_others_frames_preserve_audio_and_transcript_order() -> None:
    pipeline = HardwareFreePipeline(
        decoder=PerSourceDecoder(
            me_text="確認します。",
            others_text="承知しました。",
        )
    )
    pipeline.emit_native_stereo(AudioSource.OTHERS, start_ms=120, duration_ms=700)
    pipeline.emit_native_mono(AudioSource.ME, start_ms=100, duration_ms=700)
    pipeline.stop_and_finalize()

    assert pipeline.writer_pcm_duration_ms(AudioSource.ME) == 700
    assert pipeline.writer_pcm_duration_ms(AudioSource.OTHERS) == 700
    assert [(record.source.value, record.text) for record in pipeline.records] == [
        ("ME", "確認します。"),
        ("OTHERS", "承知しました。"),
    ]
    assert [record.sequence for record in pipeline.records] == [1, 2]
    assert pipeline.records[0].source_start_sample == 0
    assert pipeline.records[1].source_start_sample == 0
```

- [ ] **Step 2: Run and confirm the missing harness fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_audio_asr_pipeline.py -v`

Expected: FAIL because `HardwareFreePipeline` is not defined.

- [ ] **Step 3: Implement the in-test harness from public ports**

Build `HardwareFreePipeline` only inside the test module. It must connect `SoxrAudioNormalizer -> AudioDispatcher -> AsrEngine`, use real bounded `queue.Queue` objects, and use fake capture/model ports. Do not add a test-only branch to production code. Verify Writer commands reconstruct exactly the bytes accepted by ASR and that both source sample counters exclude an injected two-second pause while session timestamps retain it.

- [ ] **Step 4: Run all automated gates**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio tests/asr tests/integration/test_audio_asr_pipeline.py -v`

Expected: all tests pass.

Run: `.\.venv\Scripts\python.exe -m black --check src/flowlens/audio src/flowlens/asr tests/audio tests/asr tests/integration/test_audio_asr_pipeline.py`

Expected: all files unchanged.

Run: `.\.venv\Scripts\python.exe -m ruff check src/flowlens/audio src/flowlens/asr tests/audio tests/asr tests/integration/test_audio_asr_pipeline.py`

Expected: all checks passed.

Run: `.\.venv\Scripts\python.exe -m mypy src/flowlens/audio src/flowlens/asr`

Expected: success with no issues.

---

### Task 12: Add designated-PC audio and ASR smoke scripts

**Files:**
- Create: `src/flowlens/smoke/__init__.py`
- Create: `src/flowlens/smoke/audio.py`
- Create: `src/flowlens/smoke/asr.py`
- Create: `scripts/smoke_audio.ps1`
- Create: `scripts/smoke_asr.ps1`
- Test: `tests/audio/test_smoke_script_contract.py`

**Interfaces:**
- Consumes: installed local models, selected device IDs, built worker entrypoints, foundation `flowlens.writer.worker.run_writer_worker`, and Writer-owned WAV output.
- Produces: `flowlens.smoke.audio.main(argv: Sequence[str] | None = None) -> int`, `flowlens.smoke.asr.main(argv: Sequence[str] | None = None) -> int`, and reproducible one-minute Audio/two-minute ASR acceptance commands. These are the only hardware/CUDA-dependent gates in this plan.

- [ ] **Step 1: Write a static contract test for script parameters**

```python
from pathlib import Path


def test_smoke_scripts_require_explicit_devices_and_local_model() -> None:
    audio = Path("scripts/smoke_audio.ps1").read_text(encoding="utf-8")
    asr = Path("scripts/smoke_asr.ps1").read_text(encoding="utf-8")
    assert "[Parameter(Mandatory = $true)][string]$MicrophoneId" in audio
    assert "[Parameter(Mandatory = $true)][string]$LoopbackOutputId" in audio
    assert "[Parameter(Mandatory = $true)][string]$ModelPath" in asr
    assert "[ValidateSet(60)][int]$DurationSeconds = 60" in audio
    assert "[ValidateSet(120)][int]$DurationSeconds = 120" in asr
```

- [ ] **Step 2: Run and confirm scripts are absent**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_smoke_script_contract.py -v`

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Implement explicit, non-fallback smoke entrypoints**

`flowlens.smoke.audio.main()` parses exact `--microphone-id`, `--loopback-output-id`, `--output-directory`, and `--duration-seconds` arguments, starts the production Audio and Writer worker entrypoints through `multiprocessing.get_context("spawn")`, captures for the requested duration, performs the generic drain handshake, and validates both Writer-owned WAV files with Python's `wave` module. It returns 0 only when both files are mono, 16-bit, 16 kHz, each duration error is below 0.5% (strictly between 59.7 and 60.3 seconds for the fixed one-minute run), and queue overflow count is zero.

`smoke_audio.ps1` accepts mandatory `MicrophoneId`, `LoopbackOutputId`, and `OutputDirectory` plus `[ValidateSet(60)][int]$DurationSeconds = 60`, invokes `.\.venv\Scripts\python.exe -m flowlens.smoke.audio` with those four explicit arguments, and exits with the Python process exit code.

`flowlens.smoke.asr.main()` adds exact `--model-path`, connects the production Audio, ASR, and Writer workers, collects control messages for 120 seconds, and performs the generic Audio-stop then ASR-finalize handshake. It records monotonic speech-start, partial-emission, detected utterance-end, and commit-emission times, then calculates partial and commit p95 with the nearest-rank method. `smoke_asr.ps1` exposes `[ValidateSet(120)][int]$DurationSeconds = 120` and invokes that module. Its summary prints the resolved model path, `device=cuda`, `compute_type=float16`, partial count, committed count per source, partial p95, commit-after-end p95, maximum backlog, and whether an instructed overlapping section produced separate ME/OTHERS records. Both entrypoints return nonzero on a substituted device, wrong format, empty source, wrong model configuration, queue overflow, partial p95 above 2,000 ms, or commit p95 above 3,000 ms.

- [ ] **Step 4: Run static gates, then the designated-PC gates**

Run: `.\.venv\Scripts\python.exe -m pytest tests/audio/test_smoke_script_contract.py -v`

Expected: pass without hardware.

Run on the designated PC after explicit IDs/model path are known:

```powershell
.\scripts\smoke_audio.ps1 -MicrophoneId "input:3" -LoopbackOutputId "wasapi-output:7" -OutputDirectory "$env:LOCALAPPDATA\FlowLens\smoke\audio" -DurationSeconds 60
.\scripts\smoke_asr.ps1 -MicrophoneId "input:3" -LoopbackOutputId "wasapi-output:7" -ModelPath "$env:LOCALAPPDATA\FlowLens\models\kotoba-whisper-v2.0-faster" -OutputDirectory "$env:LOCALAPPDATA\FlowLens\smoke\asr" -DurationSeconds 120
```

Expected: both commands exit 0; both sources contain audio; WAV formats are exact; Japanese partials/commits appear; overlapping speech remains separately labeled; the fixed local Kotoba model configuration is printed; overflow count is zero.

Hardware result evidence belongs in the implementation task report, not in source code. A hardware failure must identify the exact device/model/configuration cause and must not be converted into a unit-test skip.

---

## Final Verification

- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/audio tests/asr tests/integration/test_audio_asr_pipeline.py -v`.
- [ ] Run `.\.venv\Scripts\python.exe -m black --check src/flowlens/audio src/flowlens/asr tests/audio tests/asr tests/integration/test_audio_asr_pipeline.py`.
- [ ] Run `.\.venv\Scripts\python.exe -m ruff check src/flowlens/audio src/flowlens/asr tests/audio tests/asr tests/integration/test_audio_asr_pipeline.py`.
- [ ] Run `.\.venv\Scripts\python.exe -m mypy src/flowlens/audio src/flowlens/asr`.
- [ ] Run both designated-PC smoke scripts with the IDs selected in preflight and the checksum-validated local model path.
- [ ] Confirm no test, adapter, or worker passes a Hugging Face repository ID to faster-whisper at session runtime.
- [ ] Confirm Writer queue failure safely stops both streams and emits the fatal error before shutdown.
- [ ] Confirm Audio puts every final Writer command and exactly one `AudioDrainFence` before `WORKER_STOPPED/drained=true`, sends no audio after the fence, never sends the fence to ASR, and that committed records precede ASR `WORKER_STOPPED/drained=true`.
- [ ] Confirm the full application integration plan consumes these exact worker entrypoints, commands, output payloads, and threshold semantics.
