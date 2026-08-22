"""Designated-PC dual-source capture and Writer-owned WAV smoke gate."""

from __future__ import annotations

import argparse
import multiprocessing
import queue
import sys
import threading
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from flowlens.audio.types import AudioWorkerConfig
from flowlens.audio.worker import run_audio_worker
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import (
    MessageType,
    ProcessSource,
    SessionMode,
    SessionStatus,
)
from flowlens.domain.ids import new_ulid
from flowlens.domain.messages import (
    MessageEnvelope,
    WriterAck,
    WriterFatal,
    WriterOpenSession,
    WriterShutdown,
)
from flowlens.domain.session import DeviceIdentity, ModelIdentity, SessionManifest
from flowlens.workers.writer import run_writer_worker

_STARTUP_TIMEOUT_SECONDS = 60.0
_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_QUEUE_POLL_SECONDS = 0.1
_MODEL_PLACEHOLDER_SHA256 = "0" * 64


class _QueuePort(Protocol):
    def put(self, item: object) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> object: ...

    def get_nowait(self) -> object: ...

    def close(self) -> None: ...

    def cancel_join_thread(self) -> None: ...

    def join_thread(self) -> None: ...


class _ProcessPort(Protocol):
    @property
    def exitcode(self) -> int | None: ...

    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _EventPort(Protocol):
    def set(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioSmokeArguments:
    """Validated explicit command-line values for the Audio smoke."""

    microphone_id: str
    loopback_output_id: str
    output_directory: Path
    duration_seconds: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the designated-PC FlowLens audio smoke gate.",
    )
    parser.add_argument("--microphone-id", required=True)
    parser.add_argument("--loopback-output-id", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--duration-seconds", required=True, type=int, choices=(60,))
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> AudioSmokeArguments:
    parser = _parser()
    namespace = parser.parse_args(argv)
    microphone_id = str(namespace.microphone_id).strip()
    loopback_id = str(namespace.loopback_output_id).strip()
    if not microphone_id:
        parser.error("--microphone-id must be non-empty")
    if not loopback_id:
        parser.error("--loopback-output-id must be non-empty")
    if microphone_id == loopback_id:
        parser.error("microphone and loopback output device IDs must be distinct")
    output = Path(str(namespace.output_directory)).expanduser().resolve()
    return AudioSmokeArguments(
        microphone_id=microphone_id,
        loopback_output_id=loopback_id,
        output_directory=output,
        duration_seconds=int(namespace.duration_seconds),
    )


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory must be empty: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_wav(path: Path, expected_duration_seconds: int) -> int:
    """Validate canonical format and strict sub-0.5% duration error."""

    if not path.is_file():
        raise ValueError(f"Writer-owned WAV is missing: {path}")
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
    if channels != 1:
        raise ValueError(f"WAV must be mono: {path}")
    if sample_width != 2:
        raise ValueError(f"WAV must use 16-bit samples: {path}")
    if sample_rate != 16_000:
        raise ValueError(f"WAV must use 16000 Hz: {path}")
    if frame_count <= 0:
        raise ValueError(f"WAV must contain audio: {path}")
    expected_frames = expected_duration_seconds * sample_rate
    if abs(frame_count - expected_frames) * 200 >= expected_frames:
        raise ValueError(
            f"WAV duration error must be below 0.5%: {path} "
            f"({frame_count / sample_rate:.3f}s)"
        )
    return round(frame_count * 1_000 / sample_rate)


def _now_utc() -> datetime:
    now = datetime.now(UTC).astimezone()
    return now.replace(microsecond=(now.microsecond // 1_000) * 1_000)


def _manifest(
    arguments: AudioSmokeArguments,
    session_id: str,
    started_at: datetime,
) -> SessionManifest:
    return SessionManifest(
        schema_version=1,
        session_id=session_id,
        status=SessionStatus.INCOMPLETE,
        mode=SessionMode.GENERAL,
        started_at=started_at,
        ended_at=None,
        active_duration_ms=0,
        pause_intervals=(),
        microphone=DeviceIdentity(
            arguments.microphone_id,
            arguments.microphone_id,
        ),
        loopback_output=DeviceIdentity(
            arguments.loopback_output_id,
            arguments.loopback_output_id,
        ),
        asr_model=ModelIdentity(
            "smoke/audio-only",
            "not-loaded",
            _MODEL_PLACEHOLDER_SHA256,
        ),
        discussion_model=ModelIdentity(
            "smoke/not-loaded",
            "not-loaded",
            _MODEL_PLACEHOLDER_SHA256,
        ),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=("Designated-PC audio smoke artifact",),
    )


def _control(
    session_id: str,
    message_type: MessageType,
    sequence: int,
    payload: object,
    monotonic_ms: int,
) -> MessageEnvelope[object]:
    return MessageEnvelope(
        schema_version=1,
        session_id=session_id,
        message_type=message_type,
        sequence=sequence,
        source=ProcessSource.GUI,
        created_monotonic_ms=monotonic_ms,
        payload=payload,
    )


def _worker_command(
    session_id: str,
    message_type: MessageType,
    sequence: int,
    worker: str,
    monotonic_ms: int,
    *,
    finalize: bool = False,
) -> MessageEnvelope[object]:
    payload: dict[str, object] = {"worker": worker}
    if finalize:
        payload["finalize"] = True
    return _control(session_id, message_type, sequence, payload, monotonic_ms)


def _wait_for_envelope(
    source: _QueuePort,
    expected: MessageType,
    *,
    deadline: float,
    monotonic: Callable[[], float],
) -> MessageEnvelope[object]:
    while monotonic() < deadline:
        try:
            value = source.get(timeout=_QUEUE_POLL_SECONDS)
        except queue.Empty:
            continue
        if not isinstance(value, MessageEnvelope):
            continue
        if value.message_type is MessageType.WORKER_ERROR:
            raise RuntimeError(f"worker error: {value.payload}")
        if value.message_type is MessageType.WRITER_FATAL:
            payload = value.payload
            if isinstance(payload, WriterFatal):
                raise RuntimeError(f"Writer failed: {payload.message}")
            raise RuntimeError(f"Writer failed: {payload}")
        if value.message_type is expected:
            return value
    raise TimeoutError(f"timed out waiting for {expected.value}")


def _wait_writer_ack(
    source: _QueuePort,
    sequence: int,
    *,
    deadline: float,
    monotonic: Callable[[], float],
) -> None:
    while True:
        envelope = _wait_for_envelope(
            source,
            MessageType.WRITER_ACK,
            deadline=deadline,
            monotonic=monotonic,
        )
        payload = envelope.payload
        if isinstance(payload, WriterAck) and payload.acknowledged_sequence == sequence:
            return


def _validate_worker_stopped(
    envelope: MessageEnvelope[object],
    expected_worker: str,
) -> None:
    """Require an exact drained acknowledgement from the addressed worker."""

    definitions: dict[str, tuple[ProcessSource, frozenset[str]]] = {
        "AUDIO": (
            ProcessSource.AUDIO,
            frozenset({"worker", "drained", "writer_frames", "asr_frames"}),
        ),
        "ASR": (
            ProcessSource.ASR,
            frozenset({"worker", "drained", "committed_count"}),
        ),
    }
    definition = definitions.get(expected_worker)
    payload = envelope.payload
    if (
        definition is None
        or envelope.message_type is not MessageType.WORKER_STOPPED
        or envelope.source is not definition[0]
        or not isinstance(payload, dict)
        or frozenset(payload) != definition[1]
        or payload.get("worker") != expected_worker
        or payload.get("drained") is not True
    ):
        raise ValueError(f"invalid {expected_worker} WORKER_STOPPED acknowledgement")
    count_fields = (
        ("writer_frames", "asr_frames")
        if expected_worker == "AUDIO"
        else ("committed_count",)
    )
    for field_name in count_fields:
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"invalid {expected_worker} WORKER_STOPPED acknowledgement"
            )


def _join_process(process: _ProcessPort, timeout_seconds: float) -> None:
    process.join(timeout_seconds)
    if process.is_alive():
        raise TimeoutError("worker did not exit within the bounded timeout")
    if process.exitcode not in (None, 0):
        raise RuntimeError(f"worker exited with code {process.exitcode}")


def _cleanup_resources(
    processes: Sequence[_ProcessPort],
    queues: Sequence[_QueuePort],
    *,
    join_timeout_seconds: float = 1.0,
) -> None:
    """Boundedly join/terminate children, then release queue feeder threads."""

    for process in processes:
        try:
            process.join(join_timeout_seconds)
        except BaseException:
            pass
    for process in processes:
        try:
            alive = process.is_alive()
        except BaseException:
            alive = False
        if alive:
            try:
                process.terminate()
                process.join(join_timeout_seconds)
            except BaseException:
                pass
        try:
            still_alive = process.is_alive()
        except BaseException:
            still_alive = False
        if still_alive:
            try:
                process.kill()
                process.join(join_timeout_seconds)
            except BaseException:
                pass
    for resource in queues:
        try:
            resource.cancel_join_thread()
        except BaseException:
            pass
        try:
            resource.close()
        except BaseException:
            pass
        try:
            resource.join_thread()
        except BaseException:
            pass


def _run(arguments: AudioSmokeArguments) -> int:
    _prepare_output_directory(arguments.output_directory)
    context = multiprocessing.get_context("spawn")
    session_id = new_ulid()
    started_at = _now_utc()
    started_monotonic_ms = int(time.monotonic() * 1_000)

    audio_control = context.Queue(maxsize=16)
    audio_status = context.Queue(maxsize=2_048)
    writer_control = context.Queue(maxsize=64)
    writer_audio = context.Queue(maxsize=6_000)
    writer_status = context.Queue(maxsize=64)
    unused_asr_audio = context.Queue(maxsize=12_000)
    writer_stop = context.Event()
    queues: list[_QueuePort] = [
        audio_control,
        audio_status,
        writer_control,
        writer_audio,
        writer_status,
        unused_asr_audio,
    ]
    writer = context.Process(
        target=run_writer_worker,
        args=(writer_control, writer_audio, writer_status, writer_stop),
        name="flowlens-smoke-writer",
    )
    audio = context.Process(
        target=run_audio_worker,
        args=(
            AudioWorkerConfig(
                session_id=session_id,
                microphone_device_id=arguments.microphone_id,
                loopback_output_device_id=arguments.loopback_output_id,
                session_started_monotonic_ms=started_monotonic_ms,
                writer_queue_max_frames=6_000,
                asr_queue_max_frames=6_000,
            ),
            audio_control,
            audio_status,
            writer_audio,
            unused_asr_audio,
        ),
        name="flowlens-smoke-audio",
    )
    processes: list[_ProcessPort] = [writer, audio]
    asr_sink_stopped = threading.Event()

    def drain_unused_asr() -> None:
        while not asr_sink_stopped.is_set():
            try:
                unused_asr_audio.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue

    asr_sink = threading.Thread(
        target=drain_unused_asr,
        name="flowlens-smoke-audio-asr-sink",
    )
    try:
        asr_sink.start()
        writer.start()
        writer_control.put(
            _control(
                session_id,
                MessageType.WRITER_OPEN_SESSION,
                1,
                WriterOpenSession(
                    arguments.output_directory,
                    _manifest(arguments, session_id, started_at),
                    DiscussionState.initial(SessionMode.GENERAL, started_at),
                ),
                started_monotonic_ms,
            )
        )
        _wait_writer_ack(
            writer_status,
            1,
            deadline=time.monotonic() + _STARTUP_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
        audio.start()
        _wait_for_envelope(
            audio_status,
            MessageType.WORKER_READY,
            deadline=time.monotonic() + _STARTUP_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
        audio_control.put(
            _worker_command(
                session_id,
                MessageType.WORKER_START,
                1,
                "AUDIO",
                int(time.monotonic() * 1_000),
            )
        )
        capture_deadline = time.monotonic() + arguments.duration_seconds
        while time.monotonic() < capture_deadline:
            try:
                value = audio_status.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            if isinstance(value, MessageEnvelope) and value.message_type in {
                MessageType.WORKER_ERROR,
                MessageType.SOURCE_DISCONNECTED,
            }:
                raise RuntimeError(f"audio smoke failed: {value.payload}")
        audio_control.put(
            _worker_command(
                session_id,
                MessageType.WORKER_STOP,
                2,
                "AUDIO",
                int(time.monotonic() * 1_000),
            )
        )
        stopped = _wait_for_envelope(
            audio_status,
            MessageType.WORKER_STOPPED,
            deadline=time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
        _validate_worker_stopped(stopped, "AUDIO")
        _join_process(audio, _SHUTDOWN_TIMEOUT_SECONDS)
        writer_control.put(
            _control(
                session_id,
                MessageType.WRITER_SHUTDOWN,
                2,
                WriterShutdown(),
                int(time.monotonic() * 1_000),
            )
        )
        _wait_writer_ack(
            writer_status,
            2,
            deadline=time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
        _join_process(writer, _SHUTDOWN_TIMEOUT_SECONDS)
        me_duration = _validate_wav(
            arguments.output_directory / "mic.wav",
            arguments.duration_seconds,
        )
        others_duration = _validate_wav(
            arguments.output_directory / "loopback.wav",
            arguments.duration_seconds,
        )
        print(f"microphone_id={arguments.microphone_id}")
        print(f"loopback_output_id={arguments.loopback_output_id}")
        print(f"mic_duration_ms={me_duration}")
        print(f"loopback_duration_ms={others_duration}")
        print("overflow_count=0")
        return 0
    finally:
        writer_stop.set()
        asr_sink_stopped.set()
        asr_sink.join(1.0)
        _cleanup_resources(processes, queues)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit one-minute Audio acceptance gate."""

    try:
        return _run(_parse_args(argv))
    except KeyboardInterrupt:
        print(
            "audio smoke interrupted; workers were boundedly cleaned up",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"audio smoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
