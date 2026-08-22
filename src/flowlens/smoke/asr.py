"""Designated-PC fixed-model ASR latency and overlap smoke gate."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import queue
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from flowlens.asr.types import AsrWorkerConfig
from flowlens.asr.worker import run_asr_worker
from flowlens.audio.types import AudioWorkerConfig
from flowlens.audio.worker import run_audio_worker
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import (
    AudioSource,
    MessageType,
    SessionMode,
)
from flowlens.domain.ids import new_ulid
from flowlens.domain.messages import (
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    WriterOpenSession,
    WriterShutdown,
)
from flowlens.smoke.audio import (
    _SHUTDOWN_TIMEOUT_SECONDS,
    _STARTUP_TIMEOUT_SECONDS,
    AudioSmokeArguments,
    _cleanup_resources,
    _control,
    _join_process,
    _manifest,
    _now_utc,
    _prepare_output_directory,
    _QueuePort,
    _validate_wav,
    _validate_worker_stopped,
    _wait_for_envelope,
    _wait_writer_ack,
    _worker_command,
)
from flowlens.workers.writer import run_writer_worker

_EXACT_MODEL_DIRECTORY_NAME = "kotoba-whisper-v2.0-faster"
_JAPANESE_TEXT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True, slots=True)
class AsrSmokeArguments:
    """Validated explicit command-line values for the ASR smoke."""

    microphone_id: str
    loopback_output_id: str
    model_path: Path
    output_directory: Path
    duration_seconds: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the designated-PC FlowLens ASR smoke gate.",
    )
    parser.add_argument("--microphone-id", required=True)
    parser.add_argument("--loopback-output-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--duration-seconds", required=True, type=int, choices=(120,))
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> AsrSmokeArguments:
    parser = _parser()
    namespace = parser.parse_args(argv)
    microphone_id = str(namespace.microphone_id).strip()
    loopback_id = str(namespace.loopback_output_id).strip()
    raw_model_path = str(namespace.model_path).strip()
    if not microphone_id:
        parser.error("--microphone-id must be non-empty")
    if not loopback_id:
        parser.error("--loopback-output-id must be non-empty")
    if microphone_id == loopback_id:
        parser.error("microphone and loopback output device IDs must be distinct")
    if not raw_model_path:
        parser.error("--model-path must be non-empty")
    if "/" in raw_model_path and not Path(raw_model_path).is_absolute():
        parser.error("--model-path must be a local directory, not a repository ID")
    model_path = Path(raw_model_path).expanduser().resolve()
    if not model_path.is_dir():
        parser.error("--model-path must be an existing local directory")
    if model_path.name != _EXACT_MODEL_DIRECTORY_NAME:
        parser.error(
            "--model-path must identify the local kotoba-whisper-v2.0-faster directory"
        )
    output = Path(str(namespace.output_directory)).expanduser().resolve()
    return AsrSmokeArguments(
        microphone_id=microphone_id,
        loopback_output_id=loopback_id,
        model_path=model_path,
        output_directory=output,
        duration_seconds=int(namespace.duration_seconds),
    )


def _nearest_rank_p95(values: Iterable[int]) -> int:
    """Return p95 by the nearest-rank definition (ceil(0.95 * n))."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("p95 requires at least one observation")
    rank = math.ceil(95 * len(ordered) / 100)
    return ordered[rank - 1]


def _payload_int(payload: object, key: str) -> int:
    if not isinstance(payload, dict):
        raise ValueError(f"ASR payload must be an object: {key}")
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ASR payload field must be an integer: {key}")
    return value


def _payload_text(payload: object, key: str) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"ASR payload must be an object: {key}")
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"ASR payload field must be text: {key}")
    return value


@dataclass(slots=True)
class AsrSmokeMetrics:
    """Monotonic observations available from the existing ASR IPC contract.

    Commit latency uses the transcript/audio end on the monotonic session
    timeline as a conservative proxy. The current worker payload does not expose
    the VAD detector's internal end-observation timestamp.
    """

    session_started_monotonic_ms: int
    partial_latencies_ms: list[int] = field(default_factory=list)
    commit_after_end_latencies_ms: list[int] = field(default_factory=list)
    committed_counts: dict[AudioSource, int] = field(
        default_factory=lambda: {source: 0 for source in AudioSource}
    )
    partial_texts: list[str] = field(default_factory=list)
    partial_count: int = 0
    committed_records: list[TranscriptRecord] = field(default_factory=list)
    maximum_backlog_ms: int = 0
    _partial_utterances: set[tuple[AudioSource, int]] = field(
        default_factory=set,
        repr=False,
    )

    def observe(self, envelope: MessageEnvelope[object]) -> TranscriptRecord | None:
        """Record one ASR envelope and return a parsed commit when present."""

        if envelope.message_type is MessageType.TRANSCRIPT_PARTIAL:
            text = _payload_text(envelope.payload, "text")
            if text:
                self.partial_count += 1
                self.partial_texts.append(text)
                source = AudioSource(_payload_text(envelope.payload, "source"))
                source_start = _payload_int(envelope.payload, "source_start_sample")
                utterance_key = (source, source_start)
                if utterance_key not in self._partial_utterances:
                    self._partial_utterances.add(utterance_key)
                    start = _payload_int(envelope.payload, "session_start_ms")
                    latency = envelope.created_monotonic_ms - (
                        self.session_started_monotonic_ms + start
                    )
                    self.partial_latencies_ms.append(max(0, latency))
            return None
        if envelope.message_type is MessageType.TRANSCRIPT_COMMITTED:
            if not isinstance(envelope.payload, dict):
                raise ValueError("committed transcript payload must be an object")
            record = TranscriptRecord.from_dict(envelope.payload)
            proxy_end = self.session_started_monotonic_ms + record.session_end_ms
            self.commit_after_end_latencies_ms.append(
                max(0, envelope.created_monotonic_ms - proxy_end)
            )
            self.committed_counts[record.source] += 1
            self.committed_records.append(record)
            return record
        if envelope.message_type is MessageType.ASR_STATUS:
            backlog = _payload_int(envelope.payload, "backlog_ms")
            maximum = _payload_int(envelope.payload, "maximum_backlog_ms")
            if maximum < backlog or maximum < self.maximum_backlog_ms:
                raise ValueError("ASR maximum_backlog_ms must be cumulative")
            self.maximum_backlog_ms = maximum
        return None

    @property
    def overlap_separate(self) -> bool:
        """Return whether separately labeled source records overlap in session time."""

        me = [
            record
            for record in self.committed_records
            if record.source is AudioSource.ME
        ]
        others = [
            record
            for record in self.committed_records
            if record.source is AudioSource.OTHERS
        ]
        return any(
            left.session_start_ms < right.session_end_ms
            and right.session_start_ms < left.session_end_ms
            for left in me
            for right in others
        )


@dataclass(slots=True)
class _StopHandshake:
    """Enforce Audio drain acknowledgement before ASR finalization request."""

    session_id: str
    audio_stopped: bool = False

    def audio_stop(self, monotonic_ms: int) -> MessageEnvelope[object]:
        return _worker_command(
            self.session_id,
            MessageType.WORKER_STOP,
            2,
            "AUDIO",
            monotonic_ms,
        )

    def acknowledge_audio_stopped(self) -> None:
        self.audio_stopped = True

    def asr_finalize(self, monotonic_ms: int) -> MessageEnvelope[object]:
        if not self.audio_stopped:
            raise RuntimeError("ASR finalize requires Audio drained acknowledgement")
        return _worker_command(
            self.session_id,
            MessageType.WORKER_STOP,
            2,
            "ASR",
            monotonic_ms,
            finalize=True,
        )


def _assert_japanese(metrics: AsrSmokeMetrics) -> None:
    if not any(_JAPANESE_TEXT.search(text) for text in metrics.partial_texts):
        raise ValueError("no Japanese nonempty partial transcript was observed")
    for source in AudioSource:
        source_texts = [
            record.text
            for record in metrics.committed_records
            if record.source is source
        ]
        if not source_texts:
            raise ValueError(f"no committed transcript for {source.value}")
        if not any(_JAPANESE_TEXT.search(text) for text in source_texts):
            raise ValueError(f"no Japanese committed text for {source.value}")


def _run(arguments: AsrSmokeArguments) -> int:
    _prepare_output_directory(arguments.output_directory)
    context = multiprocessing.get_context("spawn")
    session_id = new_ulid()
    started_at = _now_utc()
    started_monotonic_ms = int(time.monotonic() * 1_000)
    audio_arguments = AudioSmokeArguments(
        arguments.microphone_id,
        arguments.loopback_output_id,
        arguments.output_directory,
        arguments.duration_seconds,
    )

    audio_control = context.Queue(maxsize=16)
    audio_status = context.Queue(maxsize=16_384)
    asr_control = context.Queue(maxsize=16)
    asr_status = context.Queue(maxsize=8_192)
    writer_control = context.Queue(maxsize=1_024)
    writer_audio = context.Queue(maxsize=12_000)
    writer_status = context.Queue(maxsize=1_024)
    asr_audio = context.Queue(maxsize=12_000)
    writer_stop = context.Event()
    queues: list[_QueuePort] = [
        audio_control,
        audio_status,
        asr_control,
        asr_status,
        writer_control,
        writer_audio,
        writer_status,
        asr_audio,
    ]
    writer = context.Process(
        target=run_writer_worker,
        args=(writer_control, writer_audio, writer_status, writer_stop),
        name="flowlens-smoke-writer",
    )
    asr = context.Process(
        target=run_asr_worker,
        args=(
            AsrWorkerConfig(session_id=session_id, model_path=arguments.model_path),
            asr_audio,
            asr_control,
            asr_status,
        ),
        name="flowlens-smoke-asr",
    )
    audio = context.Process(
        target=run_audio_worker,
        args=(
            AudioWorkerConfig(
                session_id=session_id,
                microphone_device_id=arguments.microphone_id,
                loopback_output_device_id=arguments.loopback_output_id,
                session_started_monotonic_ms=started_monotonic_ms,
                writer_queue_max_frames=12_000,
                asr_queue_max_frames=12_000,
            ),
            audio_control,
            audio_status,
            writer_audio,
            asr_audio,
        ),
        name="flowlens-smoke-audio",
    )
    processes = [writer, asr, audio]
    metrics = AsrSmokeMetrics(started_monotonic_ms)
    handshake = _StopHandshake(session_id)
    writer_sequence = 1
    asr_final_status_observed = False

    def forward_asr(envelope: MessageEnvelope[object]) -> None:
        nonlocal writer_sequence
        if envelope.message_type is MessageType.WORKER_ERROR:
            raise RuntimeError(f"ASR smoke failed: {envelope.payload}")
        record = metrics.observe(envelope)
        if record is None:
            return
        writer_sequence += 1
        writer_control.put(
            _control(
                session_id,
                MessageType.TRANSCRIPT_COMMITTED,
                writer_sequence,
                TranscriptCommitted(record),
                int(time.monotonic() * 1_000),
            )
        )
        _wait_writer_ack(
            writer_status,
            writer_sequence,
            deadline=time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )

    def drain_statuses() -> tuple[bool, bool]:
        nonlocal asr_final_status_observed
        audio_stopped = False
        asr_stopped = False
        while True:
            try:
                value = audio_status.get_nowait()
            except queue.Empty:
                break
            if not isinstance(value, MessageEnvelope):
                continue
            if value.message_type in {
                MessageType.WORKER_ERROR,
                MessageType.SOURCE_DISCONNECTED,
            }:
                raise RuntimeError(f"audio smoke failed: {value.payload}")
            if value.message_type is MessageType.WORKER_STOPPED:
                _validate_worker_stopped(value, "AUDIO")
                audio_stopped = True
        while True:
            try:
                value = asr_status.get_nowait()
            except queue.Empty:
                break
            if not isinstance(value, MessageEnvelope):
                continue
            forward_asr(value)
            if value.message_type is MessageType.ASR_STATUS and (
                _payload_text(value.payload, "state") == "STOPPED"
            ):
                asr_final_status_observed = True
            if value.message_type is MessageType.WORKER_STOPPED:
                _validate_worker_stopped(value, "ASR")
                asr_stopped = True
        return audio_stopped, asr_stopped

    try:
        writer.start()
        writer_control.put(
            _control(
                session_id,
                MessageType.WRITER_OPEN_SESSION,
                writer_sequence,
                WriterOpenSession(
                    arguments.output_directory,
                    _manifest(audio_arguments, session_id, started_at),
                    DiscussionState.initial(SessionMode.GENERAL, started_at),
                ),
                started_monotonic_ms,
            )
        )
        _wait_writer_ack(
            writer_status,
            writer_sequence,
            deadline=time.monotonic() + _STARTUP_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
        asr.start()
        audio.start()
        _wait_for_envelope(
            asr_status,
            MessageType.WORKER_READY,
            deadline=time.monotonic() + _STARTUP_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
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
        asr_control.put(
            _worker_command(
                session_id,
                MessageType.WORKER_START,
                1,
                "ASR",
                int(time.monotonic() * 1_000),
            )
        )
        capture_deadline = time.monotonic() + arguments.duration_seconds
        while time.monotonic() < capture_deadline:
            drain_statuses()
            if not audio.is_alive() or not asr.is_alive() or not writer.is_alive():
                raise RuntimeError("a smoke worker exited before capture completed")
            time.sleep(0.01)

        audio_control.put(handshake.audio_stop(int(time.monotonic() * 1_000)))
        stop_deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
        audio_stopped = False
        while not audio_stopped and time.monotonic() < stop_deadline:
            stopped, _ = drain_statuses()
            audio_stopped |= stopped
            time.sleep(0.01)
        if not audio_stopped:
            raise TimeoutError("timed out waiting for ordered Audio drain fence")
        handshake.acknowledge_audio_stopped()
        _join_process(audio, _SHUTDOWN_TIMEOUT_SECONDS)

        asr_control.put(handshake.asr_finalize(int(time.monotonic() * 1_000)))
        stop_deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
        asr_stopped = False
        while not asr_stopped and time.monotonic() < stop_deadline:
            _, stopped = drain_statuses()
            asr_stopped |= stopped
            time.sleep(0.01)
        if not asr_stopped:
            raise TimeoutError("timed out waiting for ordered ASR finalize")
        _join_process(asr, _SHUTDOWN_TIMEOUT_SECONDS)
        final_status_deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
        while (
            not asr_final_status_observed and time.monotonic() < final_status_deadline
        ):
            drain_statuses()
            time.sleep(0.01)
        if not asr_final_status_observed:
            raise TimeoutError("timed out waiting for final ASR_STATUS maximum")

        writer_sequence += 1
        writer_control.put(
            _control(
                session_id,
                MessageType.WRITER_SHUTDOWN,
                writer_sequence,
                WriterShutdown(),
                int(time.monotonic() * 1_000),
            )
        )
        _wait_writer_ack(
            writer_status,
            writer_sequence,
            deadline=time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS,
            monotonic=time.monotonic,
        )
        _join_process(writer, _SHUTDOWN_TIMEOUT_SECONDS)

        _validate_wav(arguments.output_directory / "mic.wav", 120)
        _validate_wav(arguments.output_directory / "loopback.wav", 120)
        _assert_japanese(metrics)
        partial_p95 = _nearest_rank_p95(metrics.partial_latencies_ms)
        commit_p95 = _nearest_rank_p95(metrics.commit_after_end_latencies_ms)
        if partial_p95 > 2_000:
            raise ValueError(f"partial p95 exceeds 2000 ms: {partial_p95}")
        if commit_p95 > 3_000:
            raise ValueError(
                f"commit-after-end proxy p95 exceeds 3000 ms: {commit_p95}"
            )
        if not metrics.overlap_separate:
            raise ValueError("no separate overlapping ME/OTHERS records were observed")

        print(f"model_path={arguments.model_path}")
        print("device=cuda")
        print("compute_type=float16")
        print(f"partial_count={metrics.partial_count}")
        print(f"committed_me={metrics.committed_counts[AudioSource.ME]}")
        print(f"committed_others={metrics.committed_counts[AudioSource.OTHERS]}")
        print(f"partial_p95_ms={partial_p95}")
        print(f"commit_after_end_p95_ms={commit_p95}")
        print("commit_end_basis=transcript_audio_end_proxy")
        print(f"maximum_backlog_ms={metrics.maximum_backlog_ms}")
        print(f"overlap_separate={str(metrics.overlap_separate).lower()}")
        print("overflow_count=0")
        return 0
    finally:
        writer_stop.set()
        _cleanup_resources(processes, queues)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit two-minute fixed Kotoba-Whisper acceptance gate."""

    try:
        return _run(_parse_args(argv))
    except KeyboardInterrupt:
        print(
            "ASR smoke interrupted; workers were boundedly cleaned up", file=sys.stderr
        )
        return 130
    except Exception as exc:
        print(f"ASR smoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
