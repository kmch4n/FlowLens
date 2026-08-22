"""Spawn-boundary tests for the single-owner Writer Worker."""

import json
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
)
from flowlens.workers.writer import run_writer_worker
from tests.workers.writer_support import (
    assert_ack,
    control_envelope,
    make_audio_command,
    make_finalize_envelope,
    make_open_envelope,
    make_shutdown_envelope,
    make_transcript_envelope,
)

_CROSS_QUEUE_STRESS_ITERATIONS = 100
_CROSS_QUEUE_STRESS_SAMPLES = 32_768


def _stress_audio_command(index: int) -> AudioWriteCommand:
    start_sample = index * _CROSS_QUEUE_STRESS_SAMPLES
    end_sample = start_sample + _CROSS_QUEUE_STRESS_SAMPLES
    return AudioWriteCommand(
        AudioSource.ME,
        b"\x00\x00" * _CROSS_QUEUE_STRESS_SAMPLES,
        start_sample,
        end_sample,
        index * 100,
        1_000 + index * 100,
    )


def _stress_transcript_envelope(index: int) -> MessageEnvelope[object]:
    sequence = index + 1
    start_sample = index * _CROSS_QUEUE_STRESS_SAMPLES
    end_sample = start_sample + _CROSS_QUEUE_STRESS_SAMPLES
    record = TranscriptRecord(
        1,
        f"01J{sequence:023d}",
        sequence,
        AudioSource.ME,
        f"stress transcript {sequence}",
        index * 100,
        index * 100 + 100,
        start_sample,
        end_sample,
        datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
    )
    return control_envelope(
        sequence + 1,
        MessageType.TRANSCRIPT_COMMITTED,
        TranscriptCommitted(record),
    )


def test_writer_process_opens_appends_and_finalizes(tmp_path: Path) -> None:
    """The worker must persist ordered work and terminate after shutdown."""

    context = get_context("spawn")
    control = context.Queue()
    audio = context.Queue()
    responses = context.Queue()
    stop = context.Event()
    process = context.Process(
        target=run_writer_worker,
        args=(control, audio, responses, stop),
    )
    process.start()
    try:
        control.put(make_open_envelope(tmp_path / "session", sequence=1))
        assert_ack(responses, acknowledged_sequence=1)
        audio.put(make_audio_command(source_start_sample=0))
        control.put(make_transcript_envelope(sequence=2))
        assert_ack(responses, acknowledged_sequence=2)
        audio.put(AudioDrainFence())
        control.put(make_finalize_envelope(sequence=3))
        assert_ack(responses, acknowledged_sequence=3)
        control.put(make_shutdown_envelope(sequence=4))
        assert_ack(responses, acknowledged_sequence=4)
        process.join(timeout=10)
        assert process.exitcode == 0
        manifest = json.loads(
            (tmp_path / "session" / "session.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_writer_spawn_preserves_cross_queue_audio_dependency_under_stress(
    tmp_path: Path,
) -> None:
    """Independent queue feeders must not reorder transcript persistence."""

    context = get_context("spawn")
    control = context.Queue()
    audio = context.Queue()
    responses = context.Queue()
    stop = context.Event()
    session_dir = tmp_path / "session"
    process = context.Process(
        target=run_writer_worker,
        args=(control, audio, responses, stop),
    )
    process.start()
    try:
        control.put(make_open_envelope(session_dir, sequence=1))
        assert_ack(responses, acknowledged_sequence=1)
        for index in range(_CROSS_QUEUE_STRESS_ITERATIONS):
            audio.put(_stress_audio_command(index))
            control.put(_stress_transcript_envelope(index))
            assert_ack(responses, acknowledged_sequence=index + 2)
        audio.put(AudioDrainFence())
        shutdown_sequence = _CROSS_QUEUE_STRESS_ITERATIONS + 2
        control.put(make_shutdown_envelope(sequence=shutdown_sequence))
        assert_ack(responses, acknowledged_sequence=shutdown_sequence)
        process.join(timeout=10)
        assert process.exitcode == 0
        transcript_lines = (
            (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(transcript_lines) == _CROSS_QUEUE_STRESS_ITERATIONS
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for queue in (control, audio, responses):
            queue.close()
            queue.join_thread()


def test_writer_reports_fatal_and_leaves_incomplete_on_audio_gap(
    tmp_path: Path,
) -> None:
    """An audio cursor gap must fail closed without completing the session."""

    context = get_context("spawn")
    control = context.Queue()
    audio = context.Queue()
    responses = context.Queue()
    stop = context.Event()
    session_dir = tmp_path / "session"
    process = context.Process(
        target=run_writer_worker,
        args=(control, audio, responses, stop),
    )
    process.start()
    try:
        control.put(make_open_envelope(session_dir, sequence=1))
        assert_ack(responses, 1)
        audio.put(make_audio_command(source_start_sample=640))
        fatal = responses.get(timeout=5)
        assert isinstance(fatal, MessageEnvelope)
        assert fatal.message_type is MessageType.WRITER_FATAL
        assert fatal.source is ProcessSource.WRITER
        assert fatal.payload.failed_sequence == 0
        process.join(timeout=10)
        assert process.exitcode is not None
        assert process.exitcode != 0
        manifest = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "incomplete"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
