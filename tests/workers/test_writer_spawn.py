"""Spawn-boundary tests for the single-owner Writer Worker."""

import json
import random
import time
from dataclasses import replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

from flowlens.domain.enums import AudioSource, EventType, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    WriterForceCloseOutcome,
    WriterForceCloseRequest,
    WriterForceCloseResult,
)
from flowlens.workers.finalization_gate import WriterFinalizationGate
from flowlens.workers.writer import run_writer_worker
from tests.factories import make_event_record
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
    gate = WriterFinalizationGate.create(context)
    process = context.Process(
        target=run_writer_worker,
        args=(control, audio, responses, stop, gate),
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
        terminal_events = [
            json.loads(line)
            for line in (tmp_path / "session" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [
            (event["sequence"], event["event_type"]) for event in terminal_events
        ] == [(1, EventType.SESSION_COMPLETED.value)]
        result = gate.result()
        assert result is not None
        assert result.outcome is WriterForceCloseOutcome.COMPLETED
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_writer_spawn_force_close_persists_evidence_and_stays_incomplete(
    tmp_path: Path,
) -> None:
    """The spawned Writer must honor the out-of-band pre-finalize signal."""

    context = get_context("spawn")
    control = context.Queue()
    audio = context.Queue()
    responses = context.Queue()
    stop = context.Event()
    gate = WriterFinalizationGate.create(context)
    session_dir = tmp_path / "session-force"
    process = context.Process(
        target=run_writer_worker,
        args=(
            control,
            audio,
            responses,
            stop,
            gate,
        ),
    )
    process.start()
    try:
        control.put(make_open_envelope(session_dir, sequence=1))
        assert_ack(responses, acknowledged_sequence=1)
        audio.put(AudioDrainFence())
        gate.request_force_close(
            WriterForceCloseRequest(
                replace(
                    make_event_record(sequence=1),
                    event_type=EventType.FORCE_CLOSE_REQUESTED,
                )
            ),
            timeout_seconds=0.1,
        )
        control.put(make_finalize_envelope(sequence=2))
        result_envelope = responses.get(timeout=5)
        assert isinstance(result_envelope, MessageEnvelope)
        assert result_envelope.message_type is MessageType.WRITER_FORCE_CLOSE_RESULT
        assert isinstance(result_envelope.payload, WriterForceCloseResult)
        assert result_envelope.payload.outcome is WriterForceCloseOutcome.INCOMPLETE
        process.join(timeout=10)
        assert process.exitcode == 0
        manifest = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        events = [
            json.loads(line)
            for line in (session_dir / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert manifest["status"] == "incomplete"
        assert [event["event_type"] for event in events] == [
            EventType.FORCE_CLOSE_REQUESTED.value
        ]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for queue in (control, audio, responses):
            queue.close()
            queue.join_thread()


def test_real_writer_spawn_randomized_terminal_races_keep_one_sequence(
    tmp_path: Path,
) -> None:
    """Real spawned persistence must publish exactly one terminal candidate."""

    generator = random.Random(37)
    context = get_context("spawn")
    for iteration in range(12):
        control = context.Queue()
        audio = context.Queue()
        responses = context.Queue()
        stop = context.Event()
        gate = WriterFinalizationGate.create(context)
        session_dir = tmp_path / f"race-{iteration}"
        process = context.Process(
            target=run_writer_worker,
            args=(control, audio, responses, stop, gate),
        )
        process.start()
        try:
            control.put(make_open_envelope(session_dir, sequence=1))
            assert_ack(responses, acknowledged_sequence=1)
            audio.put(AudioDrainFence())
            request = WriterForceCloseRequest(
                replace(
                    make_event_record(sequence=1),
                    event_type=EventType.FORCE_CLOSE_REQUESTED,
                )
            )
            if generator.choice((True, False)):
                gate.request_force_close(request, timeout_seconds=0.1)
                control.put(make_finalize_envelope(sequence=2))
            else:
                control.put(make_finalize_envelope(sequence=2))
                time.sleep(generator.random() / 1_000)
                gate.request_force_close(request, timeout_seconds=0.1)

            response = responses.get(timeout=5)
            assert isinstance(response, MessageEnvelope)
            if response.message_type is MessageType.WRITER_ACK:
                control.put(make_shutdown_envelope(sequence=3))
                assert_ack(responses, acknowledged_sequence=3)
            else:
                assert response.message_type is MessageType.WRITER_FORCE_CLOSE_RESULT
            process.join(timeout=10)
            assert process.exitcode == 0

            manifest = json.loads(
                (session_dir / "session.json").read_text(encoding="utf-8")
            )
            terminal_events = [
                json.loads(line)
                for line in (session_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert len(terminal_events) == 1
            assert terminal_events[0]["sequence"] == 1
            result = gate.result()
            assert result is not None
            expected_status = (
                "completed"
                if result.outcome is WriterForceCloseOutcome.COMPLETED
                else "incomplete"
            )
            assert manifest["status"] == expected_status
            expected_event = (
                EventType.SESSION_COMPLETED
                if result.outcome is WriterForceCloseOutcome.COMPLETED
                else EventType.FORCE_CLOSE_REQUESTED
            )
            assert terminal_events[0]["event_type"] == expected_event.value
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            for queue in (control, audio, responses):
                queue.close()
                queue.join_thread()


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
