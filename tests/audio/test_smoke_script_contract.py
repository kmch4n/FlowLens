"""Hardware-free contracts for the designated-PC smoke entrypoints."""

from __future__ import annotations

import io
import pickle
import wave
from pathlib import Path

import pytest

from flowlens.asr.types import AsrWorkerConfig
from flowlens.asr.worker import run_asr_worker
from flowlens.audio.types import AudioWorkerConfig
from flowlens.audio.worker import run_audio_worker
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import MessageEnvelope
from flowlens.smoke.asr import (
    AsrSmokeMetrics,
    _nearest_rank_p95,
    _StopHandshake,
)
from flowlens.smoke.asr import (
    _parse_args as asr_args,
)
from flowlens.smoke.asr import (
    main as asr_main,
)
from flowlens.smoke.audio import (
    _cleanup_resources,
    _validate_wav,
    _validate_worker_stopped,
)
from flowlens.smoke.audio import (
    _parse_args as audio_args,
)
from flowlens.smoke.audio import (
    main as audio_main,
)
from flowlens.workers.writer import run_writer_worker


def _write_wav(path: Path, *, frames: int, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(bytes(frames * channels * 2))


def test_smoke_scripts_require_explicit_devices_and_local_model() -> None:
    audio = Path("scripts/smoke_audio.ps1").read_text(encoding="utf-8")
    asr = Path("scripts/smoke_asr.ps1").read_text(encoding="utf-8")
    assert "[Parameter(Mandatory = $true)][string]$MicrophoneId" in audio
    assert "[Parameter(Mandatory = $true)][string]$LoopbackOutputId" in audio
    assert "[Parameter(Mandatory = $true)][string]$ModelPath" in asr
    assert "[ValidateSet(60)][int]$DurationSeconds = 60" in audio
    assert "[ValidateSet(120)][int]$DurationSeconds = 120" in asr
    assert ".venv" in audio and "$LASTEXITCODE" in audio
    assert ".venv" in asr and "$LASTEXITCODE" in asr


def test_parsers_require_explicit_values_and_fixed_durations(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        audio_args([])
    with pytest.raises(SystemExit):
        audio_args(
            [
                "--microphone-id",
                "input:3",
                "--loopback-output-id",
                "wasapi-output:7",
                "--output-directory",
                str(tmp_path / "audio"),
                "--duration-seconds",
                "61",
            ]
        )
    with pytest.raises(SystemExit):
        asr_args(
            [
                "--microphone-id",
                "input:3",
                "--loopback-output-id",
                "wasapi-output:7",
                "--model-path",
                "kotoba-tech/kotoba-whisper-v2.0-faster",
                "--output-directory",
                str(tmp_path / "asr"),
                "--duration-seconds",
                "120",
            ]
        )


def test_asr_parser_accepts_only_existing_exact_local_model(tmp_path: Path) -> None:
    model = tmp_path / "kotoba-whisper-v2.0-faster"
    model.mkdir()
    parsed = asr_args(
        [
            "--microphone-id",
            "input:3",
            "--loopback-output-id",
            "wasapi-output:7",
            "--model-path",
            str(model),
            "--output-directory",
            str(tmp_path / "asr"),
            "--duration-seconds",
            "120",
        ]
    )
    assert parsed.model_path == model.resolve()


def test_nearest_rank_p95_uses_ceiling_rank() -> None:
    assert _nearest_rank_p95([1, 2, 3, 4, 100]) == 100
    assert _nearest_rank_p95(range(1, 101)) == 95
    with pytest.raises(ValueError, match="at least one observation"):
        _nearest_rank_p95([])


def test_wav_validation_enforces_format_nonempty_and_strict_duration(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.wav"
    lower = tmp_path / "lower.wav"
    stereo = tmp_path / "stereo.wav"
    _write_wav(valid, frames=960_000)
    _write_wav(lower, frames=955_200)
    _write_wav(stereo, frames=960_000, channels=2)
    assert _validate_wav(valid, 60) == 60_000
    with pytest.raises(ValueError, match="duration error"):
        _validate_wav(lower, 60)
    with pytest.raises(ValueError, match="mono"):
        _validate_wav(stereo, 60)


def test_asr_metrics_use_envelope_monotonic_times_without_fabrication() -> None:
    metrics = AsrSmokeMetrics(session_started_monotonic_ms=10_000)
    partial = MessageEnvelope(
        1,
        "01J00000000000000000000000",
        MessageType.TRANSCRIPT_PARTIAL,
        1,
        ProcessSource.ASR,
        11_500,
        {
            "source": "ME",
            "text": "確認",
            "session_start_ms": 100,
            "session_end_ms": 600,
            "source_start_sample": 0,
            "source_end_sample": 8_000,
        },
    )
    committed = MessageEnvelope(
        1,
        "01J00000000000000000000000",
        MessageType.TRANSCRIPT_COMMITTED,
        2,
        ProcessSource.ASR,
        12_000,
        {
            "schema_version": 1,
            "segment_id": "01J00000000000000000000001",
            "sequence": 1,
            "source": "ME",
            "text": "確認します。",
            "session_start_ms": 100,
            "session_end_ms": 1_000,
            "source_start_sample": 0,
            "source_end_sample": 16_000,
            "committed_at": "2026-08-19T12:00:02.000+09:00",
        },
    )
    metrics.observe(partial)
    metrics.observe(committed)
    assert metrics.partial_latencies_ms == [1_400]
    assert metrics.commit_after_end_latencies_ms == [1_000]
    assert metrics.committed_counts[AudioSource.ME] == 1


def test_partial_p95_counts_only_first_availability_per_utterance() -> None:
    metrics = AsrSmokeMetrics(session_started_monotonic_ms=10_000)
    for sequence, emitted_ms in enumerate(range(10_500, 20_501, 500), start=1):
        metrics.observe(
            MessageEnvelope(
                1,
                "01J00000000000000000000000",
                MessageType.TRANSCRIPT_PARTIAL,
                sequence,
                ProcessSource.ASR,
                emitted_ms,
                {
                    "source": "ME",
                    "text": f"更新{sequence}",
                    "session_start_ms": 0,
                    "session_end_ms": emitted_ms - 10_000,
                    "source_start_sample": 0,
                    "source_end_sample": (emitted_ms - 10_000) * 16,
                },
            )
        )
    metrics.observe(
        MessageEnvelope(
            1,
            "01J00000000000000000000000",
            MessageType.TRANSCRIPT_PARTIAL,
            22,
            ProcessSource.ASR,
            20_600,
            {
                "source": "ME",
                "text": "",
                "session_start_ms": 0,
                "session_end_ms": 10_600,
                "source_start_sample": 0,
                "source_end_sample": 169_600,
            },
        )
    )
    metrics.observe(
        MessageEnvelope(
            1,
            "01J00000000000000000000000",
            MessageType.TRANSCRIPT_PARTIAL,
            23,
            ProcessSource.ASR,
            21_500,
            {
                "source": "ME",
                "text": "次の発言",
                "session_start_ms": 11_000,
                "session_end_ms": 11_500,
                "source_start_sample": 176_000,
                "source_end_sample": 184_000,
            },
        )
    )
    assert metrics.partial_latencies_ms == [500, 500]
    assert _nearest_rank_p95(metrics.partial_latencies_ms) == 500
    assert metrics.partial_count == 22


def test_asr_metrics_require_authoritative_cumulative_maximum_backlog() -> None:
    metrics = AsrSmokeMetrics(session_started_monotonic_ms=10_000)
    missing = MessageEnvelope(
        1,
        "01J00000000000000000000000",
        MessageType.ASR_STATUS,
        1,
        ProcessSource.ASR,
        10_100,
        {"state": "RUNNING", "backlog_ms": 100, "analysis_paused": False},
    )
    with pytest.raises(ValueError, match="maximum_backlog_ms"):
        metrics.observe(missing)

    metrics.observe(
        MessageEnvelope(
            1,
            "01J00000000000000000000000",
            MessageType.ASR_STATUS,
            2,
            ProcessSource.ASR,
            10_200,
            {
                "state": "STOPPED",
                "backlog_ms": 0,
                "maximum_backlog_ms": 4_900,
                "analysis_paused": False,
            },
        )
    )
    assert metrics.maximum_backlog_ms == 4_900


class _Resource:
    def __init__(self) -> None:
        self.closed = 0
        self.cancelled = 0
        self.joined = 0

    def put(self, item: object) -> None:
        del item
        raise AssertionError("cleanup must not put queue items")

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        del block, timeout
        raise AssertionError("cleanup must not get queue items")

    def get_nowait(self) -> object:
        raise AssertionError("cleanup must not get queue items")

    def close(self) -> None:
        self.closed += 1

    def cancel_join_thread(self) -> None:
        self.cancelled += 1

    def join_thread(self) -> None:
        self.joined += 1


class _Process:
    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.terminated = 0
        self.killed = 0
        self.joined: list[float] = []

    @property
    def exitcode(self) -> int | None:
        return 0

    def start(self) -> None:
        raise AssertionError("cleanup must not start processes")

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated += 1
        self.alive = False

    def kill(self) -> None:
        self.killed += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.joined.append(0.0 if timeout is None else timeout)


class _UnstartedProcess(_Process):
    def join(self, timeout: float | None = None) -> None:
        raise AssertionError("can only join a started process")

    def is_alive(self) -> bool:
        raise AssertionError("can only test a started process")


def test_cleanup_terminates_live_children_and_closes_every_queue() -> None:
    processes = [
        _Process(alive=False),
        _Process(alive=True),
        _UnstartedProcess(alive=False),
    ]
    queues = [_Resource(), _Resource()]
    _cleanup_resources(processes, queues, join_timeout_seconds=0.25)
    assert processes[0].terminated == 0
    assert processes[1].terminated == 1
    assert all(process.joined for process in processes[:2])
    assert all(
        resource.cancelled == resource.closed == resource.joined == 1
        for resource in queues
    )


def test_production_spawn_targets_are_pickleable() -> None:
    output = io.BytesIO()
    pickle.dump((run_audio_worker, run_asr_worker, run_writer_worker), output)
    pickle.dump(
        (
            AudioWorkerConfig(
                "01J00000000000000000000000",
                "input:3",
                "wasapi-output:7",
                1_000,
                100,
                100,
            ),
            AsrWorkerConfig(
                "01J00000000000000000000000",
                Path("C:/models/kotoba-whisper-v2.0-faster"),
            ),
        ),
        output,
    )


def test_stop_handshake_rejects_asr_finalize_before_audio_drains() -> None:
    handshake = _StopHandshake("01J00000000000000000000000")
    audio_stop = handshake.audio_stop(1_000)
    with pytest.raises(RuntimeError, match="Audio drained"):
        handshake.asr_finalize(1_001)
    handshake.acknowledge_audio_stopped()
    asr_stop = handshake.asr_finalize(1_002)
    assert audio_stop.payload == {"worker": "AUDIO"}
    assert asr_stop.payload == {"worker": "ASR", "finalize": True}


@pytest.mark.parametrize(
    ("worker", "payload"),
    [
        (
            "AUDIO",
            {
                "worker": "AUDIO",
                "drained": False,
                "writer_frames": 1,
                "asr_frames": 1,
            },
        ),
        (
            "AUDIO",
            {
                "worker": "ASR",
                "drained": True,
                "writer_frames": 1,
                "asr_frames": 1,
            },
        ),
        (
            "AUDIO",
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": -1,
                "asr_frames": 1,
            },
        ),
        (
            "ASR",
            {"worker": "ASR", "drained": False, "committed_count": 1},
        ),
        (
            "ASR",
            {"worker": "AUDIO", "drained": True, "committed_count": 1},
        ),
        (
            "ASR",
            {"worker": "ASR", "drained": True, "committed_count": True},
        ),
    ],
)
def test_worker_stopped_validation_rejects_false_misaddressed_or_malformed(
    worker: str,
    payload: dict[str, object],
) -> None:
    envelope = MessageEnvelope(
        1,
        "01J00000000000000000000000",
        MessageType.WORKER_STOPPED,
        1,
        ProcessSource.AUDIO if worker == "AUDIO" else ProcessSource.ASR,
        1_000,
        payload,
    )
    with pytest.raises(ValueError, match="WORKER_STOPPED"):
        _validate_worker_stopped(envelope, worker)


@pytest.mark.parametrize(
    ("worker", "source", "payload"),
    [
        (
            "AUDIO",
            ProcessSource.AUDIO,
            {
                "worker": "AUDIO",
                "drained": True,
                "writer_frames": 6_000,
                "asr_frames": 6_000,
            },
        ),
        (
            "ASR",
            ProcessSource.ASR,
            {"worker": "ASR", "drained": True, "committed_count": 2},
        ),
    ],
)
def test_worker_stopped_validation_accepts_exact_drained_acknowledgement(
    worker: str,
    source: ProcessSource,
    payload: dict[str, object],
) -> None:
    envelope = MessageEnvelope(
        1,
        "01J00000000000000000000000",
        MessageType.WORKER_STOPPED,
        1,
        source,
        1_000,
        payload,
    )
    _validate_worker_stopped(envelope, worker)


def test_audio_main_returns_actionable_error_before_spawning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    result = audio_main(
        [
            "--microphone-id",
            "input:3",
            "--loopback-output-id",
            "wasapi-output:7",
            "--output-directory",
            str(output_file),
            "--duration-seconds",
            "60",
        ]
    )
    assert result == 1
    assert "output path is not a directory" in capsys.readouterr().err


def test_asr_main_returns_actionable_error_before_spawning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "kotoba-whisper-v2.0-faster"
    model.mkdir()
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    result = asr_main(
        [
            "--microphone-id",
            "input:3",
            "--loopback-output-id",
            "wasapi-output:7",
            "--model-path",
            str(model),
            "--output-directory",
            str(output_file),
            "--duration-seconds",
            "120",
        ]
    )
    assert result == 1
    assert "output path is not a directory" in capsys.readouterr().err
