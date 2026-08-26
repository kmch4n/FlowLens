"""Deterministic acceptance metric and report tests."""

import json
from pathlib import Path

from scripts.collect_acceptance import (
    AcceptanceMetrics,
    _acceptance_artifact_errors,
    collect_acceptance,
    evaluate_acceptance,
    nearest_rank_p95,
)
from scripts.validate_session import SessionValidationResult


def make_metrics(**changes: object) -> AcceptanceMetrics:
    values: dict[str, object] = {
        "partial_p95_ms": 1_500,
        "commit_p95_ms": 2_500,
        "discussion_p95_ms": 4_500,
        "max_ui_feedback_ms": 90,
        "queue_overflows": 0,
        "wav_error_percent": 0.4,
        "memory_growth_mb": 400.0,
        "gpu_oom_count": 0,
        "network_blocked": True,
        "artifacts_valid": True,
        "discussion_enabled": True,
    }
    values.update(changes)
    return AcceptanceMetrics(**values)  # type: ignore[arg-type]


def test_acceptance_thresholds_are_exact() -> None:
    metrics = make_metrics(
        partial_p95_ms=2_000,
        commit_p95_ms=3_000,
        discussion_p95_ms=5_000,
        max_ui_feedback_ms=100,
        queue_overflows=0,
        wav_error_percent=0.49,
        memory_growth_mb=499,
        gpu_oom_count=0,
        network_blocked=True,
    )
    assert evaluate_acceptance(metrics).errors == ()


def test_acceptance_rejects_analysis_that_pushes_commit_latency_over_limit() -> None:
    metrics = make_metrics(commit_p95_ms=3_001, discussion_enabled=True)
    assert "Committed ASR p95 exceeds 3000 ms" in evaluate_acceptance(metrics).errors


def test_acceptance_rejects_missing_measurements_and_exact_wav_boundary() -> None:
    metrics = make_metrics(partial_p95_ms=None, wav_error_percent=0.5)
    result = evaluate_acceptance(metrics)
    assert "Partial ASR p95 is missing" in result.errors
    assert "WAV duration error must be below 0.5 percent" in result.errors


def test_acceptance_never_treats_missing_discussion_evidence_as_disabled() -> None:
    metrics = make_metrics(discussion_p95_ms=None, discussion_enabled=False)
    result = evaluate_acceptance(metrics)
    assert "Discussion p95 is missing" in result.errors


def test_nearest_rank_p95_is_deterministic() -> None:
    assert nearest_rank_p95([100, 500, 200, 300, 400]) == 500
    assert nearest_rank_p95([]) is None


def test_acceptance_artifacts_require_both_sources_and_discussion_update() -> None:
    validation = SessionValidationResult(
        errors=(),
        sources=("ME",),
        final_revision=0,
    )
    assert _acceptance_artifact_errors(validation) == (
        "Committed transcript must contain ME and OTHERS",
        "Discussion state must advance at least once",
    )


def test_collector_uses_event_samples_and_offline_evidence(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "events.jsonl").write_text("", encoding="utf-8", newline="\n")
    application_report = tmp_path / "application.json"
    application_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exit_code": 0,
                "controller": {
                    "state": "COMPLETED",
                    "completion_available": True,
                    "latencies_ms": {
                        "partial": [1_000],
                        "commit": [2_000],
                        "discussion": [4_000],
                        "ui_feedback": [80],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"elapsed_seconds": 300, "rss_bytes": 100_000_000, "gpu_oom": False})
        + "\n"
        + json.dumps(
            {"elapsed_seconds": 1800, "rss_bytes": 200_000_000, "gpu_oom": False}
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    evidence = tmp_path / "firewall.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "program": "C:/FlowLens/FlowLens.exe",
                "rule_name": "FlowLens-Acceptance-123",
                "outbound_blocked": True,
                "active_throughout": True,
            }
        ),
        encoding="utf-8",
    )
    metrics = collect_acceptance(
        session=session,
        samples_path=samples,
        offline_evidence_path=evidence,
        application_report_path=application_report,
        artifact_errors=(),
        wav_error_percent=0.4,
        queue_overflows=0,
    )
    assert metrics.partial_p95_ms == 1_000
    assert metrics.memory_growth_mb == 100_000_000 / (1024 * 1024)
    assert evaluate_acceptance(metrics).errors == ()


def test_collector_rejects_samples_that_miss_measurement_boundaries(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    events = [
        {"event_type": "PARTIAL_AVAILABLE", "details": {"latency_ms": 1_000}},
        {"event_type": "TRANSCRIPT_COMMITTED", "details": {"latency_ms": 2_000}},
        {"event_type": "DISCUSSION_REPLACED", "details": {"latency_ms": 4_000}},
        {"event_type": "UI_FEEDBACK", "details": {"latency_ms": 80}},
    ]
    (session / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
        newline="\n",
    )
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"elapsed_seconds": 600, "rss_bytes": 100_000_000, "gpu_oom": False})
        + "\n"
        + json.dumps(
            {"elapsed_seconds": 2000, "rss_bytes": 200_000_000, "gpu_oom": False}
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    evidence = tmp_path / "firewall.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "program": "C:/FlowLens/FlowLens.exe",
                "rule_name": "FlowLens-Acceptance-123",
                "outbound_blocked": True,
                "active_throughout": True,
            }
        ),
        encoding="utf-8",
    )
    metrics = collect_acceptance(
        session=session,
        samples_path=samples,
        offline_evidence_path=evidence,
        artifact_errors=(),
        wav_error_percent=0.4,
        queue_overflows=0,
    )
    assert metrics.memory_growth_mb is None
    assert "Minute-5-to-minute-30 memory growth is missing" in (
        evaluate_acceptance(metrics).errors
    )
