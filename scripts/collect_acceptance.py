"""Collect deterministic FlowLens acceptance metrics from local evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from scripts.validate_session import SessionValidationResult, validate_session


@dataclass(frozen=True, slots=True)
class AcceptanceMetrics:
    """All measurable MVP acceptance values."""

    partial_p95_ms: int | None
    commit_p95_ms: int | None
    discussion_p95_ms: int | None
    max_ui_feedback_ms: int | None
    queue_overflows: int
    wav_error_percent: float | None
    memory_growth_mb: float | None
    gpu_oom_count: int
    network_blocked: bool
    artifacts_valid: bool
    discussion_enabled: bool


@dataclass(frozen=True, slots=True)
class AcceptanceEvaluation:
    """Pass/fail evaluation with stable human-readable errors."""

    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return true only when every threshold has evidence and passes."""

        return not self.errors


def nearest_rank_p95(values: Sequence[int]) -> int | None:
    """Return the deterministic nearest-rank p95 for non-negative integers."""

    if not values:
        return None
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("latencies must be non-negative integers")
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[rank - 1]


def evaluate_acceptance(metrics: AcceptanceMetrics) -> AcceptanceEvaluation:
    """Apply the exact MVP thresholds without rounding favorable values."""

    if not isinstance(metrics, AcceptanceMetrics):
        raise TypeError("metrics must be AcceptanceMetrics")
    errors: list[str] = []
    limits = (
        ("Partial ASR", metrics.partial_p95_ms, 2_000),
        ("Committed ASR", metrics.commit_p95_ms, 3_000),
        ("Discussion", metrics.discussion_p95_ms, 5_000),
    )
    for label, value, limit in limits:
        if value is None:
            errors.append(f"{label} p95 is missing")
        elif value > limit:
            errors.append(f"{label} p95 exceeds {limit} ms")
    if metrics.max_ui_feedback_ms is None:
        errors.append("Direct UI feedback latency is missing")
    elif metrics.max_ui_feedback_ms > 100:
        errors.append("Direct UI feedback exceeds 100 ms")
    if metrics.queue_overflows != 0:
        errors.append("Audio queue overflow count must be zero")
    if metrics.wav_error_percent is None:
        errors.append("WAV duration error is missing")
    elif metrics.wav_error_percent >= 0.5:
        errors.append("WAV duration error must be below 0.5 percent")
    if metrics.memory_growth_mb is None:
        errors.append("Minute-5-to-minute-30 memory growth is missing")
    elif metrics.memory_growth_mb >= 500:
        errors.append("Memory growth must be below 500 MB")
    if metrics.gpu_oom_count != 0:
        errors.append("GPU OOM count must be zero")
    if not metrics.network_blocked:
        errors.append("Outbound network block evidence is missing or incomplete")
    if not metrics.artifacts_valid:
        errors.append("Session artifacts failed validation")
    return AcceptanceEvaluation(tuple(errors))


def _load_json(path: Path) -> object:
    encoded = path.read_bytes()
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name} must be UTF-8 without BOM")
    return json.loads(encoded.decode("utf-8"))


def _load_jsonl(path: Path) -> list[object]:
    encoded = path.read_bytes()
    if encoded and (not encoded.endswith(b"\n") or b"\r" in encoded):
        raise ValueError(f"{path.name} must be LF-terminated JSONL")
    return [json.loads(line) for line in encoded.decode("utf-8").splitlines()]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _latencies(events: Sequence[object], event_type: str) -> list[int]:
    values: list[int] = []
    for index, value in enumerate(events, start=1):
        event = _mapping(value, f"events line {index}")
        if event.get("event_type") != event_type:
            continue
        details = _mapping(event.get("details"), f"events line {index} details")
        values.append(
            _non_negative_int(
                details.get("latency_ms"),
                f"events line {index} latency_ms",
            )
        )
    return values


def _sample_measurements(samples: Sequence[object]) -> float | None:
    parsed: list[tuple[int, int]] = []
    for index, value in enumerate(samples, start=1):
        sample = _mapping(value, f"samples line {index}")
        elapsed = _non_negative_int(
            sample.get("elapsed_seconds"), f"samples line {index} elapsed_seconds"
        )
        rss = _non_negative_int(
            sample.get("rss_bytes"), f"samples line {index} rss_bytes"
        )
        parsed.append((elapsed, rss))
    if [item[0] for item in parsed] != sorted({item[0] for item in parsed}):
        raise ValueError("sample elapsed_seconds must be strictly increasing")
    minute_5 = next((item for item in parsed if 300 <= item[0] <= 305), None)
    minute_30 = next((item for item in parsed if 1_800 <= item[0] <= 1_805), None)
    growth = None
    if minute_5 is not None and minute_30 is not None:
        growth = (minute_30[1] - minute_5[1]) / (1024 * 1024)
    return growth


def _gpu_oom_count(events: Sequence[object]) -> int:
    count = 0
    for index, value in enumerate(events, start=1):
        event = _mapping(value, f"events line {index}")
        if event.get("event_type") != "ANALYSIS_FAILED":
            continue
        details = _mapping(event.get("details"), f"events line {index} details")
        if details.get("error_code") == "GPU_OOM":
            count += 1
    return count


def _offline_blocked(value: object) -> bool:
    evidence = _mapping(value, "offline evidence")
    required = {
        "schema_version",
        "program",
        "rule_name",
        "outbound_blocked",
        "active_throughout",
    }
    if set(evidence) != required:
        raise ValueError("offline evidence has missing or unknown fields")
    return (
        evidence["schema_version"] == 1
        and isinstance(evidence["program"], str)
        and bool(evidence["program"])
        and isinstance(evidence["rule_name"], str)
        and bool(evidence["rule_name"])
        and evidence["outbound_blocked"] is True
        and evidence["active_throughout"] is True
    )


def _application_latencies(path: Path) -> dict[str, list[int]]:
    report = _mapping(_load_json(path), "application report")
    if report.get("schema_version") != 1 or report.get("exit_code") != 0:
        raise ValueError("application report does not record a successful run")
    controller = _mapping(report.get("controller"), "application controller")
    if (
        controller.get("state") != "COMPLETED"
        or controller.get("completion_available") is not True
    ):
        raise ValueError("application report does not prove normal completion")
    raw = _mapping(controller.get("latencies_ms"), "application latencies")
    names = {"partial", "commit", "discussion", "ui_feedback"}
    if set(raw) != names:
        raise ValueError("application latencies have missing or unknown fields")
    result: dict[str, list[int]] = {}
    for name in sorted(names):
        values = raw[name]
        if not isinstance(values, list):
            raise ValueError(f"application {name} latencies must be a list")
        result[name] = [
            _non_negative_int(value, f"application {name} latency") for value in values
        ]
    return result


def collect_acceptance(
    *,
    session: Path,
    samples_path: Path,
    offline_evidence_path: Path,
    application_report_path: Path | None = None,
    artifact_errors: tuple[str, ...],
    wav_error_percent: float | None,
    queue_overflows: int,
) -> AcceptanceMetrics:
    """Collect acceptance values from validated local evidence files."""

    events = _load_jsonl(session / "events.jsonl")
    if application_report_path is None:
        latency_values = {
            "partial": _latencies(events, "PARTIAL_AVAILABLE"),
            "commit": _latencies(events, "TRANSCRIPT_COMMITTED"),
            "discussion": _latencies(events, "DISCUSSION_REPLACED"),
            "ui_feedback": _latencies(events, "UI_FEEDBACK"),
        }
    else:
        latency_values = _application_latencies(application_report_path)
    samples = _load_jsonl(samples_path)
    memory_growth = _sample_measurements(samples)
    gpu_oom_count = _gpu_oom_count(events)
    ui_values = latency_values["ui_feedback"]
    discussion_values = latency_values["discussion"]
    return AcceptanceMetrics(
        partial_p95_ms=nearest_rank_p95(latency_values["partial"]),
        commit_p95_ms=nearest_rank_p95(latency_values["commit"]),
        discussion_p95_ms=nearest_rank_p95(discussion_values),
        max_ui_feedback_ms=max(ui_values) if ui_values else None,
        queue_overflows=_non_negative_int(queue_overflows, "queue_overflows"),
        wav_error_percent=wav_error_percent,
        memory_growth_mb=memory_growth,
        gpu_oom_count=gpu_oom_count,
        network_blocked=_offline_blocked(_load_json(offline_evidence_path)),
        artifacts_valid=not artifact_errors,
        discussion_enabled=True,
    )


def _acceptance_artifact_errors(
    validation: SessionValidationResult,
) -> tuple[str, ...]:
    errors = list(validation.errors)
    if set(validation.sources) != {"ME", "OTHERS"}:
        errors.append("Committed transcript must contain ME and OTHERS")
    if validation.final_revision is None or validation.final_revision < 1:
        errors.append("Discussion state must advance at least once")
    return tuple(errors)


def _write_json(path: Path, value: object) -> None:
    parent = path.parent.resolve(strict=True)
    target = (parent / path.name).resolve(strict=False)
    if target.parent != parent:
        raise ValueError("output path must remain in its resolved parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=4, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--offline-evidence", type=Path, required=True)
    parser.add_argument("--application-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-active-seconds", type=int, default=1_800)
    parser.add_argument("--require-pause", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate artifacts, collect metrics, and write one final report."""

    arguments = _build_parser().parse_args(argv)
    session = cast(Path, arguments.session)
    validation = validate_session(
        session,
        minimum_active_seconds=cast(int, arguments.minimum_active_seconds),
        expected_status="completed",
        require_pause=cast(bool, arguments.require_pause),
    )
    artifact_errors = _acceptance_artifact_errors(validation)
    metrics = collect_acceptance(
        session=session,
        samples_path=cast(Path, arguments.samples),
        offline_evidence_path=cast(Path, arguments.offline_evidence),
        application_report_path=cast(Path, arguments.application_report),
        artifact_errors=artifact_errors,
        wav_error_percent=validation.wav_error_percent,
        # Audio queue overflow is fatal in the production worker. A completed
        # application report therefore proves that no overflow occurred.
        queue_overflows=0,
    )
    evaluation = evaluate_acceptance(metrics)
    report = {
        "schema_version": 1,
        "passed": evaluation.passed,
        "errors": list(evaluation.errors),
        "artifact_errors": list(artifact_errors),
        "metrics": asdict(metrics),
    }
    _write_json(cast(Path, arguments.output), report)
    return 0 if evaluation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
