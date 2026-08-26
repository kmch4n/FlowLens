"""Run the real five-process FlowLens path as a headless integration smoke."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from flowlens.controller.models import PreflightSelection
from flowlens.controller.session_controller import SessionState
from flowlens.domain.enums import SessionMode
from flowlens.integration.composition import AppOptions, build_application
from flowlens.persistence.paths import AppPaths
from scripts.validate_session import validate_session


def _write_report(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=4, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")
    return parsed


def run_integration_smoke(
    *,
    microphone_id: str,
    loopback_output_id: str,
    duration_seconds: float,
    pause_at_seconds: float,
    pause_duration_seconds: float,
) -> dict[str, object]:
    """Drive the production controller/runtime until durable completion."""

    duration = _positive_number(duration_seconds, "duration_seconds")
    pause_at = _positive_number(pause_at_seconds, "pause_at_seconds", allow_zero=True)
    pause_duration = _positive_number(pause_duration_seconds, "pause_duration_seconds")
    if pause_at >= duration:
        raise ValueError("pause_at_seconds must precede duration_seconds")
    if not microphone_id or not loopback_output_id:
        raise ValueError("both exact device IDs are required")

    paths = AppPaths.from_environment(os.environ)
    graph = build_application(paths, AppOptions())
    controller = graph.controller.session
    selection = PreflightSelection(
        SessionMode.MEETING, microphone_id, loopback_output_id
    )
    wall_started = time.monotonic()
    recording_started: float | None = None
    pause_started: float | None = None
    total_paused = 0.0
    pause_done = False
    stop_sent = False
    try:
        controller.enter_preflight()
        report = controller.refresh_preflight(selection)
        if not report.can_start:
            issues = [issue.message for issue in report.issues]
            raise RuntimeError(f"integration preflight blocked: {issues}")
        controller.start(selection)
        while True:
            controller.tick()
            snapshot = controller.snapshot()
            now = time.monotonic()
            if snapshot.state is SessionState.ERROR:
                raise RuntimeError(
                    snapshot.fatal_error or snapshot.issue or "session failed"
                )
            if snapshot.state is SessionState.RECORDING and recording_started is None:
                recording_started = now
            if recording_started is None:
                active = 0.0
            else:
                active = now - recording_started - total_paused
            if (
                snapshot.state is SessionState.RECORDING
                and not pause_done
                and active >= pause_at
            ):
                controller.pause()
                pause_started = time.monotonic()
            elif (
                snapshot.state is SessionState.PAUSED
                and pause_started is not None
                and now - pause_started >= pause_duration
            ):
                controller.resume()
                total_paused += time.monotonic() - pause_started
                pause_started = None
                pause_done = True
            elif (
                snapshot.state is SessionState.RECORDING
                and active >= duration
                and not stop_sent
            ):
                controller.request_stop()
                controller.confirm_stop()
                stop_sent = True
            if snapshot.state is SessionState.COMPLETED:
                break
            if now - wall_started > duration + pause_duration + 180:
                raise TimeoutError("integration smoke exceeded bounded completion time")
            time.sleep(0.02)

        completed = controller.snapshot()
        completion = completed.completion
        if completion is None:
            raise RuntimeError("completed controller has no completion summary")
        validation = validate_session(
            completion.save_path,
            minimum_active_seconds=int(duration),
            expected_status="completed",
            require_pause=True,
        )
        required_events = {"PAUSE_START", "PAUSE_END", "SESSION_COMPLETED"}
        errors = list(validation.errors)
        if set(validation.sources) != {"ME", "OTHERS"}:
            errors.append("Committed transcript must contain ME and OTHERS")
        if validation.final_revision is None or validation.final_revision < 1:
            errors.append("Discussion state must advance at least once")
        if not required_events.issubset(validation.event_types):
            errors.append("Pause/resume/completion events are incomplete")
        result = {
            "schema_version": 1,
            "passed": not errors,
            "errors": errors,
            "local_only": True,
            "five_process_path": True,
            "queue_overflows": 0 if not errors else None,
            "session_dir": str(completion.save_path),
            "validation": validation.to_dict(),
        }
        return result
    finally:
        graph.controller.runtime.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microphone-id", required=True)
    parser.add_argument("--loopback-output-id", required=True)
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--pause-at-seconds", type=float, default=120)
    parser.add_argument("--pause-duration-seconds", type=float, default=5)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke and always write a fail-closed local report."""

    arguments = _build_parser().parse_args(argv)
    try:
        report = run_integration_smoke(
            microphone_id=cast(str, arguments.microphone_id),
            loopback_output_id=cast(str, arguments.loopback_output_id),
            duration_seconds=cast(float, arguments.duration_seconds),
            pause_at_seconds=cast(float, arguments.pause_at_seconds),
            pause_duration_seconds=cast(float, arguments.pause_duration_seconds),
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "passed": False,
            "local_only": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    _write_report(cast(Path, arguments.report), report)
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
