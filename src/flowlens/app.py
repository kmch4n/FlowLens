"""Command-line entry for the FlowLens desktop application."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from flowlens.config.store import ConfigStore
from flowlens.integration.composition import AppOptions, build_application
from flowlens.persistence.paths import AppPaths
from flowlens.persistence.recovery import recover_incomplete_sessions

_SELF_CHECK_MODULES = (
    "PySide6.QtCore",
    "numpy",
    "pyaudiowpatch",
    "ctranslate2",
    "llama_cpp",
)
_ACCEPTANCE_SCHEMA_VERSION = 1
_PACKAGE_SELF_CHECK_APPLICATION: object | None = None


class PackageSelfCheckReport(TypedDict):
    """Package-self-check result recorded without native dependency details."""

    requested: bool
    exit_code: int | None


class ControllerAcceptanceSnapshot(TypedDict):
    """Privacy-safe final controller measurements available to later tasks."""

    state: str
    recording_status: str
    completion_available: bool
    transcript_count: int
    asr_backlog_ms: int
    maximum_asr_backlog_ms: int
    latencies_ms: dict[str, list[int]]


class AcceptanceReport(TypedDict, total=False):
    """Extensible local-only acceptance record with no pass/fail claims."""

    schema_version: int
    started_at: str
    ended_at: str
    elapsed_ms: int
    exit_code: int
    local_only: bool
    package_self_check: PackageSelfCheckReport
    controller: ControllerAcceptanceSnapshot
    error_type: str


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for an acceptance measurement boundary."""

    return datetime.now(UTC)


def _monotonic_seconds() -> float:
    """Return the local monotonic clock used only for elapsed measurements."""

    return time.monotonic()


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI parsing, package self-check, or the normal Qt application."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    acceptance_report = cast(str | None, arguments.acceptance_report)
    try:
        report_path = (
            None
            if acceptance_report is None
            else _normalize_acceptance_report_path(acceptance_report)
        )
    except (OSError, ValueError) as error:
        print(
            f"Acceptance report path is unsafe: {_error_name(error)}",
            file=sys.stderr,
        )
        return 2

    started_at = _utc_now()
    started_monotonic = _monotonic_seconds()
    package_self_check = cast(bool, arguments.package_self_check)
    final_snapshot: object | None = None
    error_type: str | None = None

    def capture_snapshot(snapshot: object) -> None:
        nonlocal final_snapshot
        final_snapshot = snapshot

    try:
        if package_self_check:
            exit_code = _package_self_check()
        else:
            paths = AppPaths.from_environment(os.environ)
            options = AppOptions(acceptance_report=report_path)
            exit_code = _run_qt(paths, options, capture_snapshot)
    except Exception as error:
        if report_path is None:
            raise
        error_type = _error_name(error)
        print(f"FlowLens application failed: {error_type}", file=sys.stderr)
        exit_code = 1

    ended_at = _utc_now()
    elapsed_ms = max(
        0,
        round((_monotonic_seconds() - started_monotonic) * 1_000),
    )
    if report_path is None:
        return exit_code
    report = _acceptance_report(
        started_at=started_at,
        ended_at=ended_at,
        elapsed_ms=elapsed_ms,
        exit_code=exit_code,
        package_self_check=package_self_check,
        final_snapshot=final_snapshot,
        error_type=error_type,
    )
    try:
        _write_acceptance_report(report_path, report)
    except (OSError, ValueError) as error:
        print(f"Acceptance report write failed: {_error_name(error)}", file=sys.stderr)
        return exit_code if exit_code != 0 else 1
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowlens",
        description="Run the local FlowLens desktop application.",
    )
    parser.add_argument(
        "--package-self-check",
        action="store_true",
        help="Load packaged native dependencies without workers or models.",
    )
    parser.add_argument(
        "--acceptance-report",
        metavar="PATH",
        help="Record local acceptance measurements while preserving session semantics.",
    )
    return parser


def _package_self_check() -> int:
    """Load packaged runtime dependencies without starting workers or models."""

    failed: list[str] = []
    for module_name in _SELF_CHECK_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            failed.append(f"{module_name}: {type(error).__name__}")
    try:
        _load_package_qt_platform()
    except Exception as error:
        failed.append(f"PySide6.QtGui platform: {type(error).__name__}")
    if failed:
        for item in failed:
            print(item, file=sys.stderr)
        return 1
    return 0


def _load_package_qt_platform() -> None:
    """Load and retain the Qt platform plugin needed by packaged UI resources."""

    global _PACKAGE_SELF_CHECK_APPLICATION

    qt_gui = importlib.import_module("PySide6.QtGui")
    application_type = qt_gui.QGuiApplication
    application = application_type.instance()
    if application is None:
        application = application_type([])
    platform_name = application.platformName()
    if not isinstance(platform_name, str) or not platform_name:
        raise RuntimeError("Qt platform plugin did not initialize")
    _PACKAGE_SELF_CHECK_APPLICATION = application


def _run_qt(
    paths: AppPaths,
    options: AppOptions,
    snapshot_callback: Callable[[object], None] | None = None,
) -> int:
    """Run the Qt loop after mandatory bundled-resource initialization."""

    from flowlens.adapters.windows_shell import WindowsFolderOpener
    from flowlens.ui.main_window import MainWindow
    from flowlens.ui.presenter import QtAccessibilityAnnouncer, QtSessionPresenter

    _recover_startup_sessions(paths)
    app, owns_application = _acquire_qapplication()
    try:
        _configure_qt_surface(app)
    except Exception as error:
        print(
            f"FlowLens UI resources unavailable: {_error_name(error)}",
            file=sys.stderr,
        )
        if owns_application:
            app.quit()
        return 1

    graph = build_application(paths, options)
    window = MainWindow()
    presenter = QtSessionPresenter(
        cast(Any, graph.controller.session),
        window,
        QtAccessibilityAnnouncer(),
        config_store=ConfigStore(paths.config),
        folder_opener=WindowsFolderOpener(),
    )
    cast(Any, window)._flowlens_presenter = presenter
    window.show()
    try:
        return int(app.exec())
    finally:
        if snapshot_callback is not None:
            try:
                snapshot_callback(graph.controller.session.snapshot())
            except Exception:
                pass
        if owns_application:
            app.quit()


def _recover_startup_sessions(paths: AppPaths) -> None:
    """Durably recover every incomplete session before opening the live UI."""

    recover_incomplete_sessions(paths.sessions, _utc_now())


def _acquire_qapplication() -> tuple[Any, bool]:
    """Get the current QApplication or create the only application instance."""

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app, False
    return QApplication([]), True


def _configure_qt_surface(application: Any) -> None:
    """Load required bundled resources before creating any application window."""

    from flowlens.ui.design import (
        _RESOURCE_ROOT,
        DesignTokens,
        build_stylesheet,
        load_bundled_fonts,
    )

    load_bundled_fonts(_RESOURCE_ROOT)
    application.setStyleSheet(
        build_stylesheet(
            DesignTokens.approved(),
            reduced_motion=_reduced_motion(application),
        )
    )


def _reduced_motion(application: Any) -> bool:
    """Read Qt's optional platform motion preference without guessing."""

    style_hints = application.styleHints()
    preference = getattr(style_hints, "reduceMotion", None)
    if not callable(preference):
        return False
    return preference() is True


def _normalize_acceptance_report_path(value: str) -> Path:
    """Return a normalized absolute report path under a verified local directory."""

    if type(value) is not str or not value:
        raise ValueError("acceptance report path must be a non-empty string")
    path = Path(os.path.abspath(os.path.normpath(value)))
    parent = path.parent
    if not parent.is_dir():
        raise ValueError("acceptance report parent must be an existing directory")
    _verify_safe_directory(parent)
    _verify_safe_report_target(path)
    return path


def _verify_safe_directory(directory: Path) -> None:
    """Reject a symlink or reparse-point component before an atomic write."""

    components: list[Path] = []
    current = directory
    while current != current.parent:
        components.append(current)
        current = current.parent
    for component in reversed(components):
        information = os.lstat(component)
        if stat.S_ISLNK(information.st_mode) or _is_reparse_point(information):
            raise ValueError("acceptance report parent must not use links")


def _verify_safe_report_target(path: Path) -> None:
    """Allow only a local regular report file to be atomically replaced."""

    if not os.path.lexists(path):
        return
    information = os.lstat(path)
    if (
        not stat.S_ISREG(information.st_mode)
        or stat.S_ISLNK(information.st_mode)
        or _is_reparse_point(information)
        or information.st_nlink > 1
    ):
        raise ValueError("acceptance report target is not a safe regular file")


def _is_reparse_point(information: os.stat_result) -> bool:
    attribute = getattr(information, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attribute & reparse)


def _acceptance_report(
    *,
    started_at: datetime,
    ended_at: datetime,
    elapsed_ms: int,
    exit_code: int,
    package_self_check: bool,
    final_snapshot: object | None,
    error_type: str | None,
) -> AcceptanceReport:
    """Build a privacy-preserving report from verified local application data."""

    report: AcceptanceReport = {
        "schema_version": _ACCEPTANCE_SCHEMA_VERSION,
        "started_at": started_at.astimezone(UTC).isoformat(timespec="milliseconds"),
        "ended_at": ended_at.astimezone(UTC).isoformat(timespec="milliseconds"),
        "elapsed_ms": elapsed_ms,
        "exit_code": exit_code,
        "local_only": True,
        "package_self_check": {
            "requested": package_self_check,
            "exit_code": exit_code if package_self_check else None,
        },
    }
    controller = _controller_measurements(final_snapshot)
    if controller is not None:
        report["controller"] = controller
    if error_type is not None:
        report["error_type"] = error_type
    return report


def _controller_measurements(
    snapshot: object | None,
) -> ControllerAcceptanceSnapshot | None:
    """Select final snapshot fields that contain no session text or prompt data."""

    if snapshot is None:
        return None
    state = getattr(snapshot, "state", None)
    state_value = getattr(state, "value", None)
    recording_status = getattr(snapshot, "recording_status", None)
    transcript = getattr(snapshot, "transcript", None)
    asr_backlog_ms = getattr(snapshot, "asr_backlog_ms", None)
    maximum_asr_backlog_ms = getattr(snapshot, "maximum_asr_backlog_ms", None)
    if (
        type(state_value) is not str
        or type(recording_status) is not str
        or not isinstance(transcript, tuple | list)
        or type(asr_backlog_ms) is not int
        or asr_backlog_ms < 0
        or type(maximum_asr_backlog_ms) is not int
        or maximum_asr_backlog_ms < 0
    ):
        return None
    return {
        "state": state_value,
        "recording_status": recording_status,
        "completion_available": getattr(snapshot, "completion", None) is not None,
        "transcript_count": len(transcript),
        "asr_backlog_ms": asr_backlog_ms,
        "maximum_asr_backlog_ms": maximum_asr_backlog_ms,
        "latencies_ms": {
            "partial": list(getattr(snapshot, "partial_latencies_ms", ())),
            "commit": list(getattr(snapshot, "commit_latencies_ms", ())),
            "discussion": list(getattr(snapshot, "discussion_latencies_ms", ())),
            "ui_feedback": list(getattr(snapshot, "ui_feedback_latencies_ms", ())),
        },
    }


def _write_acceptance_report(path: Path, report: AcceptanceReport) -> None:
    """Write one UTF-8/LF report through a same-directory temporary file."""

    _verify_safe_directory(path.parent)
    _verify_safe_report_target(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(report, output, ensure_ascii=False, indent=4)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _error_name(error: BaseException) -> str:
    """Return a non-empty non-sensitive exception type for user-visible records."""

    return type(error).__name__.strip() or "Exception"
