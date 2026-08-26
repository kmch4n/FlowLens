"""Composition-root and entrypoint tests for the local-only application graph."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from flowlens.controller.ports import DeviceCatalog, ModelReadiness, WorkerRuntime
from flowlens.controller.session_controller import ControllerSnapshot, SessionState
from flowlens.persistence.paths import AppPaths


def fake_paths(tmp_path: Path) -> AppPaths:
    root = tmp_path.resolve()
    return AppPaths(
        root=root,
        config=root / "config.json",
        models=root / "models",
        sessions=root / "sessions",
    )


def test_composition_has_no_http_or_websocket_dependency(tmp_path: Path) -> None:
    from flowlens.integration.composition import AppOptions, build_application

    graph = build_application(fake_paths(tmp_path), AppOptions())

    import_names = graph.production_adapter_modules()
    assert all("requests" not in name for name in import_names)
    assert all("httpx" not in name for name in import_names)
    assert all("websocket" not in name for name in import_names)
    assert all("socket" not in name for name in import_names)


def test_composition_injects_hardware_adapters_behind_ports(tmp_path: Path) -> None:
    from flowlens.integration.composition import AppOptions, build_application

    graph = build_application(fake_paths(tmp_path), AppOptions())

    assert isinstance(graph.controller.preflight.device_catalog, DeviceCatalog)
    assert isinstance(graph.controller.preflight.model_readiness, ModelReadiness)
    assert isinstance(graph.controller.runtime, WorkerRuntime)
    assert graph.controller.runtime.processes == {}


def test_acceptance_report_option_preserves_graph_semantics(tmp_path: Path) -> None:
    from flowlens.integration.composition import AppOptions, build_application

    report_path = tmp_path / "acceptance.json"
    graph = build_application(
        fake_paths(tmp_path),
        AppOptions(acceptance_report=report_path),
    )

    assert graph.options.acceptance_report == report_path
    assert graph.controller.runtime.processes == {}


def test_entrypoint_freezes_multiprocessing_before_app_import() -> None:
    source = Path("src/flowlens/__main__.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    assert len(module.body) == 1
    guard = module.body[0]
    assert isinstance(guard, ast.If)
    body = guard.body
    assert isinstance(body[0], ast.Import)
    assert body[0].names[0].name == "multiprocessing"
    assert isinstance(body[1], ast.Expr)
    call = body[1].value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Attribute)
    assert call.func.attr == "freeze_support"
    assert isinstance(body[2], ast.ImportFrom)
    assert body[2].module == "flowlens.app"


def test_module_help_exits_without_starting_workers() -> None:
    environment = os.environ.copy()
    src_path = str(Path("src").resolve())
    environment["PYTHONPATH"] = (
        src_path
        if not environment.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "-m", "flowlens", "--help"],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "--package-self-check" in result.stdout
    assert "--acceptance-report" in result.stdout


def test_package_self_check_does_not_build_application(
    monkeypatch: MonkeyPatch,
) -> None:
    import flowlens.app as app

    def fail_build_application(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("self-check must not build the runtime graph")

    monkeypatch.setattr(app, "build_application", fail_build_application)
    monkeypatch.setattr(app, "_package_self_check", lambda: 0)

    assert app.main(["--package-self-check"]) == 0


def final_snapshot() -> ControllerSnapshot:
    return ControllerSnapshot(
        state=SessionState.COMPLETED,
        preflight=None,
        issue=None,
        recording_status="Completed",
        transcript=(),
        partials=(),
        discussion_state=None,
        microphone_level=0.0,
        loopback_level=0.0,
        asr_status="Stopped",
        asr_backlog_ms=0,
        maximum_asr_backlog_ms=125,
        analysis_status="Stopped",
        latest_successful_save_at=None,
        fatal_error=None,
        stop_confirmation_visible=False,
        slow_finalization_visible=False,
        completion=None,
    )


def test_qt_setup_loads_bundled_fonts_and_applies_the_approved_stylesheet(
    monkeypatch: MonkeyPatch,
) -> None:
    import flowlens.app as app
    import flowlens.ui.design as design

    calls: list[object] = []

    class FakeStyleHints:
        def reduceMotion(self) -> bool:
            return True

    class FakeApplication:
        def styleHints(self) -> FakeStyleHints:
            return FakeStyleHints()

        def setStyleSheet(self, stylesheet: str) -> None:
            calls.append(stylesheet)

    def load_fonts(root: Path) -> object:
        calls.append(root)
        return object()

    def build(tokens: object, reduced_motion: bool) -> str:
        calls.append((tokens, reduced_motion))
        return "approved stylesheet"

    monkeypatch.setattr(design, "load_bundled_fonts", load_fonts)
    monkeypatch.setattr(design, "build_stylesheet", build)

    app._configure_qt_surface(FakeApplication())

    assert calls[0] == design._RESOURCE_ROOT
    stylesheet_call = calls[1]
    assert isinstance(stylesheet_call, tuple)
    assert stylesheet_call[0] == design.DesignTokens.approved()
    assert stylesheet_call[1] is True
    assert calls[2] == "approved stylesheet"


def test_qt_resource_failure_returns_nonzero_before_building_a_window(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flowlens.app as app

    class FakeApplication:
        def quit(self) -> None:
            return None

    monkeypatch.setattr(app, "_acquire_qapplication", lambda: (FakeApplication(), True))
    monkeypatch.setattr(
        app,
        "_configure_qt_surface",
        lambda application: (_ for _ in ()).throw(RuntimeError("missing font")),
    )
    monkeypatch.setattr(
        app,
        "build_application",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    from flowlens.integration.composition import AppOptions

    assert app._run_qt(fake_paths(tmp_path), AppOptions()) == 1


def test_acceptance_report_serializes_only_local_safe_final_measurements(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flowlens.app as app

    report_path = tmp_path / "acceptance.json"
    timestamps = iter(
        (
            datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 26, 12, 0, 1, tzinfo=UTC),
        )
    )
    monotonic_values = iter((100.0, 101.234))

    def fake_run(
        paths: AppPaths,
        options: object,
        snapshot_callback: Any | None = None,
    ) -> int:
        del paths, options
        if snapshot_callback is not None:
            snapshot_callback(final_snapshot())
        return 0

    monkeypatch.setattr(app, "_utc_now", lambda: next(timestamps))
    monkeypatch.setattr(app, "_monotonic_seconds", lambda: next(monotonic_values))
    monkeypatch.setattr(app, "_run_qt", fake_run)
    monkeypatch.setattr(
        AppPaths,
        "from_environment",
        staticmethod(lambda environment: fake_paths(tmp_path)),
    )

    assert app.main(["--acceptance-report", str(report_path)]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "schema_version": 1,
        "started_at": "2026-08-26T12:00:00.000+00:00",
        "ended_at": "2026-08-26T12:00:01.000+00:00",
        "elapsed_ms": 1234,
        "exit_code": 0,
        "local_only": True,
        "package_self_check": {"requested": False, "exit_code": None},
        "controller": {
            "state": "COMPLETED",
            "recording_status": "Completed",
            "completion_available": False,
            "transcript_count": 0,
            "asr_backlog_ms": 0,
            "maximum_asr_backlog_ms": 125,
        },
    }
    assert "\ufeff" not in report_path.read_text(encoding="utf-8")


def test_acceptance_report_records_package_self_check_status(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flowlens.app as app

    report_path = tmp_path / "self-check.json"
    timestamps = iter(
        (
            datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 26, 12, 0, 1, tzinfo=UTC),
        )
    )
    monotonic_values = iter((100.0, 100.001))
    monkeypatch.setattr(app, "_utc_now", lambda: next(timestamps))
    monkeypatch.setattr(app, "_monotonic_seconds", lambda: next(monotonic_values))
    monkeypatch.setattr(app, "_package_self_check", lambda: 1)

    assert (
        app.main(["--package-self-check", "--acceptance-report", str(report_path)]) == 1
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["package_self_check"] == {"requested": True, "exit_code": 1}
    assert "controller" not in report


def test_acceptance_report_write_failure_preserves_app_exit_code_and_old_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: Any,
) -> None:
    import flowlens.app as app

    report_path = tmp_path / "acceptance.json"
    report_path.write_text("previous\n", encoding="utf-8")
    timestamps = iter(
        (
            datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 26, 12, 0, 1, tzinfo=UTC),
        )
    )
    monotonic_values = iter((100.0, 100.001))
    monkeypatch.setattr(app, "_utc_now", lambda: next(timestamps))
    monkeypatch.setattr(app, "_monotonic_seconds", lambda: next(monotonic_values))
    monkeypatch.setattr(app, "_run_qt", lambda *args, **kwargs: 7)
    monkeypatch.setattr(
        os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("replace failed")),
    )
    monkeypatch.setattr(
        AppPaths,
        "from_environment",
        staticmethod(lambda environment: fake_paths(tmp_path)),
    )

    assert app.main(["--acceptance-report", str(report_path)]) == 7
    assert report_path.read_text(encoding="utf-8") == "previous\n"
    assert "OSError" in capsys.readouterr().err


def test_acceptance_report_rejects_a_dangling_link_target(tmp_path: Path) -> None:
    import flowlens.app as app

    report_path = tmp_path / "acceptance.json"
    try:
        report_path.symlink_to(tmp_path / "missing.json")
    except OSError:
        pytest.skip("The current Windows test environment cannot create symlinks")

    with pytest.raises(ValueError, match="safe regular file"):
        app._normalize_acceptance_report_path(str(report_path))


def test_main_preserves_normal_launch_exceptions_without_a_report(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flowlens.app as app

    monkeypatch.setattr(
        app,
        "_run_qt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("run failed")),
    )
    monkeypatch.setattr(
        AppPaths,
        "from_environment",
        staticmethod(lambda environment: fake_paths(tmp_path)),
    )

    with pytest.raises(RuntimeError, match="run failed"):
        app.main([])
