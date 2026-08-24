"""Composition-root and entrypoint tests for the local-only application graph."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from pytest import MonkeyPatch

from flowlens.controller.ports import DeviceCatalog, ModelReadiness, WorkerRuntime
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
