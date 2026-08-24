"""Command-line entry for the FlowLens desktop application."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from flowlens.config.store import ConfigStore
from flowlens.integration.composition import AppOptions, build_application
from flowlens.persistence.paths import AppPaths

_SELF_CHECK_MODULES = (
    "PySide6.QtCore",
    "numpy",
    "pyaudiowpatch",
    "ctranslate2",
    "llama_cpp",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI parsing, package self-check, or the normal Qt application."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if cast(bool, arguments.package_self_check):
        return _package_self_check()
    paths = AppPaths.from_environment(os.environ)
    acceptance_report = cast(str | None, arguments.acceptance_report)
    options = AppOptions(
        acceptance_report=(
            None
            if acceptance_report is None
            else Path(acceptance_report).resolve(strict=False)
        )
    )
    return _run_qt(paths, options)


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
    """Import packaged native modules without starting workers or loading models."""

    failed: list[str] = []
    for module_name in _SELF_CHECK_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            failed.append(f"{module_name}: {type(error).__name__}")
    if failed:
        for item in failed:
            print(item, file=sys.stderr)
        return 1
    return 0


def _run_qt(paths: AppPaths, options: AppOptions) -> int:
    from PySide6.QtWidgets import QApplication

    from flowlens.adapters.windows_shell import WindowsFolderOpener
    from flowlens.ui.main_window import MainWindow
    from flowlens.ui.presenter import QtAccessibilityAnnouncer, QtSessionPresenter

    app = QApplication.instance()
    owns_application = False
    if not isinstance(app, QApplication):
        app = QApplication([])
        owns_application = True

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
    exit_code = app.exec()
    if owns_application:
        app.quit()
    return int(exit_code)
