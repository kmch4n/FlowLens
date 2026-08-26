"""Behavioral tests for the folder-package audit.

The fixtures use a non-executable placeholder for ``FlowLens.exe``.  Every
launch result is supplied through an injected probe so the test suite never
executes a fixture-controlled binary.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import pytest

_CHECK_PACKAGE_PATH = Path("scripts/check_package.py")
_SPEC = importlib.util.spec_from_file_location(
    "flowlens_package_audit", _CHECK_PACKAGE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@dataclass(frozen=True)
class ProbeResult:
    """Structural equivalent of the audit module's probe result."""

    exit_code: int | None
    timed_out: bool
    detail: str


class PackageAudit(Protocol):
    """Audit result fields used by these black-box tests."""

    errors: tuple[str, ...]


class CheckPackage(Protocol):
    """Typed boundary around the dynamically loaded script module."""

    def __call__(
        self,
        package_root: Path,
        *,
        probe: object | None = None,
        font_verifier: object | None = None,
    ) -> PackageAudit:
        """Run the package audit."""


check_package = cast(CheckPackage, _MODULE.check_package)


@dataclass
class SuccessfulProbe:
    """Deterministic executable probe used only by package-audit fixtures."""

    calls: list[tuple[Path, tuple[str, ...], int]] = field(default_factory=list)

    def run(
        self,
        executable: Path,
        arguments: tuple[str, ...],
        timeout_seconds: int,
    ) -> ProbeResult:
        self.calls.append((executable, arguments, timeout_seconds))
        return ProbeResult(exit_code=0, timed_out=False, detail="")


class AcceptBundledFonts:
    """Font verifier that keeps file-layout tests independent of Qt state."""

    def verify(self, font_paths: tuple[Path, ...]) -> tuple[str, ...]:
        assert all(path.is_file() for path in font_paths)
        return ("IBM Plex Sans JP", "IBM Plex Mono")


def make_fake_package(tmp_path: Path) -> Path:
    """Create the smallest valid package layout without a runnable executable."""

    package = tmp_path / "FlowLens"
    runtime = package / "runtime"
    licenses = package / "licenses"
    fonts = runtime / "assets" / "fonts"
    styles = runtime / "assets" / "styles"
    for directory in (
        runtime,
        licenses,
        fonts,
        styles,
        runtime / "PySide6" / "Qt" / "plugins" / "platforms",
        runtime / "llama_cpp",
        runtime / "ctranslate2",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (package / "FlowLens.exe").write_bytes(b"fixture only; never execute")
    for path in (
        runtime / "python312.dll",
        runtime / "base_library.zip",
        runtime / "PySide6" / "QtCore.pyd",
        runtime / "PySide6" / "QtGui.pyd",
        runtime / "PySide6" / "Qt" / "plugins" / "platforms" / "qwindows.dll",
        runtime / "llama_cpp" / "llama_cpp.pyd",
        runtime / "llama_cpp" / "llama.dll",
        runtime / "ctranslate2" / "ctranslate2.pyd",
        runtime / "ctranslate2" / "ctranslate2.dll",
        runtime / "_portaudiowpatch.cp312-win_amd64.pyd",
        fonts / "IBMPlexSansJP-Regular.ttf",
        fonts / "IBMPlexSansJP-SemiBold.ttf",
        fonts / "IBMPlexMono-Regular.ttf",
        styles / "flowlens.qss",
    ):
        path.write_bytes(b"fixture")
    for name in (
        "PySide6-LGPL-3.0-only.txt",
        "llama-cpp-python-MIT.txt",
        "IBM-Plex-OFL.txt",
        "Qwen3-4B-Instruct-2507-Apache-2.0.txt",
        "kotoba-whisper-v2.0-license.txt",
    ):
        (licenses / name).write_text("fixture license\n", encoding="utf-8")
    return package


def _audit(
    package: Path,
    probe: SuccessfulProbe | None = None,
) -> PackageAudit:
    return check_package(
        package,
        probe=SuccessfulProbe() if probe is None else probe,
        font_verifier=AcceptBundledFonts(),
    )


def test_package_audit_requires_structure_and_rejects_models(tmp_path: Path) -> None:
    """A valid fixture passes until a recursive model artifact appears."""

    package = make_fake_package(tmp_path)
    audit = _audit(package)
    assert audit.errors == ()
    (package / "runtime" / "qwen.gguf").write_bytes(b"model")
    audit = _audit(package)
    assert audit.errors == (
        "Runtime package must not contain model artifacts: runtime/qwen.gguf",
    )


def test_package_audit_requires_every_license_and_font(tmp_path: Path) -> None:
    """License and font resources are mandatory package contents."""

    package = make_fake_package(tmp_path)
    (package / "licenses" / "IBM-Plex-OFL.txt").unlink()
    errors = _audit(package).errors
    assert "Missing license: IBM-Plex-OFL.txt" in errors

    (package / "runtime" / "assets" / "fonts" / "IBMPlexMono-Regular.ttf").unlink()
    errors = _audit(package).errors
    assert "Missing bundled font: IBMPlexMono-Regular.ttf" in errors


def test_package_audit_probes_help_and_native_imports_without_fixture_execution(
    tmp_path: Path,
) -> None:
    """Injected probes see the two safe command paths and the fixed timeout."""

    package = make_fake_package(tmp_path)
    probe = SuccessfulProbe()
    assert _audit(package, probe).errors == ()
    assert [arguments for _, arguments, _ in probe.calls] == [
        ("--help",),
        ("--package-self-check",),
    ]
    assert all(timeout == 10 for _, _, timeout in probe.calls)


def test_package_audit_does_not_launch_a_fixture_without_an_injected_probe(
    tmp_path: Path,
) -> None:
    """The library API leaves executable launch authority with its caller."""

    package = make_fake_package(tmp_path)
    audit = check_package(package, font_verifier=AcceptBundledFonts())
    assert audit.errors == ()


def test_package_audit_rejects_model_bins_but_allows_native_library_bins(
    tmp_path: Path,
) -> None:
    """Model directories take precedence over a native-library allowlist."""

    package = make_fake_package(tmp_path)
    native_binary = package / "runtime" / "ctranslate2" / "native" / "kernel.bin"
    native_binary.parent.mkdir()
    native_binary.write_bytes(b"native")
    assert _audit(package).errors == ()

    model_binary = package / "runtime" / "kotoba-whisper-v2.0-faster" / "model.bin"
    model_binary.parent.mkdir()
    model_binary.write_bytes(b"model")
    assert _audit(package).errors == (
        "Runtime package must not contain model artifacts: "
        "runtime/kotoba-whisper-v2.0-faster/model.bin",
    )


def test_package_audit_rejects_extra_top_level_entries(tmp_path: Path) -> None:
    """The public release folder has exactly the three specified entries."""

    package = make_fake_package(tmp_path)
    (package / "README.txt").write_text("unexpected\n", encoding="utf-8")
    assert _audit(package).errors == (
        "Package top level must contain exactly FlowLens.exe, licenses, runtime; "
        "found FlowLens.exe, README.txt, licenses, runtime",
    )


def test_package_audit_rejects_a_linked_runtime_entry(tmp_path: Path) -> None:
    """No probe runs when a runtime path can escape through a link."""

    package = make_fake_package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = package / "runtime" / "models"
    try:
        os.symlink(outside, linked, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links unavailable for this test: {type(error).__name__}")
    probe = SuccessfulProbe()
    assert _audit(package, probe).errors == ("Unsafe package path: ValueError",)
    assert probe.calls == []
