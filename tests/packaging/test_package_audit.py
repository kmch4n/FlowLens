"""Behavioral tests for the folder-package audit.

The fixtures use a non-executable placeholder for ``FlowLens.exe``.  Every
launch result is supplied through an injected probe so the test suite never
executes a fixture-controlled binary.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
        source = Path("licenses") / name
        (licenses / name).write_bytes(source.read_bytes())
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


def test_package_audit_rejects_empty_model_directories_and_all_bin_artifacts(
    tmp_path: Path,
) -> None:
    """Model locations are forbidden even when empty or disguised as native data."""

    package = make_fake_package(tmp_path)
    empty_model_directory = package / "runtime" / "models"
    empty_model_directory.mkdir()
    assert _audit(package).errors == (
        "Runtime package must not contain model artifacts: runtime/models",
    )

    empty_model_directory.rmdir()
    native_binary = package / "runtime" / "ctranslate2" / "native" / "kernel.bin"
    native_binary.parent.mkdir()
    native_binary.write_bytes(b"native")
    (package / "runtime" / "qwen.bin").write_bytes(b"model")
    assert _audit(package).errors == (
        "Runtime package must not contain model artifacts: "
        "runtime/ctranslate2/native/kernel.bin",
        "Runtime package must not contain model artifacts: " "runtime/qwen.bin",
    )


def test_package_audit_rejects_replaced_license_text(tmp_path: Path) -> None:
    """Every approved license artifact must retain its pinned source text."""

    package = make_fake_package(tmp_path)
    (package / "licenses" / "IBM-Plex-OFL.txt").write_text("\n", encoding="utf-8")

    assert "Invalid license SHA-256: IBM-Plex-OFL.txt" in _audit(package).errors


def test_qt_font_verifier_loads_real_bundled_fonts_in_offscreen_subprocess() -> None:
    """Qt loads the shipped fonts after a single offscreen GUI application exists."""

    pytest.importorskip("PySide6")
    source = Path("scripts/check_package.py").resolve()
    fonts = [
        Path("assets/fonts/IBMPlexSansJP-Regular.ttf").resolve(),
        Path("assets/fonts/IBMPlexSansJP-SemiBold.ttf").resolve(),
        Path("assets/fonts/IBMPlexMono-Regular.ttf").resolve(),
    ]
    source_reference = repr(str(source))
    font_references = repr([str(font) for font in fonts])
    script = (
        "import importlib.util\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"source_path = {source_reference}\n"
        "spec = importlib.util.spec_from_file_location('qt_font_audit', source_path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "assert spec.loader is not None\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        f"font_paths = {font_references}\n"
        "verifier = module.QtFontVerifier()\n"
        "families = verifier.verify(tuple(Path(item) for item in font_paths))\n"
        "assert 'IBM Plex Sans JP' in families\n"
        "assert 'IBM Plex Mono' in families\n"
    )
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


@dataclass
class _TimedOutProcess:
    """Deterministic process double that remains alive until fallback kill."""

    pid: int = 4102
    killed: bool = False
    communicate_calls: int = 0

    def communicate(self, timeout: int) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.communicate_calls == 1:
            raise subprocess.TimeoutExpired("FlowLens.exe", timeout)
        return (b"", b"")

    def poll(self) -> int | None:
        return 0 if self.killed else None

    def kill(self) -> None:
        self.killed = True

    @property
    def returncode(self) -> int | None:
        return None if not self.killed else 1


class _WindowsHost:
    """Host double that enables Windows-only process-control behavior."""

    def is_windows(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("taskkill_failure", "expected_detail"),
    [
        (SimpleNamespace(returncode=1), "taskkill exited with code 1"),
        (subprocess.TimeoutExpired("taskkill", 5), "taskkill timed out"),
    ],
)
def test_probe_reports_taskkill_cleanup_failures_and_falls_back_to_kill(
    taskkill_failure: object,
    expected_detail: str,
) -> None:
    """Timed probes expose failed descendant cleanup instead of suppressing it."""

    process = _TimedOutProcess()
    taskkill_calls: list[tuple[object, ...]] = []

    def launch(*args: object, **kwargs: object) -> _TimedOutProcess:
        del args, kwargs
        return process

    def taskkill(*args: object, **kwargs: object) -> object:
        del kwargs
        taskkill_calls.append(args)
        if isinstance(taskkill_failure, BaseException):
            raise taskkill_failure
        return taskkill_failure

    probe = _MODULE.WindowsSubprocessProbe(
        platform=_WindowsHost(),
        process_factory=launch,
        taskkill_runner=taskkill,
    )
    result = probe.run(Path("FlowLens.exe"), ("--help",), 10)

    assert result.timed_out is True
    assert expected_detail in result.detail
    assert process.killed is True
    assert taskkill_calls == [(["taskkill", "/PID", "4102", "/T", "/F"],)]


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
