"""Fail-closed audit for a folder-based FlowLens Windows package.

``check_package`` is deliberately dependency-injected at its two environment
boundaries. Tests pass a probe rather than launching a fixture-controlled
executable, while the CLI uses the Windows subprocess probe. Font verification
uses Qt only after the package tree has passed its structural safety checks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

_EXPECTED_TOP_LEVEL = frozenset({"FlowLens.exe", "runtime", "licenses"})
_REQUIRED_LICENSES = (
    "PySide6-LGPL-3.0-only.txt",
    "llama-cpp-python-MIT.txt",
    "IBM-Plex-OFL.txt",
    "Qwen3-4B-Instruct-2507-Apache-2.0.txt",
    "kotoba-whisper-v2.0-license.txt",
)
_REQUIRED_FONTS = (
    "IBMPlexSansJP-Regular.ttf",
    "IBMPlexSansJP-SemiBold.ttf",
    "IBMPlexMono-Regular.ttf",
)
_EXPECTED_FONT_FAMILIES = frozenset({"IBM Plex Sans JP", "IBM Plex Mono"})
_MODEL_DIRECTORY_NAMES = frozenset(
    {
        "models",
        "model",
        "checkpoints",
        "checkpoint",
        "weights",
        "qwen3-4b-instruct-2507",
        "kotoba-whisper-v2.0-faster",
    }
)
_LICENSE_SHA256 = {
    "IBM-Plex-OFL.txt": (
        "d741e57d5f865e294df801f96b7b5161a88b211df65887e4358d271c9fc5fb4f"
    ),
    "PySide6-LGPL-3.0-only.txt": (
        "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118"
    ),
    "Qwen3-4B-Instruct-2507-Apache-2.0.txt": (
        "c156170b718ec29139d3653d40ed1986fd92fb7e0959b5c71f3c48f62e6636f4"
    ),
    "kotoba-whisper-v2.0-license.txt": (
        "c449d68446c278baaf454177fc2de19cf830b3ae86dfdd1e1227245d422f9fa4"
    ),
    "llama-cpp-python-MIT.txt": (
        "2aa1e22f6de50309ec505631695f95b308e980efa318e15bb45732d2c5e07a34"
    ),
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One bounded executable probe result."""

    exit_code: int | None
    timed_out: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PackageAudit:
    """All deterministic failures found while checking a package directory."""

    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the audit found no errors."""

        return not self.errors


class ExecutableProbe(Protocol):
    """Run one allowed executable command with a bounded timeout."""

    def run(
        self,
        executable: Path,
        arguments: tuple[str, ...],
        timeout_seconds: int,
    ) -> ProbeResult:
        """Return a non-throwing result for one executable invocation."""


class FontVerifier(Protocol):
    """Verify that package-local font files register their expected families."""

    def verify(self, font_paths: tuple[Path, ...]) -> tuple[str, ...]:
        """Load fonts and return every registered family."""


class HostPlatform(Protocol):
    """Minimal operating-system boundary for a testable package probe."""

    def is_windows(self) -> bool:
        """Return whether executable probing can use Windows process control."""


class CurrentPlatform:
    """Production host-platform implementation."""

    def is_windows(self) -> bool:
        """Return whether the current interpreter runs on Windows."""

        return os.name == "nt"


class _ProbeProcess(Protocol):
    """Minimal subprocess surface needed to stop an executable probe."""

    pid: int
    returncode: int | None

    def communicate(self, timeout: int) -> tuple[bytes, bytes]:
        """Collect output while applying a bounded wait."""

    def poll(self) -> int | None:
        """Return the process exit code if it has already exited."""

    def kill(self) -> None:
        """Terminate the direct process as a fallback."""


class WindowsSubprocessProbe:
    """Execute only the two FlowLens probe commands and kill timed-out trees."""

    def __init__(
        self,
        platform: HostPlatform | None = None,
        process_factory: Callable[..., _ProbeProcess] | None = None,
        taskkill_runner: Callable[..., object] | None = None,
    ) -> None:
        self._platform = CurrentPlatform() if platform is None else platform
        self._process_factory = (
            cast(Callable[..., _ProbeProcess], subprocess.Popen)
            if process_factory is None
            else process_factory
        )
        self._taskkill_runner = (
            cast(Callable[..., object], subprocess.run)
            if taskkill_runner is None
            else taskkill_runner
        )

    def run(
        self,
        executable: Path,
        arguments: tuple[str, ...],
        timeout_seconds: int,
    ) -> ProbeResult:
        """Run one command without a shell and terminate its tree on timeout."""

        if not self._platform.is_windows():
            return ProbeResult(
                exit_code=None,
                timed_out=False,
                detail="Windows executable probe requires a Windows host",
            )
        if arguments not in (("--help",), ("--package-self-check",)):
            return ProbeResult(
                exit_code=None,
                timed_out=False,
                detail="Executable probe arguments are not allowlisted",
            )
        if timeout_seconds <= 0:
            return ProbeResult(
                exit_code=None,
                timed_out=False,
                detail="Executable probe timeout must be positive",
            )
        try:
            process = self._process_factory(
                [os.fspath(executable), *arguments],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as error:
            return ProbeResult(None, False, f"launch failed: {type(error).__name__}")
        try:
            process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            cleanup_error = _terminate_process_tree(
                process,
                platform=self._platform,
                taskkill_runner=self._taskkill_runner,
            )
            detail = "timed out"
            if cleanup_error:
                detail += f"; cleanup failed: {cleanup_error}"
            return ProbeResult(None, True, detail)
        except OSError as error:
            cleanup_error = _terminate_process_tree(
                process,
                platform=self._platform,
                taskkill_runner=self._taskkill_runner,
            )
            detail = f"probe failed: {type(error).__name__}"
            if cleanup_error:
                detail += f"; cleanup failed: {cleanup_error}"
            return ProbeResult(None, False, detail)
        return ProbeResult(process.returncode, False, "")


class QtFontVerifier:
    """Verify bundled fonts through the Qt API used by the application."""

    def __init__(self) -> None:
        self._application: object | None = None

    def verify(self, font_paths: tuple[Path, ...]) -> tuple[str, ...]:
        """Load package-local fonts and return all Qt family names."""

        from PySide6.QtGui import QFontDatabase, QGuiApplication

        application = QGuiApplication.instance()
        if application is None:
            application = QGuiApplication([])
        if not isinstance(application, QGuiApplication):
            raise RuntimeError("Qt application is not a QGuiApplication")
        if not application.platformName():
            raise RuntimeError("Qt platform plugin did not initialize")
        self._application = application

        families: list[str] = []
        for path in font_paths:
            font_id = QFontDatabase.addApplicationFont(os.fspath(path))
            if font_id < 0:
                raise RuntimeError(f"could not load {path.name}")
            families.extend(QFontDatabase.applicationFontFamilies(font_id))
        return tuple(families)


@dataclass(frozen=True, slots=True)
class _ScannedPath:
    """One lstat-validated package path retained for race detection."""

    relative: PurePosixPath
    path: Path
    device: int
    inode: int
    mode: int

    def is_dir(self) -> bool:
        """Return whether the saved entry is a directory."""

        return stat.S_ISDIR(self.mode)

    def is_file(self) -> bool:
        """Return whether the saved entry is a regular file."""

        return stat.S_ISREG(self.mode)


def check_package(
    package_root: Path | str,
    *,
    probe: ExecutableProbe | None = None,
    font_verifier: FontVerifier | None = None,
) -> PackageAudit:
    """Audit one Windows package without loading models or starting workers.

    ``probe`` and ``font_verifier`` are injectable so tests never execute a
    fixture-controlled ``.exe`` or depend on a host Qt installation. The CLI
    supplies the real Windows probe and Qt verifier. Callers that omit
    ``probe`` receive the static/package-resource audit only; this prevents a
    fixture-controlled executable from running merely because a unit test
    checks folder layout.
    """

    errors: list[str] = []
    try:
        root = _absolute_path(package_root)
        entries = _safe_walk(root)
    except (OSError, ValueError) as error:
        return PackageAudit((f"Unsafe package path: {type(error).__name__}",))

    _check_top_level(entries, errors)
    _check_runtime_entries(entries, errors)
    _check_fonts(entries, errors, font_verifier)
    _check_licenses(entries, errors)
    _check_model_artifacts(entries, errors)
    if not errors and probe is not None:
        _check_executable_probes(
            entries,
            errors,
            probe,
        )
    return PackageAudit(tuple(errors))


def _absolute_path(value: Path | str) -> Path:
    if not isinstance(value, Path | str):
        raise ValueError("package root must be a path")
    return Path(os.path.abspath(os.path.normpath(os.fspath(value))))


def _safe_walk(root: Path) -> dict[PurePosixPath, _ScannedPath]:
    root_stat = _safe_lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("package root must be a directory")
    entries: dict[PurePosixPath, _ScannedPath] = {}
    pending = [
        _ScannedPath(
            relative=PurePosixPath("."),
            path=root,
            device=root_stat.st_dev,
            inode=root_stat.st_ino,
            mode=root_stat.st_mode,
        )
    ]
    while pending:
        directory = pending.pop()
        try:
            _assert_unchanged(directory)
        except (OSError, ValueError) as error:
            raise ValueError("package directory changed during audit") from error
        if not directory.is_dir():
            raise ValueError("package directory changed during audit")
        with os.scandir(directory.path) as scanned:
            children = sorted(scanned, key=lambda child: child.name.casefold())
        for child in children:
            path = Path(child.path)
            relative = directory.relative / child.name
            information = _safe_lstat(path)
            if not _is_within(root, path):
                raise ValueError("package entry escapes package root")
            snapshot = _ScannedPath(
                relative=relative,
                path=path,
                device=information.st_dev,
                inode=information.st_ino,
                mode=information.st_mode,
            )
            entries[relative] = snapshot
            if snapshot.is_dir():
                pending.append(snapshot)
            elif not snapshot.is_file():
                raise ValueError("package contains a non-regular entry")
    return entries


def _safe_lstat(path: Path) -> os.stat_result:
    information = os.lstat(path)
    if stat.S_ISLNK(information.st_mode) or _is_reparse_point(information):
        raise ValueError("package paths must not use links or reparse points")
    return information


def _is_reparse_point(information: os.stat_result) -> bool:
    attribute = getattr(information, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attribute & reparse)


def _is_within(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(path))) == os.fspath(root)
    except ValueError:
        return False


def _check_top_level(
    entries: dict[PurePosixPath, _ScannedPath],
    errors: list[str],
) -> None:
    top_level = {path.name for path in entries if len(path.parts) == 1}
    if top_level != _EXPECTED_TOP_LEVEL:
        errors.append(
            "Package top level must contain exactly FlowLens.exe, licenses, runtime; "
            f"found {', '.join(sorted(top_level))}"
        )
    _require_regular_file(entries, PurePosixPath("FlowLens.exe"), errors)
    _require_directory(entries, PurePosixPath("runtime"), errors)
    _require_directory(entries, PurePosixPath("licenses"), errors)


def _check_runtime_entries(
    entries: dict[PurePosixPath, _ScannedPath],
    errors: list[str],
) -> None:
    runtime = PurePosixPath("runtime")
    _require_any_file_matching(
        entries,
        runtime,
        lambda path: path.parent == runtime
        and path.name.casefold().startswith("python")
        and path.suffix.casefold() == ".dll",
        "Missing runtime Python DLL",
        errors,
    )
    _require_regular_file(entries, runtime / "base_library.zip", errors)
    for package in ("PySide6", "llama_cpp", "ctranslate2"):
        package_root = runtime / package
        _require_directory(entries, package_root, errors)
        _require_any_file_matching(
            entries,
            package_root,
            lambda path, root=package_root: _is_descendant(path, root)
            and path.suffix.casefold() == ".pyd",
            f"Missing runtime import extension: {package}",
            errors,
        )
    for package in ("llama_cpp", "ctranslate2"):
        package_root = runtime / package
        _require_any_file_matching(
            entries,
            package_root,
            lambda path, root=package_root: _is_descendant(path, root)
            and path.suffix.casefold() == ".dll",
            f"Missing runtime DLL: {package}",
            errors,
        )
    _require_any_file_matching(
        entries,
        runtime,
        lambda path: path.parent == runtime
        and path.name.casefold().startswith("_portaudiowpatch")
        and path.suffix.casefold() == ".pyd",
        "Missing runtime import extension: _portaudiowpatch",
        errors,
    )
    _require_regular_file(
        entries,
        runtime / "PySide6" / "Qt" / "plugins" / "platforms" / "qwindows.dll",
        errors,
    )


def _check_fonts(
    entries: dict[PurePosixPath, _ScannedPath],
    errors: list[str],
    verifier: FontVerifier | None,
) -> None:
    paths: list[Path] = []
    for name in _REQUIRED_FONTS:
        relative = PurePosixPath("runtime") / "assets" / "fonts" / name
        entry = entries.get(relative)
        if entry is None or not entry.is_file():
            errors.append(f"Missing bundled font: {name}")
            continue
        try:
            _assert_unchanged(entry)
        except (OSError, ValueError):
            errors.append(f"Unsafe bundled font: {name}")
            continue
        paths.append(entry.path)
    stylesheet = PurePosixPath("runtime") / "assets" / "styles" / "flowlens.qss"
    _require_regular_file(entries, stylesheet, errors)
    if len(paths) != len(_REQUIRED_FONTS):
        return
    active_verifier = QtFontVerifier() if verifier is None else verifier
    try:
        families = set(active_verifier.verify(tuple(paths)))
    except Exception as error:
        errors.append(f"Bundled fonts are not Qt-loadable: {type(error).__name__}")
        return
    missing = _EXPECTED_FONT_FAMILIES - families
    if missing:
        errors.append(
            "Bundled fonts are not Qt-loadable: missing " + ", ".join(sorted(missing))
        )


def _check_licenses(
    entries: dict[PurePosixPath, _ScannedPath],
    errors: list[str],
) -> None:
    for name in _REQUIRED_LICENSES:
        relative = PurePosixPath("licenses") / name
        entry = entries.get(relative)
        if entry is None or not entry.is_file():
            errors.append(f"Missing license: {name}")
            continue
        try:
            digest = _sha256_entry(entry)
        except (OSError, ValueError):
            errors.append(f"Unsafe license: {name}")
            continue
        if digest != _LICENSE_SHA256[name]:
            errors.append(f"Invalid license SHA-256: {name}")


def _check_model_artifacts(
    entries: dict[PurePosixPath, _ScannedPath],
    errors: list[str],
) -> None:
    for relative, entry in sorted(entries.items(), key=lambda item: item[0].as_posix()):
        if not relative.parts or relative.parts[0] != "runtime":
            continue
        if _is_model_artifact(relative, is_directory=entry.is_dir()):
            errors.append(
                "Runtime package must not contain model artifacts: "
                + relative.as_posix()
            )


def _is_model_artifact(relative: PurePosixPath, *, is_directory: bool) -> bool:
    """Return whether a package path is a model artifact or known model location."""

    lower_parts = tuple(part.casefold() for part in relative.parts)
    if any(part in _MODEL_DIRECTORY_NAMES for part in lower_parts):
        return True
    if is_directory:
        return False
    filename = lower_parts[-1]
    if filename.endswith(".gguf"):
        return True
    return filename.endswith(".bin")


def _check_executable_probes(
    entries: dict[PurePosixPath, _ScannedPath],
    errors: list[str],
    probe: ExecutableProbe,
) -> None:
    executable = entries.get(PurePosixPath("FlowLens.exe"))
    if executable is None or not executable.is_file():
        return
    try:
        _assert_unchanged(executable)
    except (OSError, ValueError):
        errors.append("Unsafe executable: FlowLens.exe")
        return
    for arguments in (("--help",), ("--package-self-check",)):
        try:
            result = probe.run(executable.path, arguments, 10)
        except Exception as error:
            errors.append(
                "Executable probe failed: "
                f"FlowLens.exe {' '.join(arguments)}: {type(error).__name__}"
            )
            continue
        command = f"FlowLens.exe {' '.join(arguments)}"
        if result.timed_out:
            suffix = "" if not result.detail else f" ({result.detail})"
            errors.append(
                f"Executable probe timed out after 10 seconds: {command}{suffix}"
            )
        elif result.exit_code != 0:
            suffix = "" if not result.detail else f" ({result.detail})"
            errors.append(f"Executable probe failed: {command}{suffix}")


def _assert_unchanged(entry: _ScannedPath) -> None:
    information = _safe_lstat(entry.path)
    if (
        information.st_dev != entry.device
        or information.st_ino != entry.inode
        or information.st_mode != entry.mode
    ):
        raise ValueError("package path changed during audit")


def _sha256_entry(entry: _ScannedPath) -> str:
    """Hash one lstat-validated license while rejecting path substitution races."""

    _assert_unchanged(entry)
    digest = hashlib.sha256()
    with entry.path.open("rb") as source:
        while chunk := source.read(128 * 1024):
            digest.update(chunk)
    _assert_unchanged(entry)
    return digest.hexdigest()


def _require_regular_file(
    entries: dict[PurePosixPath, _ScannedPath],
    relative: PurePosixPath,
    errors: list[str],
) -> None:
    entry = entries.get(relative)
    if entry is None or not entry.is_file():
        errors.append(f"Missing package file: {relative.as_posix()}")


def _require_directory(
    entries: dict[PurePosixPath, _ScannedPath],
    relative: PurePosixPath,
    errors: list[str],
) -> None:
    entry = entries.get(relative)
    if entry is None or not entry.is_dir():
        errors.append(f"Missing package directory: {relative.as_posix()}")


def _require_any_file_matching(
    entries: dict[PurePosixPath, _ScannedPath],
    root: PurePosixPath,
    predicate: PathPredicate,
    error: str,
    errors: list[str],
) -> None:
    if not any(
        entry.is_file() and predicate(relative) for relative, entry in entries.items()
    ):
        errors.append(error)


class PathPredicate(Protocol):
    """Predicate over a package-relative path."""

    def __call__(self, path: PurePosixPath) -> bool:
        """Return whether the path satisfies one runtime requirement."""


def _is_descendant(path: PurePosixPath, root: PurePosixPath) -> bool:
    return (
        len(path.parts) > len(root.parts)
        and path.parts[: len(root.parts)] == root.parts
    )


def _terminate_process_tree(
    process: _ProbeProcess,
    *,
    platform: HostPlatform,
    taskkill_runner: Callable[..., object],
) -> str:
    """Terminate a timed-out process tree and return any cleanup failure detail."""

    failures: list[str] = []
    if platform.is_windows():
        try:
            result = taskkill_runner(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=5,
            )
            returncode = getattr(result, "returncode", None)
            if returncode != 0:
                failures.append(f"taskkill exited with code {returncode}")
        except subprocess.TimeoutExpired:
            failures.append("taskkill timed out")
        except OSError as error:
            failures.append(f"taskkill failed: {type(error).__name__}")
    try:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        failures.append("direct process termination timed out")
    except OSError as error:
        failures.append(f"direct process termination failed: {type(error).__name__}")
    return "; ".join(failures)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit as a command-line program."""

    parser = argparse.ArgumentParser(
        description="Audit a folder-based FlowLens Windows package."
    )
    parser.add_argument("--package", required=True, metavar="PATH")
    arguments = parser.parse_args(argv)
    audit = check_package(
        arguments.package,
        probe=WindowsSubprocessProbe(),
        font_verifier=QtFontVerifier(),
    )
    if audit.passed:
        print("PASS")
        return 0
    print("FAIL", file=sys.stderr)
    for error in audit.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
