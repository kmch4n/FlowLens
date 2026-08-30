"""Allowlisted dynamic imports for local-only runtime dependencies."""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

# Dynamic imports keep native runtimes out of CLI help and parent-process
# startup. Every permitted target is a bundled/local dependency or a FlowLens
# worker module; arbitrary names can never reach Python's importer.
_ALLOWED_LOCAL_MODULES: Final[frozenset[str]] = frozenset(
    {
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "ctranslate2",
        "faster_whisper",
        "flowlens.asr.worker",
        "flowlens.audio.worker",
        "flowlens.discussion.worker",
        "flowlens.workers.writer",
        "llama_cpp",
        "numpy",
        "pyaudiowpatch",
    }
)
_CUDA_DEPENDENT_MODULES: Final[frozenset[str]] = frozenset(
    {"ctranslate2", "faster_whisper"}
)
_CUDA_12_ROOT_VARIABLES: Final[tuple[str, ...]] = (
    "CUDA_PATH_V12_8",
    "CUDA_PATH",
)
_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()


def _register_cuda_12_dll_directory(module_name: str) -> None:
    """Register one verified CUDA 12 runtime before importing CTranslate2."""

    if sys.platform != "win32" or module_name not in _CUDA_DEPENDENT_MODULES:
        return
    for variable in _CUDA_12_ROOT_VARIABLES:
        root = os.environ.get(variable)
        if not root:
            continue
        candidate = Path(root) / "bin"
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir() or not all(
            (resolved / name).is_file()
            for name in ("cublas64_12.dll", "cudart64_12.dll")
        ):
            continue
        normalized = os.path.normcase(os.fspath(resolved))
        if normalized in _REGISTERED_DLL_DIRECTORIES:
            return
        handle = os.add_dll_directory(os.fspath(resolved))
        _DLL_DIRECTORY_HANDLES.append(handle)
        _REGISTERED_DLL_DIRECTORIES.add(normalized)
        return


def import_local_module(module_name: str) -> ModuleType:
    """Import one explicitly approved local module and reject every other name."""

    if module_name not in _ALLOWED_LOCAL_MODULES:
        raise ValueError(
            f"Dynamic module is not in the offline import allowlist: {module_name}"
        )
    _register_cuda_12_dll_directory(module_name)
    return importlib.import_module(module_name)
