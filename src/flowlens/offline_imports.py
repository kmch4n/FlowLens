"""Allowlisted dynamic imports for local-only runtime dependencies."""

import importlib
from types import ModuleType
from typing import Final

# Dynamic imports keep native runtimes out of CLI help and parent-process
# startup. Every permitted target is a bundled/local dependency or a FlowLens
# worker module; arbitrary names can never reach Python's importer.
_ALLOWED_LOCAL_MODULES: Final[frozenset[str]] = frozenset(
    {
        "PySide6.QtCore",
        "PySide6.QtGui",
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


def import_local_module(module_name: str) -> ModuleType:
    """Import one explicitly approved local module and reject every other name."""

    if module_name not in _ALLOWED_LOCAL_MODULES:
        raise ValueError(
            f"Dynamic module is not in the offline import allowlist: {module_name}"
        )
    return importlib.import_module(module_name)
