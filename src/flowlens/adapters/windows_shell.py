from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


class WindowsFolderOpener:
    """Open an existing absolute session directory through the OS shell."""

    def __init__(self, shell_execute: Callable[[str], object] | None = None) -> None:
        self._shell_execute = shell_execute

    def open(self, path: Path) -> None:
        """Open an existing absolute directory after resolving symlinks."""

        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        if not path.is_absolute():
            raise ValueError("path must be absolute")
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        self._execute(str(resolved))

    def _execute(self, path: str) -> None:
        shell_execute = self._shell_execute
        if shell_execute is None:
            try:
                shell_execute = os.startfile
            except AttributeError as error:
                raise OSError("os.startfile is unavailable") from error
        shell_execute(path)
