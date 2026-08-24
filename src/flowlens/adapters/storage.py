"""Local session-root capacity and exclusive write probe."""

import os
import shutil
import stat
import uuid
from pathlib import Path

from flowlens.controller.models import StorageCheck


class LocalStorageReadiness:
    """Probe a normalized local sessions root without deleting existing files."""

    def check(self, root: Path, required_bytes: int) -> StorageCheck:
        """Create, synchronize, close, and remove one exclusive probe file."""

        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("root must be an absolute Path")
        if type(required_bytes) is not int or required_bytes < 0:
            raise ValueError("required_bytes must be a non-negative integer")
        if root.resolve(strict=False) != root:
            return StorageCheck(root.resolve(strict=False), 0, False, "unsafe_path")
        free_bytes = 0
        try:
            _ensure_safe_directory(root)
            free_bytes = shutil.disk_usage(root).free
        except (OSError, ValueError):
            return StorageCheck(root, 0, False, "unavailable")

        probe = root / f".flowlens-write-probe-{uuid.uuid4().hex}"
        descriptor: int | None = None
        owns_probe = False
        writable = False
        failure: str | None = None
        try:
            descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            owns_probe = True
            os.write(descriptor, b"FlowLens")
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            probe.unlink()
            writable = True
        except OSError:
            failure = "unwritable"
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    failure = "cleanup_failed"
            if owns_probe and probe.exists():
                try:
                    probe.unlink()
                except OSError:
                    failure = "cleanup_failed"
            if failure is not None:
                writable = False
        return StorageCheck(root, free_bytes, writable, failure)


def _ensure_safe_directory(root: Path) -> None:
    existing = root
    missing: list[Path] = []
    while not existing.exists():
        missing.append(existing)
        if existing.parent == existing:
            raise ValueError("no existing storage ancestor")
        existing = existing.parent
    _reject_reparse(existing)
    for directory in reversed(missing):
        directory.mkdir()
        _reject_reparse(directory)
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("storage root must be a canonical directory")


def _reject_reparse(path: Path) -> None:
    status = path.lstat()
    attributes = getattr(status, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(status.st_mode) or bool(attributes & reparse):
        raise ValueError("storage path must not use a reparse point")
