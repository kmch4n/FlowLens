"""No-follow-equivalent file handles and identities for recovery."""

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast


class RecoveryError(ValueError):
    """Report one artifact that prevents deterministic recovery."""

    def __init__(self, artifact_path: Path, reason: str) -> None:
        self.artifact_path = Path(artifact_path)
        self.reason = reason
        super().__init__(
            f"Recovery validation failed for {self.artifact_path}: {reason}"
        )


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Stable regular-file identity and content guard for later mutation."""

    path: Path
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    """Stable real-directory identity retained for atomic replacement guards."""

    path: Path
    device: int
    inode: int
    mode: int


@dataclass(frozen=True, slots=True)
class OpenArtifact:
    """One no-follow-equivalent descriptor held throughout inspection."""

    path: Path
    descriptor: int
    device: int
    inode: int
    mode: int
    link_count: int


def require_safe_directory(path: Path) -> None:
    """Require one real directory rather than a link or reparse point."""

    try:
        status = path.lstat()
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    require_safe_directory_status(path, status)


def require_safe_directory_status(path: Path, status: os.stat_result) -> None:
    """Validate already captured directory metadata."""

    if status_is_reparse(status):
        raise RecoveryError(
            path, "directory must not be a symbolic link or reparse point"
        )
    if not stat.S_ISDIR(status.st_mode):
        raise RecoveryError(path, "path must be a directory")


def open_guarded_artifact(path: Path, *, writable: bool = False) -> OpenArtifact:
    """Open a single-link regular file with an identity sandwich."""

    try:
        before = path.lstat()
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    _require_safe_regular_status(path, before)

    binary_flag = cast(int, vars(os).get("O_BINARY", 0))
    no_follow_flag = cast(int, vars(os).get("O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        access_flag = os.O_RDWR if writable else os.O_RDONLY
        descriptor = os.open(path, access_flag | binary_flag | no_follow_flag)
        opened_status = os.fstat(descriptor)
        _require_safe_regular_status(path, opened_status)
        after = path.lstat()
        _require_safe_regular_status(path, after)
        _require_same_file(path, before, opened_status)
        _require_same_file(path, opened_status, after)
        return OpenArtifact(
            path=path,
            descriptor=descriptor,
            device=opened_status.st_dev,
            inode=opened_status.st_ino,
            mode=opened_status.st_mode,
            link_count=opened_status.st_nlink,
        )
    except BaseException as primary_error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as close_error:
                primary_error.add_note(
                    f"Artifact descriptor cleanup failed for {path}: {close_error}"
                )
        if isinstance(primary_error, RecoveryError):
            raise
        raise RecoveryError(path, str(primary_error)) from primary_error


def close_artifact(opened: OpenArtifact) -> None:
    """Close one held recovery descriptor."""

    try:
        os.close(opened.descriptor)
    except OSError as error:
        raise RecoveryError(opened.path, f"descriptor close failed: {error}") from error


def read_open_artifact(opened: OpenArtifact) -> bytes:
    """Read all bytes from the held descriptor, never from its path."""

    try:
        os.lseek(opened.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(opened.descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise RecoveryError(opened.path, str(error)) from error


def hash_open_artifact(opened: OpenArtifact) -> tuple[int, str]:
    """Stream size and SHA-256 from the held descriptor."""

    try:
        os.lseek(opened.descriptor, 0, os.SEEK_SET)
        digest = sha256()
        size = 0
        while chunk := os.read(opened.descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        return size, digest.hexdigest()
    except OSError as error:
        raise RecoveryError(opened.path, str(error)) from error


def build_artifact_identity(
    opened: OpenArtifact,
    encoded: bytes,
) -> ArtifactIdentity:
    """Bind in-memory bytes to current descriptor and path metadata."""

    return build_artifact_identity_from_digest(
        opened,
        len(encoded),
        sha256(encoded).hexdigest(),
    )


def build_artifact_identity_from_digest(
    opened: OpenArtifact,
    size: int,
    digest: str,
) -> ArtifactIdentity:
    """Bind a streaming digest to current descriptor and path metadata."""

    try:
        status = os.fstat(opened.descriptor)
    except OSError as error:
        raise RecoveryError(opened.path, str(error)) from error
    require_opened_identity(opened, status)
    if status.st_size != size:
        raise RecoveryError(opened.path, "artifact changed while being read")
    _require_current_path_identity(opened)
    return ArtifactIdentity(
        path=opened.path,
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        link_count=status.st_nlink,
        size=size,
        modified_ns=status.st_mtime_ns,
        sha256=digest,
    )


def verify_artifact_identity(
    opened: OpenArtifact,
    identity: ArtifactIdentity,
) -> None:
    """Revalidate a held descriptor and current path against a typed guard."""

    try:
        status = os.fstat(opened.descriptor)
    except OSError as error:
        raise RecoveryError(opened.path, str(error)) from error
    require_opened_identity(opened, status)
    if (
        status.st_dev != identity.device
        or status.st_ino != identity.inode
        or status.st_size != identity.size
        or status.st_mtime_ns != identity.modified_ns
        or status.st_mode != identity.mode
        or status.st_nlink != identity.link_count
    ):
        raise RecoveryError(opened.path, "artifact changed after inspection")
    _require_current_path_identity(opened)
    size, digest = hash_open_artifact(opened)
    if size != identity.size or digest != identity.sha256:
        raise RecoveryError(opened.path, "artifact changed after inspection")


def with_verified_artifact[ResultT](
    identity: ArtifactIdentity,
    operation: Callable[[int], ResultT],
) -> ResultT:
    """Run mutation against the same guarded descriptor that was revalidated."""

    opened = open_guarded_artifact(identity.path, writable=True)
    primary_error: BaseException | None = None
    try:
        verify_artifact_identity(opened, identity)
        result = operation(opened.descriptor)
        status = os.fstat(opened.descriptor)
        require_opened_identity(opened, status)
        _require_current_path_identity(opened)
        return result
    except BaseException as error:
        if isinstance(error, RecoveryError):
            primary_error = error
            raise
        wrapped_error = RecoveryError(identity.path, str(error))
        primary_error = wrapped_error
        raise wrapped_error from error
    finally:
        try:
            close_artifact(opened)
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Artifact descriptor cleanup failed for {identity.path}: "
                f"{close_error}"
            )


def capture_directory_identity(path: Path) -> DirectoryIdentity:
    """Capture one safe directory for later replacement-plan validation."""

    try:
        status = path.lstat()
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    require_safe_directory_status(path, status)
    return DirectoryIdentity(path, status.st_dev, status.st_ino, status.st_mode)


def verify_directory_identity(identity: DirectoryIdentity) -> None:
    """Require a replacement-plan directory to remain the same real directory."""

    try:
        status = identity.path.lstat()
    except OSError as error:
        raise RecoveryError(identity.path, str(error)) from error
    require_safe_directory_status(identity.path, status)
    if (
        status.st_dev != identity.device
        or status.st_ino != identity.inode
        or status.st_mode != identity.mode
    ):
        raise RecoveryError(identity.path, "directory changed after inspection")


def is_reparse_point(path: Path) -> bool:
    """Return whether lstat identifies a link or Windows reparse point."""

    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    return status_is_reparse(status)


def _require_safe_regular_status(path: Path, status: os.stat_result) -> None:
    if status_is_reparse(status):
        raise RecoveryError(path, "symbolic links and reparse points are unsafe")
    if not stat.S_ISREG(status.st_mode):
        raise RecoveryError(path, "artifact must be a regular file")
    if status.st_nlink != 1:
        raise RecoveryError(path, "artifact link count must be 1")


def _require_same_file(
    path: Path,
    first: os.stat_result,
    second: os.stat_result,
) -> None:
    if first.st_dev != second.st_dev or first.st_ino != second.st_ino:
        raise RecoveryError(path, "artifact identity changed while opening")


def require_opened_identity(opened: OpenArtifact, status: os.stat_result) -> None:
    """Require current descriptor metadata to retain its opening identity."""

    _require_safe_regular_status(opened.path, status)
    if status.st_dev != opened.device or status.st_ino != opened.inode:
        raise RecoveryError(opened.path, "opened artifact identity changed")


def _require_current_path_identity(opened: OpenArtifact) -> None:
    try:
        current = opened.path.lstat()
    except OSError as error:
        raise RecoveryError(opened.path, f"artifact path changed: {error}") from error
    _require_safe_regular_status(opened.path, current)
    if current.st_dev != opened.device or current.st_ino != opened.inode:
        raise RecoveryError(opened.path, "artifact path identity changed")


def status_is_reparse(status: os.stat_result) -> bool:
    """Inspect POSIX link mode and Windows file attributes."""

    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)
