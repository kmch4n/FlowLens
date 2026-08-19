"""Local application paths and deterministic session-folder naming."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flowlens.domain.ids import new_ulid

_SESSION_ID_PATTERN = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Application-owned paths rooted in the local app-data directory."""

    root: Path
    config: Path
    models: Path
    sessions: Path

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "AppPaths":
        """Resolve application paths from an injected environment mapping.

        Args:
            environment: Environment variables containing ``LOCALAPPDATA``.

        Returns:
            Paths rooted at the local FlowLens application directory.

        Raises:
            RuntimeError: If ``LOCALAPPDATA`` is absent or empty.
        """

        local_app_data = environment.get("LOCALAPPDATA")
        if not local_app_data:
            raise RuntimeError("LOCALAPPDATA is required to resolve application paths")
        root = Path(local_app_data) / "FlowLens"
        return cls(
            root=root,
            config=root / "config.json",
            models=root / "models",
            sessions=root / "sessions",
        )


def _require_session_id(session_id: str) -> str:
    """Validate a session ID before it is incorporated into a path."""

    if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError("session_id must contain 26 uppercase Crockford characters")
    return session_id


def session_directory_name(started_at: datetime, session_id: str) -> str:
    """Build a Windows-safe session directory name.

    Args:
        started_at: Time the session began, including a UTC offset.
        session_id: Uppercase Crockford ULID assigned to the session.

    Returns:
        The deterministic timestamp-and-ID directory name.

    Raises:
        ValueError: If the timestamp is naive or the session ID is invalid.
    """

    if started_at.utcoffset() is None:
        raise ValueError("started_at must include a timezone")
    return f"{started_at:%Y%m%dT%H%M%S%z}_{_require_session_id(session_id)}"


def new_session_directory(
    sessions_root: Path,
    started_at: datetime,
    id_factory: Callable[[], str] = new_ulid,
) -> Path:
    """Return the location for a new session without creating it.

    Args:
        sessions_root: Parent directory that stores session directories.
        started_at: Time the session began, including a UTC offset.
        id_factory: Factory that supplies one shared session identifier.

    Returns:
        The session directory path, whether or not it already exists.
    """

    session_id = id_factory()
    return sessions_root / session_directory_name(started_at, session_id)
