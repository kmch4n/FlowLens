from datetime import datetime
from pathlib import Path

import pytest

from flowlens.persistence.paths import (
    AppPaths,
    new_session_directory,
    session_directory_name,
)

SESSION_ID = "01J00000000000000000000000"
STARTED_AT = datetime.fromisoformat("2026-08-19T12:34:56+09:00")


def test_paths_are_rooted_at_local_appdata(tmp_path: Path) -> None:
    paths = AppPaths.from_environment({"LOCALAPPDATA": str(tmp_path)})

    assert paths.config == tmp_path / "FlowLens" / "config.json"
    assert paths.models == tmp_path / "FlowLens" / "models"
    assert paths.sessions == tmp_path / "FlowLens" / "sessions"


def test_missing_localappdata_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="LOCALAPPDATA"):
        AppPaths.from_environment({})


def test_session_folder_name_is_windows_safe() -> None:
    name = session_directory_name(STARTED_AT, SESSION_ID)

    assert name == "20260819T123456+0900_01J00000000000000000000000"
    assert ":" not in name


def test_session_folder_name_rejects_naive_datetime() -> None:
    naive_started_at = datetime.fromisoformat("2026-08-19T12:34:56")

    with pytest.raises(ValueError, match="timezone"):
        session_directory_name(naive_started_at, SESSION_ID)


@pytest.mark.parametrize(
    "session_id",
    [
        "01J0000000000000000000000",
        "01J000000000000000000000000",
        "01J00000000000000000000000/",
        "..\\J00000000000000000000000",
        "01J0000000000000000000000I",
        "01j00000000000000000000000",
    ],
)
def test_session_folder_name_rejects_invalid_or_unsafe_session_id(
    session_id: str,
) -> None:
    with pytest.raises(ValueError, match="session_id"):
        session_directory_name(STARTED_AT, session_id)


def test_session_directory_generation_accepts_deterministic_id_factory(
    tmp_path: Path,
) -> None:
    path = new_session_directory(
        tmp_path,
        STARTED_AT,
        id_factory=lambda: SESSION_ID,
    )

    assert path == tmp_path / "20260819T123456+0900_01J00000000000000000000000"
    assert not path.exists()


def test_session_directory_generation_calls_id_factory_once(tmp_path: Path) -> None:
    calls = 0

    def id_factory() -> str:
        nonlocal calls
        calls += 1
        return SESSION_ID

    new_session_directory(tmp_path, STARTED_AT, id_factory=id_factory)

    assert calls == 1


def test_session_directory_generation_returns_existing_collision_path(
    tmp_path: Path,
) -> None:
    existing_path = tmp_path / "20260819T123456+0900_01J00000000000000000000000"
    existing_path.mkdir()

    path = new_session_directory(
        tmp_path,
        STARTED_AT,
        id_factory=lambda: SESSION_ID,
    )

    assert path == existing_path
    assert path.is_dir()


def test_session_directory_generation_rejects_invalid_factory_id(
    tmp_path: Path,
) -> None:
    def factory() -> str:
        return "../unsafe"

    with pytest.raises(ValueError, match="session_id"):
        new_session_directory(tmp_path, STARTED_AT, id_factory=factory)
