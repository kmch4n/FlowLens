from __future__ import annotations

from pathlib import Path

import pytest

from flowlens.adapters.windows_shell import WindowsFolderOpener


def test_folder_opener_rejects_relative_or_missing_path(tmp_path: Path) -> None:
    called: list[str] = []
    opener = WindowsFolderOpener(shell_execute=called.append)

    with pytest.raises(ValueError):
        opener.open(Path("relative"))
    with pytest.raises(FileNotFoundError):
        opener.open(tmp_path / "missing")

    assert called == []


def test_folder_opener_accepts_existing_absolute_directory(tmp_path: Path) -> None:
    called: list[str] = []
    opener = WindowsFolderOpener(shell_execute=called.append)

    opener.open(tmp_path)

    assert called == [str(tmp_path.resolve())]


def test_folder_opener_rejects_files(tmp_path: Path) -> None:
    file_path = tmp_path / "session.json"
    file_path.write_text("{}", encoding="utf-8")
    called: list[str] = []
    opener = WindowsFolderOpener(shell_execute=called.append)

    with pytest.raises(NotADirectoryError):
        opener.open(file_path)

    assert called == []


def test_folder_opener_resolves_directory_symlink_before_opening(
    tmp_path: Path,
) -> None:
    target = tmp_path / "session"
    target.mkdir()
    link = tmp_path / "session-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")
    called: list[str] = []
    opener = WindowsFolderOpener(shell_execute=called.append)

    opener.open(link)

    assert called == [str(target.resolve())]


def test_folder_opener_propagates_shell_execute_errors(tmp_path: Path) -> None:
    def fail(_: str) -> None:
        raise OSError("shell unavailable")

    opener = WindowsFolderOpener(shell_execute=fail)

    with pytest.raises(OSError, match="shell unavailable"):
        opener.open(tmp_path)
