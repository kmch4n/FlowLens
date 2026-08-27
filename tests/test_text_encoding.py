from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".claude",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".superpowers",
    ".venv",
    ".venv-py311-backup",
    "__pycache__",
    "build",
    "dist",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".in",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".qss",
    ".spec",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore"}


def _repository_text_paths() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(REPOSITORY_ROOT.rglob("*"))
        if path.is_file()
        and not any(part in EXCLUDED_DIRECTORIES for part in path.parts)
        and (path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES)
    )


@pytest.mark.parametrize(
    "path",
    _repository_text_paths(),
    ids=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
)
def test_repository_text_is_utf8_without_bom_and_uses_lf(path: Path) -> None:
    data = path.read_bytes()

    assert not data.startswith(b"\xef\xbb\xbf")
    data.decode("utf-8")
    assert b"\r" not in data


def test_bundled_ibm_plex_license_is_utf8_without_bom_and_uses_lf() -> None:
    path = REPOSITORY_ROOT / "assets" / "fonts" / "OFL.txt"
    data = path.read_bytes()

    assert not data.startswith(b"\xef\xbb\xbf")
    data.decode("utf-8")
    assert b"\r" not in data
