import os
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from flowlens.config.model import AppConfig
from flowlens.config.store import ConfigLoadError, ConfigStore


def test_missing_config_returns_defaults_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "config.json"

    assert ConfigStore(path).load() == AppConfig.default()
    assert not path.exists()


def test_save_is_utf8_without_bom_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)

    store.save(AppConfig.default())

    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert store.load() == AppConfig.default()


@pytest.mark.parametrize(
    "contents, reason",
    [
        ('{"schema_version": 2}', "schema_version"),
        ("{not valid json", "Expecting property name"),
    ],
)
def test_invalid_config_is_reported_without_changing_its_bytes(
    tmp_path: Path,
    contents: str,
    reason: str,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigLoadError, match=reason):
        ConfigStore(path).load()

    assert path.read_text(encoding="utf-8") == contents


def test_invalid_utf8_config_is_reported_without_changing_its_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    original = b'{"schema_version": "\xff"}'
    path.write_bytes(original)

    with pytest.raises(ConfigLoadError, match="utf-8"):
        ConfigStore(path).load()

    assert path.read_bytes() == original


def test_deep_json_is_reported_without_changing_its_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = "[" * 3_000 + "]" * 3_000
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="maximum recursion depth"):
        ConfigStore(path).load()

    assert path.read_text(encoding="utf-8") == original


def test_save_removes_its_temp_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    path = tmp_path / "nested" / "config.json"
    store = ConfigStore(path)

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.save(AppConfig.default())

    assert not path.with_name("config.json.tmp").exists()
