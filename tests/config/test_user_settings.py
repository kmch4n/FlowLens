"""Settings round-trip independently of the legacy window configuration."""

from pathlib import Path

import pytest

from flowlens.config.user_settings import SettingsStore, UserSettings


def test_settings_round_trip_and_default_without_writing(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    assert store.load() == UserSettings()
    assert not store.path.exists()
    selected = UserSettings("low", 800, 18)
    store.save(selected)
    assert store.load() == selected
    assert selected.min_speech_rms == 400


@pytest.mark.parametrize(
    "values",
    [("bad", 450, 16), ("normal", 10, 16), ("normal", 450, 99), ("normal", True, 16)],
)
def test_invalid_settings_are_rejected(values: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        UserSettings(*values)  # type: ignore[arg-type]


def test_corrupt_settings_use_defaults_and_preserve_original_on_save(
    tmp_path: Path,
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.path.write_bytes(b"broken settings")
    settings, warning = store.load_for_use()
    assert settings == UserSettings()
    assert warning
    assert store.path.read_bytes() == b"broken settings"
    store.save(UserSettings("low", 600, 18))
    backups = list(tmp_path.glob("settings.json.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"broken settings"
    assert store.load().sensitivity == "low"
