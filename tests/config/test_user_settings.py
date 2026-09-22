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
