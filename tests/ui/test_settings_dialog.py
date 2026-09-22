"""Settings changes remain local until explicitly saved."""

from pathlib import Path

from pytestqt.qtbot import QtBot

from flowlens.config.user_settings import SettingsStore
from flowlens.ui.settings_dialog import SettingsDialog


def test_cancel_does_not_write(qtbot: QtBot, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    dialog = SettingsDialog(store)
    qtbot.addWidget(dialog)
    dialog.sensitivity.setCurrentIndex(0)
    dialog.reject()
    assert not store.path.exists()


def test_save_persists_selected_values(qtbot: QtBot, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    dialog = SettingsDialog(store)
    qtbot.addWidget(dialog)
    dialog.sensitivity.setCurrentIndex(0)
    dialog.silence.setValue(800)
    dialog.text_size.setValue(18)
    with qtbot.waitSignal(dialog.saved):
        dialog._save()
    current = store.load()
    assert (current.min_speech_rms, current.silence_end_ms, current.text_size) == (
        400,
        800,
        18,
    )
