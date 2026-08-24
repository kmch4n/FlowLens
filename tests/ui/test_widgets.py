from pytestqt.qtbot import QtBot

from flowlens.ui.widgets import InputMeter, StatefulButton, StatusIndicator


def test_stateful_button_exposes_all_eight_states(qtbot: QtBot) -> None:
    button = StatefulButton("Start session")
    qtbot.addWidget(button)
    assert set(button.supported_states) == {
        "default",
        "hover",
        "focus",
        "active",
        "disabled",
        "loading",
        "error",
        "success",
    }
    button.set_ui_state("loading", "Checking models")
    assert button.isEnabled() is False
    assert button.accessibleDescription() == "Checking models"
    button.set_ui_state("error", "Model checksum failed")
    assert button.property("uiState") == "error"
    assert "Model checksum failed" in button.accessibleDescription()


def test_primary_controls_have_44_by_44_minimum(qtbot: QtBot) -> None:
    button = StatefulButton("Stop")
    qtbot.addWidget(button)
    assert button.minimumWidth() >= 44
    assert button.minimumHeight() >= 44


def test_status_indicator_pairs_state_with_text_and_icon(qtbot: QtBot) -> None:
    indicator = StatusIndicator("ASR")
    qtbot.addWidget(indicator)
    indicator.set_status("error", "ASR worker stopped")
    assert indicator.property("uiState") == "error"
    assert "Error" in indicator.text()
    assert "!" in indicator.text()
    assert indicator.accessibleDescription() == "Error: ASR worker stopped"
    indicator.set_status("success", "Audio ready")
    assert "Ready" in indicator.text()
    assert "✓" in indicator.text()


def test_input_meter_clamps_levels_and_keeps_accessible_name(qtbot: QtBot) -> None:
    meter = InputMeter("Microphone")
    qtbot.addWidget(meter)
    assert meter.minimumHeight() >= 44
    assert meter.accessibleName() == "Microphone input level"
    meter.set_level(1.25)
    assert meter.value() == 100
    meter.set_level(-0.25)
    assert meter.value() == 0
