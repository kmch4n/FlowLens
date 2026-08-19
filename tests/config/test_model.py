import pytest

from flowlens.config.model import AppConfig, Rect, clamp_window
from flowlens.domain._validation import ContractValidationError


def test_default_config_has_only_specified_preferences() -> None:
    value = AppConfig.default().to_dict()

    assert value == {
        "schema_version": 1,
        "window": {
            "x": 100,
            "y": 100,
            "width": 1280,
            "height": 800,
            "maximized": False,
            "always_on_top": False,
        },
        "devices": {"microphone_id": "", "loopback_output_id": ""},
        "last_mode": "MEETING",
    }


@pytest.mark.parametrize(
    "value",
    [
        {},
        {
            "schema_version": 1,
            "window": {
                "x": 100,
                "y": 100,
                "width": 1280,
                "height": 800,
                "maximized": False,
                "always_on_top": False,
            },
            "devices": {"microphone_id": "", "loopback_output_id": ""},
            "last_mode": "MEETING",
            "unexpected": True,
        },
        {
            "schema_version": 1,
            "window": {
                "x": 100,
                "y": 100,
                "width": 1280,
                "height": 800,
                "maximized": False,
            },
            "devices": {"microphone_id": "", "loopback_output_id": ""},
            "last_mode": "MEETING",
        },
    ],
)
def test_from_dict_rejects_missing_or_unknown_config_keys(value: object) -> None:
    with pytest.raises(ContractValidationError):
        AppConfig.from_dict(value)


def test_from_dict_rejects_invalid_config_field_types() -> None:
    window: dict[str, object] = {
        "x": 100,
        "y": 100,
        "width": 1280,
        "height": 800,
        "maximized": 1,
        "always_on_top": False,
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "window": window,
        "devices": {"microphone_id": "", "loopback_output_id": ""},
        "last_mode": "MEETING",
    }

    with pytest.raises(ContractValidationError, match="maximized"):
        AppConfig.from_dict(value)


def test_window_is_clamped_to_display_with_largest_intersection() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=1800,
        y=900,
        width=1280,
        height=800,
    )

    clamped = clamp_window(
        saved,
        (Rect(0, 0, 1920, 1080), Rect(1920, 0, 1920, 1080)),
    )

    assert (clamped.x, clamped.y, clamped.width, clamped.height) == (
        1920,
        280,
        1280,
        800,
    )


def test_offscreen_window_moves_to_primary_display() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=9000,
        y=9000,
        width=1280,
        height=800,
    )

    clamped = clamp_window(saved, (Rect(0, 0, 1920, 1080),))

    assert (clamped.x, clamped.y) == (640, 280)


def test_window_clamps_inside_negative_coordinate_display() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=-1800,
        y=900,
        width=1280,
        height=800,
    )

    clamped = clamp_window(
        saved,
        (Rect(-1920, 0, 1920, 1080), Rect(0, 0, 1920, 1080)),
    )

    assert (clamped.x, clamped.y, clamped.width, clamped.height) == (
        -1800,
        280,
        1280,
        800,
    )


def test_window_size_is_limited_by_selected_display() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=10,
        y=10,
        width=400,
        height=300,
    )

    clamped = clamp_window(saved, (Rect(0, 0, 800, 500),))

    assert (clamped.x, clamped.y, clamped.width, clamped.height) == (0, 0, 800, 500)


def test_equal_intersections_choose_the_first_display() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=1000,
        y=0,
        width=1000,
        height=800,
    )

    clamped = clamp_window(
        saved,
        (Rect(0, 0, 1400, 1080), Rect(1600, 0, 1400, 1080)),
    )

    assert (clamped.x, clamped.y) == (400, 0)


def test_zero_intersections_choose_the_first_display() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=9000,
        y=9000,
        width=1280,
        height=800,
    )

    clamped = clamp_window(
        saved,
        (Rect(-1920, 0, 1920, 1080), Rect(0, 0, 1920, 1080)),
    )

    assert (clamped.x, clamped.y) == (-1280, 280)


def test_window_flags_are_preserved_while_clamping() -> None:
    saved = AppConfig.default().window.with_geometry(
        x=9000,
        y=9000,
        width=1280,
        height=800,
    )
    saved = type(saved)(
        x=saved.x,
        y=saved.y,
        width=saved.width,
        height=saved.height,
        maximized=True,
        always_on_top=True,
    )

    clamped = clamp_window(saved, (Rect(0, 0, 1920, 1080),))

    assert (clamped.maximized, clamped.always_on_top) == (True, True)
