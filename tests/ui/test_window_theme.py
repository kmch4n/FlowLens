"""Native theme requests remain best effort and preserve window controls."""

import ctypes
import sys
from types import SimpleNamespace
from typing import Any

from pytest import MonkeyPatch

from flowlens.ui import window_theme


def test_dark_caption_falls_back_to_legacy_attribute(monkeypatch: MonkeyPatch) -> None:
    calls: list[int] = []

    class Api:
        def __call__(self, window: Any, attribute: int, value: Any, size: int) -> int:
            calls.append(attribute)
            return 0 if attribute == 19 else -1

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(dwmapi=SimpleNamespace(DwmSetWindowAttribute=Api())),
        raising=False,
    )
    assert window_theme.apply_dark_caption(123)
    assert calls == [20, 19]


def test_non_windows_is_safe(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert not window_theme.apply_dark_caption(123)
