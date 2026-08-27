"""Hallmark Midnight design contract for the FlowLens Qt surface."""

import sys
from dataclasses import astuple, dataclass
from pathlib import Path
from string import Formatter

from PySide6.QtGui import QFontDatabase

_SOURCE_RESOURCE_ROOT = Path(__file__).resolve().parents[3] / "assets"
_EXPECTED_PLACEHOLDERS = {
    "background",
    "surface",
    "elevated_surface",
    "rule",
    "primary_text",
    "muted_text",
    "accent",
    "focus",
    "error",
    "success",
    "font_interface",
    "font_mono",
    "motion_duration",
}


@dataclass(frozen=True, slots=True)
class DesignTokens:
    """Immutable color and typography tokens approved for the MVP."""

    background: str
    surface: str
    elevated_surface: str
    rule: str
    primary_text: str
    muted_text: str
    accent: str
    focus: str
    error: str
    success: str
    font_interface: str = "IBM Plex Sans JP"
    font_mono: str = "IBM Plex Mono"

    @classmethod
    def approved(cls) -> "DesignTokens":
        """Return the approved Hallmark Midnight token set."""

        return cls(
            background="#0D1117",
            surface="#131922",
            elevated_surface="#19212C",
            rule="#2A3441",
            primary_text="#E6EDF3",
            muted_text="#9AA7B5",
            accent="#D6A13D",
            focus="#78A9FF",
            error="#E16A6A",
            success="#55B982",
        )

    def values(self) -> tuple[str, ...]:
        """Return every token value in stable declaration order."""

        return astuple(self)


@dataclass(frozen=True, slots=True)
class FontFamilies:
    """Qt-verified family names for bundled FlowLens fonts."""

    interface: str
    mono: str
    font_ids: tuple[int, ...]


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG 2.1 contrast ratio for two opaque hex colors."""

    foreground_luminance = _relative_luminance(foreground)
    background_luminance = _relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def resolve_resource_root() -> Path:
    """Return the source or PyInstaller-owned asset root for this process."""

    if getattr(sys, "frozen", False) is True:
        bundle_root = getattr(sys, "_MEIPASS", None)
        if type(bundle_root) is not str or not bundle_root:
            raise RuntimeError("Frozen application resource root is unavailable")
        return Path(bundle_root) / "assets"
    return _SOURCE_RESOURCE_ROOT


def build_stylesheet(
    tokens: DesignTokens,
    reduced_motion: bool,
    resource_root: Path | None = None,
) -> str:
    """Resolve the QSS template against semantic tokens exactly once."""

    root = resolve_resource_root() if resource_root is None else resource_root
    template = (root / "styles" / "flowlens.qss").read_text(encoding="utf-8")
    placeholders = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }
    unknown = placeholders - _EXPECTED_PLACEHOLDERS
    missing = _EXPECTED_PLACEHOLDERS - placeholders
    if unknown or missing:
        raise ValueError(
            "Unexpected FlowLens stylesheet placeholders: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )

    values = {
        "background": tokens.background,
        "surface": tokens.surface,
        "elevated_surface": tokens.elevated_surface,
        "rule": tokens.rule,
        "primary_text": tokens.primary_text,
        "muted_text": tokens.muted_text,
        "accent": tokens.accent,
        "focus": tokens.focus,
        "error": tokens.error,
        "success": tokens.success,
        "font_interface": tokens.font_interface,
        "font_mono": tokens.font_mono,
        "motion_duration": "0ms" if reduced_motion else "120ms",
    }
    return template.format(**values)


def load_bundled_fonts(resource_root: Path) -> FontFamilies:
    """Load bundled IBM Plex fonts and verify their Qt family names."""

    fonts_root = resource_root / "fonts"
    font_paths = (
        fonts_root / "IBMPlexSansJP-Regular.ttf",
        fonts_root / "IBMPlexSansJP-SemiBold.ttf",
        fonts_root / "IBMPlexMono-Regular.ttf",
    )
    font_ids = tuple(_load_font(path) for path in font_paths)
    families = tuple(
        family
        for font_id in font_ids
        for family in QFontDatabase.applicationFontFamilies(font_id)
    )
    missing = {
        family
        for family in ("IBM Plex Sans JP", "IBM Plex Mono")
        if family not in families
    }
    if missing:
        raise RuntimeError(
            "Bundled font family names were not registered: "
            + ", ".join(sorted(missing))
        )
    return FontFamilies("IBM Plex Sans JP", "IBM Plex Mono", font_ids)


def _load_font(path: Path) -> int:
    font_id = QFontDatabase.addApplicationFont(str(path))
    if font_id < 0:
        raise RuntimeError(f"Bundled font could not be loaded: {path}")
    return font_id


def _relative_luminance(hex_color: str) -> float:
    red, green, blue = _hex_channels(hex_color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _hex_channels(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color: {hex_color}")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return (
        _linearized(red / 255),
        _linearized(green / 255),
        _linearized(blue / 255),
    )


def _linearized(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return float(((channel + 0.055) / 1.055) ** 2.4)
