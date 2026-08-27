import re
import sys
from pathlib import Path

from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot

from flowlens.ui.design import (
    DesignTokens,
    build_stylesheet,
    contrast_ratio,
    load_bundled_fonts,
    resolve_resource_root,
)


def test_tokens_match_approved_midnight_contract() -> None:
    assert (
        Path("requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        .count("PySide6==6.11.2")
        == 1
    )
    assert (
        Path("requirements-dev.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        .count("pytest-qt==4.5.0")
        == 1
    )
    tokens = DesignTokens.approved()
    assert tokens.background == "#0D1117"
    assert tokens.surface == "#131922"
    assert tokens.elevated_surface == "#19212C"
    assert tokens.rule == "#2A3441"
    assert tokens.primary_text == "#E6EDF3"
    assert tokens.muted_text == "#9AA7B5"
    assert tokens.accent == "#D6A13D"
    assert tokens.focus == "#78A9FF"
    assert tokens.error == "#E16A6A"
    assert tokens.success == "#55B982"
    assert "#000000" not in tokens.values()
    assert "#FFFFFF" not in tokens.values()
    assert contrast_ratio(tokens.primary_text, tokens.background) >= 4.5
    assert contrast_ratio(tokens.focus, tokens.background) >= 3.0


def test_stylesheet_has_hallmark_stamp_and_bans() -> None:
    stylesheet = build_stylesheet(DesignTokens.approved(), reduced_motion=False)
    assert stylesheet.startswith(
        "/* Hallmark · genre: atmospheric · macrostructure: Workbench · "
        "theme: Midnight · tone: technical-austere · enrichment: none */"
    )
    lowered = stylesheet.lower()
    for banned in (
        "gradient",
        "qgraphicsdropshadoweffect",
        "border-radius: 999",
        "transition-all",
    ):
        assert banned not in lowered


def test_stylesheet_resolves_tokens_once_without_placeholders() -> None:
    stylesheet = build_stylesheet(DesignTokens.approved(), reduced_motion=True)
    assert "{background}" not in stylesheet
    assert "duration: 0ms" in stylesheet
    assert 'font-family: "IBM Plex Sans JP"' in stylesheet
    assert 'font-family: "IBM Plex Mono"' in stylesheet


def test_qss_padding_shorthand_uses_four_pixel_spacing_grid() -> None:
    stylesheet = build_stylesheet(DesignTokens.approved(), reduced_motion=True)
    violations: list[str] = []
    for match in re.finditer(r"padding:\s*([^;]+);", stylesheet):
        declaration = match.group(0)
        values = [int(value) for value in re.findall(r"(\d+)px", declaration)]
        if any(value % 4 != 0 for value in values):
            violations.append(declaration)
    assert violations == []


def test_bundled_fonts_load_expected_qt_family_names(qtbot: QtBot) -> None:
    del qtbot
    families = load_bundled_fonts(Path("assets"))
    assert families.interface == "IBM Plex Sans JP"
    assert families.mono == "IBM Plex Mono"


def test_resource_root_uses_source_assets_when_not_frozen(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    root = resolve_resource_root()

    assert root == Path(__file__).resolve().parents[2] / "assets"


def test_resource_root_uses_pyinstaller_bundle_assets_when_frozen(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "runtime"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)

    assert resolve_resource_root() == bundle_root / "assets"


def test_stylesheet_reads_from_selected_package_resource_root(tmp_path: Path) -> None:
    styles = tmp_path / "assets" / "styles"
    styles.mkdir(parents=True)
    source = Path("assets/styles/flowlens.qss").read_text(encoding="utf-8")
    (styles / "flowlens.qss").write_text(source, encoding="utf-8", newline="\n")

    stylesheet = build_stylesheet(
        DesignTokens.approved(),
        reduced_motion=True,
        resource_root=tmp_path / "assets",
    )

    assert stylesheet.startswith("/* Hallmark")
