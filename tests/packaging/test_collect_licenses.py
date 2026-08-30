"""Behavioral tests for dependency-license collection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import ClassVar, Protocol, cast

import pytest

_SCRIPT = Path("scripts/collect_licenses.py")
_SPEC = importlib.util.spec_from_file_location("flowlens_collect_licenses", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class DistributionPort(Protocol):
    """Distribution fields used by the collector."""

    version: str
    files: list[Path]

    def locate_file(self, path: Path) -> Path: ...


class FakeDistribution:
    """One installed distribution rooted in a controlled test directory."""

    def __init__(self, root: Path, version: str, license_name: str) -> None:
        self.version = version
        self.files = [Path(license_name)]
        self._root = root

    def locate_file(self, path: Path) -> Path:
        return self._root / path


def test_runtime_requirements_pin_the_complete_license_inventory() -> None:
    """A clean install must reproduce every inventoried runtime version."""

    requirements = {}
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator == "=="
        requirements[name] = version

    assert requirements == _MODULE.PINNED_DEPENDENCIES


def test_collector_inventories_every_pinned_dependency(tmp_path: Path) -> None:
    """Removing any Python distribution from the closure must fail this contract."""

    installed = tmp_path / "installed"
    installed.mkdir()
    distributions: dict[str, DistributionPort] = {}
    pinned = cast(dict[str, str], _MODULE.PINNED_DEPENDENCIES)
    fallbacks = cast(dict[str, str], _MODULE.FALLBACK_LICENSES)
    for name, version in pinned.items():
        license_name = f"{name}.dist-info/licenses/LICENSE.txt"
        path = installed / license_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"license for {name}\n", encoding="utf-8", newline="\n")
        distributions[name] = FakeDistribution(installed, version, license_name)

    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    for name, filename in fallbacks.items():
        (fallback_root / filename).write_text(
            f"fallback for {name}\n",
            encoding="utf-8",
            newline="\n",
        )

    destination = tmp_path / "licenses"
    manifest = _MODULE.collect_dependency_licenses(
        destination,
        fallback_root,
        distribution_loader=lambda name: distributions[name],
    )

    assert {item["name"] for item in manifest["dependencies"]} == set(pinned)
    assert (
        json.loads(
            (destination / "dependency-licenses.json").read_text(encoding="utf-8")
        )
        == manifest
    )
    for item in manifest["dependencies"]:
        assert item["version"] == pinned[item["name"]]
        assert item["license_files"]
        for license_file in item["license_files"]:
            assert (destination / license_file["path"]).is_file()


def test_collector_fails_closed_for_wrong_installed_version(tmp_path: Path) -> None:
    """A stale build environment cannot silently produce a false inventory."""

    class WrongDistribution:
        version = "0.0.0"
        files: ClassVar[list[Path]] = []

        def locate_file(self, path: Path) -> Path:
            return tmp_path / path

    with pytest.raises(ValueError, match="version mismatch"):
        _MODULE.collect_dependency_licenses(
            tmp_path / "licenses",
            tmp_path,
            distribution_loader=lambda name: WrongDistribution(),
        )
