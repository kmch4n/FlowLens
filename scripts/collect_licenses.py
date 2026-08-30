"""Collect pinned runtime dependency licenses into a package inventory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

PINNED_DEPENDENCIES = {
    "av": "18.1.0",
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.1",
    "colorama": "0.4.6",
    "ctranslate2": "4.7.2",
    "diskcache": "5.6.3",
    "faster-whisper": "1.2.1",
    "filelock": "3.32.4",
    "flatbuffers": "25.12.19",
    "fsspec": "2026.7.0",
    "huggingface_hub": "0.36.2",
    "idna": "3.19",
    "Jinja2": "3.1.6",
    "llama-cpp-python": "0.3.35",
    "MarkupSafe": "3.0.3",
    "numpy": "2.5.2",
    "onnxruntime": "1.29.0",
    "packaging": "26.3",
    "protobuf": "4.25.9",
    "PyAudioWPatch": "0.2.12.8",
    "PySide6": "6.11.2",
    "PySide6_Addons": "6.11.2",
    "PySide6_Essentials": "6.11.2",
    "python-ulid": "3.1.0",
    "PyYAML": "6.0.3",
    "requests": "2.34.2",
    "setuptools": "81.0.0",
    "shiboken6": "6.11.2",
    "soxr": "1.1.0",
    "tokenizers": "0.22.2",
    "tqdm": "4.70.0",
    "typing_extensions": "4.16.0",
    "urllib3": "2.7.0",
    "webrtcvad-wheels": "2.0.14",
}
FALLBACK_LICENSES = {
    "ctranslate2": "CTranslate2-MIT.txt",
    "flatbuffers": "Qwen3-4B-Instruct-2507-Apache-2.0.txt",
    "PySide6": "PySide6-LGPL-3.0-only.txt",
    "PySide6_Addons": "PySide6-LGPL-3.0-only.txt",
    "PySide6_Essentials": "PySide6-LGPL-3.0-only.txt",
    "shiboken6": "PySide6-LGPL-3.0-only.txt",
    "tokenizers": "Qwen3-4B-Instruct-2507-Apache-2.0.txt",
}


class DistributionPort(Protocol):
    """Installed-distribution fields required by the collector."""

    version: str
    files: Sequence[Path] | None

    def locate_file(self, path: Path) -> Path: ...


DistributionLoader = Callable[[str], DistributionPort]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(name: str) -> str:
    value = "".join(
        character.lower() if character.isalnum() else "-" for character in name
    ).strip("-")
    if not value:
        raise ValueError("dependency name has no safe path component")
    return value


def _is_license_path(path: Path) -> bool:
    lowered = path.name.casefold()
    return "license" in lowered or "licence" in lowered or "copying" in lowered


def _copy_license(
    source: Path,
    destination: Path,
    relative: Path,
) -> dict[str, str]:
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise ValueError(f"license source is not a regular nonempty file: {source}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"path": relative.as_posix(), "sha256": _sha256(target)}


def collect_dependency_licenses(
    destination: Path,
    fallback_root: Path,
    *,
    distribution_loader: DistributionLoader | None = None,
) -> dict[str, object]:
    """Copy every pinned dependency license and write its hash inventory."""

    destination.mkdir(parents=True, exist_ok=True)
    dependency_root = destination / "dependencies"
    if dependency_root.exists():
        raise ValueError("dependency license destination already exists")
    dependency_root.mkdir()
    dependencies: list[dict[str, object]] = []
    active_loader = (
        cast(DistributionLoader, importlib.metadata.distribution)
        if distribution_loader is None
        else distribution_loader
    )
    for name, expected_version in PINNED_DEPENDENCIES.items():
        distribution = active_loader(name)
        if distribution.version != expected_version:
            raise ValueError(
                f"dependency version mismatch for {name}: "
                f"expected {expected_version}, found {distribution.version}"
            )
        directory = Path("dependencies") / _safe_component(name)
        copied: list[dict[str, str]] = []
        fallback_name = FALLBACK_LICENSES.get(name)
        if fallback_name is not None:
            copied.append(
                _copy_license(
                    fallback_root / fallback_name,
                    destination,
                    directory / fallback_name,
                )
            )
        else:
            candidates = sorted(
                (Path(item) for item in distribution.files or ()),
                key=lambda item: item.as_posix().casefold(),
            )
            for index, relative_source in enumerate(
                item for item in candidates if _is_license_path(item)
            ):
                copied.append(
                    _copy_license(
                        Path(distribution.locate_file(relative_source)),
                        destination,
                        directory / f"{index:02d}-{relative_source.name}",
                    )
                )
        if not copied:
            raise ValueError(f"no license files found for dependency: {name}")
        dependencies.append(
            {
                "name": name,
                "version": expected_version,
                "license_files": copied,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dependencies": dependencies,
    }
    manifest_path = destination / "dependency-licenses.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--fallback-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Collect licenses for the active pinned build environment."""

    arguments = _build_parser().parse_args(argv)
    collect_dependency_licenses(
        cast(Path, arguments.destination),
        cast(Path, arguments.fallback_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
