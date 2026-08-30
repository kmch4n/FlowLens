"""Safety regressions for the Windows package build cleanup boundary."""

from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only PowerShell")


def _invoke_target_validator(
    repository_root: Path, relative_path: str
) -> subprocess.CompletedProcess[str]:
    """Load only build-script helpers and validate one non-destructive target."""

    source = Path("scripts/build_windows.ps1").read_text(encoding="utf-8")
    helpers = source.split("$repositoryRoot =", maxsplit=1)[0]
    command = (
        helpers
        + "\ntry {\n"
        + "    Remove-ValidatedPackageTarget -RepositoryRoot '"
        + str(repository_root).replace("'", "''")
        + "' -RelativePath '"
        + relative_path.replace("'", "''")
        + "' | Out-Null\n"
        + "    exit 0\n"
        + "}\ncatch {\n"
        + "    exit 7\n"
        + "}\n"
    )
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_build_cleanup_rejects_parent_junctions_and_non_package_targets(
    tmp_path: Path,
) -> None:
    """Cleanup cannot traverse a parent junction or accept an arbitrary child."""

    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    junction = repository / "build"
    creation = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if creation.returncode != 0:
        pytest.skip("Windows junction creation is unavailable for this test")

    assert _invoke_target_validator(repository, "build\\FlowLens").returncode == 7
    ordinary_repository = tmp_path / "ordinary-repository"
    ordinary_repository.mkdir()
    assert (
        _invoke_target_validator(ordinary_repository, "build\\unexpected").returncode
        == 7
    )


def test_build_cleanup_allows_an_explicit_repository_root_junction(
    tmp_path: Path,
) -> None:
    """A trusted workspace junction must not block an exact package target."""

    backing_repository = tmp_path / "backing-repository"
    backing_repository.mkdir()
    repository = tmp_path / "repository-junction"
    creation = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(repository), str(backing_repository)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if creation.returncode != 0:
        pytest.skip("Windows junction creation is unavailable for this test")

    assert _invoke_target_validator(repository, "build\\FlowLens").returncode == 0


def test_build_uses_a_supported_parameter_to_create_the_license_directory() -> None:
    """PowerShell New-Item accepts Path, not LiteralPath."""

    source = Path("scripts/build_windows.ps1").read_text(encoding="utf-8")

    assert "New-Item -ItemType Directory -Path $licenseDestination" in source
    assert "New-Item -ItemType Directory -LiteralPath" not in source
