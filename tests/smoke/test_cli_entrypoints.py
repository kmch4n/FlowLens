import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    "check_package.py",
    "validate_session.py",
    "collect_acceptance.py",
    "smoke_discussion.py",
    "smoke_integration.py",
)
KNOWN_REPOSITORY_ARTIFACTS = (
    REPOSITORY_ROOT / "build" / "FlowLens",
    REPOSITORY_ROOT / "dist" / "FlowLens",
    REPOSITORY_ROOT / "build" / "reports" / "discussion-smoke.json",
    REPOSITORY_ROOT / "build" / "reports" / "integration-smoke.json",
    REPOSITORY_ROOT / "build" / "reports" / "acceptance-30m.json",
    REPOSITORY_ROOT / "build" / "reports" / "acceptance-recovery.json",
)


def _artifact_snapshot(paths: tuple[Path, ...]) -> tuple[tuple[str, bool, int], ...]:
    return tuple(
        (
            str(path),
            path.exists(),
            path.stat().st_mtime_ns if path.exists() else 0,
        )
        for path in paths
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_python_verification_script_help_is_side_effect_free(
    script: str,
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "guard"
    guard_root.mkdir()
    marker = tmp_path / "forbidden-side-effect.txt"
    (guard_root / "sitecustomize.py").write_text(
        """
import multiprocessing.process
import os
import socket
import subprocess
from pathlib import Path

marker = Path(os.environ["FLOWLENS_SIDE_EFFECT_MARKER"])

def forbidden(*args, **kwargs):
    marker.write_text("forbidden side effect", encoding="utf-8")
    raise RuntimeError("CLI help attempted a forbidden side effect")

socket.socket = forbidden
socket.create_connection = forbidden
multiprocessing.process.BaseProcess.start = forbidden
subprocess.Popen = forbidden
""".lstrip(),
        encoding="utf-8",
        newline="\n",
    )
    for module in ("llama_cpp", "faster_whisper", "pyaudiowpatch"):
        (guard_root / f"{module}.py").write_text(
            'raise RuntimeError("CLI help imported a native runtime")\n',
            encoding="utf-8",
            newline="\n",
        )

    local_app_data = tmp_path / "local-app-data"
    isolated_cwd = tmp_path / "cwd"
    isolated_cwd.mkdir()
    isolated_artifacts = (
        local_app_data / "FlowLens",
        isolated_cwd / "build",
        isolated_cwd / "dist",
        isolated_cwd / "discussion-smoke.json",
        isolated_cwd / "integration-smoke.json",
        isolated_cwd / "acceptance-30m.json",
        isolated_cwd / "acceptance-recovery.json",
    )
    all_artifacts = KNOWN_REPOSITORY_ARTIFACTS + isolated_artifacts
    before = _artifact_snapshot(all_artifacts)
    environment = os.environ.copy()
    environment.update(
        {
            "APPDATA": str(tmp_path / "app-data"),
            "FLOWLENS_SIDE_EFFECT_MARKER": str(marker),
            "LOCALAPPDATA": str(local_app_data),
            "PYTHONPATH": str(guard_root),
            "TEMP": str(tmp_path / "temp"),
            "TMP": str(tmp_path / "temp"),
        }
    )

    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / script), "--help"],
        cwd=isolated_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not marker.exists()
    assert _artifact_snapshot(all_artifacts) == before
