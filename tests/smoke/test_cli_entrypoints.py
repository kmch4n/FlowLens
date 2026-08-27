import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script",
    ["scripts/collect_acceptance.py", "scripts/smoke_integration.py"],
)
def test_python_smoke_script_supports_direct_help_invocation(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
