"""Reject unbounded smoke durations before accessing real devices."""

import json
from pathlib import Path

import pytest

from scripts import smoke_integration


@pytest.mark.parametrize(
    "option",
    ["--duration-seconds", "--pause-at-seconds", "--pause-duration-seconds"],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_times_fail_before_application_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    def forbidden_build(*args: object, **kwargs: object) -> object:
        pytest.fail("Invalid times reached application construction")

    monkeypatch.setattr(smoke_integration, "build_application", forbidden_build)
    report = tmp_path / "report.json"
    result = smoke_integration.main(
        [
            "--microphone-id=fixture-mic",
            "--loopback-output-id=fixture-output",
            f"{option}={value}",
            f"--report={report}",
        ]
    )

    assert result == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["error_type"] == "ValueError"
    assert "finite" in payload["error"]
