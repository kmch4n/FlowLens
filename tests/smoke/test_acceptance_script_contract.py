"""Static safety contract for the elevated Windows acceptance harness."""

from pathlib import Path


def _script() -> str:
    return Path("scripts/run_acceptance.ps1").read_text(encoding="utf-8")


def test_acceptance_script_enforces_requested_active_duration_and_pause() -> None:
    script = _script()
    assert "--minimum-active-seconds $minimumActiveSeconds" in script
    assert "--require-pause" in script
    assert "$minimumActiveSeconds = $MinimumActiveMinutes * 60" in script


def test_acceptance_script_always_stops_its_recorded_process_tree() -> None:
    script = _script()
    assert "$null -ne $rootProcess -and -not $rootProcess.HasExited) {" in script
    assert (
        "$null -ne $rootProcess -and -not $rootProcess.HasExited -and $RecoveryCheck"
        not in script
    )


def test_acceptance_script_removes_only_its_verified_exact_rule() -> None:
    script = _script()
    assert "if (-not (Test-OwnedFirewallRule -RuleName $ruleName" in script
    assert "Refusing to remove a firewall rule that is no longer owned" in script


def test_acceptance_script_checks_application_completed_before_collection() -> None:
    script = _script()
    assert "Assert-CompletedApplicationReport" in script
    assert "completion_available" in script
    assert "--queue-overflows 0" not in script


def test_integration_smoke_uses_the_production_composition_root() -> None:
    script = Path("scripts/smoke_integration.py").read_text(encoding="utf-8")
    assert "build_application(paths, AppOptions())" in script
    assert "production_worker_targets" not in script


def test_integration_smoke_excludes_worker_startup_from_active_duration() -> None:
    script = Path("scripts/smoke_integration.py").read_text(encoding="utf-8")
    assert "recording_started: float | None = None" in script
    assert "recording_started = now" in script
    assert "active = now - recording_started - total_paused" in script
