"""SessionWriter bootstrap-claim lifecycle tests."""

import os
from pathlib import Path

import pytest

from flowlens.persistence.json_files import AtomicJsonFile
from flowlens.persistence.session_writer import SessionWriter, _BootstrapClaim
from flowlens.persistence.wav_sink import WavSink
from tests.factories import make_discussion_state, make_manifest

_REQUIRED_ARTIFACTS = {
    "session.json",
    "mic.wav",
    "loopback.wav",
    "transcript.jsonl",
    "discussion-state.json",
    "state-history.jsonl",
    "events.jsonl",
}


def _inject_claim_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_close: bool,
    fail_unlink: bool,
) -> None:
    """Inject claim-only cleanup errors while leaving other cleanup real."""

    original_acquire = _BootstrapClaim.acquire
    original_close = os.close
    original_unlink = Path.unlink
    claim_descriptors: set[int] = set()

    def track_claim_acquire(
        cls: type[_BootstrapClaim],
        session_dir: Path,
    ) -> _BootstrapClaim:
        del cls
        claim = original_acquire(session_dir)
        assert claim.descriptor is not None
        claim_descriptors.add(claim.descriptor)
        return claim

    def injected_close(descriptor: int) -> None:
        original_close(descriptor)
        if fail_close and descriptor in claim_descriptors:
            raise OSError("injected cleanup failure")

    def injected_unlink(path: Path, missing_ok: bool = False) -> None:
        if fail_unlink and path.name == ".flowlens-bootstrap.claim":
            raise OSError("injected cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        _BootstrapClaim,
        "acquire",
        classmethod(track_claim_acquire),
    )
    monkeypatch.setattr("flowlens.persistence.session_writer.os.close", injected_close)
    monkeypatch.setattr(Path, "unlink", injected_unlink)


@pytest.mark.parametrize(
    "failure_stage",
    ["claim", "manifest_json", "discussion_json"],
)
def test_construction_failure_releases_claim_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Every post-open construction failure must release its fd and claim."""

    session_dir = tmp_path / "session"
    original_close = os.close
    original_claim_init = _BootstrapClaim.__init__
    original_atomic_init = AtomicJsonFile.__init__
    claim_close_count = 0

    def track_claim_close(descriptor: int) -> None:
        nonlocal claim_close_count
        claim_close_count += 1
        original_close(descriptor)

    def fail_claim_init(
        self: _BootstrapClaim,
        path: Path,
        descriptor: int | None,
        owns_path: bool = True,
    ) -> None:
        del self, path, descriptor, owns_path
        raise OSError("claim construction failed")

    atomic_init_count = 0
    target_atomic_init = 1 if failure_stage == "manifest_json" else 2

    def fail_selected_atomic_init(
        self: AtomicJsonFile,
        path: Path,
    ) -> None:
        nonlocal atomic_init_count
        atomic_init_count += 1
        if atomic_init_count == target_atomic_init:
            raise OSError(f"{failure_stage} construction failed")
        original_atomic_init(self, path)

    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.close",
        track_claim_close,
    )
    if failure_stage == "claim":
        monkeypatch.setattr(_BootstrapClaim, "__init__", fail_claim_init)
        failure_message = "claim construction failed"
    else:
        monkeypatch.setattr(AtomicJsonFile, "__init__", fail_selected_atomic_init)
        failure_message = f"{failure_stage} construction failed"

    with pytest.raises(OSError, match=failure_message):
        SessionWriter.open(session_dir, make_manifest(), make_discussion_state())

    assert claim_close_count == 1
    assert session_dir.exists()
    assert not tuple(session_dir.iterdir())

    monkeypatch.setattr("flowlens.persistence.session_writer.os.close", original_close)
    monkeypatch.setattr(_BootstrapClaim, "__init__", original_claim_init)
    monkeypatch.setattr(AtomicJsonFile, "__init__", original_atomic_init)

    writer = SessionWriter.open(session_dir, make_manifest(), make_discussion_state())
    assert {path.name for path in session_dir.iterdir()} == _REQUIRED_ARTIFACTS
    writer.close_incomplete()


@pytest.mark.parametrize("cleanup_mode", ["close", "unlink", "both"])
def test_primary_bootstrap_error_reports_each_claim_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
) -> None:
    """Every claim cleanup problem must be noted directly on the primary error."""

    primary_error = OSError("primary bootstrap failure")

    def fail_wav_open(cls: type[WavSink], path: Path) -> WavSink:
        del cls, path
        raise primary_error

    monkeypatch.setattr(WavSink, "open", classmethod(fail_wav_open))
    _inject_claim_cleanup_failures(
        monkeypatch,
        fail_close=cleanup_mode in {"close", "both"},
        fail_unlink=cleanup_mode in {"unlink", "both"},
    )

    with pytest.raises(OSError) as raised:
        SessionWriter.open(
            tmp_path / "session",
            make_manifest(),
            make_discussion_state(),
        )

    assert raised.value is primary_error
    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    if cleanup_mode in {"close", "both"}:
        assert "claim descriptor close" in notes
    if cleanup_mode in {"unlink", "both"}:
        assert "claim unlink" in notes
    assert (tmp_path / "session" / "session.json").exists()


@pytest.mark.parametrize("cleanup_mode", ["close", "unlink", "both"])
def test_claim_cleanup_only_failure_is_diagnostic_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
) -> None:
    """Claim cleanup errors after bootstrap must be surfaced without a Writer."""

    session_dir = tmp_path / "session"
    _inject_claim_cleanup_failures(
        monkeypatch,
        fail_close=cleanup_mode in {"close", "both"},
        fail_unlink=cleanup_mode in {"unlink", "both"},
    )

    with pytest.raises(OSError, match="injected cleanup failure") as raised:
        SessionWriter.open(session_dir, make_manifest(), make_discussion_state())

    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    if cleanup_mode in {"close", "both"}:
        assert "claim descriptor close" in notes
    if cleanup_mode in {"unlink", "both"}:
        assert "claim unlink" in notes
    assert _REQUIRED_ARTIFACTS <= {path.name for path in session_dir.iterdir()}
    for artifact in _REQUIRED_ARTIFACTS:
        (session_dir / artifact).unlink()
