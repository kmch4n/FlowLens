"""Read-only incomplete-session recovery inspection tests."""

import json
import os
import stat
import struct
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import flowlens.persistence._recovery_artifacts as artifact_guard
import flowlens.persistence.recovery as recovery
from flowlens.domain._validation import json_dumps
from flowlens.domain.discussion import StateHistoryRecord
from flowlens.domain.enums import AudioSource, SessionStatus
from flowlens.domain.messages import AudioWriteCommand
from flowlens.persistence.json_files import JsonlRepairPlan, encode_jsonl_record
from flowlens.persistence.recovery import (
    RecoveryError,
    RecoveryReport,
    find_incomplete_sessions,
    inspect_incomplete_session,
)
from flowlens.persistence.session_writer import SessionWriter
from tests.factories import (
    make_discussion_state,
    make_event_record,
    make_manifest,
    make_transcript_record,
)
from tests.persistence.recovery_support import create_session_fixture

_SESSION_ID = "01J00000000000000000000000"
_REQUIRED_ARTIFACTS = {
    "session.json",
    "mic.wav",
    "loopback.wav",
    "transcript.jsonl",
    "discussion-state.json",
    "state-history.jsonl",
    "events.jsonl",
}


def _artifact_bytes(session_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes() for path in session_dir.iterdir() if path.is_file()
    }


def _create_populated_session(session_dir: Path) -> Path:
    writer = SessionWriter.open(
        session_dir,
        make_manifest(),
        make_discussion_state(),
    )
    writer.append_audio(
        AudioWriteCommand(AudioSource.ME, b"\x01\x00" * 12_800, 0, 12_800, 0, 800)
    )
    writer.append_transcript(make_transcript_record(1))
    writer.replace_discussion_state(0, make_discussion_state(1))
    writer.append_event(make_event_record(1))
    writer.close_incomplete()
    return session_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(encode_jsonl_record(record) for record in records))


def _create_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")


def _create_file_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"file symlinks are unavailable: {error}")


def test_scan_returns_only_valid_incomplete_sessions_in_name_order(
    tmp_path: Path,
) -> None:
    create_session_fixture(
        tmp_path / "20260819T120000+0900_01J00000000000000000000000",
        "incomplete",
    )
    create_session_fixture(
        tmp_path / "20260819T130000+0900_01J00000000000000000000001",
        "completed",
    )
    create_session_fixture(
        tmp_path / "20260819T110000+0900_01J00000000000000000000002",
        "incomplete",
    )
    (tmp_path / "ordinary-file").write_text("ignored", encoding="utf-8")
    (tmp_path / "directory-without-manifest").mkdir()

    assert [path.name for path in find_incomplete_sessions(tmp_path)] == [
        "20260819T110000+0900_01J00000000000000000000002",
        "20260819T120000+0900_01J00000000000000000000000",
    ]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"\xef\xbb\xbf{}", "UTF-8"),
        (b'{"schema_version":', "JSON"),
        (b'{"unexpected":true}', "SessionManifest"),
    ],
)
def test_scan_rejects_malformed_manifest_instead_of_skipping_it(
    tmp_path: Path,
    payload: bytes,
    reason: str,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest_path = session_dir / "session.json"
    manifest_path.write_bytes(payload)

    with pytest.raises(RecoveryError, match=rf"session\.json.*{reason}"):
        find_incomplete_sessions(tmp_path)

    assert manifest_path.read_bytes() == payload


def test_scan_rejects_a_linked_session_directory(tmp_path: Path) -> None:
    real_dir = create_session_fixture(tmp_path / "real-session", "incomplete")
    linked_dir = tmp_path / "linked-session"
    _create_directory_symlink(linked_dir, real_dir)

    with pytest.raises(RecoveryError, match="linked-session.*reparse"):
        find_incomplete_sessions(tmp_path)


def test_scan_rejects_a_broken_link_named_session_manifest(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = session_dir / "session.json"
    _create_file_symlink(manifest, tmp_path / "missing-manifest.json")

    with pytest.raises(RecoveryError, match=r"session\.json.*reparse"):
        find_incomplete_sessions(tmp_path)


def test_scan_lstats_broken_sessions_root_before_exists_shortcut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions_root = tmp_path / "broken-sessions-link"
    original_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def fake_lstat(path: Path) -> os.stat_result:
        if path == sessions_root:
            return cast(
                os.stat_result,
                SimpleNamespace(
                    st_mode=stat.S_IFDIR,
                    st_file_attributes=reparse_flag,
                ),
            )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(RecoveryError, match=r"broken-sessions-link.*reparse"):
        find_incomplete_sessions(sessions_root)


def test_recovery_report_has_the_exact_frozen_slotted_contract(tmp_path: Path) -> None:
    assert [field.name for field in fields(RecoveryReport)] == [
        "session_id",
        "session_dir",
        "discarded_jsonl_tail_bytes",
        "repaired_wav_headers",
        "transcript_entry_count",
        "final_discussion_state_revision",
        "active_duration_ms",
    ]
    report = RecoveryReport(_SESSION_ID, tmp_path, {}, (), 0, 0, 0)
    assert not hasattr(report, "__dict__")
    with pytest.raises(FrozenInstanceError):
        report.active_duration_ms = 1  # type: ignore[misc]


def test_complete_but_schema_invalid_jsonl_record_blocks_recovery(
    tmp_path: Path,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    path = session_dir / "events.jsonl"
    path.write_text(
        '{"schema_version":1,"unexpected":true}\n',
        encoding="utf-8",
        newline="\n",
    )
    original = path.read_bytes()

    with pytest.raises(RecoveryError, match=r"events\.jsonl.*line 1"):
        inspect_incomplete_session(session_dir)

    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["missing", "extra", "directory"])
def test_inspection_requires_exactly_seven_regular_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    if mutation == "missing":
        (session_dir / "events.jsonl").unlink()
        match = r"events\.jsonl.*missing"
    elif mutation == "extra":
        (session_dir / "unexpected.tmp").write_bytes(b"keep")
        match = r"unexpected\.tmp.*unexpected"
    else:
        (session_dir / "events.jsonl").unlink()
        (session_dir / "events.jsonl").mkdir()
        match = r"events\.jsonl.*regular file"
    before = {path.name for path in session_dir.iterdir()}

    with pytest.raises(RecoveryError, match=match):
        inspect_incomplete_session(session_dir)

    assert {path.name for path in session_dir.iterdir()} == before


def test_inspection_rejects_linked_artifact_without_reading_target(
    tmp_path: Path,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(b'{"secret":true}\n')
    artifact = session_dir / "events.jsonl"
    artifact.unlink()
    _create_file_symlink(artifact, outside)

    with pytest.raises(RecoveryError, match=r"events\.jsonl.*reparse"):
        inspect_incomplete_session(session_dir)

    assert outside.read_bytes() == b'{"secret":true}\n'


def test_inspection_rejects_jsonl_swapped_to_external_symlink_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    artifact = session_dir / "events.jsonl"
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(encode_jsonl_record(make_event_record().to_dict()))
    outside_before = outside.read_bytes()
    original_inspect = recovery._inspect_jsonl_tail_bytes
    swapped = False

    def swap_before_inspect(path: Path, encoded: bytes):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if path == artifact and not swapped:
            artifact.unlink()
            _create_file_symlink(artifact, outside)
            swapped = True
        return original_inspect(path, encoded)

    monkeypatch.setattr(recovery, "_inspect_jsonl_tail_bytes", swap_before_inspect)

    with pytest.raises(RecoveryError, match=r"events\.jsonl"):
        inspect_incomplete_session(session_dir)

    assert outside.read_bytes() == outside_before


def test_inspection_rejects_hard_linked_artifact(tmp_path: Path) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    artifact = session_dir / "events.jsonl"
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(encode_jsonl_record(make_event_record().to_dict()))
    artifact.unlink()
    os.link(outside, artifact)
    before = outside.read_bytes()

    with pytest.raises(RecoveryError, match=r"events\.jsonl.*link count"):
        inspect_incomplete_session(session_dir)

    assert outside.read_bytes() == before


def test_inspection_records_tail_header_and_snapshot_repairs_without_mutation(
    tmp_path: Path,
) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    transcript = session_dir / "transcript.jsonl"
    transcript.write_bytes(transcript.read_bytes() + b'{"schema_version":')
    event_path = session_dir / "events.jsonl"
    event_path.write_bytes(event_path.read_bytes().removesuffix(b"\n"))
    mic_path = session_dir / "mic.wav"
    mic = bytearray(mic_path.read_bytes())
    struct.pack_into("<I", mic, 4, 0)
    struct.pack_into("<I", mic, 40, 0)
    mic_path.write_bytes(mic)
    (session_dir / "discussion-state.json").write_text(
        json_dumps(make_discussion_state().to_dict()),
        encoding="utf-8",
        newline="\n",
    )
    before = _artifact_bytes(session_dir)

    inspection = inspect_incomplete_session(session_dir)

    assert inspection.report.discarded_jsonl_tail_bytes == {
        "transcript.jsonl": len(b'{"schema_version":')
    }
    assert inspection.report.repaired_wav_headers == ("mic.wav",)
    assert inspection.report.transcript_entry_count == 1
    assert inspection.report.final_discussion_state_revision == 1
    assert inspection.report.active_duration_ms == 800
    assert inspection.transcript_entry_count == 1
    assert inspection.final_discussion_state_revision == 1
    assert inspection.active_duration_ms == 800
    assert inspection.next_event_sequence == 2
    assert inspection.snapshot_replacement == make_discussion_state(1)
    assert {
        plan.path.name for plan in inspection.jsonl_repair_plans if plan.append_final_lf
    } == {"events.jsonl"}
    assert _artifact_bytes(session_dir) == before


@pytest.mark.parametrize("artifact_name", ["session.json", "discussion-state.json"])
def test_inspection_rejects_valid_snapshot_replaced_after_it_was_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    artifact = session_dir / artifact_name
    original_compare = recovery._compare_snapshot_to_history

    def replace_before_return(*args, **kwargs):  # type: ignore[no-untyped-def]
        if artifact_name == "session.json":
            replacement = make_manifest(status=SessionStatus.COMPLETED).to_dict()
        else:
            replacement = replace(
                make_discussion_state(), current_focus="changed after read"
            ).to_dict()
        artifact.write_text(json_dumps(replacement), encoding="utf-8", newline="\n")
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(
        recovery,
        "_compare_snapshot_to_history",
        replace_before_return,
    )

    with pytest.raises(RecoveryError, match=rf"{artifact_name}.*changed"):
        inspect_incomplete_session(session_dir)


def test_inspection_retains_typed_artifact_and_snapshot_guards(
    tmp_path: Path,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")

    inspection = inspect_incomplete_session(session_dir)

    assert {identity.path.name for identity in inspection.artifact_identities} == (
        _REQUIRED_ARTIFACTS
    )
    assert all(identity.link_count == 1 for identity in inspection.artifact_identities)
    assert inspection.manifest_plan.identity.path.name == "session.json"
    assert inspection.manifest_plan.value.status is SessionStatus.INCOMPLETE
    assert inspection.snapshot_plan.identity.path.name == "discussion-state.json"
    assert inspection.snapshot_plan.value == make_discussion_state()


def test_retained_identity_rejects_change_before_task13_mutation(
    tmp_path: Path,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    inspection = inspect_incomplete_session(session_dir)
    manifest_path = session_dir / "session.json"
    manifest_path.write_text(
        json_dumps(make_manifest(status=SessionStatus.COMPLETED).to_dict()),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(RecoveryError, match=r"session\.json.*changed"):
        recovery._with_verified_artifact(
            inspection.manifest_plan.identity, lambda _descriptor: None
        )


def test_guarded_mutation_never_writes_external_target_after_postverify_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    inspection = inspect_incomplete_session(session_dir)
    artifact = session_dir / "events.jsonl"
    identity = next(
        item for item in inspection.artifact_identities if item.path == artifact
    )
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(b"outside")
    outside_before = outside.read_bytes()
    original_verify = artifact_guard.verify_artifact_identity
    swap_attempted = False

    def verify_then_swap(opened, expected):  # type: ignore[no-untyped-def]
        nonlocal swap_attempted
        original_verify(opened, expected)
        swap_attempted = True
        artifact.unlink()
        _create_file_symlink(artifact, outside)

    monkeypatch.setattr(artifact_guard, "verify_artifact_identity", verify_then_swap)

    def mutate(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, b"caller mutation")

    try:
        recovery._with_verified_artifact(identity, mutate)
    except RecoveryError:
        pass

    assert swap_attempted
    assert outside.read_bytes() == outside_before


def test_inspection_retains_atomic_replace_directory_guards(tmp_path: Path) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")

    inspection = inspect_incomplete_session(session_dir)

    assert inspection.session_directory_identity.path == session_dir
    assert inspection.parent_directory_identity.path == session_dir.parent


def test_no_path_only_directory_mutation_helper_is_exposed() -> None:
    assert not hasattr(artifact_guard, "with_verified_directories")
    assert not hasattr(recovery, "_with_verified_directories")


def test_jsonl_tail_inspection_does_not_call_path_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    jsonl_paths = {
        session_dir / name for name in _REQUIRED_ARTIFACTS if name.endswith(".jsonl")
    }
    original_read_bytes = Path.read_bytes

    def forbid_jsonl_path_read(path: Path) -> bytes:
        if path in jsonl_paths:
            raise AssertionError(f"path reader reached {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_jsonl_path_read)

    inspection = inspect_incomplete_session(session_dir)

    assert inspection.report.transcript_entry_count == 0


def test_cleanup_attempts_all_seven_handles_when_one_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    opened_artifacts = {}
    close_attempts: list[int] = []
    original_open = recovery._open_exact_artifacts
    original_close = recovery._close_artifact

    def capture_opened(path: Path):  # type: ignore[no-untyped-def]
        opened = original_open(path)
        opened_artifacts.update(opened)
        return opened

    def fail_first_close(opened):  # type: ignore[no-untyped-def]
        close_attempts.append(opened.descriptor)
        original_close(opened)
        if len(close_attempts) == 1:
            raise RecoveryError(opened.path, "injected close failure")

    monkeypatch.setattr(recovery, "_open_exact_artifacts", capture_opened)
    monkeypatch.setattr(recovery, "_close_artifact", fail_first_close)

    try:
        with pytest.raises(RecoveryError, match="injected close failure"):
            inspect_incomplete_session(session_dir)
        assert len(close_attempts) == 7
    finally:
        for opened in opened_artifacts.values():
            try:
                original_close(opened)
            except RecoveryError:
                pass


def test_cleanup_preserves_primary_validation_error_and_adds_close_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    (session_dir / "events.jsonl").write_bytes(b"invalid complete line\n")
    close_attempts = 0
    original_close = recovery._close_artifact

    def fail_first_close(opened):  # type: ignore[no-untyped-def]
        nonlocal close_attempts
        close_attempts += 1
        original_close(opened)
        if close_attempts == 1:
            raise RecoveryError(opened.path, "injected cleanup failure")

    monkeypatch.setattr(recovery, "_close_artifact", fail_first_close)

    with pytest.raises(RecoveryError, match=r"events\.jsonl.*Invalid JSONL") as error:
        inspect_incomplete_session(session_dir)

    assert close_attempts == 7
    assert any("injected cleanup failure" in note for note in error.value.__notes__)


@pytest.mark.parametrize("artifact_name", ["transcript.jsonl", "events.jsonl"])
def test_inspection_requires_contiguous_sequences(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    path = session_dir / artifact_name
    records = _read_jsonl(path)
    records[0]["sequence"] = 2
    _write_jsonl(path, records)

    with pytest.raises(RecoveryError, match=rf"{artifact_name}.*expected.*1.*got 2"):
        inspect_incomplete_session(session_dir)


def test_inspection_rejects_duplicate_transcript_segment_ids(tmp_path: Path) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    path = session_dir / "transcript.jsonl"
    records = _read_jsonl(path)
    second = make_transcript_record(2).to_dict()
    second["segment_id"] = records[0]["segment_id"]
    records.append(second)
    _write_jsonl(path, records)

    with pytest.raises(RecoveryError, match=r"transcript\.jsonl.*segment_id"):
        inspect_incomplete_session(session_dir)


@pytest.mark.parametrize("kind", ["revision_gap", "session_id", "mode"])
def test_inspection_validates_state_history_continuity_and_identity(
    tmp_path: Path,
    kind: str,
) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    path = session_dir / "state-history.jsonl"
    records = _read_jsonl(path)
    if kind == "revision_gap":
        replacement_state = make_discussion_state(2)
        records[0] = StateHistoryRecord(
            1, _SESSION_ID, 1, 2, replacement_state
        ).to_dict()
        match = r"expected previous_revision 0.*got 1"
    elif kind == "session_id":
        records[0]["session_id"] = "01J00000000000000000000009"
        match = r"session_id.*manifest"
    else:
        state_value = records[0]["state"]
        assert isinstance(state_value, dict)
        state_value["mode"] = "INTERVIEW"
        match = r"mode.*manifest"
    _write_jsonl(path, records)

    with pytest.raises(RecoveryError, match=match):
        inspect_incomplete_session(session_dir)


def test_inspection_rejects_event_session_mismatch(tmp_path: Path) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    path = session_dir / "events.jsonl"
    records = _read_jsonl(path)
    records[0]["session_id"] = "01J00000000000000000000009"
    _write_jsonl(path, records)

    with pytest.raises(RecoveryError, match=r"events\.jsonl.*session_id.*manifest"):
        inspect_incomplete_session(session_dir)


@pytest.mark.parametrize("snapshot_case", ["newer", "different"])
def test_inspection_rejects_unproved_or_divergent_snapshot(
    tmp_path: Path,
    snapshot_case: str,
) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    snapshot = make_discussion_state(2 if snapshot_case == "newer" else 1).to_dict()
    if snapshot_case == "different":
        snapshot["current_focus"] = "unproved state"
    (session_dir / "discussion-state.json").write_text(
        json_dumps(snapshot), encoding="utf-8", newline="\n"
    )

    with pytest.raises(RecoveryError, match=r"discussion-state\.json.*history"):
        inspect_incomplete_session(session_dir)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("truncated", "44-byte header"),
        ("non_pcm", "PCM format 1"),
        ("stereo", "mono"),
    ],
)
def test_inspection_rejects_invalid_wav_payload_or_fixed_format(
    tmp_path: Path,
    damage: str,
    reason: str,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    path = session_dir / "mic.wav"
    encoded = bytearray(path.read_bytes())
    if damage == "truncated":
        encoded = encoded[:43]
    elif damage == "non_pcm":
        struct.pack_into("<H", encoded, 20, 3)
    elif damage == "stereo":
        struct.pack_into("<H", encoded, 22, 2)
    path.write_bytes(encoded)

    with pytest.raises(RecoveryError, match=rf"mic\.wav.*{reason}"):
        inspect_incomplete_session(session_dir)


def test_inspection_retains_an_odd_wav_tail_outside_the_valid_pcm_length(
    tmp_path: Path,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    path = session_dir / "mic.wav"
    path.write_bytes(path.read_bytes() + b"\xff")
    before = path.read_bytes()

    inspection = inspect_incomplete_session(session_dir)

    assert inspection.report.repaired_wav_headers == ()
    assert inspection.report.active_duration_ms == 0
    assert path.read_bytes() == before


def test_inspection_rejects_oversized_wav_before_reading_its_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    path = session_dir / "mic.wav"
    oversized_size = 44 + (0xFFFFFFFF - 36) + 2
    monkeypatch.setattr(
        recovery,
        "_wav_file_size",
        lambda candidate: (
            oversized_size
            if candidate.path == path
            else os.fstat(candidate.descriptor).st_size
        ),
        raising=False,
    )

    with pytest.raises(RecoveryError, match=r"mic\.wav.*RIFF size limit"):
        inspect_incomplete_session(session_dir)


def test_inspection_rejects_jsonl_changed_after_tail_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    path = session_dir / "transcript.jsonl"
    original_inspect = recovery._inspect_jsonl_tail_bytes
    changed = False

    def change_after_plan(candidate: Path, encoded: bytes) -> JsonlRepairPlan:
        nonlocal changed
        plan = original_inspect(candidate, encoded)
        if candidate == path and not changed:
            record = make_transcript_record(1).to_dict()
            record["text"] = "changed during inspection"
            candidate.write_bytes(encode_jsonl_record(record))
            changed = True
        return plan

    monkeypatch.setattr(recovery, "_inspect_jsonl_tail_bytes", change_after_plan)

    with pytest.raises(RecoveryError, match=r"transcript\.jsonl.*changed"):
        inspect_incomplete_session(session_dir)


def test_inspection_rejects_transcript_offsets_beyond_retained_audio(
    tmp_path: Path,
) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    path = session_dir / "transcript.jsonl"
    records = _read_jsonl(path)
    records[0]["source_end_sample"] = 12_801
    _write_jsonl(path, records)

    with pytest.raises(RecoveryError, match=r"transcript\.jsonl.*retained audio"):
        inspect_incomplete_session(session_dir)


def test_inspection_is_read_only_when_a_late_validation_fails(tmp_path: Path) -> None:
    session_dir = _create_populated_session(tmp_path / "session")
    transcript = session_dir / "transcript.jsonl"
    transcript.write_bytes(transcript.read_bytes() + b'{"schema_version":')
    event_path = session_dir / "events.jsonl"
    event_path.write_bytes(event_path.read_bytes() + b"not-json\n")
    before = _artifact_bytes(session_dir)

    with pytest.raises(RecoveryError, match=r"events\.jsonl"):
        inspect_incomplete_session(session_dir)

    assert _artifact_bytes(session_dir) == before


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only unsafe artifact type")
def test_inspection_rejects_a_fifo_artifact(tmp_path: Path) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    path = session_dir / "events.jsonl"
    path.unlink()
    os.mkfifo(path)  # type: ignore[attr-defined]

    with pytest.raises(RecoveryError, match=r"events\.jsonl.*regular file"):
        inspect_incomplete_session(session_dir)

    path.unlink()


def test_inspection_rejects_completed_session(tmp_path: Path) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "completed")

    with pytest.raises(RecoveryError, match=r"session\.json.*INCOMPLETE"):
        inspect_incomplete_session(session_dir)


def test_required_artifact_fixture_remains_exact(tmp_path: Path) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    assert {path.name for path in session_dir.iterdir()} == _REQUIRED_ARTIFACTS
