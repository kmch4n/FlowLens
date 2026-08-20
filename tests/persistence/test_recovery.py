"""End-to-end incomplete-session recovery tests."""

import json
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, cast
from unittest.mock import MagicMock

import pytest

import flowlens.persistence._recovery_artifacts as recovery_artifacts
import flowlens.persistence.recovery as recovery
from flowlens.domain.enums import EventType, ProcessSource
from flowlens.domain.messages import EventRecord
from flowlens.persistence._recovery_artifacts import ArtifactIdentity, DirectoryAnchor
from flowlens.persistence.json_files import AtomicJsonFile, JsonlAppender
from flowlens.persistence.recovery import (
    RecoveryError,
    recover_incomplete_session,
    recover_incomplete_sessions,
)
from tests.persistence.recovery_support import (
    aware_recovery_time,
    corrupt_wav_sizes,
    create_interrupted_session,
    load_manifest_status,
)


def _load_events(session_dir: Path) -> list[dict[str, object]]:
    path = session_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recovery_repairs_tails_appends_event_and_marks_recovered_last(
    tmp_path: Path,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    transcript_path = session_dir / "transcript.jsonl"
    transcript_prefix = transcript_path.read_bytes()
    transcript_path.write_bytes(transcript_prefix + b'{"schema_version":')
    corrupt_wav_sizes(session_dir / "mic.wav")

    report = recover_incomplete_session(session_dir, aware_recovery_time())

    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    events = _load_events(session_dir)
    assert manifest["status"] == "recovered"
    assert manifest["ended_at"] == "2026-08-19T13:00:00.000+09:00"
    assert manifest["active_duration_ms"] == 800
    assert manifest["transcript_entry_count"] == 1
    assert events[-1]["event_type"] == "SESSION_RECOVERED"
    details = events[-1]["details"]
    assert isinstance(details, dict)
    discarded = details["discarded_jsonl_tail_bytes"]
    assert isinstance(discarded, dict)
    assert discarded["transcript.jsonl"] > 0
    assert details["repaired_wav_headers"] == ["loopback.wav", "mic.wav"]
    assert report.session_id == manifest["session_id"]
    assert transcript_path.read_bytes() == transcript_prefix
    with wave.open(str(session_dir / "mic.wav"), "rb") as reader:
        assert reader.getnframes() == 12_800


def test_recovery_preserves_pcm_payload_and_existing_valid_records(
    tmp_path: Path,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    before_pcm = (session_dir / "loopback.wav").read_bytes()[44:]
    before_transcript = (session_dir / "transcript.jsonl").read_bytes()
    recover_incomplete_session(session_dir, aware_recovery_time())
    assert (session_dir / "loopback.wav").read_bytes()[44:] == before_pcm
    assert (session_dir / "transcript.jsonl").read_bytes() == before_transcript


def test_global_recovery_does_not_recover_already_recovered_session_twice(
    tmp_path: Path,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    first = recover_incomplete_sessions(tmp_path, aware_recovery_time())
    second = recover_incomplete_sessions(tmp_path, aware_recovery_time())
    assert [report.session_dir for report in first] == [session_dir]
    assert second == ()
    assert (session_dir / "events.jsonl").read_text(encoding="utf-8").count(
        "SESSION_RECOVERED"
    ) == 1


def test_manifest_remains_incomplete_when_recovery_event_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(recovery, "_replace_bytes_anchored", fail_publish)
    with pytest.raises(OSError, match="disk full"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"


def test_retry_reuses_durable_recovery_event_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    original_replace = AtomicJsonFile.replace
    failed = False

    def fail_recovered_manifest_once(
        self: AtomicJsonFile,
        value: object,
    ) -> None:
        nonlocal failed
        is_recovered = isinstance(value, dict) and value.get("status") == "recovered"
        if self.path.name == "session.json" and is_recovered and not failed:
            failed = True
            raise OSError("manifest replace failed")
        original_replace(self, value)

    monkeypatch.setattr(AtomicJsonFile, "replace", fail_recovered_manifest_once)
    with pytest.raises(OSError, match="manifest replace failed"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"

    monkeypatch.setattr(AtomicJsonFile, "replace", original_replace)
    recover_incomplete_session(session_dir, aware_recovery_time())
    assert (session_dir / "events.jsonl").read_text(encoding="utf-8").count(
        "SESSION_RECOVERED"
    ) == 1
    assert load_manifest_status(session_dir) == "recovered"


def test_recovery_requires_timezone_aware_timestamp(tmp_path: Path) -> None:
    session_dir = create_interrupted_session(tmp_path)
    with pytest.raises(ValueError, match="timezone"):
        recover_incomplete_session(session_dir, datetime(2026, 8, 19, 13))
    assert load_manifest_status(session_dir) == "incomplete"


def test_recovery_reconstructs_pauses_and_closes_final_open_pause(
    tmp_path: Path,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    path = session_dir / "events.jsonl"
    with path.open("ab") as file:
        for sequence, event_type, session_time_ms in (
            (2, EventType.PAUSE_START, 100),
            (3, EventType.PAUSE_END, 200),
            (4, EventType.PAUSE_START, 500),
        ):
            record = EventRecord(
                1,
                "01J00000000000000000000000",
                sequence,
                event_type,
                ProcessSource.GUI,
                session_time_ms,
                aware_recovery_time(),
                {},
            )
            encoded = json.dumps(record.to_dict(), separators=(",", ":")) + "\n"
            file.write(encoded.encode())

    recover_incomplete_session(session_dir, aware_recovery_time())
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert manifest["pause_intervals"] == [
        {"started_ms": 100, "ended_ms": 200},
        {"started_ms": 500, "ended_ms": 800},
    ]
    assert any("unmatched PAUSE_START" in note for note in manifest["recovery_notes"])


def test_pause_end_without_start_blocks_recovery(tmp_path: Path) -> None:
    session_dir = create_interrupted_session(tmp_path)
    path = session_dir / "events.jsonl"
    event = EventRecord(
        1,
        "01J00000000000000000000000",
        2,
        EventType.PAUSE_END,
        ProcessSource.GUI,
        100,
        aware_recovery_time(),
        {},
    )
    with path.open("ab") as file:
        file.write((json.dumps(event.to_dict(), separators=(",", ":")) + "\n").encode())
    with pytest.raises(RecoveryError, match="PAUSE_END"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"


@pytest.mark.skipif(os.name == "nt", reason="Windows held handles prevent the swap")
def test_event_path_swap_mutates_verified_handle_not_replacement_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    event_path = session_dir / "events.jsonl"
    retained_path = session_dir / "retained-events"
    external_path = tmp_path / "external-events"
    external_path.write_bytes(b"external")
    original_append = JsonlAppender.append

    def swap_then_append(self: JsonlAppender, value: object) -> None:
        event_path.rename(retained_path)
        os.link(external_path, event_path)
        original_append(self, value)

    monkeypatch.setattr(JsonlAppender, "append", swap_then_append)
    with pytest.raises(RecoveryError, match="identity"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert external_path.read_bytes() == b"external"
    assert event_path.read_bytes() == b"external"
    assert b"SESSION_RECOVERED" in retained_path.read_bytes()
    assert load_manifest_status(session_dir) == "incomplete"


def test_event_fsync_failure_retries_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    original_publish = recovery._replace_bytes_anchored
    failed = False

    def fail_after_publish_once(*args: object, **kwargs: object) -> None:
        nonlocal failed
        original_publish(*args, **kwargs)  # type: ignore[arg-type]
        if not failed:
            failed = True
            raise OSError("event fsync acknowledgement lost")

    monkeypatch.setattr(recovery, "_replace_bytes_anchored", fail_after_publish_once)
    with pytest.raises(OSError, match="acknowledgement lost"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"

    monkeypatch.setattr(recovery, "_replace_bytes_anchored", original_publish)
    recover_incomplete_session(session_dir, aware_recovery_time())
    events = (session_dir / "events.jsonl").read_text(encoding="utf-8")
    assert events.count("SESSION_RECOVERED") == 1
    assert load_manifest_status(session_dir) == "recovered"


def test_session_directory_swap_cannot_mutate_replacement_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    retained_dir = tmp_path / "retained-session"
    replacement_manifest = b"replacement target"
    original_publish = recovery._replace_bytes_anchored

    def swap_directory_then_publish(*args: object, **kwargs: object) -> None:
        session_dir.rename(retained_dir)
        session_dir.mkdir()
        (session_dir / "session.json").write_bytes(replacement_manifest)
        original_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        recovery,
        "_replace_bytes_anchored",
        swap_directory_then_publish,
    )
    with pytest.raises((OSError, RecoveryError)):
        recover_incomplete_session(session_dir, aware_recovery_time())

    if retained_dir.exists():
        assert load_manifest_status(retained_dir) == "incomplete"
        assert (session_dir / "session.json").read_bytes() == replacement_manifest
    else:
        assert load_manifest_status(session_dir) == "incomplete"


@pytest.mark.parametrize("target_name", ["session.json", "discussion-state.json"])
def test_atomic_target_swap_is_rejected_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    if target_name == "discussion-state.json":
        writer_state = json.loads(
            (session_dir / target_name).read_text(encoding="utf-8")
        )
        writer_state["revision"] = 0
        (session_dir / target_name).write_text(
            json.dumps(writer_state, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        # Force a planned snapshot replacement without changing history bytes.
        monkeypatch.setattr(
            recovery,
            "_compare_snapshot_to_history",
            lambda snapshot, final_state, has_history, snapshot_path: final_state,
        )
    target_path = session_dir / target_name
    original_bytes = target_path.read_bytes()
    retained_original = tmp_path / f"retained-{target_name}"
    replacement_source = tmp_path / f"replacement-{target_name}"
    replacement_bytes = b"replacement sentinel"
    replacement_source.write_bytes(replacement_bytes)
    original_replace = DirectoryAnchor.replace

    def swap_before_replace(
        self: DirectoryAnchor,
        source: Path,
        target: Path,
        expected_identity: ArtifactIdentity,
    ) -> None:
        if target.name == target_name:
            os.replace(target, retained_original)
            os.replace(replacement_source, target)
        original_replace(self, source, target, expected_identity)

    monkeypatch.setattr(DirectoryAnchor, "replace", swap_before_replace)
    with pytest.raises(RecoveryError, match="changed|identity"):
        recover_incomplete_session(session_dir, aware_recovery_time())

    assert retained_original.read_bytes() == original_bytes
    assert target_path.read_bytes() == replacement_bytes
    assert sorted(path.name for path in session_dir.iterdir()) == [
        "discussion-state.json",
        "events.jsonl",
        "loopback.wav",
        "mic.wav",
        "session.json",
        "state-history.jsonl",
        "transcript.jsonl",
    ]


def test_repair_intent_survives_failure_before_all_repairs_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    transcript_path = session_dir / "transcript.jsonl"
    transcript_path.write_bytes(transcript_path.read_bytes() + b'{"schema_version":')
    original_repair = recovery._apply_wav_plan_same_handle
    failed = False

    def fail_after_first_wav_repair(
        opened: object,
        plan: object,
    ) -> None:
        nonlocal failed
        original_repair(opened, plan)  # type: ignore[arg-type]
        if not failed:
            failed = True
            raise OSError("crash after first WAV repair")

    monkeypatch.setattr(
        recovery,
        "_apply_wav_plan_same_handle",
        fail_after_first_wav_repair,
    )
    with pytest.raises(OSError, match="crash after first WAV repair"):
        recover_incomplete_session(session_dir, aware_recovery_time())

    first_events = _load_events(session_dir)
    assert first_events[-1]["event_type"] == "SESSION_RECOVERED"
    intent_details = first_events[-1]["details"]
    monkeypatch.setattr(recovery, "_apply_wav_plan_same_handle", original_repair)
    report = recover_incomplete_session(session_dir, aware_recovery_time())

    assert _load_events(session_dir)[-1]["details"] == intent_details
    assert report.discarded_jsonl_tail_bytes["transcript.jsonl"] > 0
    assert report.repaired_wav_headers == ("loopback.wav", "mic.wav")
    assert load_manifest_status(session_dir) == "recovered"


def test_event_append_failure_precedes_every_planned_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    transcript_path = session_dir / "transcript.jsonl"
    transcript_path.write_bytes(transcript_path.read_bytes() + b"torn")
    before = {
        path.name: path.read_bytes() for path in session_dir.iterdir() if path.is_file()
    }

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise OSError("intent append failed")

    monkeypatch.setattr(recovery, "_replace_bytes_anchored", fail_publish)
    with pytest.raises(OSError, match="intent append failed"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert {
        path.name: path.read_bytes() for path in session_dir.iterdir() if path.is_file()
    } == before


def test_recovery_intent_atomically_replaces_a_torn_event_tail(
    tmp_path: Path,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    event_path = session_dir / "events.jsonl"
    event_path.write_bytes(event_path.read_bytes() + b'{"torn":')

    report = recover_incomplete_session(session_dir, aware_recovery_time())

    events = _load_events(session_dir)
    assert [event["event_type"] for event in events] == [
        "SESSION_START",
        "SESSION_RECOVERED",
    ]
    assert report.discarded_jsonl_tail_bytes["events.jsonl"] == len(b'{"torn":')
    assert events[-1]["details"] == {
        "discarded_jsonl_tail_bytes": {"events.jsonl": len(b'{"torn":')},
        "repaired_wav_headers": ["loopback.wav", "mic.wav"],
    }


def test_clean_recovery_intent_never_uses_direct_jsonl_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)

    def forbid_append(self: JsonlAppender, value: object) -> None:
        raise AssertionError("direct JSONL append is forbidden")

    monkeypatch.setattr(JsonlAppender, "append", forbid_append)
    recover_incomplete_session(session_dir, aware_recovery_time())
    assert _load_events(session_dir)[-1]["event_type"] == "SESSION_RECOVERED"


def test_clean_intent_replace_failure_leaves_events_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    event_path = session_dir / "events.jsonl"
    before = event_path.read_bytes()
    original_replace = DirectoryAnchor.replace

    def fail_event_replace(
        self: DirectoryAnchor,
        source: Path,
        target: Path,
        expected_identity: ArtifactIdentity,
    ) -> None:
        if target.name == "events.jsonl":
            raise OSError("event intent replace failed")
        original_replace(self, source, target, expected_identity)

    monkeypatch.setattr(DirectoryAnchor, "replace", fail_event_replace)
    with pytest.raises(OSError, match="event intent replace failed"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert event_path.read_bytes() == before
    assert load_manifest_status(session_dir) == "incomplete"


@pytest.mark.parametrize("failure_stage", ["partial_write", "flush", "fsync", "close"])
def test_clean_intent_temp_failure_leaves_events_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    event_path = session_dir / "events.jsonl"
    before = event_path.read_bytes()
    original_create = DirectoryAnchor.create_binary_temp
    temp_descriptors: set[int] = set()

    def create_failing_temp(
        self: DirectoryAnchor,
        target: Path,
    ) -> tuple[Path, BinaryIO]:
        temp_path, file = original_create(self, target)
        temp_descriptors.add(file.fileno())
        wrapped = MagicMock(wraps=file)
        if failure_stage == "partial_write":

            def fail_partial(data: object) -> int:
                view = memoryview(data)  # type: ignore[arg-type]
                file.write(view[:1])
                raise OSError("partial intent write")

            wrapped.write.side_effect = fail_partial
        elif failure_stage == "flush":
            wrapped.flush.side_effect = OSError("intent flush failed")
        elif failure_stage == "close":

            def fail_close() -> None:
                file.close()
                raise OSError("intent close failed")

            wrapped.close.side_effect = fail_close
        return temp_path, cast(BinaryIO, wrapped)

    monkeypatch.setattr(DirectoryAnchor, "create_binary_temp", create_failing_temp)
    if failure_stage == "fsync":
        original_fsync = recovery.os.fsync  # type: ignore[attr-defined]

        def fail_temp_fsync(descriptor: int) -> None:
            if descriptor in temp_descriptors:
                raise OSError("intent fsync failed")
            original_fsync(descriptor)

        monkeypatch.setattr("flowlens.persistence.recovery.os.fsync", fail_temp_fsync)

    with pytest.raises(OSError, match="intent|partial"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert event_path.read_bytes() == before
    assert load_manifest_status(session_dir) == "incomplete"
    assert sorted(path.name for path in session_dir.iterdir()) == [
        "discussion-state.json",
        "events.jsonl",
        "loopback.wav",
        "mic.wav",
        "session.json",
        "state-history.jsonl",
        "transcript.jsonl",
    ]


@pytest.mark.parametrize("substitution", ["transcript", "event", "pcm"])
def test_same_length_substitution_is_rejected_before_recovered_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    original_reparse = recovery._reparse_repaired_records

    def substitute_after_reparse(
        inspection: Any,
        opened: Any,
    ) -> tuple[object, object, object]:
        result = original_reparse(inspection, opened)
        if substitution == "pcm":
            path = session_dir / "mic.wav"
            encoded = bytearray(path.read_bytes())
            encoded[-1] ^= 1
            path.write_bytes(encoded)
        else:
            name = (
                "transcript.jsonl" if substitution == "transcript" else "events.jsonl"
            )
            path = session_dir / name
            lines = path.read_text(encoding="utf-8").splitlines()
            value = json.loads(lines[0])
            if substitution == "transcript":
                value["text"] = value["text"].replace("方", "法", 1)
            else:
                value["source"] = "ASR"
            lines[0] = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        return result

    monkeypatch.setattr(recovery, "_reparse_repaired_records", substitute_after_reparse)
    with pytest.raises(RecoveryError, match="changed|differs"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"


def test_verify_failure_closes_newly_opened_transaction_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = create_interrupted_session(tmp_path)
    inspection = recovery.inspect_incomplete_session(session_dir)
    opened_descriptors: list[int] = []
    original_open = recovery._open_guarded_artifact  # type: ignore[attr-defined]

    def capture_open(path: Path, *, writable: bool = False):  # type: ignore[no-untyped-def]
        opened = original_open(path, writable=writable)
        opened_descriptors.append(opened.descriptor)
        return opened

    def fail_verify(opened: object, identity: object) -> None:
        raise RecoveryError(session_dir / "session.json", "injected verify failure")

    monkeypatch.setattr(recovery, "_open_guarded_artifact", capture_open)
    monkeypatch.setattr(recovery, "_verify_artifact_identity", fail_verify)
    with pytest.raises(RecoveryError, match="injected verify failure"):
        recovery._open_transaction_artifacts(inspection)
    assert opened_descriptors
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_posix_anchor_acquire_preserves_primary_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = recovery_artifacts.capture_directory_identity(tmp_path)
    fake_descriptor = 9876
    monkeypatch.setattr(
        "flowlens.persistence._recovery_artifacts.sys.platform", "linux"
    )
    monkeypatch.setattr(
        "flowlens.persistence._recovery_artifacts.os.open",
        lambda path, flags: fake_descriptor,
    )
    monkeypatch.setattr(
        "flowlens.persistence._recovery_artifacts.os.fstat",
        lambda descriptor: tmp_path.stat(),
    )
    monkeypatch.setattr(
        recovery_artifacts,
        "verify_directory_identity",
        lambda value: (_ for _ in ()).throw(RecoveryError(tmp_path, "primary verify")),
    )
    monkeypatch.setattr(
        "flowlens.persistence._recovery_artifacts.os.close",
        lambda descriptor: (_ for _ in ()).throw(OSError("cleanup close")),
    )

    with pytest.raises(RecoveryError, match="primary verify") as captured:
        recovery_artifacts.open_directory_anchor(identity)
    assert any("cleanup close" in note for note in captured.value.__notes__)
