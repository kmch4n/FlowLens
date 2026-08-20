"""SessionWriter bootstrap and ownership tests."""

import json
import os
import stat
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import AudioSource, SessionMode, SessionStatus
from flowlens.domain.messages import AudioWriteCommand
from flowlens.persistence.json_files import AtomicJsonFile, JsonlAppender
from flowlens.persistence.session_writer import (
    PersistenceInvariantError,
    SessionWriter,
    WriterOwnershipError,
)
from flowlens.persistence.wav_sink import WavSink
from tests.factories import make_discussion_state, make_event_record, make_manifest

_REQUIRED_ARTIFACTS = {
    "session.json",
    "mic.wav",
    "loopback.wav",
    "transcript.jsonl",
    "discussion-state.json",
    "state-history.jsonl",
    "events.jsonl",
}


def _create_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")


def test_open_creates_exactly_seven_required_artifacts(tmp_path: Path) -> None:
    """Removing any bootstrap artifact or adding a stray file must fail this test."""

    manifest = make_manifest(status=SessionStatus.INCOMPLETE)
    state = DiscussionState.initial(manifest.mode, manifest.started_at)
    writer = SessionWriter.open(tmp_path / "session", manifest, state)
    assert {path.name for path in (tmp_path / "session").iterdir()} == (
        _REQUIRED_ARTIFACTS
    )
    assert (
        json.loads((tmp_path / "session" / "session.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "incomplete"
    )
    for name in ("mic.wav", "loopback.wav"):
        with wave.open(str(tmp_path / "session" / name), "rb") as reader:
            assert reader.getnframes() == 0
    writer.close_incomplete()


def test_open_accepts_an_existing_empty_directory(tmp_path: Path) -> None:
    """Treating a caller-created empty target as a conflict is a regression."""

    session_dir = tmp_path / "session"
    session_dir.mkdir()

    writer = SessionWriter.open(session_dir, make_manifest(), make_discussion_state())

    assert len(tuple(session_dir.iterdir())) == 7
    writer.close_incomplete()


def test_open_rejects_a_symlink_session_directory_before_writing(
    tmp_path: Path,
) -> None:
    """A linked target must not redirect the seven session writes."""

    real_dir = tmp_path / "real-session"
    real_dir.mkdir()
    session_dir = tmp_path / "session"
    _create_directory_symlink(session_dir, real_dir)

    writer: SessionWriter | None = None
    try:
        writer = SessionWriter.open(
            session_dir,
            make_manifest(),
            make_discussion_state(),
        )
    except PersistenceInvariantError as error:
        assert "reparse" in str(error)
    else:
        pytest.fail("symlink session directory was accepted")
    finally:
        if writer is not None:
            writer.close_incomplete()

    assert not tuple(real_dir.iterdir())


def test_open_rejects_an_immediate_symlink_parent_before_writing(
    tmp_path: Path,
) -> None:
    """A linked immediate parent must not redirect session directory creation."""

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    _create_directory_symlink(linked_parent, real_parent)
    session_dir = linked_parent / "session"

    writer: SessionWriter | None = None
    try:
        writer = SessionWriter.open(
            session_dir,
            make_manifest(),
            make_discussion_state(),
        )
    except PersistenceInvariantError as error:
        assert "reparse" in str(error)
    else:
        pytest.fail("immediate symlink parent was accepted")
    finally:
        if writer is not None:
            writer.close_incomplete()

    assert not (real_parent / "session").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows file attributes only")
def test_open_rejects_windows_reparse_attribute_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows junction-like reparse attributes must not depend on is_symlink."""

    session_dir = tmp_path / "session"
    protected_parent = session_dir.parent
    original_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def fake_lstat(path: Path) -> os.stat_result:
        if path == protected_parent:
            return cast(
                os.stat_result,
                SimpleNamespace(
                    st_mode=stat.S_IFDIR,
                    st_file_attributes=reparse_flag,
                ),
            )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(PersistenceInvariantError, match="reparse"):
        SessionWriter.open(
            session_dir,
            make_manifest(),
            make_discussion_state(),
        )

    assert not session_dir.exists()


def test_concurrent_openers_acquire_exactly_one_session_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an exclusive bootstrap claim both callers can own the artifacts."""

    session_dir = tmp_path / "session"
    ready = threading.Barrier(2)
    attempt_context = threading.local()
    manifest_writers: list[int] = []
    manifest_writers_lock = threading.Lock()
    original_prepare = SessionWriter._prepare_empty_session_directory
    original_replace = AtomicJsonFile.replace

    def synchronized_prepare(path: Path) -> None:
        original_prepare(path)
        ready.wait(timeout=5)

    monkeypatch.setattr(
        SessionWriter,
        "_prepare_empty_session_directory",
        staticmethod(synchronized_prepare),
    )

    def track_manifest_write(file: AtomicJsonFile, value: object) -> None:
        if file.path.name == "session.json":
            with manifest_writers_lock:
                manifest_writers.append(attempt_context.identifier)
        original_replace(file, value)

    monkeypatch.setattr(AtomicJsonFile, "replace", track_manifest_write)

    def attempt_open(
        identifier: int,
    ) -> tuple[SessionWriter | None, BaseException | None]:
        attempt_context.identifier = identifier
        try:
            return (
                SessionWriter.open(
                    session_dir,
                    make_manifest(),
                    make_discussion_state(),
                ),
                None,
            )
        except BaseException as error:
            return None, error

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = tuple(executor.map(attempt_open, range(2)))

    writers = tuple(writer for writer, _ in attempts if writer is not None)
    errors = tuple(error for _, error in attempts if error is not None)
    winner_indexes = tuple(
        index for index, (writer, _) in enumerate(attempts) if writer is not None
    )
    try:
        assert len(writers) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], FileExistsError)
        assert manifest_writers == [winner_indexes[0]]
        assert {path.name for path in session_dir.iterdir()} == {
            "session.json",
            "mic.wav",
            "loopback.wav",
            "transcript.jsonl",
            "discussion-state.json",
            "state-history.jsonl",
            "events.jsonl",
        }
    finally:
        for writer in writers:
            writer.close_incomplete()


def test_open_rejects_nonempty_directory_without_mutation(tmp_path: Path) -> None:
    """Opening over an existing artifact must never truncate or add files."""

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    marker = session_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        SessionWriter.open(session_dir, make_manifest(), make_discussion_state())

    assert tuple(session_dir.iterdir()) == (marker,)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_open_rejects_completed_manifest_before_creating_directory(
    tmp_path: Path,
) -> None:
    """A terminal manifest must never bootstrap mutable session artifacts."""

    session_dir = tmp_path / "session"

    with pytest.raises(PersistenceInvariantError, match="INCOMPLETE"):
        SessionWriter.open(
            session_dir,
            make_manifest(status=SessionStatus.COMPLETED),
            make_discussion_state(),
        )

    assert not session_dir.exists()


@pytest.mark.parametrize(
    ("manifest", "state", "match"),
    [
        (
            make_manifest(),
            DiscussionState.initial(
                SessionMode.INTERVIEW,
                make_manifest().started_at,
            ),
            "mode",
        ),
        (
            replace(make_manifest(), final_discussion_state_revision=1),
            make_discussion_state(),
            "revision",
        ),
        (
            replace(make_manifest(), transcript_entry_count=1),
            make_discussion_state(),
            "transcript_entry_count",
        ),
    ],
)
def test_open_rejects_inconsistent_manifest_and_initial_state(
    tmp_path: Path,
    manifest: object,
    state: DiscussionState,
    match: str,
) -> None:
    """Bootstrap metadata must describe the empty artifacts being created."""

    with pytest.raises(PersistenceInvariantError, match=match):
        SessionWriter.open(
            tmp_path / "session",
            manifest,  # type: ignore[arg-type]
            state,
        )

    assert not (tmp_path / "session").exists()


@pytest.mark.parametrize(
    "tamper_target",
    ["manifest_schema", "manifest_device", "discussion_key_points"],
)
def test_open_strictly_revalidates_tampered_domain_objects_before_writing(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    """Frozen-object tampering must not bypass strict persisted wire validation."""

    manifest = make_manifest()
    state = make_discussion_state()
    if tamper_target == "manifest_schema":
        object.__setattr__(manifest, "schema_version", 2)
    elif tamper_target == "manifest_device":
        object.__setattr__(manifest.microphone, "device_id", "")
    else:
        object.__setattr__(state, "key_points", (1,))
    session_dir = tmp_path / "session"

    writer: SessionWriter | None = None
    try:
        writer = SessionWriter.open(session_dir, manifest, state)
    except PersistenceInvariantError:
        pass
    else:
        pytest.fail("tampered bootstrap domain object was accepted")
    finally:
        if writer is not None:
            writer.close_incomplete()

    assert not session_dir.exists()


def test_open_uses_canonical_manifest_copy_after_validation(tmp_path: Path) -> None:
    """Caller mutation after open must not alter the Writer's session identity."""

    manifest = make_manifest()
    writer = SessionWriter.open(
        tmp_path / "session",
        manifest,
        make_discussion_state(),
    )
    object.__setattr__(manifest, "session_id", "01J00000000000000000000009")

    writer.append_event(make_event_record())

    persisted = json.loads(
        (writer.session_dir / "events.jsonl").read_text(encoding="utf-8")
    )
    assert persisted["session_id"] == "01J00000000000000000000000"
    writer.close_incomplete()


@pytest.mark.parametrize("sync_interval_seconds", [0.0, -1.0, float("inf")])
def test_open_rejects_invalid_sync_interval_before_creating_directory(
    tmp_path: Path,
    sync_interval_seconds: float,
) -> None:
    """A broken durability interval must not leave a half-open session."""

    with pytest.raises(PersistenceInvariantError, match="sync_interval_seconds"):
        SessionWriter.open(
            tmp_path / "session",
            make_manifest(),
            make_discussion_state(),
            sync_interval_seconds=sync_interval_seconds,
        )

    assert not (tmp_path / "session").exists()


def test_open_failure_closes_wav_handle_and_preserves_created_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later bootstrap error must not leak an earlier WAV handle."""

    original_open = WavSink.open
    open_count = 0

    def fail_second_open(cls: type[WavSink], path: Path) -> WavSink:
        del cls
        nonlocal open_count
        open_count += 1
        if open_count == 2:
            raise OSError("loopback open failed")
        return original_open(path)

    monkeypatch.setattr(WavSink, "open", classmethod(fail_second_open))
    session_dir = tmp_path / "session"

    with pytest.raises(OSError, match="loopback open failed"):
        SessionWriter.open(session_dir, make_manifest(), make_discussion_state())

    mic_path = session_dir / "mic.wav"
    assert mic_path.exists()
    assert (session_dir / "session.json").exists()
    mic_path.unlink()
    assert not mic_path.exists()


def test_open_failure_after_all_handles_closes_every_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final snapshot error must close WAV and JSONL handles without rollback."""

    original_replace = AtomicJsonFile.replace

    def fail_discussion_snapshot(file: AtomicJsonFile, value: object) -> None:
        if file.path.name == "discussion-state.json":
            raise OSError("snapshot bootstrap failed")
        original_replace(file, value)

    monkeypatch.setattr(AtomicJsonFile, "replace", fail_discussion_snapshot)
    session_dir = tmp_path / "session"

    with pytest.raises(OSError, match="snapshot bootstrap failed"):
        SessionWriter.open(session_dir, make_manifest(), make_discussion_state())

    for name in (
        "mic.wav",
        "loopback.wav",
        "transcript.jsonl",
        "state-history.jsonl",
        "events.jsonl",
    ):
        path = session_dir / name
        assert path.exists()
        path.unlink()
    assert (session_dir / "session.json").exists()


@pytest.mark.parametrize("failure_point", ["getpid", "monotonic"])
def test_runtime_metadata_failure_happens_before_resource_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Fallible runtime metadata must be captured before creating the session."""

    wav_opens: list[str] = []
    jsonl_opens: list[str] = []
    original_wav_open = WavSink.open
    original_jsonl_open = JsonlAppender.open

    def track_wav_open(cls: type[WavSink], path: Path) -> WavSink:
        del cls
        wav_opens.append(path.name)
        return original_wav_open(path)

    def track_jsonl_open(
        cls: type[JsonlAppender],
        path: Path,
    ) -> JsonlAppender:
        del cls
        jsonl_opens.append(path.name)
        return original_jsonl_open(path)

    def fail_runtime_metadata() -> int:
        raise OSError(f"{failure_point} failed")

    monkeypatch.setattr(WavSink, "open", classmethod(track_wav_open))
    monkeypatch.setattr(JsonlAppender, "open", classmethod(track_jsonl_open))
    if failure_point == "getpid":
        monkeypatch.setattr(
            "flowlens.persistence.session_writer.os.getpid",
            fail_runtime_metadata,
        )
    else:
        monkeypatch.setattr(
            "flowlens.persistence.session_writer.time.monotonic",
            fail_runtime_metadata,
        )

    with pytest.raises(OSError, match=f"{failure_point} failed"):
        SessionWriter.open(
            tmp_path / "session",
            make_manifest(),
            make_discussion_state(),
        )

    assert wav_opens == []
    assert jsonl_opens == []
    assert not (tmp_path / "session").exists()


def test_constructor_failure_closes_every_opened_handle_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction belongs inside the bootstrap resource cleanup boundary."""

    close_counts = {
        "mic.wav": 0,
        "loopback.wav": 0,
        "transcript.jsonl": 0,
        "state-history.jsonl": 0,
        "events.jsonl": 0,
    }
    claim_close_count = 0
    original_wav_close = WavSink.close_incomplete
    original_jsonl_close = JsonlAppender.close
    original_os_close = os.close

    def track_wav_close(sink: WavSink) -> None:
        close_counts[sink.path.name] += 1
        original_wav_close(sink)

    def track_jsonl_close(appender: JsonlAppender) -> None:
        close_counts[appender.path.name] += 1
        original_jsonl_close(appender)

    def track_claim_close(descriptor: int) -> None:
        nonlocal claim_close_count
        claim_close_count += 1
        original_os_close(descriptor)

    def fail_constructor(self: SessionWriter, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise OSError("constructor failed")

    monkeypatch.setattr(WavSink, "close_incomplete", track_wav_close)
    monkeypatch.setattr(JsonlAppender, "close", track_jsonl_close)
    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.close", track_claim_close
    )
    monkeypatch.setattr(SessionWriter, "__init__", fail_constructor)

    with pytest.raises(OSError, match="constructor failed"):
        SessionWriter.open(
            tmp_path / "session",
            make_manifest(),
            make_discussion_state(),
        )

    assert close_counts == {
        "mic.wav": 1,
        "loopback.wav": 1,
        "transcript.jsonl": 1,
        "state-history.jsonl": 1,
        "events.jsonl": 1,
    }
    assert claim_close_count == 1
    assert not (tmp_path / "session" / ".flowlens-bootstrap.claim").exists()


def test_mutation_rejects_a_process_other_than_the_opening_owner(
    open_writer: SessionWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the PID ownership guard must allow this mutation and fail."""

    mic_path = open_writer.session_dir / "mic.wav"
    before = mic_path.stat().st_size
    monkeypatch.setattr(
        "flowlens.persistence.session_writer.os.getpid",
        lambda: open_writer.owner_pid + 1,
    )

    with pytest.raises(WriterOwnershipError, match="owner PID"):
        open_writer.append_audio(
            AudioWriteCommand(AudioSource.ME, b"\x00\x00", 0, 1, 0, 0)
        )

    assert mic_path.stat().st_size == before


def test_closed_writer_rejects_later_mutation(tmp_path: Path) -> None:
    """A close that accidentally leaves mutation enabled must fail this test."""

    writer = SessionWriter.open(
        tmp_path / "session",
        make_manifest(),
        make_discussion_state(),
    )
    writer.close_incomplete()

    with pytest.raises(RuntimeError, match="closed"):
        writer.append_audio(AudioWriteCommand(AudioSource.ME, b"\x00\x00", 0, 1, 0, 0))
