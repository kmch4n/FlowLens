import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self, TextIO, cast

import pytest

from flowlens.persistence.json_files import (
    AtomicJsonFile,
    JsonlAppender,
    JsonlValidationError,
    apply_jsonl_tail_repair,
    inspect_jsonl_tail,
    validate_and_repair_jsonl_tail,
)


class _ShortWriteFile:
    def __init__(self, maximum_write: int = 3) -> None:
        self.contents = bytearray()
        self.events: list[str] = []
        self.closed = False
        self._maximum_write = maximum_write
        self._active_writes = 0
        self.overlapped = False
        self._state_lock = threading.Lock()

    def write(self, data: bytes | memoryview) -> int:
        with self._state_lock:
            self._active_writes += 1
            if self._active_writes > 1:
                self.overlapped = True
        try:
            chunk = bytes(data[: self._maximum_write])
            time.sleep(0.001)
            self.contents.extend(chunk)
            self.events.append(f"write:{len(chunk)}")
            return len(chunk)
        finally:
            with self._state_lock:
                self._active_writes -= 1

    def flush(self) -> None:
        self.events.append("flush")

    def fileno(self) -> int:
        return 41

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class _ShortTextFile:
    def __init__(self, file: TextIO, maximum_write: int = 2) -> None:
        self._file = file
        self._maximum_write = maximum_write

    def __enter__(self) -> Self:
        self._file.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._file.__exit__(exception_type, exception, traceback)

    def write(self, data: str) -> int:
        return self._file.write(data[: self._maximum_write])

    def flush(self) -> None:
        self._file.flush()

    def fileno(self) -> int:
        return self._file.fileno()

    def close(self) -> None:
        self._file.close()


class _CloseFailingTextFile(_ShortTextFile):
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        super().close()
        raise OSError("secondary close failure")


def _create_short_text_temp(
    target_path: Path,
    maximum_write: int,
) -> tuple[Path, TextIO]:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    temp_path = Path(temp_name)
    file = os.fdopen(
        file_descriptor,
        encoding="utf-8",
        mode="w",
        newline="\n",
    )
    return temp_path, cast(TextIO, _ShortTextFile(file, maximum_write))


def _create_close_failing_text_temp(target_path: Path) -> tuple[Path, TextIO]:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    temp_path = Path(temp_name)
    file = os.fdopen(
        file_descriptor,
        encoding="utf-8",
        mode="w",
        newline="\n",
    )
    return temp_path, cast(TextIO, _CloseFailingTextFile(file))


def test_jsonl_flushes_each_record_and_uses_one_compact_lf_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    appender = JsonlAppender.open(path)
    try:
        appender.append({"schema_version": 1, "text": "保存"})

        assert path.read_bytes() == '{"schema_version":1,"text":"保存"}\n'.encode()
    finally:
        appender.close()


def test_atomic_json_is_indented_utf8_and_has_no_leftover_temp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "discussion-state.json"

    AtomicJsonFile(path).replace({"revision": 1, "current_focus": "方針"})

    raw = path.read_bytes()
    assert json.loads(raw.decode("utf-8"))["revision"] == 1
    assert raw.startswith(b'{\n    "revision"')
    assert raw.endswith(b"\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert not path.with_name("discussion-state.json.tmp").exists()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_overlapping_atomic_replaces_use_independent_same_directory_temps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.json"
    original = b'{"status":"old"}\n'
    path.write_bytes(original)
    fsync_barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    actual_replace_lock = threading.Lock()
    second_replace_entered = threading.Event()
    sources: list[Path] = []
    targets_during_temp_writes: list[bytes] = []
    active_replace_calls = 0
    first_replace_probe_started = False
    replace_calls_overlapped = False

    def overlap_fsync(file_descriptor: int) -> None:
        with state_lock:
            targets_during_temp_writes.append(path.read_bytes())
        fsync_barrier.wait(timeout=5)
        os.fsync(file_descriptor)

    def overlap_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        nonlocal active_replace_calls
        nonlocal first_replace_probe_started
        nonlocal replace_calls_overlapped
        source_path = Path(source)
        target_path = Path(target)
        with state_lock:
            sources.append(source_path)
            active_replace_calls += 1
            should_wait_for_peer = not first_replace_probe_started
            first_replace_probe_started = True
            if active_replace_calls > 1:
                replace_calls_overlapped = True
                second_replace_entered.set()
        if should_wait_for_peer:
            second_replace_entered.wait(timeout=0.25)
        try:
            with actual_replace_lock:
                os.replace(source_path, target_path)
        finally:
            with state_lock:
                active_replace_calls -= 1

    atomic_file = AtomicJsonFile(
        path,
        _fsync=overlap_fsync,
        _replace=overlap_replace,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(atomic_file.replace, {"writer": writer})
            for writer in (1, 2)
        ]
        for future in futures:
            future.result(timeout=5)

    assert len(set(sources)) == 2
    assert all(source.parent == path.parent for source in sources)
    assert targets_during_temp_writes == [original, original]
    assert not replace_calls_overlapped
    assert json.loads(path.read_text(encoding="utf-8")) in [
        {"writer": 1},
        {"writer": 2},
    ]
    assert all(not source.exists() for source in sources)


def test_preexisting_fixed_temp_hard_link_is_never_opened_or_removed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.json"
    original = b'{"status":"old"}\n'
    path.write_bytes(original)
    stale_temp_path = path.with_name("session.json.tmp")
    os.link(path, stale_temp_path)
    created_sources: list[Path] = []

    def fail_before_replace(
        source: os.PathLike[str],
        target: os.PathLike[str],
    ) -> None:
        source_path = Path(source)
        created_sources.append(source_path)
        assert source_path != stale_temp_path
        assert Path(target).read_bytes() == original
        raise OSError("stop before replace")

    with pytest.raises(OSError, match="stop before replace"):
        AtomicJsonFile(path, _replace=fail_before_replace).replace({"status": "new"})

    assert path.read_bytes() == original
    assert stale_temp_path.read_bytes() == original
    assert len(created_sources) == 1
    assert not created_sources[0].exists()


def test_atomic_json_fsyncs_same_directory_temp_before_atomic_replace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "session.json"
    operations: list[str] = []

    def track_fsync(file_descriptor: int) -> None:
        assert os.fstat(file_descriptor).st_size > 0
        operations.append("fsync")

    def track_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path != path.with_name("session.json.tmp")
        assert source_path.parent == target_path.parent == path.parent
        assert source_path.exists()
        operations.append("replace")
        os.replace(source_path, target_path)

    AtomicJsonFile(path, _fsync=track_fsync, _replace=track_replace).replace(
        {"status": "incomplete"}
    )

    assert operations == ["fsync", "replace"]
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "incomplete"


def test_atomic_json_preserves_target_and_cleans_temp_when_replace_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.json"
    original = b'{"status":"old"}\n'
    path.write_bytes(original)
    sources: list[Path] = []

    def fail_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        sources.append(Path(source))
        raise OSError(f"replace failed: {source} -> {target}")

    with pytest.raises(OSError, match="replace failed"):
        AtomicJsonFile(path, _replace=fail_replace).replace({"status": "new"})

    assert path.read_bytes() == original
    assert len(sources) == 1
    assert not sources[0].exists()


def test_cleanup_error_does_not_mask_primary_replace_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.json"
    path.write_bytes(b'{"status":"old"}\n')

    def fail_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        raise OSError(f"primary replace failure: {source} -> {target}")

    def fail_cleanup(temp_path: Path) -> None:
        os.unlink(temp_path)
        raise OSError(f"secondary cleanup failure: {temp_path}")

    with pytest.raises(OSError, match="primary replace failure") as raised:
        AtomicJsonFile(
            path,
            _replace=fail_replace,
            _remove_temp=fail_cleanup,
        ).replace({"status": "new"})

    assert any("secondary cleanup failure" in note for note in raised.value.__notes__)


def test_cleanup_error_does_not_mask_primary_write_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.json"

    def create_no_progress_writer(target_path: Path) -> tuple[Path, TextIO]:
        return _create_short_text_temp(target_path, maximum_write=0)

    def fail_cleanup(temp_path: Path) -> None:
        os.unlink(temp_path)
        raise OSError(f"secondary cleanup failure: {temp_path}")

    with pytest.raises(OSError, match="file write made no progress") as raised:
        AtomicJsonFile(
            path,
            _create_temp=create_no_progress_writer,
            _remove_temp=fail_cleanup,
        ).replace({"status": "new"})

    assert any("secondary cleanup failure" in note for note in raised.value.__notes__)


def test_close_error_does_not_mask_primary_fsync_error(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    original = b'{"status":"old"}\n'
    path.write_bytes(original)
    created_temps: list[Path] = []

    def create_close_failing_writer(target_path: Path) -> tuple[Path, TextIO]:
        temp_path, file = _create_close_failing_text_temp(target_path)
        created_temps.append(temp_path)
        return temp_path, file

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError(f"primary fsync failure for {file_descriptor}")

    with pytest.raises(OSError, match="primary fsync failure") as raised:
        AtomicJsonFile(
            path,
            _create_temp=create_close_failing_writer,
            _fsync=fail_fsync,
        ).replace({"status": "new"})

    assert any("secondary close failure" in note for note in raised.value.__notes__)
    assert path.read_bytes() == original
    assert len(created_temps) == 1
    assert not created_temps[0].exists()


def test_close_only_error_is_surfaced_and_cleans_owned_temp(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    original = b'{"status":"old"}\n'
    path.write_bytes(original)
    created_temps: list[Path] = []

    def create_close_failing_writer(target_path: Path) -> tuple[Path, TextIO]:
        temp_path, file = _create_close_failing_text_temp(target_path)
        created_temps.append(temp_path)
        return temp_path, file

    with pytest.raises(OSError, match="secondary close failure"):
        AtomicJsonFile(
            path,
            _create_temp=create_close_failing_writer,
        ).replace({"status": "new"})

    assert path.read_bytes() == original
    assert len(created_temps) == 1
    assert not created_temps[0].exists()


def test_atomic_json_retries_short_temp_writes_before_replace(tmp_path: Path) -> None:
    path = tmp_path / "discussion-state.json"

    def create_short_writer(target_path: Path) -> tuple[Path, TextIO]:
        return _create_short_text_temp(target_path, maximum_write=2)

    AtomicJsonFile(path, _create_temp=create_short_writer).replace(
        {"revision": 1, "current_focus": "方針"}
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "revision": 1,
        "current_focus": "方針",
    }
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_atomic_json_preserves_target_and_cleans_temp_when_fsync_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "discussion-state.json"
    original = b'{"revision":1}\n'
    path.write_bytes(original)

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError(f"fsync failed for {file_descriptor}")

    with pytest.raises(OSError, match="fsync failed"):
        AtomicJsonFile(path, _fsync=fail_fsync).replace({"revision": 2})

    assert path.read_bytes() == original
    assert not path.with_name("discussion-state.json.tmp").exists()


def test_jsonl_retries_short_writes_before_flushing() -> None:
    output = _ShortWriteFile(maximum_write=2)
    appender = JsonlAppender(Path("events.jsonl"), cast(BinaryIO, output))

    appender.append({"sequence": 1})

    assert bytes(output.contents) == b'{"sequence":1}\n'
    assert output.events[-1] == "flush"
    appender.close()


def test_jsonl_sync_flushes_before_injected_fsync() -> None:
    output = _ShortWriteFile()

    def track_fsync(file_descriptor: int) -> None:
        output.events.append(f"fsync:{file_descriptor}")

    appender = JsonlAppender(
        Path("events.jsonl"),
        cast(BinaryIO, output),
        _fsync=track_fsync,
    )
    appender.append({"sequence": 1})
    assert all(not event.startswith("fsync:") for event in output.events)
    output.events.clear()

    appender.sync()

    assert output.events == ["flush", "fsync:41"]
    appender.close()


def test_jsonl_lock_keeps_concurrent_short_writes_as_complete_records() -> None:
    output = _ShortWriteFile(maximum_write=1)
    appender = JsonlAppender(Path("events.jsonl"), cast(BinaryIO, output))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda sequence: appender.append({"sequence": sequence}),
                range(8),
            )
        )
    appender.close()

    records = [json.loads(line) for line in bytes(output.contents).splitlines()]
    assert not output.overlapped
    assert sorted(record["sequence"] for record in records) == list(range(8))


def test_jsonl_rejects_no_progress_write_without_flushing() -> None:
    output = _ShortWriteFile(maximum_write=0)
    appender = JsonlAppender(Path("events.jsonl"), cast(BinaryIO, output))

    with pytest.raises(OSError, match="no progress"):
        appender.append({"sequence": 1})

    assert bytes(output.contents) == b""
    assert "flush" not in output.events
    appender.close()


def test_jsonl_rejects_nan_before_writing_any_bytes() -> None:
    output = _ShortWriteFile()
    appender = JsonlAppender(Path("events.jsonl"), cast(BinaryIO, output))

    with pytest.raises(ValueError, match="Out of range float values"):
        appender.append({"value": float("nan")})

    assert bytes(output.contents) == b""
    assert "flush" not in output.events
    appender.close()


def test_truncated_final_fragment_is_the_only_discarded_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcript.jsonl"
    good = b'{"sequence":1}\n{"sequence":2}\n'
    fragment = b'{"sequence":'
    path.write_bytes(good + fragment)

    result = validate_and_repair_jsonl_tail(path)

    assert path.read_bytes() == good
    assert result.valid_record_count == 2
    assert result.discarded_tail_bytes == len(fragment)
    assert not result.appended_final_lf


def test_valid_final_record_without_lf_is_preserved_and_terminated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    original = b'{"sequence":1}'
    path.write_bytes(original)

    result = validate_and_repair_jsonl_tail(path)

    assert path.read_bytes() == original + b"\n"
    assert result.valid_record_count == 1
    assert result.discarded_tail_bytes == 0
    assert result.appended_final_lf


def test_invalid_complete_middle_line_is_not_silently_removed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state-history.jsonl"
    original = b'{"revision":1}\nnot-json\n{"revision":2}\n'
    path.write_bytes(original)

    with pytest.raises(JsonlValidationError, match="line 2"):
        validate_and_repair_jsonl_tail(path)

    assert path.read_bytes() == original


def test_inspection_is_read_only_until_its_plan_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    original = b'{"sequence":1}'
    path.write_bytes(original)

    plan = inspect_jsonl_tail(path)

    assert path.read_bytes() == original
    result = apply_jsonl_tail_repair(path, plan)
    assert path.read_bytes() == original + b"\n"
    assert result.appended_final_lf


def test_apply_rejects_same_size_file_changed_after_inspection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"sequence":1}')
    plan = inspect_jsonl_tail(path)
    changed = b'{"sequence":2}'
    path.write_bytes(changed)

    with pytest.raises(JsonlValidationError, match="changed after inspection"):
        apply_jsonl_tail_repair(path, plan)

    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    "original",
    [
        b"[]\n",
        b"NaN\n",
        b'{"sequence":1}\r\n',
        b'{"sequence":1}\n\xff\n',
        b'{"sequence":1}\x85',
    ],
)
def test_invalid_complete_line_is_rejected_without_mutation(
    tmp_path: Path,
    original: bytes,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(original)

    with pytest.raises(JsonlValidationError, match="line"):
        validate_and_repair_jsonl_tail(path)

    assert path.read_bytes() == original


def test_invalid_utf8_final_fragment_is_discarded_exactly(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    good = b'{"sequence":1}\n'
    fragment = b'{"sequence":2,"text":"\xff'
    path.write_bytes(good + fragment)

    result = validate_and_repair_jsonl_tail(path)

    assert path.read_bytes() == good
    assert result.valid_record_count == 1
    assert result.discarded_tail_bytes == len(fragment)
    assert not result.appended_final_lf


def test_complete_valid_file_requires_no_repair(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    original = '{"sequence":1,"text":"保存"}\n'.encode()
    path.write_bytes(original)

    result = validate_and_repair_jsonl_tail(path)

    assert path.read_bytes() == original
    assert result.valid_record_count == 1
    assert result.discarded_tail_bytes == 0
    assert not result.appended_final_lf
