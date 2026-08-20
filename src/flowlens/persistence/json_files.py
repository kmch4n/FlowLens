"""Crash-safe JSON snapshots and append-only JSONL files."""

import json
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Self, TextIO

from flowlens.domain._validation import json_dumps

_Fsync = Callable[[int], None]
_CreateTemp = Callable[[Path], tuple[Path, TextIO]]
_RemoveTemp = Callable[[Path], None]
_Replace = Callable[[Path, Path], None]
_atomic_replace_lock = threading.Lock()


def _sync_file(file_descriptor: int) -> None:
    os.fsync(file_descriptor)


def _replace_file(source: Path, target: Path) -> None:
    os.replace(source, target)


def _create_temp_file(target: Path) -> tuple[Path, TextIO]:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temp_path = Path(temp_name)
    try:
        file = os.fdopen(
            file_descriptor,
            encoding="utf-8",
            mode="w",
            newline="\n",
        )
    except BaseException as primary_error:
        try:
            os.close(file_descriptor)
        except OSError as close_error:
            primary_error.add_note(
                f"Temporary descriptor cleanup failed: {close_error}"
            )
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            primary_error.add_note(
                f"Temporary file cleanup failed for {temp_path}: {cleanup_error}"
            )
        raise
    return temp_path, file


def _remove_temp_file(path: Path) -> None:
    path.unlink()


def encode_jsonl_record(value: object) -> bytes:
    """Encode one compact UTF-8 JSON record terminated by LF."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class JsonlValidationError(ValueError):
    """Raised when a complete JSONL record is not a UTF-8 JSON object."""


@dataclass(frozen=True, slots=True)
class JsonlRepairPlan:
    """Read-only tail inspection result guarded by file identity metadata."""

    path: Path
    expected_size: int
    expected_sha256: str
    valid_record_count: int
    discarded_tail_bytes: int
    append_final_lf: bool


@dataclass(frozen=True, slots=True)
class JsonlRepairResult:
    """Summary of an applied JSONL tail repair."""

    valid_record_count: int
    discarded_tail_bytes: int
    appended_final_lf: bool


@dataclass(frozen=True, slots=True)
class AtomicJsonFile:
    """Atomically replace one indented JSON snapshot."""

    path: Path
    _create_temp: _CreateTemp = field(
        default=_create_temp_file,
        repr=False,
        compare=False,
    )
    _fsync: _Fsync = field(default=_sync_file, repr=False, compare=False)
    _replace: _Replace = field(default=_replace_file, repr=False, compare=False)
    _remove_temp: _RemoveTemp = field(
        default=_remove_temp_file,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def replace(self, value: object) -> None:
        """Durably write a sibling temporary file before replacing the target."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        primary_error: BaseException | None = None
        try:
            temp_path, file = self._create_temp(self.path)
            file_operation_error: BaseException | None = None
            try:
                _write_text_all(file, json_dumps(value))
                file.flush()
                self._fsync(file.fileno())
            except BaseException as error:
                file_operation_error = error
                raise
            finally:
                try:
                    file.close()
                except BaseException as close_error:
                    if file_operation_error is None:
                        raise
                    file_operation_error.add_note(
                        f"Temporary file close failed for {temp_path}: {close_error}"
                    )
            with _atomic_replace_lock:
                self._replace(temp_path, self.path)
            temp_path = None
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if temp_path is not None:
                try:
                    self._remove_temp(temp_path)
                except FileNotFoundError:
                    pass
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        f"Temporary file cleanup failed for {temp_path}: "
                        f"{cleanup_error}"
                    )


class JsonlAppender:
    """Thread-safe binary appender for compact JSON object records."""

    def __init__(
        self,
        path: Path,
        file: BinaryIO,
        *,
        _fsync: _Fsync = _sync_file,
    ) -> None:
        self.path = Path(path)
        self._file = file
        self._fsync = _fsync
        self._lock = threading.Lock()

    @classmethod
    def open(cls, path: Path, *, _fsync: _Fsync = _sync_file) -> Self:
        """Open a JSONL file for binary append, creating its parent directory."""

        normalized_path = Path(path)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            normalized_path,
            normalized_path.open(mode="ab"),
            _fsync=_fsync,
        )

    def append(self, value: object) -> None:
        """Append and flush one compact UTF-8 JSON record."""

        encoded = encode_jsonl_record(value)
        with self._lock:
            _write_all(self._file, encoded)
            self._file.flush()

    def sync(self) -> None:
        """Flush userspace buffers and synchronize the file descriptor."""

        with self._lock:
            self._file.flush()
            self._fsync(self._file.fileno())

    def close(self) -> None:
        """Close the append stream after flushing pending bytes."""

        with self._lock:
            self._file.close()


def inspect_jsonl_tail(path: Path) -> JsonlRepairPlan:
    """Inspect JSONL bytes without modifying the file."""

    normalized_path = Path(path)
    encoded = normalized_path.read_bytes()
    lines = encoded.splitlines(keepends=True)
    valid_record_count = 0
    discarded_tail_bytes = 0
    append_final_lf = False

    for line_number, line in enumerate(lines, start=1):
        is_final_line = line_number == len(lines)
        if line.endswith(b"\r\n") or _ends_with_non_lf_separator(line):
            raise JsonlValidationError(
                f"Invalid JSONL record at {normalized_path}, line {line_number}: "
                "line must end with LF"
            )
        if line.endswith(b"\n"):
            _decode_json_object(line[:-1], normalized_path, line_number)
            valid_record_count += 1
            continue
        if not is_final_line:
            raise JsonlValidationError(
                f"Invalid JSONL record at {normalized_path}, line {line_number}: "
                "line must end with LF"
            )
        if _is_json_object(line):
            valid_record_count += 1
            append_final_lf = True
        else:
            discarded_tail_bytes = len(line)

    return JsonlRepairPlan(
        path=normalized_path,
        expected_size=len(encoded),
        expected_sha256=sha256(encoded).hexdigest(),
        valid_record_count=valid_record_count,
        discarded_tail_bytes=discarded_tail_bytes,
        append_final_lf=append_final_lf,
    )


def apply_jsonl_tail_repair(
    path: Path,
    plan: JsonlRepairPlan,
) -> JsonlRepairResult:
    """Apply a previously inspected tail repair if the file is unchanged."""

    normalized_path = Path(path)
    if normalized_path != plan.path:
        raise JsonlValidationError(
            f"JSONL repair plan is for {plan.path}, not {normalized_path}"
        )

    current = normalized_path.read_bytes()
    if (
        len(current) != plan.expected_size
        or sha256(current).hexdigest() != plan.expected_sha256
    ):
        raise JsonlValidationError(
            f"JSONL file changed after inspection: {normalized_path}"
        )

    if plan.discarded_tail_bytes or plan.append_final_lf:
        with normalized_path.open(mode="r+b") as file:
            if plan.discarded_tail_bytes:
                file.truncate(plan.expected_size - plan.discarded_tail_bytes)
            else:
                file.seek(0, os.SEEK_END)
                _write_all(file, b"\n")
            file.flush()
            os.fsync(file.fileno())

    return JsonlRepairResult(
        valid_record_count=plan.valid_record_count,
        discarded_tail_bytes=plan.discarded_tail_bytes,
        appended_final_lf=plan.append_final_lf,
    )


def validate_and_repair_jsonl_tail(path: Path) -> JsonlRepairResult:
    """Inspect and conservatively repair only the final JSONL fragment."""

    normalized_path = Path(path)
    return apply_jsonl_tail_repair(
        normalized_path,
        inspect_jsonl_tail(normalized_path),
    )


def _decode_json_object(encoded: bytes, path: Path, line_number: int) -> None:
    try:
        decoded = encoded.decode("utf-8")
        parsed = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise JsonlValidationError(
            f"Invalid JSONL record at {path}, line {line_number}: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise JsonlValidationError(
            f"Invalid JSONL record at {path}, line {line_number}: "
            "record must be a JSON object"
        )


def _is_json_object(encoded: bytes) -> bool:
    try:
        decoded = encoded.decode("utf-8")
        parsed = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        return False
    return isinstance(parsed, dict)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _ends_with_non_lf_separator(line: bytes) -> bool:
    return line.endswith((b"\r", b"\x0b", b"\x0c", b"\x1c", b"\x1d", b"\x1e", b"\x85"))


def _write_all(file: BinaryIO, encoded: bytes) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = file.write(remaining)
        if written is None or written <= 0:
            raise OSError("file write made no progress")
        remaining = remaining[written:]


def _write_text_all(file: TextIO, encoded: str) -> None:
    remaining = encoded
    while remaining:
        written = file.write(remaining)
        if written is None or written <= 0:
            raise OSError("file write made no progress")
        remaining = remaining[written:]
