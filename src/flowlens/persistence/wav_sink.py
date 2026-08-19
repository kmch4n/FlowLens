"""Canonical mono PCM WAV persistence and crash header repair."""

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self

_HEADER_SIZE = 44
_SAMPLE_WIDTH = 2
_SAMPLE_RATE = 16_000
_CHANNEL_COUNT = 1
_MAX_UINT32 = 0xFFFFFFFF
_MAX_PCM_BYTES = ((_MAX_UINT32 - 36) // _SAMPLE_WIDTH) * _SAMPLE_WIDTH


@dataclass(frozen=True, slots=True)
class WavRepairResult:
    """Summary of fixed-format WAV header repair."""

    original_pcm_bytes: int
    valid_pcm_bytes: int
    header_changed: bool


class WavSink:
    """Owned binary sink for canonical 16 kHz mono signed 16-bit PCM."""

    def __init__(self, path: Path, file: BinaryIO) -> None:
        self.path = Path(path)
        self._file = file
        self._sample_count = 0
        self._closed = False
        self._failed = False

    @classmethod
    def open(cls, path: Path) -> Self:
        """Create or truncate one canonical WAV file."""

        normalized_path = Path(path)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        file = normalized_path.open("w+b")
        try:
            _write_all(file, _initial_header())
        except BaseException as primary_error:
            try:
                file.close()
            except BaseException as close_error:
                primary_error.add_note(
                    f"WAV file close failed for {normalized_path}: {close_error}"
                )
            raise
        return cls(normalized_path, file)

    def append(self, pcm: bytes) -> int:
        """Append PCM and fail terminally if an error follows a partial write."""

        self._ensure_open()
        if len(pcm) % _SAMPLE_WIDTH != 0:
            raise ValueError("PCM data must contain complete 16-bit samples")
        pcm_bytes = self._sample_count * _SAMPLE_WIDTH + len(pcm)
        if pcm_bytes > _MAX_PCM_BYTES:
            raise OverflowError("PCM data exceeds the canonical RIFF size limit")
        first_sample = self._sample_count
        write_started = False

        def mark_write_started() -> None:
            nonlocal write_started
            write_started = True

        try:
            _write_all(self._file, pcm, on_progress=mark_write_started)
        except BaseException as primary_error:
            if write_started:
                self._failed = True
                self._closed = True
                _close_preserving_primary(self._file, self.path, primary_error)
            raise
        self._sample_count += len(pcm) // _SAMPLE_WIDTH
        return first_sample

    def sync(self) -> None:
        """Flush and durably synchronize pending header and PCM bytes."""

        self._ensure_open()
        self._file.flush()
        os.fsync(self._file.fileno())

    def finalize(self) -> None:
        """Write final chunk sizes, synchronize them, and close the file."""

        self._ensure_open()

        def publish_header() -> None:
            pcm_bytes = self._sample_count * _SAMPLE_WIDTH
            self._file.seek(4)
            _write_all(self._file, struct.pack("<I", 36 + pcm_bytes))
            self._file.seek(40)
            _write_all(self._file, struct.pack("<I", pcm_bytes))
            self._file.flush()
            os.fsync(self._file.fileno())

        self._run_terminal_operation(publish_header)

    def close_incomplete(self) -> None:
        """Synchronize and close once without publishing final chunk sizes.

        Already closed or failed sinks reject the call without closing again.
        """

        self._ensure_open()

        def synchronize() -> None:
            self._file.flush()
            os.fsync(self._file.fileno())

        self._run_terminal_operation(synchronize)

    def _ensure_open(self) -> None:
        if self._failed:
            raise RuntimeError(f"WAV sink is failed: {self.path}")
        if self._closed:
            raise RuntimeError(f"WAV sink is closed: {self.path}")

    def _run_terminal_operation(self, operation: Callable[[], None]) -> None:
        primary_error: BaseException | None = None
        try:
            operation()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            self._closed = True
            try:
                self._file.close()
            except BaseException as close_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"WAV file close failed for {self.path}: {close_error}"
                )


def repair_wav_header(path: Path) -> WavRepairResult:
    """Repair fixed-size chunk lengths from the complete PCM payload."""

    normalized_path = Path(path)
    file = normalized_path.open("r+b")
    primary_error: BaseException | None = None
    try:
        header = file.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE:
            raise ValueError("WAV file must contain a complete 44-byte header")
        _validate_canonical_header(header)
        original_pcm_bytes = _file_size(normalized_path) - _HEADER_SIZE
        valid_pcm_bytes = (original_pcm_bytes // _SAMPLE_WIDTH) * _SAMPLE_WIDTH
        if valid_pcm_bytes > _MAX_PCM_BYTES:
            raise OverflowError("PCM payload exceeds the canonical RIFF size limit")
        riff_size = 36 + valid_pcm_bytes
        original_riff_size = struct.unpack_from("<I", header, 4)[0]
        original_data_size = struct.unpack_from("<I", header, 40)[0]
        file.seek(4)
        _write_all(file, struct.pack("<I", riff_size))
        file.seek(40)
        _write_all(file, struct.pack("<I", valid_pcm_bytes))
        file.flush()
        os.fsync(file.fileno())
        result = WavRepairResult(
            original_pcm_bytes=original_pcm_bytes,
            valid_pcm_bytes=valid_pcm_bytes,
            header_changed=(
                original_riff_size != riff_size or original_data_size != valid_pcm_bytes
            ),
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            file.close()
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"WAV file close failed for {normalized_path}: {close_error}"
            )
    return result


def _initial_header() -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0,
        b"WAVE",
        b"fmt ",
        16,
        1,
        _CHANNEL_COUNT,
        _SAMPLE_RATE,
        _SAMPLE_RATE * _SAMPLE_WIDTH * _CHANNEL_COUNT,
        _SAMPLE_WIDTH * _CHANNEL_COUNT,
        _SAMPLE_WIDTH * 8,
        b"data",
        0,
    )


def _validate_canonical_header(header: bytes) -> None:
    if header[0:4] != b"RIFF":
        raise ValueError("WAV header must start with RIFF")
    if header[8:12] != b"WAVE":
        raise ValueError("WAV header must contain the WAVE marker")
    if header[12:16] != b"fmt ":
        raise ValueError("WAV header must contain the fmt marker")
    if struct.unpack_from("<I", header, 16)[0] != 16:
        raise ValueError("WAV format chunk size must be 16")
    if struct.unpack_from("<H", header, 20)[0] != 1:
        raise ValueError("WAV must use PCM format 1")
    if struct.unpack_from("<H", header, 22)[0] != _CHANNEL_COUNT:
        raise ValueError("WAV must be mono")
    if struct.unpack_from("<I", header, 24)[0] != _SAMPLE_RATE:
        raise ValueError("WAV sample rate must be 16,000 Hz")
    if struct.unpack_from("<I", header, 28)[0] != _SAMPLE_RATE * _SAMPLE_WIDTH:
        raise ValueError("WAV byte rate must be 32,000")
    if struct.unpack_from("<H", header, 32)[0] != _SAMPLE_WIDTH:
        raise ValueError("WAV block alignment must be 2")
    if struct.unpack_from("<H", header, 34)[0] != _SAMPLE_WIDTH * 8:
        raise ValueError("WAV samples must be 16-bit")
    if header[36:40] != b"data":
        raise ValueError("WAV header must contain the data marker")


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _close_preserving_primary(
    file: BinaryIO,
    path: object,
    primary_error: BaseException,
) -> None:
    try:
        file.close()
    except BaseException as close_error:
        primary_error.add_note(f"WAV file close failed for {path}: {close_error}")


def _write_all(
    file: BinaryIO,
    encoded: bytes,
    *,
    on_progress: Callable[[], None] | None = None,
) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = file.write(remaining)
        if written is None or written <= 0:
            raise OSError("WAV file write made no progress")
        if on_progress is not None:
            on_progress()
        remaining = remaining[written:]
