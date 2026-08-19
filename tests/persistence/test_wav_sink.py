import struct
import wave
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self, cast

import pytest

from flowlens.persistence.wav_sink import WavSink, repair_wav_header


class _TrackingBinaryFile:
    def __init__(
        self,
        *,
        maximum_write: int | None = None,
        fail_flush: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.contents = bytearray(44)
        self.events: list[str] = []
        self.position = 44
        self.closed = False
        self.maximum_write = maximum_write
        self.fail_flush = fail_flush
        self.fail_close = fail_close

    def write(self, data: bytes | memoryview) -> int:
        encoded = bytes(data)
        if self.maximum_write is not None:
            encoded = encoded[: self.maximum_write]
        end = self.position + len(encoded)
        if end > len(self.contents):
            self.contents.extend(b"\x00" * (end - len(self.contents)))
        self.contents[self.position : end] = encoded
        self.position = end
        self.events.append(f"write:{len(encoded)}")
        return len(encoded)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence != 0:
            raise ValueError("test file only supports absolute seeks")
        self.position = offset
        self.events.append(f"seek:{offset}")
        return self.position

    def flush(self) -> None:
        self.events.append("flush")
        if self.fail_flush:
            raise OSError("primary flush failure")

    def fileno(self) -> int:
        return 41

    def close(self) -> None:
        self.events.append("close")
        self.closed = True
        if self.fail_close:
            raise OSError("secondary close failure")


class _ScriptedWriteFile(_TrackingBinaryFile):
    def __init__(
        self,
        actions: list[int | BaseException],
        *,
        fail_close: bool = False,
    ) -> None:
        super().__init__(fail_close=fail_close)
        self._actions = actions

    def write(self, data: bytes | memoryview) -> int:
        action = self._actions.pop(0)
        if isinstance(action, BaseException):
            self.events.append("write:error")
            raise action
        encoded = bytes(data[:action])
        end = self.position + len(encoded)
        if end > len(self.contents):
            self.contents.extend(b"\x00" * (end - len(self.contents)))
        self.contents[self.position : end] = encoded
        self.position = end
        self.events.append(f"write:{len(encoded)}")
        return len(encoded)


class _RepairBinaryFile(_TrackingBinaryFile):
    def __init__(
        self,
        *,
        primary_stage: str | None = None,
        primary_error: BaseException | None = None,
        fail_close: bool = False,
    ) -> None:
        super().__init__(fail_close=fail_close)
        self.contents = bytearray(
            struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                0,
                b"WAVE",
                b"fmt ",
                16,
                1,
                1,
                16_000,
                32_000,
                2,
                16,
                b"data",
                0,
            )
            + b"\x01\x00"
        )
        self.position = 0
        self._primary_stage = primary_stage
        self._primary_error = primary_error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def read(self, size: int = -1) -> bytes:
        end = len(self.contents) if size < 0 else self.position + size
        encoded = bytes(self.contents[self.position : end])
        self.position += len(encoded)
        self.events.append(f"read:{len(encoded)}")
        return encoded

    def write(self, data: bytes | memoryview) -> int:
        if self._primary_stage == "write":
            self.events.append("write:error")
            assert self._primary_error is not None
            raise self._primary_error
        return super().write(data)

    def flush(self) -> None:
        if self._primary_stage == "flush":
            self.events.append("flush:error")
            assert self._primary_error is not None
            raise self._primary_error
        super().flush()


class _RepairPathProxy:
    def __init__(self, file: _RepairBinaryFile) -> None:
        self._file = file

    def open(self, mode: str) -> BinaryIO:
        assert mode == "r+b"
        return cast(BinaryIO, self._file)


def test_sink_writes_canonical_wav_and_returns_sample_offsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    assert sink.append(b"\x00\x00" * 320) == 0
    assert sink.append(b"\x01\x00" * 160) == 320
    sink.finalize()

    with wave.open(str(path), "rb") as reader:
        assert (
            reader.getframerate(),
            reader.getsampwidth(),
            reader.getnchannels(),
        ) == (16_000, 2, 1)
        assert reader.getnframes() == 480


def test_repair_uses_file_size_without_changing_pcm_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loopback.wav"
    sink = WavSink.open(path)
    pcm = b"\x02\x00" * 320
    sink.append(pcm)
    sink.close_incomplete()
    with path.open("r+b") as file:
        file.seek(4)
        file.write(struct.pack("<I", 0))
        file.seek(40)
        file.write(struct.pack("<I", 0))

    before_payload = path.read_bytes()[44:]
    result = repair_wav_header(path)

    assert path.read_bytes()[44:] == before_payload
    assert result.valid_pcm_bytes == len(pcm)
    with wave.open(str(path), "rb") as reader:
        assert reader.getnframes() == 320


@pytest.mark.parametrize("primary_stage", ["write", "flush", "fsync"])
def test_repair_preserves_primary_error_when_close_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    primary_stage: str,
) -> None:
    primary_error = OSError(f"primary {primary_stage} failure")
    output = _RepairBinaryFile(
        primary_stage=primary_stage,
        primary_error=primary_error,
        fail_close=True,
    )
    path_proxy = _RepairPathProxy(output)
    monkeypatch.setattr("flowlens.persistence.wav_sink.Path", lambda _: path_proxy)
    monkeypatch.setattr("flowlens.persistence.wav_sink._file_size", lambda _: 46)

    def fsync(file_descriptor: int) -> None:
        if primary_stage == "fsync":
            raise primary_error

    monkeypatch.setattr("flowlens.persistence.wav_sink.os.fsync", fsync)

    with pytest.raises(OSError) as raised:
        repair_wav_header(Path("mic.wav"))

    assert raised.value is primary_error
    assert output.events.count("close") == 1
    assert output.closed
    assert any("secondary close failure" in note for note in raised.value.__notes__)


def test_repair_surfaces_close_only_error(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _RepairBinaryFile(fail_close=True)
    path_proxy = _RepairPathProxy(output)
    monkeypatch.setattr("flowlens.persistence.wav_sink.Path", lambda _: path_proxy)
    monkeypatch.setattr("flowlens.persistence.wav_sink._file_size", lambda _: 46)
    monkeypatch.setattr("flowlens.persistence.wav_sink.os.fsync", lambda _: None)

    with pytest.raises(OSError, match="secondary close failure"):
        repair_wav_header(Path("mic.wav"))

    assert output.events.count("close") == 1
    assert output.closed


def test_append_rejects_partial_sample_without_changing_payload_or_offset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    try:
        assert sink.append(b"\x01\x00") == 0

        with pytest.raises(ValueError, match="complete 16-bit samples"):
            sink.append(b"\x02")

        assert sink.append(b"\x03\x00") == 1
    finally:
        sink.finalize()

    assert path.read_bytes()[44:] == b"\x01\x00\x03\x00"


def test_append_retries_short_binary_writes_and_syncs_only_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _TrackingBinaryFile(maximum_write=1)

    def track_fsync(file_descriptor: int) -> None:
        output.events.append(f"fsync:{file_descriptor}")

    monkeypatch.setattr("flowlens.persistence.wav_sink.os.fsync", track_fsync)
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))

    assert sink.append(b"\x01\x00\x02\x00") == 0
    assert bytes(output.contents[44:]) == b"\x01\x00\x02\x00"
    assert "flush" not in output.events
    assert all(not event.startswith("fsync:") for event in output.events)

    sink.sync()

    assert output.events[-2:] == ["flush", "fsync:41"]
    sink.close_incomplete()


def test_partial_append_then_zero_progress_fails_sink_without_offset_reuse() -> None:
    output = _ScriptedWriteFile([1, 0])
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))

    with pytest.raises(OSError, match="write made no progress"):
        sink.append(b"\x01\x00")

    assert bytes(output.contents[44:]) == b"\x01"
    assert output.events.count("close") == 1
    terminal_events = list(output.events)
    terminal_bytes = bytes(output.contents)
    for operation in (
        lambda: sink.append(b"\x02\x00"),
        sink.sync,
        sink.finalize,
        sink.close_incomplete,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert output.events == terminal_events
    assert bytes(output.contents) == terminal_bytes


def test_partial_append_preserves_primary_error_when_failed_close_also_fails() -> None:
    primary_error = OSError("primary append failure")
    output = _ScriptedWriteFile([1, primary_error], fail_close=True)
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))

    with pytest.raises(OSError) as raised:
        sink.append(b"\x01\x00")

    assert raised.value is primary_error
    assert output.events.count("close") == 1
    assert output.closed
    assert any("secondary close failure" in note for note in raised.value.__notes__)
    with pytest.raises(RuntimeError, match="failed"):
        sink.close_incomplete()
    assert output.events.count("close") == 1


def test_open_closes_owned_binary_handle_when_initial_header_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _TrackingBinaryFile(maximum_write=0, fail_close=True)
    modes: list[str] = []

    class _OpenPathProxy:
        def __init__(self, value: object) -> None:
            self._path = Path(cast(str | Path, value))

        @property
        def parent(self) -> Path:
            return self._path.parent

        def open(self, mode: str) -> BinaryIO:
            modes.append(mode)
            return cast(BinaryIO, output)

        def __fspath__(self) -> str:
            return str(self._path)

    monkeypatch.setattr("flowlens.persistence.wav_sink.Path", _OpenPathProxy)

    with pytest.raises(OSError, match="write made no progress") as raised:
        WavSink.open(tmp_path / "mic.wav")

    assert modes == ["w+b"]
    assert output.events.count("close") == 1
    assert output.closed
    assert any("secondary close failure" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("close_method", ["finalize", "close_incomplete"])
def test_terminal_close_preserves_fsync_error_and_attempts_close_once(
    monkeypatch: pytest.MonkeyPatch,
    close_method: str,
) -> None:
    output = _TrackingBinaryFile(fail_close=True)

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError(f"primary fsync failure for {file_descriptor}")

    monkeypatch.setattr("flowlens.persistence.wav_sink.os.fsync", fail_fsync)
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))

    with pytest.raises(OSError, match="primary fsync failure") as raised:
        getattr(sink, close_method)()

    assert output.events.count("close") == 1
    assert output.closed
    assert any("secondary close failure" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("close_method", ["finalize", "close_incomplete"])
def test_terminal_close_preserves_flush_error_and_attempts_close_once(
    close_method: str,
) -> None:
    output = _TrackingBinaryFile(fail_flush=True, fail_close=True)
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))

    with pytest.raises(OSError, match="primary flush failure") as raised:
        getattr(sink, close_method)()

    assert output.events.count("close") == 1
    assert output.closed
    assert any("secondary close failure" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("close_method", ["finalize", "close_incomplete"])
def test_terminal_close_surfaces_close_only_error(
    monkeypatch: pytest.MonkeyPatch,
    close_method: str,
) -> None:
    output = _TrackingBinaryFile(fail_close=True)
    monkeypatch.setattr("flowlens.persistence.wav_sink.os.fsync", lambda _: None)
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))

    with pytest.raises(OSError, match="secondary close failure"):
        getattr(sink, close_method)()

    assert output.events.count("close") == 1
    assert output.closed


@pytest.mark.parametrize("close_method", ["finalize", "close_incomplete"])
def test_closed_sink_rejects_every_later_operation(
    tmp_path: Path,
    close_method: str,
) -> None:
    sink = WavSink.open(tmp_path / "mic.wav")
    getattr(sink, close_method)()

    for operation in (
        lambda: sink.append(b"\x00\x00"),
        sink.sync,
        sink.finalize,
        sink.close_incomplete,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_append_rejects_pcm_that_would_overflow_riff_size_before_writing() -> None:
    output = _TrackingBinaryFile()
    sink = WavSink(Path("mic.wav"), cast(BinaryIO, output))
    sink._sample_count = (0xFFFFFFFF - 36) // 2
    before = bytes(output.contents)

    with pytest.raises(OverflowError, match="RIFF"):
        sink.append(b"\x00\x00")

    assert bytes(output.contents) == before


def test_incomplete_close_preserves_zero_sizes_and_exact_pcm_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mic.wav"
    pcm = b"\x00\x0a\x0d\xff"
    sink = WavSink.open(path)
    sink.append(pcm)
    sink.close_incomplete()

    encoded = path.read_bytes()
    assert encoded[0:4] == b"RIFF"
    assert encoded[4:8] == b"\x00\x00\x00\x00"
    assert encoded[8:12] == b"WAVE"
    assert encoded[36:40] == b"data"
    assert encoded[40:44] == b"\x00\x00\x00\x00"
    assert encoded[44:] == pcm


def test_reopening_path_truncates_prior_wav_and_restarts_offsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mic.wav"
    first = WavSink.open(path)
    first.append(b"\x01\x00" * 10)
    first.finalize()

    second = WavSink.open(path)
    assert second.append(b"\x02\x00") == 0
    second.finalize()

    assert path.read_bytes()[44:] == b"\x02\x00"
    with wave.open(str(path), "rb") as reader:
        assert reader.getnframes() == 1


@pytest.mark.parametrize("length", [0, 43])
def test_repair_rejects_truncated_header_without_mutation(
    tmp_path: Path,
    length: int,
) -> None:
    path = tmp_path / "mic.wav"
    original = bytes(range(44))[:length]
    path.write_bytes(original)

    with pytest.raises(ValueError, match="44-byte header"):
        repair_wav_header(path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("offset", "replacement", "match"),
    [
        (0, b"RIFX", "RIFF"),
        (8, b"FAIL", "WAVE"),
        (12, b"fail", "fmt"),
        (16, struct.pack("<I", 18), "format chunk size"),
        (20, struct.pack("<H", 3), "PCM format"),
        (22, struct.pack("<H", 2), "mono"),
        (24, struct.pack("<I", 48_000), "16,000 Hz"),
        (28, struct.pack("<I", 64_000), "byte rate"),
        (32, struct.pack("<H", 4), "block alignment"),
        (34, struct.pack("<H", 24), "16-bit"),
        (36, b"FAIL", "data"),
    ],
)
def test_repair_rejects_noncanonical_header_without_mutation(
    tmp_path: Path,
    offset: int,
    replacement: bytes,
    match: str,
) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    sink.append(b"\x01\x00")
    sink.finalize()
    damaged = bytearray(path.read_bytes())
    damaged[offset : offset + len(replacement)] = replacement
    original = bytes(damaged)
    path.write_bytes(original)

    with pytest.raises(ValueError, match=match):
        repair_wav_header(path)

    assert path.read_bytes() == original


def test_repair_trims_size_to_complete_sample_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    sink.append(b"\x01\x00")
    sink.close_incomplete()
    with path.open("ab") as file:
        file.write(b"\xff")
    payload = path.read_bytes()[44:]

    first = repair_wav_header(path)
    second = repair_wav_header(path)

    assert path.read_bytes()[44:] == payload == b"\x01\x00\xff"
    assert first.original_pcm_bytes == 3
    assert first.valid_pcm_bytes == 2
    assert first.header_changed
    assert second.original_pcm_bytes == 3
    assert second.valid_pcm_bytes == 2
    assert not second.header_changed


def test_repair_rejects_payload_beyond_riff_limit_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    sink.close_incomplete()
    original = path.read_bytes()

    oversized_size = 44 + (0xFFFFFFFF - 36) + 2
    monkeypatch.setattr(
        "flowlens.persistence.wav_sink._file_size",
        lambda _: oversized_size,
    )

    with pytest.raises(OverflowError, match="RIFF"):
        repair_wav_header(path)

    assert path.read_bytes() == original


def test_repair_releases_handle_when_fsync_fails_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    sink.append(b"\x01\x00")
    sink.close_incomplete()

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError(f"fsync failure for {file_descriptor}")

    monkeypatch.setattr("flowlens.persistence.wav_sink.os.fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failure"):
        repair_wav_header(path)

    path.unlink()
    assert not path.exists()
