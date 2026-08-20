"""Single-owner creation and append APIs for one session artifact bundle."""

import math
import os
import stat
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

from flowlens.domain.discussion import DiscussionState, StateHistoryRecord
from flowlens.domain.enums import AudioSource, SessionStatus
from flowlens.domain.messages import AudioWriteCommand, EventRecord, TranscriptRecord
from flowlens.domain.session import SessionManifest
from flowlens.persistence.json_files import AtomicJsonFile, JsonlAppender
from flowlens.persistence.wav_sink import WavSink

_SOURCE_RANK = {
    AudioSource.ME: 0,
    AudioSource.OTHERS: 1,
}
_BOOTSTRAP_CLAIM_NAME = ".flowlens-bootstrap.claim"


class PersistenceInvariantError(ValueError):
    """Raised before a write when a session persistence invariant is violated."""


class WriterOwnershipError(RuntimeError):
    """Raised when a process other than the opening PID attempts a mutation."""


class _WriterState(Enum):
    OPEN = "open"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(slots=True)
class _BootstrapClaim:
    """One process-owned exclusive claim for a session bootstrap directory."""

    path: Path
    descriptor: int | None
    owns_path: bool = True

    @classmethod
    def acquire(cls, session_dir: Path) -> "_BootstrapClaim":
        path = session_dir / _BOOTSTRAP_CLAIM_NAME
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            return cls(path=path, descriptor=descriptor)
        except BaseException as primary_error:
            failures = cls._release_parts(path, descriptor, owns_path=True)
            cls._attach_failures(primary_error, path, failures)
            raise

    def release(
        self,
        primary_error: BaseException | None = None,
    ) -> BaseException | None:
        descriptor = self.descriptor
        self.descriptor = None
        failures = self._release_parts(self.path, descriptor, self.owns_path)
        if not any(operation == "unlink" for operation, _ in failures):
            self.owns_path = False
        if not failures:
            return primary_error
        if primary_error is None:
            primary_error = failures[0][1]
        self._attach_failures(primary_error, self.path, failures)
        return primary_error

    @staticmethod
    def _release_parts(
        path: Path,
        descriptor: int | None,
        owns_path: bool,
    ) -> list[tuple[str, BaseException]]:
        failures: list[tuple[str, BaseException]] = []
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                failures.append(("descriptor close", error))
        if owns_path:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except BaseException as error:
                failures.append(("unlink", error))
        return failures

    @staticmethod
    def _attach_failures(
        primary_error: BaseException,
        path: Path,
        failures: Iterable[tuple[str, BaseException]],
    ) -> None:
        for operation, cleanup_error in failures:
            primary_error.add_note(
                f"Bootstrap claim {operation} failed for {path}: {cleanup_error}"
            )


class SessionWriter:
    """Own and mutate the seven persistent artifacts for one active session."""

    def __init__(
        self,
        session_dir: Path,
        manifest: SessionManifest,
        initial_state: DiscussionState,
        microphone_sink: WavSink,
        loopback_sink: WavSink,
        transcript_log: JsonlAppender,
        state_history_log: JsonlAppender,
        event_log: JsonlAppender,
        *,
        owner_pid: int,
        sync_interval_seconds: float,
        opened_monotonic: float,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.owner_pid = owner_pid
        self.opened_monotonic = opened_monotonic
        self._sync_interval_seconds = sync_interval_seconds
        self._manifest = manifest
        self._manifest_file = AtomicJsonFile(self.session_dir / "session.json")
        self._discussion_file = AtomicJsonFile(
            self.session_dir / "discussion-state.json"
        )
        self._microphone_sink = microphone_sink
        self._loopback_sink = loopback_sink
        self._transcript_log = transcript_log
        self._state_history_log = state_history_log
        self._event_log = event_log
        self._source_cursors = {
            AudioSource.ME: 0,
            AudioSource.OTHERS: 0,
        }
        self._next_transcript_sequence = 1
        self._next_event_sequence = 1
        self._discussion_revision = initial_state.revision
        self._transcript_segment_ids: set[str] = set()
        self._last_transcript_order: tuple[int, int] | None = None
        self._last_event_session_time_ms: int | None = None
        self._state = _WriterState.OPEN
        self._resources_closed = False

    @classmethod
    def open(
        cls,
        session_dir: Path,
        manifest: SessionManifest,
        initial_state: DiscussionState,
        *,
        sync_interval_seconds: float = 1.0,
    ) -> Self:
        """Create exactly seven artifacts for a new incomplete session."""

        normalized_dir = Path(session_dir)
        manifest = cls._canonical_manifest(manifest)
        initial_state = cls._canonical_discussion_state(
            initial_state,
            "initial discussion state",
        )
        cls._validate_open_inputs(
            manifest,
            initial_state,
            sync_interval_seconds,
        )
        cls._validate_storage_path(normalized_dir)
        owner_pid = os.getpid()
        opened_monotonic = time.monotonic()
        cls._prepare_empty_session_directory(normalized_dir)
        cls._validate_storage_path(normalized_dir)
        claim = _BootstrapClaim.acquire(normalized_dir)

        microphone_sink: WavSink | None = None
        loopback_sink: WavSink | None = None
        transcript_log: JsonlAppender | None = None
        state_history_log: JsonlAppender | None = None
        event_log: JsonlAppender | None = None
        primary_error: BaseException | None = None

        try:
            manifest_file = AtomicJsonFile(normalized_dir / "session.json")
            discussion_file = AtomicJsonFile(normalized_dir / "discussion-state.json")
            if set(normalized_dir.iterdir()) != {claim.path}:
                raise FileExistsError(
                    "session directory changed while acquiring bootstrap ownership: "
                    f"{normalized_dir}"
                )
            manifest_file.replace(manifest.to_dict())
            microphone_sink = WavSink.open(normalized_dir / "mic.wav")
            loopback_sink = WavSink.open(normalized_dir / "loopback.wav")
            transcript_log = JsonlAppender.open(normalized_dir / "transcript.jsonl")
            state_history_log = JsonlAppender.open(
                normalized_dir / "state-history.jsonl"
            )
            event_log = JsonlAppender.open(normalized_dir / "events.jsonl")
            discussion_file.replace(initial_state.to_dict())
            microphone_sink.sync()
            loopback_sink.sync()
            cls._publish_empty_wav_header(microphone_sink.path)
            cls._publish_empty_wav_header(loopback_sink.path)
            transcript_log.sync()
            state_history_log.sync()
            event_log.sync()
            writer = cls(
                normalized_dir,
                manifest,
                initial_state,
                microphone_sink,
                loopback_sink,
                transcript_log,
                state_history_log,
                event_log,
                owner_pid=owner_pid,
                sync_interval_seconds=sync_interval_seconds,
                opened_monotonic=opened_monotonic,
            )
        except BaseException as error:
            primary_error = error
            cls._close_bootstrap_resources(
                microphone_sink,
                loopback_sink,
                transcript_log,
                state_history_log,
                event_log,
                primary_error,
            )
            raise
        finally:
            cleanup_error = claim.release(primary_error)
            if primary_error is None and cleanup_error is not None:
                cls._close_bootstrap_resources(
                    microphone_sink,
                    loopback_sink,
                    transcript_log,
                    state_history_log,
                    event_log,
                    cleanup_error,
                )
                raise cleanup_error
        return writer

    def append_audio(self, command: AudioWriteCommand) -> None:
        """Append one contiguous PCM command to its source-specific WAV."""

        self._ensure_mutable()
        self._validate_audio_command(command)
        expected_start = self._source_cursors[command.source]
        if command.source_start_sample != expected_start:
            raise PersistenceInvariantError(
                f"expected source_start_sample {expected_start} for "
                f"{command.source.value}, got {command.source_start_sample}"
            )
        sink = self._sink_for(command.source)
        try:
            persisted_start = sink.append(command.pcm_s16le)
        except BaseException as primary_error:
            self._fail_closed(primary_error)
            raise
        if persisted_start != expected_start:
            cursor_error = RuntimeError(
                f"WAV sink cursor mismatch for {command.source.value}: "
                f"expected {expected_start}, got {persisted_start}"
            )
            self._fail_closed(cursor_error)
            raise cursor_error
        self._source_cursors[command.source] = command.source_end_sample

    def append_transcript(self, record: TranscriptRecord) -> None:
        """Append one committed transcript in merged chronological order."""

        self._ensure_mutable()
        record = self._canonical_transcript_record(record)
        if record.sequence != self._next_transcript_sequence:
            raise PersistenceInvariantError(
                f"expected transcript sequence {self._next_transcript_sequence}, "
                f"got {record.sequence}"
            )
        if record.segment_id in self._transcript_segment_ids:
            raise PersistenceInvariantError(
                f"duplicate transcript segment_id {record.segment_id}"
            )
        if record.source not in self._source_cursors:
            raise PersistenceInvariantError("transcript source must be ME or OTHERS")
        persisted_cursor = self._source_cursors[record.source]
        if record.source_end_sample > persisted_cursor:
            raise PersistenceInvariantError(
                f"transcript source_end_sample {record.source_end_sample} exceeds "
                f"persisted audio cursor {persisted_cursor} for {record.source.value}"
            )
        order = (record.session_start_ms, _SOURCE_RANK[record.source])
        if self._last_transcript_order is not None:
            if order < self._last_transcript_order:
                raise PersistenceInvariantError(
                    "transcript records must remain in chronological order"
                )
        try:
            self._transcript_log.append(record.to_dict())
        except BaseException as primary_error:
            self._fail_closed(primary_error)
            raise
        self._next_transcript_sequence += 1
        self._transcript_segment_ids.add(record.segment_id)
        self._last_transcript_order = order

    def replace_discussion_state(
        self,
        previous_revision: int,
        state: DiscussionState,
    ) -> None:
        """Durably append history before atomically publishing its live snapshot."""

        self._ensure_mutable()
        if not isinstance(previous_revision, int) or isinstance(
            previous_revision, bool
        ):
            raise PersistenceInvariantError("previous_revision must be an integer")
        state = self._canonical_discussion_state(state, "discussion state")
        if previous_revision != self._discussion_revision:
            raise PersistenceInvariantError(
                f"expected previous_revision {self._discussion_revision}, "
                f"got {previous_revision}"
            )
        expected_revision = previous_revision + 1
        if state.revision != expected_revision:
            raise PersistenceInvariantError(
                f"expected discussion revision {expected_revision}, "
                f"got {state.revision}"
            )
        if state.mode is not self._manifest.mode:
            raise PersistenceInvariantError(
                "discussion state mode must match the session manifest mode"
            )
        history = self._canonical_state_history_record(
            StateHistoryRecord(
                schema_version=1,
                session_id=self._manifest.session_id,
                previous_revision=previous_revision,
                new_revision=state.revision,
                state=state,
            )
        )
        try:
            self._state_history_log.append(history.to_dict())
        except BaseException as primary_error:
            self._fail_closed(primary_error)
            raise
        try:
            self._discussion_file.replace(state.to_dict())
        except BaseException as primary_error:
            self._fail_closed(primary_error)
            raise
        self._discussion_revision = state.revision

    def append_event(self, record: EventRecord) -> None:
        """Append one session event with contiguous identity and chronology."""

        self._ensure_mutable()
        record = self._canonical_event_record(record)
        if record.sequence != self._next_event_sequence:
            raise PersistenceInvariantError(
                f"expected event sequence {self._next_event_sequence}, "
                f"got {record.sequence}"
            )
        if record.session_id != self._manifest.session_id:
            raise PersistenceInvariantError(
                "event session_id must match the session manifest"
            )
        if self._last_event_session_time_ms is not None:
            if record.session_time_ms < self._last_event_session_time_ms:
                raise PersistenceInvariantError(
                    "event records must remain in chronological order"
                )
        try:
            self._event_log.append(record.to_dict())
        except BaseException as primary_error:
            self._fail_closed(primary_error)
            raise
        self._next_event_sequence += 1
        self._last_event_session_time_ms = record.session_time_ms

    def close_incomplete(self) -> None:
        """Synchronize and close all resources without changing the manifest."""

        self._ensure_owner()
        if self._state is not _WriterState.OPEN:
            return
        self._state = _WriterState.CLOSED
        primary_error = self._close_owned_resources()
        if primary_error is not None:
            self._state = _WriterState.FAILED
            raise primary_error

    @classmethod
    def _validate_open_inputs(
        cls,
        manifest: SessionManifest,
        initial_state: DiscussionState,
        sync_interval_seconds: float,
    ) -> None:
        if not isinstance(manifest, SessionManifest):
            raise PersistenceInvariantError("manifest must be a SessionManifest")
        if not isinstance(initial_state, DiscussionState):
            raise PersistenceInvariantError("initial_state must be a DiscussionState")
        if manifest.status is not SessionStatus.INCOMPLETE:
            raise PersistenceInvariantError(
                "session manifest status must be INCOMPLETE when opening"
            )
        if manifest.transcript_entry_count != 0:
            raise PersistenceInvariantError(
                "transcript_entry_count must be zero for new session artifacts"
            )
        if manifest.mode is not initial_state.mode:
            raise PersistenceInvariantError(
                "initial discussion mode must match the session manifest mode"
            )
        if manifest.final_discussion_state_revision != initial_state.revision:
            raise PersistenceInvariantError(
                "manifest final discussion revision must match the initial state "
                "revision"
            )
        if (
            isinstance(sync_interval_seconds, bool)
            or not isinstance(sync_interval_seconds, int | float)
            or not math.isfinite(sync_interval_seconds)
            or sync_interval_seconds <= 0
        ):
            raise PersistenceInvariantError(
                "sync_interval_seconds must be a positive finite number"
            )

    @staticmethod
    def _canonical_manifest(manifest: SessionManifest) -> SessionManifest:
        if not isinstance(manifest, SessionManifest):
            raise PersistenceInvariantError("manifest must be a SessionManifest")
        try:
            return SessionManifest.from_dict(manifest.to_dict())
        except Exception as error:
            raise PersistenceInvariantError(
                f"invalid session manifest: {error}"
            ) from error

    @staticmethod
    def _canonical_discussion_state(
        state: DiscussionState,
        record_name: str,
    ) -> DiscussionState:
        if not isinstance(state, DiscussionState):
            raise PersistenceInvariantError(f"{record_name} must be a DiscussionState")
        try:
            return DiscussionState.from_dict(state.to_dict())
        except Exception as error:
            raise PersistenceInvariantError(
                f"invalid {record_name}: {error}"
            ) from error

    @staticmethod
    def _canonical_transcript_record(record: TranscriptRecord) -> TranscriptRecord:
        if not isinstance(record, TranscriptRecord):
            raise PersistenceInvariantError("record must be a TranscriptRecord")
        try:
            return TranscriptRecord.from_dict(record.to_dict())
        except Exception as error:
            raise PersistenceInvariantError(
                f"invalid transcript record: {error}"
            ) from error

    @staticmethod
    def _canonical_event_record(record: EventRecord) -> EventRecord:
        if not isinstance(record, EventRecord):
            raise PersistenceInvariantError("record must be an EventRecord")
        try:
            return EventRecord.from_dict(record.to_dict())
        except Exception as error:
            raise PersistenceInvariantError(f"invalid event record: {error}") from error

    @staticmethod
    def _canonical_state_history_record(
        record: StateHistoryRecord,
    ) -> StateHistoryRecord:
        try:
            return StateHistoryRecord.from_dict(record.to_dict())
        except Exception as error:
            raise PersistenceInvariantError(
                f"invalid state history record: {error}"
            ) from error

    @staticmethod
    def _prepare_empty_session_directory(session_dir: Path) -> None:
        if session_dir.exists():
            if not session_dir.is_dir():
                raise FileExistsError(
                    f"session path already exists and is not a directory: {session_dir}"
                )
            if next(session_dir.iterdir(), None) is not None:
                raise FileExistsError(
                    f"session directory already exists and is non-empty: {session_dir}"
                )
            return
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if not session_dir.is_dir():
                raise
            if next(session_dir.iterdir(), None) is not None:
                raise FileExistsError(
                    f"session directory already exists and is non-empty: {session_dir}"
                ) from None

    @classmethod
    def _validate_storage_path(cls, session_dir: Path) -> None:
        """Reject immediate redirection; root containment belongs to the caller."""

        if cls._is_reparse_point(session_dir):
            raise PersistenceInvariantError(
                "session directory must not be a symbolic link or reparse point: "
                f"{session_dir}"
            )
        if cls._is_reparse_point(session_dir.parent):
            raise PersistenceInvariantError(
                "session directory parent must not be a symbolic link or reparse "
                f"point: {session_dir.parent}"
            )

    @staticmethod
    def _is_reparse_point(path: Path) -> bool:
        try:
            status = path.lstat()
        except FileNotFoundError:
            return False
        attributes = getattr(status, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)

    @staticmethod
    def _publish_empty_wav_header(path: Path) -> None:
        """Make a new zero-frame WAV readable while its append sink stays open."""

        with path.open("r+b", buffering=0) as file:
            file.seek(4)
            encoded = struct.pack("<I", 36)
            written = file.write(encoded)
            if written != len(encoded):
                raise OSError(f"short WAV header write for {path}")
            file.flush()
            os.fsync(file.fileno())

    @classmethod
    def _close_bootstrap_resources(
        cls,
        microphone_sink: WavSink | None,
        loopback_sink: WavSink | None,
        transcript_log: JsonlAppender | None,
        state_history_log: JsonlAppender | None,
        event_log: JsonlAppender | None,
        primary_error: BaseException,
    ) -> None:
        operations: list[tuple[str, Callable[[], None]]] = []
        if microphone_sink is not None:
            operations.append(("mic.wav", microphone_sink.close_incomplete))
        if loopback_sink is not None:
            operations.append(("loopback.wav", loopback_sink.close_incomplete))
        if transcript_log is not None:
            operations.append(("transcript.jsonl", transcript_log.close))
        if state_history_log is not None:
            operations.append(("state-history.jsonl", state_history_log.close))
        if event_log is not None:
            operations.append(("events.jsonl", event_log.close))
        cls._run_cleanup_operations(reversed(operations), primary_error)

    def _validate_audio_command(self, command: AudioWriteCommand) -> None:
        if not isinstance(command, AudioWriteCommand):
            raise PersistenceInvariantError("command must be an AudioWriteCommand")
        if not isinstance(command.source, AudioSource):
            raise PersistenceInvariantError("audio source must be ME or OTHERS")
        if not isinstance(command.pcm_s16le, bytes):
            raise PersistenceInvariantError("pcm_s16le must be bytes")
        if len(command.pcm_s16le) % 2 != 0:
            raise PersistenceInvariantError(
                "pcm_s16le must contain complete 16-bit samples"
            )
        self._require_non_negative_runtime_int(
            command.source_start_sample,
            "source_start_sample",
        )
        self._require_non_negative_runtime_int(
            command.source_end_sample,
            "source_end_sample",
        )
        self._require_non_negative_runtime_int(
            command.session_start_ms,
            "session_start_ms",
        )
        self._require_non_negative_runtime_int(
            command.captured_monotonic_ms,
            "captured_monotonic_ms",
        )
        sample_count = len(command.pcm_s16le) // 2
        if command.source_end_sample - command.source_start_sample != sample_count:
            raise PersistenceInvariantError(
                "source sample range must match the PCM sample count"
            )

    @staticmethod
    def _require_non_negative_runtime_int(value: object, field_name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PersistenceInvariantError(
                f"{field_name} must be a non-negative integer"
            )

    def _sink_for(self, source: AudioSource) -> WavSink:
        if source is AudioSource.ME:
            return self._microphone_sink
        return self._loopback_sink

    def _ensure_mutable(self) -> None:
        self._ensure_owner()
        if self._state is _WriterState.FAILED:
            raise RuntimeError("SessionWriter is failed and cannot mutate artifacts")
        if self._state is _WriterState.CLOSED:
            raise RuntimeError("SessionWriter is closed and cannot mutate artifacts")

    def _ensure_owner(self) -> None:
        actual_pid = os.getpid()
        if actual_pid != self.owner_pid:
            raise WriterOwnershipError(
                f"SessionWriter owner PID is {self.owner_pid}, current PID is "
                f"{actual_pid}"
            )

    def _fail_closed(self, primary_error: BaseException) -> None:
        self._state = _WriterState.FAILED
        cleanup_error = self._close_owned_resources(primary_error)
        if cleanup_error is not None and cleanup_error is not primary_error:
            primary_error.add_note(f"Session resource cleanup failed: {cleanup_error}")

    def _close_owned_resources(
        self,
        primary_error: BaseException | None = None,
    ) -> BaseException | None:
        if self._resources_closed:
            return primary_error
        self._resources_closed = True
        operations: list[tuple[str, Callable[[], None]]] = [
            ("events.jsonl", lambda: self._close_jsonl(self._event_log)),
            (
                "state-history.jsonl",
                lambda: self._close_jsonl(self._state_history_log),
            ),
            (
                "transcript.jsonl",
                lambda: self._close_jsonl(self._transcript_log),
            ),
            ("loopback.wav", self._loopback_sink.close_incomplete),
            ("mic.wav", self._microphone_sink.close_incomplete),
        ]
        return self._run_cleanup_operations(operations, primary_error)

    @staticmethod
    def _close_jsonl(appender: JsonlAppender) -> None:
        primary_error: BaseException | None = None
        try:
            appender.sync()
        except BaseException as error:
            primary_error = error
        try:
            appender.close()
        except BaseException as close_error:
            if primary_error is None:
                primary_error = close_error
            else:
                primary_error.add_note(
                    f"JSONL close failed for {appender.path}: {close_error}"
                )
        if primary_error is not None:
            raise primary_error

    @staticmethod
    def _run_cleanup_operations(
        operations: Iterable[tuple[str, Callable[[], None]]],
        primary_error: BaseException | None,
    ) -> BaseException | None:
        first_error = primary_error
        for name, operation in operations:
            try:
                operation()
            except BaseException as cleanup_error:
                if first_error is None:
                    first_error = cleanup_error
                else:
                    first_error.add_note(
                        f"Session resource cleanup failed for {name}: {cleanup_error}"
                    )
        return first_error
