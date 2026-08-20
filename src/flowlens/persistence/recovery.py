"""Read-only discovery and typed inspection for incomplete sessions."""

import json
import os
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from flowlens.domain.discussion import DiscussionState, StateHistoryRecord
from flowlens.domain.enums import AudioSource, SessionStatus
from flowlens.domain.messages import EventRecord, TranscriptRecord
from flowlens.domain.session import SessionManifest
from flowlens.persistence._recovery_artifacts import (
    ArtifactIdentity as _ArtifactIdentity,
)
from flowlens.persistence._recovery_artifacts import (
    DirectoryIdentity as _DirectoryIdentity,
)
from flowlens.persistence._recovery_artifacts import (
    OpenArtifact as _OpenArtifact,
)
from flowlens.persistence._recovery_artifacts import (
    RecoveryError as RecoveryError,
)
from flowlens.persistence._recovery_artifacts import (
    build_artifact_identity as _build_artifact_identity,
)
from flowlens.persistence._recovery_artifacts import (
    build_artifact_identity_from_digest as _build_artifact_identity_from_digest,
)
from flowlens.persistence._recovery_artifacts import (
    capture_directory_identity as _capture_directory_identity,
)
from flowlens.persistence._recovery_artifacts import (
    close_artifact as _close_artifact_impl,
)
from flowlens.persistence._recovery_artifacts import (
    is_reparse_point as _is_reparse_point,
)
from flowlens.persistence._recovery_artifacts import (
    open_guarded_artifact as _open_guarded_artifact,
)
from flowlens.persistence._recovery_artifacts import (
    read_open_artifact as _read_open_artifact,
)
from flowlens.persistence._recovery_artifacts import (
    require_opened_identity as _require_opened_identity,
)
from flowlens.persistence._recovery_artifacts import (
    require_safe_directory_status as _require_safe_directory_status,
)
from flowlens.persistence._recovery_artifacts import (
    verify_artifact_identity as _verify_artifact_identity,
)
from flowlens.persistence._recovery_artifacts import (
    verify_directory_identity as _verify_directory_identity,
)
from flowlens.persistence._recovery_artifacts import (
    with_verified_artifact as _run_with_verified_artifact,
)
from flowlens.persistence.json_files import (
    JsonlRepairPlan,
)
from flowlens.persistence.json_files import (
    _inspect_jsonl_tail_bytes as _inspect_jsonl_tail_bytes_impl,
)

_REQUIRED_ARTIFACT_NAMES = frozenset(
    {
        "session.json",
        "mic.wav",
        "loopback.wav",
        "transcript.jsonl",
        "discussion-state.json",
        "state-history.jsonl",
        "events.jsonl",
    }
)
_JSONL_ARTIFACT_NAMES = (
    "transcript.jsonl",
    "state-history.jsonl",
    "events.jsonl",
)
_WAV_ARTIFACT_NAMES = ("mic.wav", "loopback.wav")
_WAV_HEADER_SIZE = 44
_WAV_SAMPLE_WIDTH = 2
_WAV_SAMPLE_RATE = 16_000
_MAX_PCM_BYTES = ((0xFFFFFFFF - 36) // _WAV_SAMPLE_WIDTH) * _WAV_SAMPLE_WIDTH


def _close_artifact(opened: _OpenArtifact) -> None:
    """Close one artifact through a patchable recovery seam."""

    _close_artifact_impl(opened)


def _inspect_jsonl_tail_bytes(path: Path, encoded: bytes) -> JsonlRepairPlan:
    """Inspect held JSONL bytes through a patchable recovery seam."""

    return _inspect_jsonl_tail_bytes_impl(path, encoded)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Public summary shared by recovery inspection and mutation."""

    session_id: str
    session_dir: Path
    discarded_jsonl_tail_bytes: dict[str, int]
    repaired_wav_headers: tuple[str, ...]
    transcript_entry_count: int
    final_discussion_state_revision: int
    active_duration_ms: int


@dataclass(frozen=True, slots=True)
class _JsonArtifactPlan[ValueT]:
    """Typed JSON value tied to the exact file bytes that produced it."""

    identity: _ArtifactIdentity
    value: ValueT


@dataclass(frozen=True, slots=True)
class _WavRepairPlan:
    """Read-only WAV validation result guarded by file identity metadata."""

    path: Path
    identity: _ArtifactIdentity
    expected_size: int
    expected_sha256: str
    original_pcm_bytes: int
    valid_pcm_bytes: int
    header_changed: bool


@dataclass(frozen=True, slots=True)
class _RecoveryInspection:
    """Typed mutation plan produced without changing session artifacts."""

    manifest: SessionManifest
    manifest_plan: _JsonArtifactPlan[SessionManifest]
    snapshot_plan: _JsonArtifactPlan[DiscussionState]
    session_directory_identity: _DirectoryIdentity
    parent_directory_identity: _DirectoryIdentity
    artifact_identities: tuple[_ArtifactIdentity, ...]
    report: RecoveryReport
    jsonl_repair_plans: tuple[JsonlRepairPlan, ...]
    wav_repair_plans: tuple[_WavRepairPlan, ...]
    snapshot_replacement: DiscussionState | None
    transcript_records: tuple[TranscriptRecord, ...]
    state_history_records: tuple[StateHistoryRecord, ...]
    event_records: tuple[EventRecord, ...]
    transcript_entry_count: int
    final_discussion_state_revision: int
    active_duration_ms: int
    next_event_sequence: int


def _with_verified_artifact[ResultT](
    identity: _ArtifactIdentity,
    operation: Callable[[int], ResultT],
) -> ResultT:
    """Run a future mutation on its revalidated, no-follow descriptor."""

    return _run_with_verified_artifact(identity, operation)


def find_incomplete_sessions(sessions_root: Path) -> tuple[Path, ...]:
    """Return incomplete session directories in deterministic name order."""

    normalized_root = Path(sessions_root)
    try:
        root_status = normalized_root.lstat()
    except FileNotFoundError:
        return ()
    except OSError as error:
        raise RecoveryError(normalized_root, str(error)) from error
    _require_safe_directory_status(normalized_root, root_status)

    incomplete: list[Path] = []
    try:
        entries = sorted(normalized_root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise RecoveryError(normalized_root, str(error)) from error
    for entry in entries:
        if _is_reparse_point(entry):
            raise RecoveryError(entry, "symbolic links and reparse points are unsafe")
        try:
            if not entry.is_dir():
                continue
        except OSError as error:
            raise RecoveryError(entry, str(error)) from error
        manifest_path = entry / "session.json"
        if _is_reparse_point(manifest_path):
            raise RecoveryError(
                manifest_path,
                "symbolic links and reparse points are unsafe",
            )
        if not manifest_path.exists():
            continue
        opened = _open_guarded_artifact(manifest_path)
        try:
            encoded = _read_open_artifact(opened)
            identity = _build_artifact_identity(opened, encoded)
            _verify_artifact_identity(opened, identity)
            manifest = _load_manifest(manifest_path, encoded)
        finally:
            _close_artifact(opened)
        if manifest.status is SessionStatus.INCOMPLETE:
            incomplete.append(entry)
    return tuple(incomplete)


def inspect_incomplete_session(session_dir: Path) -> _RecoveryInspection:
    """Validate one incomplete session and return a read-only repair plan."""

    normalized_dir = Path(session_dir)
    session_directory_identity = _capture_directory_identity(normalized_dir)
    parent_directory_identity = _capture_directory_identity(normalized_dir.parent)
    opened_artifacts = _open_exact_artifacts(normalized_dir)
    artifacts = {name: opened.path for name, opened in opened_artifacts.items()}
    identities: dict[str, _ArtifactIdentity] = {}
    try:
        manifest_encoded = _read_open_artifact(opened_artifacts["session.json"])
        manifest_identity = _build_artifact_identity(
            opened_artifacts["session.json"], manifest_encoded
        )
        identities["session.json"] = manifest_identity
        manifest = _load_manifest(artifacts["session.json"], manifest_encoded)
        if manifest.status is not SessionStatus.INCOMPLETE:
            raise RecoveryError(
                artifacts["session.json"],
                "session status must be INCOMPLETE for recovery inspection",
            )

        jsonl_results = tuple(
            _inspect_open_jsonl(opened_artifacts[name])
            for name in _JSONL_ARTIFACT_NAMES
        )
        jsonl_plans = tuple(result[0] for result in jsonl_results)
        jsonl_encoded = {
            plan.path.name: encoded for plan, encoded, _identity in jsonl_results
        }
        identities.update(
            {plan.path.name: identity for plan, _encoded, identity in jsonl_results}
        )
        plans_by_name = {plan.path.name: plan for plan in jsonl_plans}
        transcript_records = _parse_jsonl_records(
            plans_by_name["transcript.jsonl"],
            jsonl_encoded["transcript.jsonl"],
            TranscriptRecord.from_dict,
        )
        state_history_records = _parse_jsonl_records(
            plans_by_name["state-history.jsonl"],
            jsonl_encoded["state-history.jsonl"],
            StateHistoryRecord.from_dict,
        )
        event_records = _parse_jsonl_records(
            plans_by_name["events.jsonl"],
            jsonl_encoded["events.jsonl"],
            EventRecord.from_dict,
        )

        wav_plans = tuple(
            _inspect_open_wav(opened_artifacts[name]) for name in _WAV_ARTIFACT_NAMES
        )
        identities.update({plan.path.name: plan.identity for plan in wav_plans})
        wav_sample_counts = {
            plan.path.name: plan.valid_pcm_bytes // _WAV_SAMPLE_WIDTH
            for plan in wav_plans
        }
        _validate_transcripts(transcript_records, wav_sample_counts, artifacts)
        _validate_state_history(state_history_records, manifest, artifacts)
        _validate_events(event_records, manifest, artifacts)

        snapshot_encoded = _read_open_artifact(
            opened_artifacts["discussion-state.json"]
        )
        snapshot_identity = _build_artifact_identity(
            opened_artifacts["discussion-state.json"], snapshot_encoded
        )
        identities["discussion-state.json"] = snapshot_identity
        snapshot = _load_discussion_state(
            artifacts["discussion-state.json"], snapshot_encoded
        )
        if snapshot.mode is not manifest.mode:
            raise RecoveryError(
                artifacts["discussion-state.json"],
                "discussion mode must match the session manifest mode",
            )
        final_state = (
            state_history_records[-1].state if state_history_records else snapshot
        )
        snapshot_replacement = _compare_snapshot_to_history(
            snapshot,
            final_state,
            bool(state_history_records),
            artifacts["discussion-state.json"],
        )

        artifact_identities = tuple(
            identities[name] for name in sorted(_REQUIRED_ARTIFACT_NAMES)
        )
        for identity in artifact_identities:
            _verify_artifact_identity(opened_artifacts[identity.path.name], identity)
        _verify_directory_identity(session_directory_identity)
        _verify_directory_identity(parent_directory_identity)

        discarded_tails = {
            plan.path.name: plan.discarded_tail_bytes
            for plan in jsonl_plans
            if plan.discarded_tail_bytes
        }
        repaired_wavs = tuple(
            sorted(plan.path.name for plan in wav_plans if plan.header_changed)
        )
        max_samples = max(wav_sample_counts.values(), default=0)
        report = RecoveryReport(
            session_id=manifest.session_id,
            session_dir=normalized_dir,
            discarded_jsonl_tail_bytes=discarded_tails,
            repaired_wav_headers=repaired_wavs,
            transcript_entry_count=len(transcript_records),
            final_discussion_state_revision=final_state.revision,
            active_duration_ms=max_samples * 1_000 // _WAV_SAMPLE_RATE,
        )
        return _RecoveryInspection(
            manifest=manifest,
            manifest_plan=_JsonArtifactPlan(manifest_identity, manifest),
            snapshot_plan=_JsonArtifactPlan(snapshot_identity, snapshot),
            session_directory_identity=session_directory_identity,
            parent_directory_identity=parent_directory_identity,
            artifact_identities=artifact_identities,
            report=report,
            jsonl_repair_plans=jsonl_plans,
            wav_repair_plans=wav_plans,
            snapshot_replacement=snapshot_replacement,
            transcript_records=transcript_records,
            state_history_records=state_history_records,
            event_records=event_records,
            transcript_entry_count=report.transcript_entry_count,
            final_discussion_state_revision=report.final_discussion_state_revision,
            active_duration_ms=report.active_duration_ms,
            next_event_sequence=len(event_records) + 1,
        )
    finally:
        primary_error = sys.exception()
        close_error = _close_open_artifacts(
            tuple(opened_artifacts.values()), primary_error
        )
        if primary_error is None and close_error is not None:
            raise close_error


def _open_exact_artifacts(session_dir: Path) -> dict[str, _OpenArtifact]:
    try:
        entries = tuple(session_dir.iterdir())
    except OSError as error:
        raise RecoveryError(session_dir, str(error)) from error
    actual_names = {path.name for path in entries}
    missing = sorted(_REQUIRED_ARTIFACT_NAMES - actual_names)
    if missing:
        path = session_dir / missing[0]
        raise RecoveryError(path, "required artifact is missing")
    unexpected = sorted(actual_names - _REQUIRED_ARTIFACT_NAMES)
    if unexpected:
        path = session_dir / unexpected[0]
        raise RecoveryError(path, "unexpected session artifact")

    paths = {path.name: path for path in entries}
    opened: dict[str, _OpenArtifact] = {}
    try:
        for name in sorted(_REQUIRED_ARTIFACT_NAMES):
            opened[name] = _open_guarded_artifact(paths[name])
    except BaseException as primary_error:
        _close_open_artifacts(tuple(opened.values()), primary_error)
        raise
    return opened


def _close_open_artifacts(
    opened_artifacts: tuple[_OpenArtifact, ...],
    primary_error: BaseException | None,
) -> BaseException | None:
    """Attempt every close while retaining the primary failure."""

    surfaced_error = primary_error
    for opened in opened_artifacts:
        try:
            _close_artifact(opened)
        except BaseException as close_error:
            if surfaced_error is None:
                surfaced_error = close_error
            else:
                surfaced_error.add_note(
                    f"Artifact cleanup failed for {opened.path}: {close_error}"
                )
    return surfaced_error


def _load_manifest(path: Path, encoded: bytes) -> SessionManifest:
    value = _load_json_object(path, encoded)
    try:
        return SessionManifest.from_dict(value)
    except Exception as error:
        raise RecoveryError(path, str(error)) from error


def _load_discussion_state(path: Path, encoded: bytes) -> DiscussionState:
    value = _load_json_object(path, encoded)
    try:
        return DiscussionState.from_dict(value)
    except Exception as error:
        raise RecoveryError(path, str(error)) from error


def _load_json_object(path: Path, encoded: bytes) -> dict[str, object]:
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise RecoveryError(path, "UTF-8 BOM is not permitted")
    try:
        decoded = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecoveryError(path, f"invalid UTF-8: {error}") from error
    try:
        value = json.loads(decoded, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RecoveryError(path, f"invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(path, "JSON document must be an object")
    return cast(dict[str, object], value)


def _inspect_open_jsonl(
    opened: _OpenArtifact,
) -> tuple[JsonlRepairPlan, bytes, _ArtifactIdentity]:
    encoded = _read_open_artifact(opened)
    identity = _build_artifact_identity(opened, encoded)
    try:
        plan = _inspect_jsonl_tail_bytes(opened.path, encoded)
    except Exception as error:
        raise RecoveryError(opened.path, str(error)) from error
    return plan, encoded, identity


def _parse_jsonl_records[RecordT](
    plan: JsonlRepairPlan,
    encoded: bytes,
    parser: Callable[[object], RecordT],
) -> tuple[RecordT, ...]:
    if (
        len(encoded) != plan.expected_size
        or sha256(encoded).hexdigest() != plan.expected_sha256
    ):
        raise RecoveryError(plan.path, "JSONL file changed during inspection")
    retained_size = plan.expected_size - plan.discarded_tail_bytes
    retained = encoded[:retained_size]
    records: list[RecordT] = []
    for line_number, line in enumerate(retained.splitlines(), start=1):
        try:
            decoded = line.decode("utf-8")
            value = json.loads(decoded, parse_constant=_reject_json_constant)
            record = parser(value)
        except Exception as error:
            raise RecoveryError(plan.path, f"line {line_number}: {error}") from error
        records.append(record)
    if len(records) != plan.valid_record_count:
        raise RecoveryError(plan.path, "JSONL record count changed during inspection")
    return tuple(records)


def _inspect_open_wav(opened: _OpenArtifact) -> _WavRepairPlan:
    path = opened.path
    try:
        expected_size = _wav_file_size(opened)
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    if expected_size < _WAV_HEADER_SIZE:
        raise RecoveryError(path, "WAV file must contain a complete 44-byte header")
    original_pcm_bytes = expected_size - _WAV_HEADER_SIZE
    valid_pcm_bytes = (original_pcm_bytes // _WAV_SAMPLE_WIDTH) * _WAV_SAMPLE_WIDTH
    if valid_pcm_bytes > _MAX_PCM_BYTES:
        raise RecoveryError(path, "PCM payload exceeds the canonical RIFF size limit")

    try:
        opened_status = os.fstat(opened.descriptor)
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    _require_opened_identity(opened, opened_status)
    if opened_status.st_size != expected_size:
        raise RecoveryError(path, "WAV path identity or size changed during inspection")
    try:
        os.lseek(opened.descriptor, 0, os.SEEK_SET)
        header_parts: list[bytes] = []
        remaining_header = _WAV_HEADER_SIZE
        while remaining_header:
            chunk = os.read(opened.descriptor, remaining_header)
            if not chunk:
                break
            header_parts.append(chunk)
            remaining_header -= len(chunk)
        header = b"".join(header_parts)
        digest = sha256(header)
        total_size = len(header)
        while chunk := os.read(opened.descriptor, 1024 * 1024):
            digest.update(chunk)
            total_size += len(chunk)
    except OSError as error:
        raise RecoveryError(path, str(error)) from error
    if total_size != expected_size:
        raise RecoveryError(path, "WAV file changed during inspection")
    try:
        _validate_canonical_wav_header(header)
    except ValueError as error:
        raise RecoveryError(path, str(error)) from error
    expected_riff_size = 36 + valid_pcm_bytes
    original_riff_size = struct.unpack_from("<I", header, 4)[0]
    original_data_size = struct.unpack_from("<I", header, 40)[0]
    identity = _build_artifact_identity_from_digest(
        opened,
        total_size,
        digest.hexdigest(),
    )
    return _WavRepairPlan(
        path=path,
        identity=identity,
        expected_size=expected_size,
        expected_sha256=digest.hexdigest(),
        original_pcm_bytes=original_pcm_bytes,
        valid_pcm_bytes=valid_pcm_bytes,
        header_changed=(
            original_riff_size != expected_riff_size
            or original_data_size != valid_pcm_bytes
        ),
    )


def _wav_file_size(opened: _OpenArtifact) -> int:
    return os.fstat(opened.descriptor).st_size


def _validate_canonical_wav_header(header: bytes) -> None:
    checks = (
        (header[0:4] == b"RIFF", "WAV header must start with RIFF"),
        (header[8:12] == b"WAVE", "WAV header must contain the WAVE marker"),
        (header[12:16] == b"fmt ", "WAV header must contain the fmt marker"),
        (
            struct.unpack_from("<I", header, 16)[0] == 16,
            "WAV format chunk size must be 16",
        ),
        (struct.unpack_from("<H", header, 20)[0] == 1, "WAV must use PCM format 1"),
        (struct.unpack_from("<H", header, 22)[0] == 1, "WAV must be mono"),
        (
            struct.unpack_from("<I", header, 24)[0] == 16_000,
            "WAV sample rate must be 16,000 Hz",
        ),
        (
            struct.unpack_from("<I", header, 28)[0] == 32_000,
            "WAV byte rate must be 32,000",
        ),
        (struct.unpack_from("<H", header, 32)[0] == 2, "WAV block alignment must be 2"),
        (struct.unpack_from("<H", header, 34)[0] == 16, "WAV samples must be 16-bit"),
        (header[36:40] == b"data", "WAV header must contain the data marker"),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)


def _validate_transcripts(
    records: tuple[TranscriptRecord, ...],
    wav_sample_counts: dict[str, int],
    artifacts: dict[str, Path],
) -> None:
    expected_sequence = 1
    segment_ids: set[str] = set()
    last_order: tuple[int, int] | None = None
    source_ranks = {AudioSource.ME: 0, AudioSource.OTHERS: 1}
    source_wavs = {AudioSource.ME: "mic.wav", AudioSource.OTHERS: "loopback.wav"}
    for line_number, record in enumerate(records, start=1):
        if record.sequence != expected_sequence:
            raise RecoveryError(
                artifacts["transcript.jsonl"],
                f"line {line_number}: expected sequence {expected_sequence}, "
                f"got {record.sequence}",
            )
        if record.segment_id in segment_ids:
            raise RecoveryError(
                artifacts["transcript.jsonl"],
                f"line {line_number}: duplicate segment_id {record.segment_id}",
            )
        wav_name = source_wavs[record.source]
        if record.source_end_sample > wav_sample_counts[wav_name]:
            raise RecoveryError(
                artifacts["transcript.jsonl"],
                f"line {line_number}: source_end_sample exceeds retained audio",
            )
        order = (record.session_start_ms, source_ranks[record.source])
        if last_order is not None and order < last_order:
            raise RecoveryError(
                artifacts["transcript.jsonl"],
                f"line {line_number}: transcript order is not chronological",
            )
        expected_sequence += 1
        segment_ids.add(record.segment_id)
        last_order = order


def _validate_state_history(
    records: tuple[StateHistoryRecord, ...],
    manifest: SessionManifest,
    artifacts: dict[str, Path],
) -> None:
    expected_previous_revision = 0
    for line_number, record in enumerate(records, start=1):
        if record.previous_revision != expected_previous_revision:
            raise RecoveryError(
                artifacts["state-history.jsonl"],
                f"line {line_number}: expected previous_revision "
                f"{expected_previous_revision}, got {record.previous_revision}",
            )
        if record.session_id != manifest.session_id:
            raise RecoveryError(
                artifacts["state-history.jsonl"],
                f"line {line_number}: session_id must match the manifest",
            )
        if record.state.mode is not manifest.mode:
            raise RecoveryError(
                artifacts["state-history.jsonl"],
                f"line {line_number}: discussion mode must match the manifest",
            )
        expected_previous_revision = record.new_revision


def _validate_events(
    records: tuple[EventRecord, ...],
    manifest: SessionManifest,
    artifacts: dict[str, Path],
) -> None:
    expected_sequence = 1
    last_session_time_ms: int | None = None
    for line_number, record in enumerate(records, start=1):
        if record.sequence != expected_sequence:
            raise RecoveryError(
                artifacts["events.jsonl"],
                f"line {line_number}: expected sequence {expected_sequence}, "
                f"got {record.sequence}",
            )
        if record.session_id != manifest.session_id:
            raise RecoveryError(
                artifacts["events.jsonl"],
                f"line {line_number}: session_id must match the manifest",
            )
        if (
            last_session_time_ms is not None
            and record.session_time_ms < last_session_time_ms
        ):
            raise RecoveryError(
                artifacts["events.jsonl"],
                f"line {line_number}: event order is not chronological",
            )
        expected_sequence += 1
        last_session_time_ms = record.session_time_ms


def _compare_snapshot_to_history(
    snapshot: DiscussionState,
    final_state: DiscussionState,
    has_history: bool,
    snapshot_path: Path,
) -> DiscussionState | None:
    if not has_history:
        if snapshot.revision != 0:
            raise RecoveryError(
                snapshot_path,
                "snapshot revision is newer than the empty state history",
            )
        return None
    if snapshot.revision > final_state.revision:
        raise RecoveryError(
            snapshot_path,
            "snapshot revision is newer than state history",
        )
    if snapshot.revision < final_state.revision:
        return final_state
    if snapshot != final_state:
        raise RecoveryError(snapshot_path, "snapshot differs from final state history")
    return None


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")
