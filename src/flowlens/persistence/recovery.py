"""Read-only discovery and typed inspection for incomplete sessions."""

import json
import os
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import cast

from flowlens.domain.discussion import DiscussionState, StateHistoryRecord
from flowlens.domain.enums import (
    AudioSource,
    EventType,
    ProcessSource,
    SessionStatus,
)
from flowlens.domain.messages import EventRecord, JsonValue, TranscriptRecord
from flowlens.domain.session import PauseInterval, SessionManifest
from flowlens.persistence._recovery_artifacts import (
    ArtifactIdentity as _ArtifactIdentity,
)
from flowlens.persistence._recovery_artifacts import (
    DirectoryAnchor as _DirectoryAnchor,
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
    close_directory_anchor as _close_directory_anchor,
)
from flowlens.persistence._recovery_artifacts import (
    is_reparse_point as _is_reparse_point,
)
from flowlens.persistence._recovery_artifacts import (
    open_directory_anchor as _open_directory_anchor,
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
    verify_opened_path_identity as _verify_opened_path_identity,
)
from flowlens.persistence._recovery_artifacts import (
    with_verified_artifact as _run_with_verified_artifact,
)
from flowlens.persistence.json_files import (
    AtomicJsonFile,
    JsonlRepairPlan,
    encode_jsonl_record,
)
from flowlens.persistence.json_files import (
    _inspect_jsonl_tail_bytes as _inspect_jsonl_tail_bytes_impl,
)
from flowlens.persistence.recovery_contract import (
    RecoveryPauseContractError,
    reconstruct_recovered_pause_intervals,
    recovered_terminal_time_ms,
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
    pcm_sha256: str
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


def recover_incomplete_sessions(
    sessions_root: Path,
    recovered_at: datetime,
) -> tuple[RecoveryReport, ...]:
    """Recover every incomplete session in deterministic filename order."""

    _require_aware_recovery_time(recovered_at)
    return tuple(
        recover_incomplete_session(session_dir, recovered_at)
        for session_dir in find_incomplete_sessions(sessions_root)
    )


def recover_incomplete_session(
    session_dir: Path,
    recovered_at: datetime,
) -> RecoveryReport:
    """Repair and durably finalize one inspected incomplete session."""

    _require_aware_recovery_time(recovered_at)
    inspection = inspect_incomplete_session(session_dir)
    if recovered_at < inspection.manifest.started_at:
        raise ValueError("recovered_at must not precede the session start")
    return _run_recovery_transaction(inspection, recovered_at)


def _require_aware_recovery_time(recovered_at: datetime) -> None:
    if not isinstance(recovered_at, datetime):
        raise TypeError("recovered_at must be a datetime")
    try:
        offset = recovered_at.utcoffset()
    except (OverflowError, ValueError) as error:
        raise ValueError("recovered_at timezone is invalid") from error
    if recovered_at.tzinfo is None or offset is None:
        raise ValueError("recovered_at must include a timezone")


def _run_recovery_transaction(
    inspection: _RecoveryInspection,
    recovered_at: datetime,
) -> RecoveryReport:
    session_anchor: _DirectoryAnchor | None = None
    parent_anchor: _DirectoryAnchor | None = None
    opened: dict[str, _OpenArtifact] = {}
    primary_error: BaseException | None = None
    result: RecoveryReport | None = None
    try:
        parent_anchor = _open_directory_anchor(inspection.parent_directory_identity)
        session_anchor = _open_directory_anchor(inspection.session_directory_identity)
        opened = _open_transaction_artifacts(inspection)
        result = _apply_recovery_transaction(
            inspection,
            recovered_at,
            session_anchor,
            opened,
        )
    except BaseException as error:
        primary_error = error
    finally:
        primary_error = _run_transaction_post_validation(
            inspection,
            session_anchor,
            parent_anchor,
            primary_error,
        )
        primary_error = _cleanup_recovery_transaction(
            tuple(opened.values()),
            session_anchor,
            parent_anchor,
            primary_error,
        )
    if primary_error is not None:
        raise primary_error
    if result is None:
        raise RuntimeError("recovery transaction produced no result")
    return result


def _open_transaction_artifacts(
    inspection: _RecoveryInspection,
) -> dict[str, _OpenArtifact]:
    opened: dict[str, _OpenArtifact] = {}
    try:
        for identity in inspection.artifact_identities:
            artifact = _open_guarded_artifact(identity.path, writable=True)
            opened[identity.path.name] = artifact
            _verify_artifact_identity(artifact, identity)
    except BaseException as error:
        _close_open_artifacts(tuple(opened.values()), error)
        raise
    return opened


def _apply_recovery_transaction(
    inspection: _RecoveryInspection,
    recovered_at: datetime,
    session_anchor: _DirectoryAnchor,
    opened: dict[str, _OpenArtifact],
) -> RecoveryReport:
    existing_recovery = _existing_durable_recovery_event(inspection)
    base_events = tuple(
        event
        for event in inspection.event_records
        if event.event_type is not EventType.SESSION_RECOVERED
    )
    pause_intervals, pause_notes = _reconstruct_pause_intervals(
        base_events,
        inspection.active_duration_ms,
        inspection.manifest_plan.identity.path,
    )
    if existing_recovery is None:
        recovery_event = EventRecord(
            schema_version=1,
            session_id=inspection.manifest.session_id,
            sequence=len(base_events) + 1,
            event_type=EventType.SESSION_RECOVERED,
            source=ProcessSource.GUI,
            session_time_ms=recovered_terminal_time_ms(
                base_events,
                inspection.active_duration_ms,
            ),
            created_at=recovered_at,
            details=_recovery_event_details(inspection.report),
        )
        event_time = recovered_at
        report = inspection.report
    else:
        recovery_event = existing_recovery
        event_time = recovery_event.created_at
        report = _report_from_recovery_event(inspection, recovery_event)
        _validate_recovery_intent(inspection, report)

    notes = _build_recovery_notes(inspection, report, pause_notes)
    recovered_manifest = replace(
        inspection.manifest,
        status=SessionStatus.RECOVERED,
        ended_at=event_time,
        active_duration_ms=inspection.active_duration_ms,
        pause_intervals=pause_intervals,
        transcript_entry_count=inspection.transcript_entry_count,
        final_discussion_state_revision=(inspection.final_discussion_state_revision),
        recovery_notes=inspection.manifest.recovery_notes + notes,
    )

    if existing_recovery is None:
        events_plan = next(
            plan
            for plan in inspection.jsonl_repair_plans
            if plan.path.name == "events.jsonl"
        )
        _publish_repaired_event_intent(
            session_anchor,
            opened,
            events_plan,
            recovery_event,
        )
    for plan in inspection.jsonl_repair_plans:
        if existing_recovery is None and plan.path.name == "events.jsonl":
            continue
        _apply_jsonl_plan_same_handle(opened[plan.path.name], plan)
        _verify_opened_path_identity(opened[plan.path.name])
    for wav_plan in inspection.wav_repair_plans:
        _apply_wav_plan_same_handle(opened[wav_plan.path.name], wav_plan)
        _verify_opened_path_identity(opened[wav_plan.path.name])
    if inspection.snapshot_replacement is not None:
        _release_atomic_target(opened, "discussion-state.json")
        _replace_json_anchored(
            session_anchor,
            inspection.snapshot_plan.identity.path,
            inspection.snapshot_replacement.to_dict(),
            inspection.snapshot_plan.identity,
        )
    transcripts, state_history, events = _reparse_repaired_records(
        inspection,
        opened,
    )
    if (
        len(transcripts) != inspection.transcript_entry_count
        or (
            state_history[-1].state.revision
            if state_history
            else inspection.snapshot_plan.value.revision
        )
        != inspection.final_discussion_state_revision
        or not events
        or events[-1] != recovery_event
    ):
        raise RecoveryError(
            inspection.manifest_plan.identity.path,
            "retained records changed after repair",
        )
    _post_validate_repaired_content(
        inspection,
        recovered_manifest,
        recovery_event,
    )
    _release_atomic_target(opened, "session.json")
    _replace_json_anchored(
        session_anchor,
        inspection.manifest_plan.identity.path,
        recovered_manifest.to_dict(),
        inspection.manifest_plan.identity,
    )
    _post_validate_recovered_manifest(inspection, recovered_manifest)
    return replace(
        report,
        transcript_entry_count=inspection.transcript_entry_count,
        final_discussion_state_revision=(inspection.final_discussion_state_revision),
        active_duration_ms=inspection.active_duration_ms,
    )


def _post_validate_repaired_content(
    inspection: _RecoveryInspection,
    recovered_manifest: SessionManifest,
    recovery_event: EventRecord,
) -> None:
    opened = _open_exact_artifacts(inspection.report.session_dir)
    primary_error: BaseException | None = None
    try:
        manifest_bytes = _read_open_artifact(opened["session.json"])
        if _load_manifest(opened["session.json"].path, manifest_bytes) != (
            inspection.manifest
        ):
            raise RecoveryError(
                opened["session.json"].path,
                "incomplete manifest changed before final replacement",
            )
        snapshot_bytes = _read_open_artifact(opened["discussion-state.json"])
        snapshot = _load_discussion_state(
            opened["discussion-state.json"].path,
            snapshot_bytes,
        )
        expected_snapshot = (
            inspection.snapshot_replacement or inspection.snapshot_plan.value
        )
        if snapshot != expected_snapshot:
            raise RecoveryError(
                opened["discussion-state.json"].path,
                "recovered discussion snapshot differs from retained history",
            )

        parsed: dict[str, tuple[object, ...]] = {}
        parsers: tuple[tuple[str, Callable[[object], object]], ...] = (
            ("transcript.jsonl", TranscriptRecord.from_dict),
            ("state-history.jsonl", StateHistoryRecord.from_dict),
            ("events.jsonl", EventRecord.from_dict),
        )
        for name, parser in parsers:
            encoded = _read_open_artifact(opened[name])
            plan = _inspect_jsonl_tail_bytes_impl(opened[name].path, encoded)
            if plan.discarded_tail_bytes or plan.append_final_lf:
                raise RecoveryError(
                    opened[name].path, "recovered JSONL tail is invalid"
                )
            parsed[name] = _parse_jsonl_records(plan, encoded, parser)
        if len(parsed["transcript.jsonl"]) != recovered_manifest.transcript_entry_count:
            raise RecoveryError(
                opened["transcript.jsonl"].path,
                "recovered transcript count differs from the manifest",
            )
        events = cast(tuple[EventRecord, ...], parsed["events.jsonl"])
        if not events or events[-1] != recovery_event:
            raise RecoveryError(
                opened["events.jsonl"].path,
                "recovery intent is not the final durable event",
            )
        wav_samples: dict[str, int] = {}
        for expected_wav in inspection.wav_repair_plans:
            current_wav = _inspect_open_wav(opened[expected_wav.path.name])
            if (
                current_wav.header_changed
                or current_wav.valid_pcm_bytes != expected_wav.valid_pcm_bytes
                or current_wav.pcm_sha256 != expected_wav.pcm_sha256
            ):
                raise RecoveryError(
                    current_wav.path,
                    "recovered WAV content differs from the inspected PCM payload",
                )
            wav_samples[current_wav.path.name] = (
                current_wav.valid_pcm_bytes // _WAV_SAMPLE_WIDTH
            )
        artifacts = {name: artifact.path for name, artifact in opened.items()}
        transcripts = cast(tuple[TranscriptRecord, ...], parsed["transcript.jsonl"])
        state_history = cast(
            tuple[StateHistoryRecord, ...], parsed["state-history.jsonl"]
        )
        base_events = tuple(
            event
            for event in events
            if event.event_type is not EventType.SESSION_RECOVERED
        )
        expected_base_events = tuple(
            event
            for event in inspection.event_records
            if event.event_type is not EventType.SESSION_RECOVERED
        )
        if transcripts != inspection.transcript_records:
            raise RecoveryError(
                opened["transcript.jsonl"].path,
                "retained transcript records changed after inspection",
            )
        if state_history != inspection.state_history_records:
            raise RecoveryError(
                opened["state-history.jsonl"].path,
                "retained state history changed after inspection",
            )
        if base_events != expected_base_events:
            raise RecoveryError(
                opened["events.jsonl"].path,
                "retained base events changed after inspection",
            )
        _validate_transcripts(transcripts, wav_samples, artifacts)
        _validate_state_history(state_history, recovered_manifest, artifacts)
        _validate_events(events, recovered_manifest, artifacts)
        final_revision = (
            state_history[-1].state.revision if state_history else snapshot.revision
        )
        if final_revision != recovered_manifest.final_discussion_state_revision:
            raise RecoveryError(
                opened["state-history.jsonl"].path,
                "recovered final revision differs from the manifest",
            )
        pauses, _notes = _reconstruct_pause_intervals(
            base_events,
            recovered_manifest.active_duration_ms,
            opened["events.jsonl"].path,
        )
        if pauses != recovered_manifest.pause_intervals:
            raise RecoveryError(
                opened["events.jsonl"].path,
                "reconstructed pauses differ from the recovered manifest",
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = _close_open_artifacts(tuple(opened.values()), primary_error)
        if primary_error is None and close_error is not None:
            raise close_error


def _post_validate_recovered_manifest(
    inspection: _RecoveryInspection,
    recovered_manifest: SessionManifest,
) -> None:
    path = inspection.manifest_plan.identity.path
    opened = _open_guarded_artifact(path)
    primary_error: BaseException | None = None
    try:
        if _load_manifest(path, _read_open_artifact(opened)) != recovered_manifest:
            raise RecoveryError(
                path,
                "recovered manifest content differs from the preflight document",
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _close_artifact(opened)
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Recovered manifest validation cleanup failed for {path}: "
                f"{close_error}"
            )


def _validate_recovery_intent(
    inspection: _RecoveryInspection,
    intent_report: RecoveryReport,
) -> None:
    current_tails = {
        plan.path.name: plan.discarded_tail_bytes
        for plan in inspection.jsonl_repair_plans
        if plan.discarded_tail_bytes
    }
    for name, count in current_tails.items():
        if intent_report.discarded_jsonl_tail_bytes.get(name) != count:
            raise RecoveryError(
                inspection.manifest_plan.identity.path,
                "durable recovery intent does not match pending JSONL repair",
            )
    pending_wavs = {
        plan.path.name for plan in inspection.wav_repair_plans if plan.header_changed
    }
    if not pending_wavs.issubset(set(intent_report.repaired_wav_headers)):
        raise RecoveryError(
            inspection.manifest_plan.identity.path,
            "durable recovery intent does not match pending WAV repair",
        )


def _release_atomic_target(
    opened: dict[str, _OpenArtifact],
    name: str,
) -> None:
    """Release a verified Windows replacement target before atomic publication."""

    artifact = opened[name]
    _close_artifact(artifact)
    del opened[name]


def _publish_repaired_event_intent(
    anchor: _DirectoryAnchor,
    opened: dict[str, _OpenArtifact],
    plan: JsonlRepairPlan,
    event: EventRecord,
) -> None:
    event_bytes = encode_jsonl_record(event.to_dict())
    original = _read_open_artifact(opened["events.jsonl"])
    retained = original[: plan.expected_size - plan.discarded_tail_bytes]
    if plan.append_final_lf:
        retained += b"\n"
    encoded = retained + event_bytes
    identity = _build_artifact_identity(opened["events.jsonl"], original)
    _release_atomic_target(opened, "events.jsonl")
    _replace_bytes_anchored(anchor, plan.path, encoded, identity)
    opened["events.jsonl"] = _open_guarded_artifact(plan.path, writable=True)


def _replace_bytes_anchored(
    anchor: _DirectoryAnchor,
    path: Path,
    encoded: bytes,
    expected_identity: _ArtifactIdentity,
) -> None:
    _verify_directory_identity(anchor.identity)
    temp_path: Path | None = None
    primary_error: BaseException | None = None
    try:
        temp_path, file = anchor.create_binary_temp(path)
        file_error: BaseException | None = None
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = file.write(remaining)
                if written is None or written <= 0:
                    raise OSError("binary temporary write made no progress")
                remaining = remaining[written:]
            file.flush()
            os.fsync(file.fileno())
        except BaseException as error:
            file_error = error
            raise
        finally:
            try:
                file.close()
            except BaseException as close_error:
                if file_error is None:
                    raise
                file_error.add_note(
                    f"Binary temporary cleanup failed for {temp_path}: {close_error}"
                )
        anchor.replace(temp_path, path, expected_identity)
        temp_path = None
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if temp_path is not None:
            try:
                anchor.remove(temp_path)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Binary temporary cleanup failed for {temp_path}: "
                    f"{cleanup_error}"
                )


def _apply_jsonl_plan_same_handle(
    opened: _OpenArtifact,
    plan: JsonlRepairPlan,
) -> None:
    if not plan.discarded_tail_bytes and not plan.append_final_lf:
        return
    if plan.discarded_tail_bytes:
        os.ftruncate(opened.descriptor, plan.expected_size - plan.discarded_tail_bytes)
    else:
        os.lseek(opened.descriptor, 0, os.SEEK_END)
        _write_descriptor_all(opened.descriptor, b"\n")
    os.fsync(opened.descriptor)


def _apply_wav_plan_same_handle(
    opened: _OpenArtifact,
    plan: _WavRepairPlan,
) -> None:
    if not plan.header_changed:
        return
    os.lseek(opened.descriptor, 4, os.SEEK_SET)
    _write_descriptor_all(
        opened.descriptor, struct.pack("<I", 36 + plan.valid_pcm_bytes)
    )
    os.lseek(opened.descriptor, 40, os.SEEK_SET)
    _write_descriptor_all(opened.descriptor, struct.pack("<I", plan.valid_pcm_bytes))
    os.fsync(opened.descriptor)


def _write_descriptor_all(descriptor: int, encoded: bytes) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("file write made no progress")
        remaining = remaining[written:]


def _replace_json_anchored(
    anchor: _DirectoryAnchor,
    path: Path,
    value: object,
    expected_identity: _ArtifactIdentity,
) -> None:
    _verify_directory_identity(anchor.identity)

    def replace_if_unchanged(source: Path, target: Path) -> None:
        anchor.replace(source, target, expected_identity)

    atomic = AtomicJsonFile(
        path,
        _create_temp=anchor.create_text_temp,
        _replace=replace_if_unchanged,
        _remove_temp=anchor.remove,
    )
    atomic.replace(value)


def _reparse_repaired_records(
    inspection: _RecoveryInspection,
    opened: dict[str, _OpenArtifact],
) -> tuple[
    tuple[TranscriptRecord, ...],
    tuple[StateHistoryRecord, ...],
    tuple[EventRecord, ...],
]:
    parsers: tuple[tuple[str, Callable[[object], object]], ...] = (
        ("transcript.jsonl", TranscriptRecord.from_dict),
        ("state-history.jsonl", StateHistoryRecord.from_dict),
        ("events.jsonl", EventRecord.from_dict),
    )
    parsed: dict[str, tuple[object, ...]] = {}
    for name, parser in parsers:
        encoded = _read_open_artifact(opened[name])
        plan = _inspect_jsonl_tail_bytes_impl(opened[name].path, encoded)
        if plan.discarded_tail_bytes or plan.append_final_lf:
            raise RecoveryError(
                opened[name].path, "JSONL repair did not produce a durable tail"
            )
        parsed[name] = _parse_jsonl_records(plan, encoded, parser)
    transcripts = cast(tuple[TranscriptRecord, ...], parsed["transcript.jsonl"])
    state_history = cast(tuple[StateHistoryRecord, ...], parsed["state-history.jsonl"])
    events = cast(tuple[EventRecord, ...], parsed["events.jsonl"])
    artifacts = {
        name: identity.path
        for name, identity in (
            (identity.path.name, identity)
            for identity in inspection.artifact_identities
        )
    }
    wav_samples = {
        plan.path.name: plan.valid_pcm_bytes // _WAV_SAMPLE_WIDTH
        for plan in inspection.wav_repair_plans
    }
    _validate_transcripts(transcripts, wav_samples, artifacts)
    _validate_state_history(state_history, inspection.manifest, artifacts)
    _validate_events(events, inspection.manifest, artifacts)
    return transcripts, state_history, events


def _recovery_event_details(report: RecoveryReport) -> dict[str, JsonValue]:
    return {
        "discarded_jsonl_tail_bytes": dict(report.discarded_jsonl_tail_bytes),
        "repaired_wav_headers": list(report.repaired_wav_headers),
    }


def _existing_durable_recovery_event(
    inspection: _RecoveryInspection,
) -> EventRecord | None:
    recovered = [
        (index, event)
        for index, event in enumerate(inspection.event_records)
        if event.event_type is EventType.SESSION_RECOVERED
    ]
    if not recovered:
        return None
    index, event = recovered[-1]
    if len(recovered) != 1 or index != len(inspection.event_records) - 1:
        raise RecoveryError(
            inspection.manifest_plan.identity.path,
            "SESSION_RECOVERED must be the unique final event",
        )
    if event.source is not ProcessSource.GUI:
        raise RecoveryError(
            inspection.manifest_plan.identity.path,
            "SESSION_RECOVERED source must be GUI",
        )
    _recovery_details_values(event, inspection.manifest_plan.identity.path)
    return event


def _recovery_details_values(
    event: EventRecord,
    path: Path,
) -> tuple[dict[str, int], tuple[str, ...]]:
    if set(event.details) != {
        "discarded_jsonl_tail_bytes",
        "repaired_wav_headers",
    }:
        raise RecoveryError(path, "SESSION_RECOVERED details have an invalid shape")
    discarded_value = event.details["discarded_jsonl_tail_bytes"]
    repaired_value = event.details["repaired_wav_headers"]
    if not isinstance(discarded_value, dict) or not all(
        isinstance(name, str)
        and name in _JSONL_ARTIFACT_NAMES
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        for name, count in discarded_value.items()
    ):
        raise RecoveryError(
            path, "SESSION_RECOVERED discarded-tail details are invalid"
        )
    if not isinstance(repaired_value, list) or not all(
        isinstance(name, str) and name in _WAV_ARTIFACT_NAMES for name in repaired_value
    ):
        raise RecoveryError(path, "SESSION_RECOVERED WAV details are invalid")
    repaired = tuple(cast(list[str], repaired_value))
    if repaired != tuple(sorted(set(repaired))):
        raise RecoveryError(
            path, "SESSION_RECOVERED WAV details must be unique and sorted"
        )
    return cast(dict[str, int], discarded_value), repaired


def _report_from_recovery_event(
    inspection: _RecoveryInspection,
    event: EventRecord,
) -> RecoveryReport:
    discarded, repaired = _recovery_details_values(
        event, inspection.manifest_plan.identity.path
    )
    return replace(
        inspection.report,
        discarded_jsonl_tail_bytes=dict(discarded),
        repaired_wav_headers=repaired,
    )


def _reconstruct_pause_intervals(
    events: tuple[EventRecord, ...],
    active_duration_ms: int,
    path: Path,
) -> tuple[tuple[PauseInterval, ...], tuple[str, ...]]:
    try:
        intervals, open_pause = reconstruct_recovered_pause_intervals(
            events,
            active_duration_ms,
        )
    except RecoveryPauseContractError as error:
        raise RecoveryError(path, str(error)) from error
    notes: tuple[str, ...] = ()
    if open_pause is not None:
        notes = (
            f"Closed unmatched PAUSE_START at {open_pause} ms at recovery boundary "
            f"{active_duration_ms} ms.",
        )
    return intervals, notes


def _build_recovery_notes(
    inspection: _RecoveryInspection,
    report: RecoveryReport,
    pause_notes: tuple[str, ...],
) -> tuple[str, ...]:
    notes = [
        f"Discarded {count} torn JSONL tail bytes from {name}."
        for name, count in sorted(report.discarded_jsonl_tail_bytes.items())
    ]
    wav_plans = {plan.path.name: plan for plan in inspection.wav_repair_plans}
    notes.extend(
        f"Repaired {name} WAV header for {wav_plans[name].valid_pcm_bytes} PCM bytes."
        for name in report.repaired_wav_headers
    )
    notes.extend(pause_notes)
    return tuple(notes)


def _run_transaction_post_validation(
    inspection: _RecoveryInspection,
    session_anchor: _DirectoryAnchor | None,
    parent_anchor: _DirectoryAnchor | None,
    primary_error: BaseException | None,
) -> BaseException | None:
    operations: list[tuple[str, Callable[[], None]]] = []
    if session_anchor is not None:
        operations.append(
            (
                "session directory",
                lambda: _verify_directory_identity(
                    inspection.session_directory_identity
                ),
            )
        )
    if parent_anchor is not None:
        operations.append(
            (
                "parent directory",
                lambda: _verify_directory_identity(
                    inspection.parent_directory_identity
                ),
            )
        )
    if session_anchor is not None:
        operations.extend(
            (
                f"artifact {identity.path.name}",
                partial(_validate_current_artifact_path, identity.path),
            )
            for identity in inspection.artifact_identities
        )
    surfaced = primary_error
    for label, operation in operations:
        try:
            operation()
        except BaseException as validation_error:
            if surfaced is None:
                surfaced = validation_error
            else:
                surfaced.add_note(
                    f"Recovery post-validation failed for {label}: {validation_error}"
                )
    return surfaced


def _validate_current_artifact_path(path: Path) -> None:
    opened = _open_guarded_artifact(path)
    primary_error: BaseException | None = None
    try:
        _verify_opened_path_identity(opened)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _close_artifact(opened)
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Post-validation artifact cleanup failed for {path}: {close_error}"
            )


def _cleanup_recovery_transaction(
    opened: tuple[_OpenArtifact, ...],
    session_anchor: _DirectoryAnchor | None,
    parent_anchor: _DirectoryAnchor | None,
    primary_error: BaseException | None,
) -> BaseException | None:
    surfaced = _close_open_artifacts(opened, primary_error)
    for anchor in (session_anchor, parent_anchor):
        if anchor is None:
            continue
        try:
            _close_directory_anchor(anchor)
        except BaseException as close_error:
            if surfaced is None:
                surfaced = close_error
            else:
                surfaced.add_note(
                    f"Directory anchor cleanup failed for {anchor.identity.path}: "
                    f"{close_error}"
                )
    return surfaced


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
        pcm_digest = sha256()
        total_size = len(header)
        while chunk := os.read(opened.descriptor, 1024 * 1024):
            digest.update(chunk)
            pcm_digest.update(chunk)
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
        pcm_sha256=pcm_digest.hexdigest(),
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
