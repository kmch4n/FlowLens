"""Validate one persisted FlowLens session without modifying it."""

from __future__ import annotations

import argparse
import json
import os
import stat
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from flowlens.domain.discussion import DiscussionState, StateHistoryRecord
from flowlens.domain.enums import EventType
from flowlens.domain.messages import EventRecord, TranscriptRecord
from flowlens.domain.session import SessionManifest
from flowlens.persistence.recovery_contract import (
    RecoveryPauseContractError,
    reconstruct_recovered_pause_intervals,
    recovered_terminal_time_ms,
)

REQUIRED_ARTIFACTS = frozenset(
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


@dataclass(frozen=True, slots=True)
class SessionValidationResult:
    """Deterministic read-only validation result."""

    errors: tuple[str, ...]
    mic_format: tuple[int, int, int] | None = None
    loopback_format: tuple[int, int, int] | None = None
    mic_duration_ms: float | None = None
    loopback_duration_ms: float | None = None
    wav_error_percent: float | None = None
    active_duration_ms: int | None = None
    transcript_count: int = 0
    sources: tuple[str, ...] = ()
    final_revision: int | None = None
    event_types: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Return true only when every validator passed."""

        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Serialize the result for smoke reports."""

        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "active_duration_ms": self.active_duration_ms,
            "wav_error_percent": self.wav_error_percent,
            "mic_format": list(self.mic_format) if self.mic_format else None,
            "loopback_format": (
                list(self.loopback_format) if self.loopback_format else None
            ),
            "mic_duration_ms": self.mic_duration_ms,
            "loopback_duration_ms": self.loopback_duration_ms,
            "transcript_count": self.transcript_count,
            "sources": list(self.sources),
            "final_revision": self.final_revision,
            "event_types": list(self.event_types),
        }


def _is_link_or_reparse(path: Path) -> bool:
    information = os.lstat(path)
    attributes = getattr(information, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(information.st_mode) or bool(attributes & reparse)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _decode_json(encoded: bytes, name: str) -> object:
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{name} must be UTF-8 without BOM")
    try:
        text = encoded.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must contain strict UTF-8 JSON") from error


def _read_json(path: Path) -> object:
    return _decode_json(path.read_bytes(), path.name)


def _read_jsonl(path: Path) -> list[object]:
    encoded = path.read_bytes()
    if not encoded:
        return []
    if not encoded.endswith(b"\n"):
        raise ValueError(
            f"{path.name} must end with a complete LF-terminated JSON line"
        )
    if b"\r" in encoded:
        raise ValueError(f"{path.name} must use LF line endings")
    values: list[object] = []
    for line_number, line in enumerate(encoded.splitlines(), start=1):
        if not line:
            raise ValueError(f"{path.name} line {line_number} must not be blank")
        values.append(_decode_json(line, f"{path.name} line {line_number}"))
    return values


def _wav_contract(path: Path) -> tuple[tuple[int, int, int], float]:
    encoded = path.read_bytes()
    if len(encoded) < 12 or encoded[:4] != b"RIFF" or encoded[8:12] != b"WAVE":
        raise ValueError("missing RIFF/WAVE header")
    riff_size = struct.unpack_from("<I", encoded, 4)[0]
    if riff_size + 8 < len(encoded):
        raise ValueError("surplus bytes after RIFF chunk")
    if riff_size + 8 > len(encoded):
        raise ValueError("RIFF chunk is truncated")

    format_values: tuple[int, int, int, int, int, int] | None = None
    data_size: int | None = None
    offset = 12
    while offset < len(encoded):
        if offset + 8 > len(encoded):
            raise ValueError("WAV chunk header is truncated")
        chunk_id = encoded[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", encoded, offset + 4)[0]
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(encoded):
            if chunk_id == b"data":
                raise ValueError("PCM payload is truncated")
            raise ValueError("WAV chunk payload is truncated")
        padded_end = payload_end + (chunk_size & 1)
        if padded_end > len(encoded):
            raise ValueError("WAV chunk padding is truncated")
        if chunk_id == b"fmt ":
            if format_values is not None or chunk_size < 16:
                raise ValueError("WAV must contain one complete fmt chunk")
            format_values = struct.unpack_from("<HHIIHH", encoded, payload_start)
        elif chunk_id == b"data":
            if data_size is not None:
                raise ValueError("WAV must contain exactly one data chunk")
            if padded_end != len(encoded):
                raise ValueError("surplus bytes after PCM payload")
            data_size = chunk_size
        offset = padded_end

    if format_values is None or data_size is None:
        raise ValueError("WAV must contain fmt and data chunks")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = format_values
    if audio_format != 1:
        raise ValueError("compressed WAV is not supported")
    if bits == 0 or bits % 8 != 0 or channels == 0 or sample_rate == 0:
        raise ValueError("PCM format values must be positive and byte-aligned")
    sample_width = bits // 8
    expected_frame_bytes = channels * sample_width
    if block_align != expected_frame_bytes:
        raise ValueError("PCM block alignment does not match frame bytes")
    if byte_rate != sample_rate * expected_frame_bytes:
        raise ValueError("PCM byte rate does not match frame bytes")
    if data_size % expected_frame_bytes != 0:
        raise ValueError("PCM payload is not an exact number of frames")
    frame_count = data_size // expected_frame_bytes
    duration_ms = frame_count * 1_000 / sample_rate
    return (channels, sample_width, sample_rate), duration_ms


def _append_contract_error(errors: list[str], name: str, error: Exception) -> None:
    errors.append(f"{name} is invalid: {type(error).__name__}: {error}")


def _pause_intervals(
    events: list[EventRecord], errors: list[str]
) -> tuple[tuple[int, int], ...]:
    open_pause: int | None = None
    intervals: list[tuple[int, int]] = []
    for event in events:
        if event.event_type is EventType.PAUSE_START:
            if open_pause is not None:
                errors.append("events.jsonl contains nested PAUSE_START events")
            else:
                open_pause = event.session_time_ms
        elif event.event_type is EventType.PAUSE_END:
            if open_pause is None:
                errors.append("events.jsonl contains PAUSE_END without PAUSE_START")
            elif event.session_time_ms < open_pause:
                errors.append("PAUSE_END must not precede PAUSE_START")
                open_pause = None
            else:
                intervals.append((open_pause, event.session_time_ms))
                open_pause = None
    if open_pause is not None:
        errors.append("events.jsonl contains an unclosed PAUSE_START")
    return tuple(intervals)


def _validate_terminal_events(
    events: list[EventRecord],
    expected_status: str,
    errors: list[str],
) -> None:
    completed = [
        event for event in events if event.event_type is EventType.SESSION_COMPLETED
    ]
    recovered = [
        event for event in events if event.event_type is EventType.SESSION_RECOVERED
    ]
    if expected_status == "completed":
        if len(completed) != 1 or not events or events[-1] is not completed[0]:
            errors.append("events.jsonl must end with exactly one SESSION_COMPLETED")
        if recovered:
            errors.append("Completed session must not contain SESSION_RECOVERED")
    else:
        if completed:
            errors.append("Recovered session must not contain SESSION_COMPLETED")
        if len(recovered) != 1 or not events or events[-1] is not recovered[0]:
            errors.append("events.jsonl must end with exactly one SESSION_RECOVERED")


def validate_session(
    session_dir: Path,
    *,
    minimum_active_seconds: int,
    expected_status: str,
    require_pause: bool = False,
) -> SessionValidationResult:
    """Validate the exact seven-artifact contract without repairing anything."""

    if not isinstance(session_dir, Path):
        raise TypeError("session_dir must be a Path")
    if (
        not isinstance(minimum_active_seconds, int)
        or isinstance(minimum_active_seconds, bool)
        or minimum_active_seconds < 0
    ):
        raise ValueError("minimum_active_seconds must be a non-negative integer")
    if expected_status not in ("completed", "recovered"):
        raise ValueError("expected_status must be completed or recovered")
    if type(require_pause) is not bool:
        raise TypeError("require_pause must be a boolean")
    try:
        if _is_link_or_reparse(session_dir):
            return SessionValidationResult(
                errors=("Session directory must not be a link or reparse point",)
            )
    except OSError:
        return SessionValidationResult(errors=("Session directory does not exist",))
    if not session_dir.is_dir():
        return SessionValidationResult(errors=("Session directory does not exist",))

    errors: list[str] = []
    entries = {entry.name: entry for entry in session_dir.iterdir()}
    for missing in sorted(REQUIRED_ARTIFACTS - entries.keys()):
        errors.append(f"Missing required artifact: {missing}")
    for unexpected in sorted(entries.keys() - REQUIRED_ARTIFACTS):
        errors.append(f"Unexpected artifact: {unexpected}")
    for name in sorted(REQUIRED_ARTIFACTS & entries.keys()):
        path = entries[name]
        try:
            if _is_link_or_reparse(path) or not path.is_file():
                errors.append(f"Artifact must be a regular non-linked file: {name}")
        except OSError:
            errors.append(f"Artifact could not be inspected: {name}")

    manifest: SessionManifest | None = None
    if "session.json" in entries:
        try:
            manifest = SessionManifest.from_dict(_read_json(entries["session.json"]))
        except Exception as error:
            _append_contract_error(errors, "session.json", error)
    if manifest is not None:
        if manifest.status.value != expected_status:
            errors.append(
                "session.json status must be "
                f"{expected_status}, got {manifest.status.value}"
            )
        minimum_ms = minimum_active_seconds * 1_000
        if manifest.active_duration_ms < minimum_ms:
            errors.append(f"Active duration is below {minimum_active_seconds} seconds")

    formats: dict[str, tuple[int, int, int]] = {}
    durations: dict[str, float] = {}
    for name in ("mic.wav", "loopback.wav"):
        if name not in entries:
            continue
        try:
            wav_format, duration_ms = _wav_contract(entries[name])
            formats[name] = wav_format
            durations[name] = duration_ms
            if wav_format != (1, 2, 16_000):
                errors.append(f"{name} must be 16 kHz, 16-bit, mono PCM")
        except Exception as error:
            _append_contract_error(errors, name, error)

    transcripts: list[TranscriptRecord] = []
    if "transcript.jsonl" in entries:
        try:
            transcripts = [
                TranscriptRecord.from_dict(value)
                for value in _read_jsonl(entries["transcript.jsonl"])
            ]
            if [item.sequence for item in transcripts] != list(
                range(1, len(transcripts) + 1)
            ):
                errors.append("transcript.jsonl sequences must be contiguous from 1")
        except Exception as error:
            if str(error).startswith("transcript.jsonl must"):
                errors.append(str(error))
            else:
                _append_contract_error(errors, "transcript.jsonl", error)

    final_state: DiscussionState | None = None
    if "discussion-state.json" in entries:
        try:
            final_state = DiscussionState.from_dict(
                _read_json(entries["discussion-state.json"])
            )
        except Exception as error:
            _append_contract_error(errors, "discussion-state.json", error)

    history: list[StateHistoryRecord] = []
    if "state-history.jsonl" in entries:
        try:
            history = [
                StateHistoryRecord.from_dict(value)
                for value in _read_jsonl(entries["state-history.jsonl"])
            ]
            if [item.new_revision for item in history] != list(
                range(1, len(history) + 1)
            ):
                errors.append("state-history.jsonl revisions must be contiguous from 1")
        except Exception as error:
            if str(error).startswith("state-history.jsonl must"):
                errors.append(str(error))
            else:
                _append_contract_error(errors, "state-history.jsonl", error)

    events: list[EventRecord] = []
    event_pause_intervals: tuple[tuple[int, int], ...] = ()
    if "events.jsonl" in entries:
        try:
            events = [
                EventRecord.from_dict(value)
                for value in _read_jsonl(entries["events.jsonl"])
            ]
            if [item.sequence for item in events] != list(range(1, len(events) + 1)):
                errors.append("events.jsonl sequences must be contiguous from 1")
            if any(
                later.session_time_ms < earlier.session_time_ms
                for earlier, later in pairwise(events)
            ):
                errors.append("events.jsonl session times must be nondecreasing")
            if manifest is not None and any(
                event.session_id != manifest.session_id for event in events
            ):
                errors.append("events.jsonl session IDs must match session.json")
            if expected_status == "recovered" and manifest is not None:
                base_events = tuple(
                    event
                    for event in events
                    if event.event_type is not EventType.SESSION_RECOVERED
                )
                try:
                    recovered_pauses, _closed_open_pause = (
                        reconstruct_recovered_pause_intervals(
                            base_events,
                            manifest.active_duration_ms,
                        )
                    )
                    event_pause_intervals = tuple(
                        (interval.started_ms, interval.ended_ms)
                        for interval in recovered_pauses
                    )
                except RecoveryPauseContractError as error:
                    errors.append(
                        f"events.jsonl recovery pause contract is invalid: {error}"
                    )
            else:
                event_pause_intervals = _pause_intervals(events, errors)
            _validate_terminal_events(events, expected_status, errors)
            if require_pause:
                pause_starts = sum(
                    event.event_type is EventType.PAUSE_START for event in events
                )
                pause_ends = sum(
                    event.event_type is EventType.PAUSE_END for event in events
                )
                if (pause_starts, pause_ends) != (1, 1):
                    errors.append(
                        "events.jsonl must contain exactly one pause/resume pair"
                    )
        except Exception as error:
            if str(error).startswith("events.jsonl must"):
                errors.append(str(error))
            else:
                _append_contract_error(errors, "events.jsonl", error)

    if manifest is not None:
        if manifest.transcript_entry_count != len(transcripts):
            errors.append("session.json transcript_entry_count does not match log")
        if final_state is not None:
            if manifest.final_discussion_state_revision != final_state.revision:
                errors.append("session.json final revision does not match snapshot")
            if final_state.mode is not manifest.mode:
                errors.append("discussion-state.json mode does not match session.json")
        if len(history) != manifest.final_discussion_state_revision:
            errors.append("state-history.jsonl length must match final revision")
        if history:
            if final_state is None or history[-1].state != final_state:
                errors.append(
                    "Final history state does not match discussion-state.json"
                )
            if any(item.session_id != manifest.session_id for item in history):
                errors.append("state-history.jsonl session IDs must match session.json")
            if any(item.state.mode is not manifest.mode for item in history):
                errors.append("state-history.jsonl modes must match session.json")
        manifest_intervals = tuple(
            (interval.started_ms, interval.ended_ms)
            for interval in manifest.pause_intervals
        )
        if event_pause_intervals != manifest_intervals:
            errors.append("session.json pause intervals do not match events.jsonl")
        expected_terminal_type = (
            EventType.SESSION_COMPLETED
            if expected_status == "completed"
            else EventType.SESSION_RECOVERED
        )
        terminal_events = [
            event for event in events if event.event_type is expected_terminal_type
        ]
        if len(terminal_events) == 1 and events and events[-1] is terminal_events[0]:
            terminal = terminal_events[0]
            if manifest.ended_at != terminal.created_at:
                errors.append(
                    "session.json ended_at must match terminal event created_at"
                )
            if expected_status == "completed":
                paused_ms = sum(end - start for start, end in event_pause_intervals)
                if terminal.session_time_ms - paused_ms != manifest.active_duration_ms:
                    errors.append(
                        "session.json active duration does not match event timeline"
                    )
            else:
                retained_base_events = tuple(
                    event
                    for event in events
                    if event.event_type is not EventType.SESSION_RECOVERED
                )
                recovery_boundary_ms = recovered_terminal_time_ms(
                    retained_base_events,
                    manifest.active_duration_ms,
                )
                if terminal.session_time_ms != recovery_boundary_ms:
                    errors.append(
                        "SESSION_RECOVERED time does not match recovery boundary"
                    )

    wav_error: float | None = None
    if manifest is not None and len(durations) == 2:
        if manifest.active_duration_ms == 0:
            wav_error = 0.0 if all(value == 0 for value in durations.values()) else None
            if wav_error is None:
                errors.append(
                    "WAV duration cannot be compared with zero active duration"
                )
        else:
            wav_error = max(
                abs(value - manifest.active_duration_ms)
                / manifest.active_duration_ms
                * 100
                for value in durations.values()
            )

    return SessionValidationResult(
        errors=tuple(errors),
        mic_format=formats.get("mic.wav"),
        loopback_format=formats.get("loopback.wav"),
        mic_duration_ms=durations.get("mic.wav"),
        loopback_duration_ms=durations.get("loopback.wav"),
        wav_error_percent=wav_error,
        active_duration_ms=(manifest.active_duration_ms if manifest else None),
        transcript_count=len(transcripts),
        sources=tuple(sorted({item.source.value for item in transcripts})),
        final_revision=(final_state.revision if final_state else None),
        event_types=tuple(item.event_type.value for item in events),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--minimum-active-seconds", type=int, required=True)
    status = parser.add_mutually_exclusive_group(required=True)
    status.add_argument("--require-completed", action="store_true")
    status.add_argument("--require-recovered", action="store_true")
    parser.add_argument("--require-pause", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validator CLI and emit deterministic JSON."""

    arguments = _build_parser().parse_args(argv)
    expected = "completed" if arguments.require_completed else "recovered"
    result = validate_session(
        cast(Path, arguments.session_dir),
        minimum_active_seconds=cast(int, arguments.minimum_active_seconds),
        expected_status=expected,
        require_pause=cast(bool, arguments.require_pause),
    )
    json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=4)
    sys.stdout.write("\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
