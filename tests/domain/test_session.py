from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import cast

import pytest

from flowlens.domain.enums import SessionMode, SessionStatus
from flowlens.domain.session import (
    DeviceIdentity,
    ModelIdentity,
    PauseInterval,
    SessionManifest,
)

START = datetime.fromisoformat("2026-08-19T12:00:00+09:00")
SESSION_ID = "01J00000000000000000000000"


def make_manifest() -> SessionManifest:
    return SessionManifest(
        schema_version=1,
        session_id=SESSION_ID,
        status=SessionStatus.INCOMPLETE,
        mode=SessionMode.MEETING,
        started_at=START,
        ended_at=None,
        active_duration_ms=0,
        pause_intervals=(),
        microphone=DeviceIdentity("mic-1", "USB Microphone"),
        loopback_output=DeviceIdentity("out-1", "Speakers"),
        asr_model=ModelIdentity(
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            "rev-a",
            "a" * 64,
        ),
        discussion_model=ModelIdentity(
            "Qwen/Qwen3-4B-Instruct-2507",
            "rev-b",
            "b" * 64,
        ),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=(),
    )


def test_manifest_round_trip_contains_every_required_field() -> None:
    manifest = make_manifest()

    assert SessionManifest.from_dict(manifest.to_dict()) == manifest
    assert list(manifest.to_dict()) == [
        "schema_version",
        "session_id",
        "status",
        "mode",
        "started_at",
        "ended_at",
        "active_duration_ms",
        "pause_intervals",
        "microphone",
        "loopback_output",
        "asr_model",
        "discussion_model",
        "application_version",
        "transcript_entry_count",
        "final_discussion_state_revision",
        "recovery_notes",
    ]


def test_manifest_normalizes_started_at_to_milliseconds() -> None:
    manifest = replace(
        make_manifest(),
        started_at=datetime.fromisoformat("2026-08-19T12:00:00.125999+09:00"),
    )

    assert manifest.started_at == datetime.fromisoformat(
        "2026-08-19T12:00:00.125+09:00"
    )
    assert SessionManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_normalizes_non_null_ended_at_to_milliseconds() -> None:
    manifest = replace(
        make_manifest(),
        status=SessionStatus.COMPLETED,
        ended_at=datetime.fromisoformat("2026-08-19T12:30:00.125999+09:00"),
    )

    assert manifest.ended_at == datetime.fromisoformat("2026-08-19T12:30:00.125+09:00")
    assert SessionManifest.from_dict(manifest.to_dict()) == manifest


def test_completed_manifest_requires_end_time() -> None:
    with pytest.raises(ValueError, match="ended_at"):
        replace(make_manifest(), status=SessionStatus.COMPLETED)


def test_recovered_manifest_requires_end_time() -> None:
    with pytest.raises(ValueError, match="ended_at"):
        replace(make_manifest(), status=SessionStatus.RECOVERED)


def test_pause_interval_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="ended_ms"):
        PauseInterval(started_ms=900, ended_ms=800)


def test_nested_models_use_exact_wire_shapes() -> None:
    manifest = make_manifest().to_dict()

    assert manifest["microphone"] == {
        "device_id": "mic-1",
        "display_name": "USB Microphone",
    }
    assert manifest["asr_model"] == {
        "repository": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "revision": "rev-a",
        "sha256": "a" * 64,
    }
    assert PauseInterval(100, 200).to_dict() == {
        "started_ms": 100,
        "ended_ms": 200,
    }


def test_manifest_from_dict_defensively_copies_nested_inputs() -> None:
    serialized = replace(
        make_manifest(),
        pause_intervals=(PauseInterval(100, 200),),
        recovery_notes=("recovered after restart",),
    ).to_dict()
    pauses = cast(list[object], serialized["pause_intervals"])
    recovery_notes = cast(list[object], serialized["recovery_notes"])
    restored = SessionManifest.from_dict(serialized)

    pauses.append({"started_ms": 300, "ended_ms": 400})
    recovery_notes.append("later mutation")
    cast(list[object], restored.to_dict()["pause_intervals"]).clear()

    assert restored.pause_intervals == (PauseInterval(100, 200),)
    assert restored.recovery_notes == ("recovered after restart",)
    assert restored.to_dict()["pause_intervals"] == [
        {"started_ms": 100, "ended_ms": 200}
    ]


@pytest.mark.parametrize("changed_key", ["missing", "unknown"])
def test_manifest_rejects_non_exact_keys(changed_key: str) -> None:
    serialized = make_manifest().to_dict()
    if changed_key == "missing":
        del serialized["mode"]
    else:
        serialized["extra"] = None

    with pytest.raises(ValueError, match=changed_key):
        SessionManifest.from_dict(serialized)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DeviceIdentity("", "USB Microphone"),
        lambda: DeviceIdentity("mic-1", ""),
        lambda: ModelIdentity("", "rev-a", "a" * 64),
        lambda: ModelIdentity("repo", "", "a" * 64),
        lambda: ModelIdentity("repo", "rev-a", "A" * 64),
    ],
)
def test_identity_models_reject_invalid_required_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: replace(value, schema_version=2), "schema_version"),
        (
            lambda value: replace(value, session_id="not-a-session-id"),
            "session_id",
        ),
        (lambda value: replace(value, active_duration_ms=-1), "active_duration_ms"),
        (
            lambda value: replace(value, transcript_entry_count=-1),
            "transcript_entry_count",
        ),
        (
            lambda value: replace(value, final_discussion_state_revision=-1),
            "final_discussion_state_revision",
        ),
        (
            lambda value: replace(value, application_version=""),
            "application_version",
        ),
    ],
)
def test_manifest_rejects_invalid_scalar_invariants(
    mutate: Callable[[SessionManifest], SessionManifest],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        mutate(make_manifest())


def test_manifest_rejects_naive_wall_clock() -> None:
    with pytest.raises(ValueError, match="started_at"):
        replace(
            make_manifest(),
            started_at=datetime.fromisoformat("2026-08-19T12:00:00"),
        )
