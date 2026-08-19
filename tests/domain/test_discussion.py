from datetime import datetime
from typing import cast

import pytest

from flowlens.domain.discussion import DiscussionState, StateHistoryRecord
from flowlens.domain.enums import SessionMode

NOW = datetime.fromisoformat("2026-08-19T12:35:02.125+09:00")
SESSION_ID = "01J00000000000000000000000"


def make_state() -> DiscussionState:
    return DiscussionState(
        revision=7,
        mode=SessionMode.INTERVIEW,
        current_focus="志望理由",
        key_points=("業務改善に関わった経験",),
        confirmed_outcomes=("完全ローカル動作をMVPの必須条件とする",),
        follow_up_items=("具体的な成果を確認する",),
        updated_at=NOW,
    )


def test_state_history_matches_spec_shape() -> None:
    state = make_state()
    record = StateHistoryRecord(1, SESSION_ID, 6, 7, state)
    restored = StateHistoryRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.to_dict()["state"] == state.to_dict()
    assert "evidence_ids" not in str(restored.to_dict())


def test_initial_state_uses_revision_zero_and_empty_values() -> None:
    state = DiscussionState.initial(SessionMode.GENERAL, NOW)

    assert state.revision == 0
    assert state.current_focus == ""
    assert state.key_points == ()
    assert state.confirmed_outcomes == ()
    assert state.follow_up_items == ()


def test_discussion_state_serializes_exact_keys_and_millisecond_time() -> None:
    serialized = make_state().to_dict()

    assert list(serialized) == [
        "revision",
        "mode",
        "current_focus",
        "key_points",
        "confirmed_outcomes",
        "follow_up_items",
        "updated_at",
    ]
    assert serialized["updated_at"] == "2026-08-19T12:35:02.125+09:00"
    assert serialized["key_points"] == ["業務改善に関わった経験"]


def test_discussion_state_normalizes_updated_at_to_milliseconds() -> None:
    state = DiscussionState(
        revision=0,
        mode=SessionMode.GENERAL,
        current_focus="",
        key_points=(),
        confirmed_outcomes=(),
        follow_up_items=(),
        updated_at=datetime.fromisoformat("2026-08-19T12:35:02.125999+09:00"),
    )

    assert state.updated_at == datetime.fromisoformat("2026-08-19T12:35:02.125+09:00")
    assert DiscussionState.from_dict(state.to_dict()) == state


def test_discussion_state_from_dict_defensively_copies_lists() -> None:
    serialized = make_state().to_dict()
    key_points = cast(list[object], serialized["key_points"])
    restored = DiscussionState.from_dict(serialized)

    key_points.append("後から追加")
    cast(list[object], restored.to_dict()["key_points"]).append("出力だけ変更")

    assert restored.key_points == ("業務改善に関わった経験",)
    assert restored.to_dict()["key_points"] == ["業務改善に関わった経験"]


@pytest.mark.parametrize("changed_key", ["missing", "unknown"])
def test_discussion_state_rejects_non_exact_keys(changed_key: str) -> None:
    serialized = make_state().to_dict()
    if changed_key == "missing":
        del serialized["mode"]
    else:
        serialized["evidence_ids"] = []

    with pytest.raises(ValueError, match=changed_key):
        DiscussionState.from_dict(serialized)


@pytest.mark.parametrize(
    ("previous_revision", "new_revision", "match"),
    [
        (-1, 0, "previous_revision"),
        (6, 8, "new_revision"),
        (7, 7, "previous_revision"),
    ],
)
def test_state_history_rejects_invalid_revision_progression(
    previous_revision: int,
    new_revision: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        StateHistoryRecord(1, SESSION_ID, previous_revision, new_revision, make_state())


def test_discussion_state_rejects_naive_updated_at() -> None:
    with pytest.raises(ValueError, match="updated_at"):
        DiscussionState.initial(
            SessionMode.MEETING,
            datetime.fromisoformat("2026-08-19T12:35:02.125"),
        )
