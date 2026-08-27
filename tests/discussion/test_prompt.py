"""Deterministic and safety-constrained discussion prompts."""

import json

import pytest

from flowlens.discussion.prompt import build_messages
from flowlens.domain.enums import AudioSource, SessionMode
from tests.discussion.factories import make_record, make_request


@pytest.mark.parametrize(
    ("mode", "labels", "forbidden"),
    [
        (
            SessionMode.MEETING,
            (
                "Current focus",
                "Key points",
                "Decisions / confirmations",
                "Unresolved / next actions",
            ),
            "Current question / topic",
        ),
        (
            SessionMode.INTERVIEW,
            (
                "Current question / topic",
                "Answer highlights",
                "Confirmed content",
                "Follow-ups / points to clarify",
            ),
            "unresolved issues",
        ),
        (
            SessionMode.GENERAL,
            (
                "Current topic",
                "Key points",
                "Confirmed items",
                "Items to revisit",
            ),
            "Decisions / confirmations",
        ),
    ],
)
def test_prompt_uses_complete_mode_semantics_and_anti_advice_rules(
    mode: SessionMode,
    labels: tuple[str, ...],
    forbidden: str,
) -> None:
    messages = build_messages(make_request(mode=mode))
    prompt = "\n".join(message.content for message in messages)

    assert [message.role for message in messages] == ["system", "user"]
    assert all(label in prompt for label in labels)
    assert forbidden not in prompt
    assert "Return a complete replacement snapshot" in prompt
    assert "Do not invent facts, decisions, questions, or next actions" in prompt
    assert "Do not suggest what anyone should say" in prompt
    assert "Do not give advice or recommendations" in prompt
    assert "Do not evaluate participants or their answers" in prompt
    assert "Do not generate pros and cons" in prompt
    assert "Do not attribute state items to speakers" in prompt
    assert "Do not include evidence IDs or hidden recommendations" in prompt
    for field_name in (
        "revision",
        "mode",
        "current_focus",
        "key_points",
        "confirmed_outcomes",
        "follow_up_items",
        "updated_at",
        "analyzed_through_sequence",
    ):
        assert field_name in prompt


def test_prompt_has_deterministic_four_space_japanese_json_and_full_state() -> None:
    records = (
        make_record(sequence=1, text="日本語の発言", source=AudioSource.ME),
        make_record(sequence=2, text="別の発言", source=AudioSource.OTHERS),
    )
    request = make_request(records=records)

    first = build_messages(request)
    second = build_messages(request)

    assert first == second
    state_json = json.dumps(
        request.current_state.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
    )
    transcript_json = json.dumps(
        [
            {"source": "ME", "text": "日本語の発言"},
            {"source": "OTHERS", "text": "別の発言"},
        ],
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
    )
    assert state_json in first[1].content
    assert transcript_json in first[1].content
    assert "\\u65e5" not in first[1].content
    assert "segment_id" not in first[1].content
    assert "source_start_sample" not in first[1].content


def test_prompt_is_complete_even_without_new_transcript_records() -> None:
    messages = build_messages(make_request(records=()))

    assert len(messages) == 2
    assert "[]" in messages[1].content
    assert "current_state" in messages[1].content


def test_prompt_defines_full_state_transition_semantics() -> None:
    prompt = "\n".join(message.content for message in build_messages(make_request()))

    assert "Replace current_focus when the active topic changes" in prompt
    assert (
        "Keep only key_points relevant to the current focus; "
        "do not retain stale points from a previous focus"
    ) in prompt
    assert (
        "Accumulate confirmed_outcomes across the session; do not remove a prior "
        "outcome unless the transcript explicitly contradicts or corrects it"
    ) in prompt
    assert (
        "Keep each follow_up_items entry until the conversation resolves or "
        "supersedes it"
    ) in prompt
