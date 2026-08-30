"""Deterministic anti-advice prompts for discussion state replacement."""

import json

from flowlens.discussion.contracts import ChatMessage, DiscussionRequest
from flowlens.domain.enums import SessionMode

_MODE_LABELS: dict[SessionMode, tuple[str, str, str, str]] = {
    SessionMode.MEETING: (
        "Current focus",
        "Key points",
        "Decisions / confirmations",
        "Unresolved / next actions",
    ),
    SessionMode.INTERVIEW: (
        "Current question / topic",
        "Answer highlights",
        "Confirmed content",
        "Follow-ups / points to clarify",
    ),
    SessionMode.GENERAL: (
        "Current topic",
        "Key points",
        "Confirmed items",
        "Items to revisit",
    ),
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=4)


def _system_content(request: DiscussionRequest) -> str:
    focus, points, outcomes, follow_up = _MODE_LABELS[request.current_state.mode]
    mode_rules = (
        (
            "- Preserve explicit confirmation wording in confirmed_outcomes "
            "when the meeting transcript confirms, agrees, or decides an outcome; "
            "copy a qualifier such as 確定, 合意, or 決定 into that field and "
            "do not paraphrase the qualifier away.",
        )
        if request.current_state.mode is SessionMode.MEETING
        else ()
    )
    return "\n".join(
        (
            "Organize only what is explicitly supported by the transcript.",
            "Return a complete replacement snapshot, never a patch or mutation list.",
            "Do not invent facts, decisions, questions, or next actions.",
            "Do not give advice or recommendations.",
            "Do not suggest what anyone should say.",
            "Do not evaluate participants or their answers.",
            "Do not generate pros and cons.",
            "Do not attribute state items to speakers.",
            "Do not include evidence IDs or hidden recommendations.",
            f"Mode: {request.current_state.mode.value}",
            "State transition rules:",
            "- Replace current_focus when the active topic changes.",
            "- Keep only key_points relevant to the current focus; do not retain "
            "stale points from a previous focus.",
            "- Accumulate confirmed_outcomes across the session; do not remove a "
            "prior outcome unless the transcript explicitly contradicts or "
            "corrects it.",
            "- Keep each follow_up_items entry until the conversation resolves or "
            "supersedes it.",
            "- When the focus changes, prior focus and key points leave the live "
            "snapshot; history is retained externally.",
            "Field semantics:",
            "- revision: use the requested revision constant.",
            "- mode: preserve the session mode constant.",
            f"- current_focus ({focus}): the topic presently being discussed.",
            f"- key_points ({points}): concise points explicitly present "
            "in the conversation.",
            f"- confirmed_outcomes ({outcomes}): only content explicitly confirmed.",
            f"- follow_up_items ({follow_up}): only open content explicitly raised.",
            *mode_rules,
            "- updated_at: use the requested timestamp constant.",
            "- analyzed_through_sequence: use the requested watermark constant.",
        )
    )


def _user_content(request: DiscussionRequest) -> str:
    transcript = [
        {"source": record.source.value, "text": record.text}
        for record in request.records
    ]
    return "\n".join(
        (
            "Create the next complete discussion state from this input.",
            f"requested_revision: {request.requested_revision}",
            f"updated_at: {request.updated_at.isoformat(timespec='milliseconds')}",
            "analyzed_through_sequence: " f"{request.analyzed_through_sequence}",
            "current_state:",
            _json(request.current_state.to_dict()),
            "new_transcript_records:",
            _json(transcript),
        )
    )


def build_messages(request: DiscussionRequest) -> tuple[ChatMessage, ...]:
    """Build exactly one system and one user message."""

    return (
        ChatMessage(role="system", content=_system_content(request)),
        ChatMessage(role="user", content=_user_content(request)),
    )
