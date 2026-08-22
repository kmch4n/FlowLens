"""Bounded chronological transcript context selection."""

from collections.abc import Callable, Sequence

from flowlens.discussion.contracts import DiscussionContextError
from flowlens.domain.messages import TranscriptRecord

__all__ = ["DiscussionContextError", "select_recent_records"]

_MAX_RECORDS = 60
_MAX_TRANSCRIPT_TOKENS = 6_000


def _validate_sequences(records: Sequence[TranscriptRecord]) -> None:
    previous: int | None = None
    for record in records:
        if not isinstance(record, TranscriptRecord):
            raise DiscussionContextError(
                "records must contain only TranscriptRecord values"
            )
        if previous is not None:
            if record.sequence == previous:
                raise DiscussionContextError("record sequence is duplicate")
            if record.sequence < previous:
                raise DiscussionContextError("record sequence is out of order")
        previous = record.sequence


def _count_record_tokens(
    record: TranscriptRecord,
    count_tokens: Callable[[str], int],
) -> int:
    try:
        count = count_tokens(record.text)
    except Exception as error:
        raise DiscussionContextError("transcript tokenization failed") from error
    if not isinstance(count, int) or isinstance(count, bool):
        raise DiscussionContextError("tokenizer count must be an integer")
    if count < 0:
        raise DiscussionContextError("tokenizer count must be non-negative")
    if count > _MAX_TRANSCRIPT_TOKENS:
        raise DiscussionContextError(
            "one transcript record exceeds the 6000 token limit"
        )
    return count


def select_recent_records(
    records: Sequence[TranscriptRecord],
    count_tokens: Callable[[str], int],
) -> tuple[TranscriptRecord, ...]:
    """Keep newest whole records within both deterministic context caps."""

    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise DiscussionContextError("records must be a sequence")
    _validate_sequences(records)
    selected = list(records[-_MAX_RECORDS:])
    counts = [_count_record_tokens(record, count_tokens) for record in selected]
    total = sum(counts)
    while total > _MAX_TRANSCRIPT_TOKENS:
        total -= counts.pop(0)
        selected.pop(0)
    return tuple(selected)
