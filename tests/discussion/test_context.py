"""Bounded transcript context selection."""

from collections.abc import Callable

import pytest

from flowlens.discussion.context import (
    DiscussionContextError,
    select_recent_records,
)
from flowlens.domain.messages import TranscriptRecord
from tests.discussion.factories import make_record


@pytest.mark.parametrize(
    ("count", "first_sequence"),
    [(59, 1), (60, 1), (61, 2)],
)
def test_context_enforces_exact_record_cap(count: int, first_sequence: int) -> None:
    records = tuple(
        make_record(sequence=index, text="一") for index in range(1, count + 1)
    )

    selected = select_recent_records(records, lambda text: 1)

    assert len(selected) == min(count, 60)
    assert selected[0].sequence == first_sequence
    assert [record.sequence for record in selected] == sorted(
        record.sequence for record in selected
    )


@pytest.mark.parametrize(
    ("token_count", "accepted"),
    [(5_999, True), (6_000, True), (6_001, False)],
)
def test_context_enforces_exact_single_record_token_cap(
    token_count: int,
    accepted: bool,
) -> None:
    record = make_record(text="境界")

    if accepted:
        assert select_recent_records((record,), lambda text: token_count) == (record,)
    else:
        with pytest.raises(DiscussionContextError, match="6000"):
            select_recent_records((record,), lambda text: token_count)


def test_context_counts_exact_text_and_evicts_oldest_whole_records() -> None:
    records = (
        make_record(sequence=1, text="a" * 3_000),
        make_record(sequence=2, text="b" * 3_000),
        make_record(sequence=3, text="c" * 3_000),
    )
    counted: list[str] = []

    def count_tokens(text: str) -> int:
        counted.append(text)
        return len(text)

    selected = select_recent_records(records, count_tokens)

    assert [record.sequence for record in selected] == [2, 3]
    assert counted == [record.text for record in records]
    assert all(len(record.text) == 3_000 for record in selected)


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            (make_record(sequence=1), make_record(sequence=1)),
            "duplicate",
        ),
        (
            (make_record(sequence=2), make_record(sequence=1)),
            "out of order",
        ),
    ],
)
def test_context_rejects_duplicate_or_out_of_order_sequences(
    records: tuple[TranscriptRecord, ...],
    match: str,
) -> None:
    with pytest.raises(DiscussionContextError, match=match):
        select_recent_records(records, lambda text: 1)


@pytest.mark.parametrize(
    ("counter", "match"),
    [
        (lambda text: True, "integer"),
        (lambda text: -1, "non-negative"),
        (lambda text: 1.5, "integer"),
    ],
)
def test_context_rejects_invalid_tokenizer_counts(
    counter: Callable[[str], int],
    match: str,
) -> None:
    with pytest.raises(DiscussionContextError, match=match):
        select_recent_records((make_record(),), counter)


def test_context_wraps_tokenizer_failure_without_transcript_text() -> None:
    secret = "private transcript content"

    def fail(text: str) -> int:
        raise RuntimeError(f"tokenizer failed for {text}")

    with pytest.raises(DiscussionContextError) as captured:
        select_recent_records((make_record(text=secret),), fail)

    assert secret not in str(captured.value)


def test_context_returns_empty_immutable_tuple() -> None:
    assert select_recent_records((), len) == ()
