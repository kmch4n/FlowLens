"""Stable transcript commit and chronological release tests."""

import pickle
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from flowlens.asr.commit import (
    ChronologicalCommitBuffer,
    StablePrefixTracker,
    choose_split_ms,
    is_transcript_content,
)
from flowlens.asr.types import CommitCandidate, DecodedToken, DecodeHypothesis
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import TranscriptRecord

NOW = datetime(2026, 8, 19, 12, 34, 56, 789_000, tzinfo=UTC)


def fixed_now() -> datetime:
    """Return a pickle-safe deterministic wall clock value."""

    return NOW


def hypothesis(*texts: str) -> DecodeHypothesis:
    """Build a hypothesis with deterministic 200 ms token spans."""

    return DecodeHypothesis(
        tuple(
            DecodedToken(text, index * 200, (index + 1) * 200)
            for index, text in enumerate(texts)
        )
    )


def candidate(
    source: AudioSource,
    text: str,
    *,
    start_ms: int,
) -> CommitCandidate:
    """Build one valid commit candidate with deterministic spans."""

    return CommitCandidate(
        source=source,
        text=text,
        session_start_ms=start_ms,
        session_end_ms=start_ms + 200,
        source_start_sample=start_ms * 16,
        source_end_sample=(start_ms + 200) * 16,
        committed_at=NOW,
    )


def make_commit_buffer() -> ChronologicalCommitBuffer:
    """Build a buffer with deterministic IDs and wall time."""

    identifiers: Iterator[str] = iter(
        (
            "01J00000000000000000000001",
            "01J00000000000000000000002",
            "01J00000000000000000000003",
        )
    )
    return ChronologicalCommitBuffer(
        segment_id_factory=lambda: next(identifiers),
        now=lambda: NOW,
    )


def test_prefix_commits_only_after_two_matches_and_twelve_hundred_ms_age() -> None:
    tracker = StablePrefixTracker(stable_age_ms=1_200)
    assert tracker.observe(hypothesis("今回", "は", "方針"), 0, final=False) == ()
    assert tracker.observe(hypothesis("今回", "は", "方針"), 500, final=False) == ()
    assert tracker.observe(hypothesis("今回", "は", "方針"), 1_199, final=False) == ()

    committed = tracker.observe(
        hypothesis("今回", "は", "方針"),
        1_200,
        final=False,
    )

    assert tuple(token.text for token in committed) == ("今回", "は", "方針")


def test_final_decode_commits_remaining_text_once() -> None:
    tracker = StablePrefixTracker(stable_age_ms=1_200)
    tracker.observe(hypothesis("確認", "します"), 0, final=False)

    first = tracker.observe(hypothesis("確認", "します"), 450, final=True)
    second = tracker.observe(hypothesis("確認", "します"), 900, final=True)

    assert "".join(token.text for token in first) == "確認します"
    assert second == ()


def test_changed_already_committed_positions_are_not_emitted_again() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    initial = hypothesis("確定", "部分")
    tracker.observe(initial, 0, final=False)
    assert tracker.observe(initial, 100, final=False) == initial.tokens

    changed = hypothesis("変更", "部分")

    assert tracker.observe(changed, 200, final=True) == ()


def test_changed_suffix_must_stabilize_before_it_can_follow_stable_prefix() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    tracker.observe(hypothesis("安定", "旧案"), 0, final=False)
    tracker.observe(hypothesis("安定", "新案"), 50, final=False)

    committed = tracker.observe(hypothesis("安定", "新案"), 100, final=False)

    assert tuple(token.text for token in committed) == ("安定",)
    assert tracker.observe(hypothesis("安定", "新案"), 150, final=False)[0].text == (
        "新案"
    )


def test_token_timestamp_change_resets_consecutive_identity_age() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    original = DecodeHypothesis((DecodedToken("同文", 0, 200),))
    shifted = DecodeHypothesis((DecodedToken("同文", 20, 220),))
    tracker.observe(original, 0, final=False)

    assert tracker.observe(shifted, 100, final=False) == ()
    assert tracker.observe(shifted, 199, final=False) == ()
    assert tracker.observe(shifted, 200, final=False) == shifted.tokens


def test_shrinking_final_hypothesis_does_not_repeat_committed_tokens() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    full = hypothesis("一", "二", "三")
    tracker.observe(full, 0, final=False)
    tracker.observe(full, 100, final=False)

    assert tracker.observe(hypothesis("一"), 200, final=True) == ()


def test_final_commits_only_tail_after_an_earlier_stable_prefix() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    tracker.observe(hypothesis("確定", "旧"), 0, final=False)
    assert tracker.observe(hypothesis("確定", "新"), 100, final=False)[0].text == (
        "確定"
    )

    committed = tracker.observe(hypothesis("確定", "新"), 120, final=True)

    assert tuple(token.text for token in committed) == ("新",)


def test_tracker_survives_pickle_with_commit_state_intact() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    decoded = hypothesis("確定")
    tracker.observe(decoded, 0, final=False)
    tracker.observe(decoded, 100, final=False)

    restored = pickle.loads(pickle.dumps(tracker))

    assert isinstance(restored, StablePrefixTracker)
    assert restored.observe(decoded, 200, final=True) == ()


@pytest.mark.parametrize("stable_age_ms", [0, -1, True])
def test_tracker_rejects_invalid_stable_age(stable_age_ms: int) -> None:
    with pytest.raises(ValueError, match="stable_age_ms"):
        StablePrefixTracker(stable_age_ms)


def test_tracker_rejects_retrograde_decode_time_and_non_boolean_final() -> None:
    tracker = StablePrefixTracker(stable_age_ms=100)
    tracker.observe(hypothesis("発言"), 100, final=False)

    with pytest.raises(ValueError, match="decoded_at_ms"):
        tracker.observe(hypothesis("発言"), 99, final=False)
    with pytest.raises(ValueError, match="final"):
        tracker.observe(hypothesis("発言"), 100, final=cast(bool, 1))


@pytest.mark.parametrize("decoded_at_ms", [-1, True])
def test_tracker_rejects_invalid_decode_time(decoded_at_ms: int) -> None:
    with pytest.raises(ValueError, match="decoded_at_ms"):
        StablePrefixTracker(100).observe(
            hypothesis("発言"),
            decoded_at_ms,
            final=False,
        )


def test_tracker_rejects_non_hypothesis() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        StablePrefixTracker(100).observe(
            cast(DecodeHypothesis, object()),
            0,
            final=False,
        )


def test_split_prefers_latest_punctuation_between_ten_and_twelve_seconds() -> None:
    decoded = DecodeHypothesis(
        (
            DecodedToken("前半。", 0, 10_400),
            DecodedToken("続き", 10_400, 11_800),
        )
    )

    assert choose_split_ms(decoded) == 10_400


def test_split_falls_back_to_twelve_seconds_without_punctuation() -> None:
    decoded = DecodeHypothesis((DecodedToken("継続発言", 0, 11_900),))

    assert choose_split_ms(decoded) == 12_000


def test_split_uses_latest_inclusive_boundary_and_rounds_down() -> None:
    decoded = DecodeHypothesis(
        (
            DecodedToken("最初。", 0, 10_019),
            DecodedToken("最後?", 10_019, 11_999),
        )
    )

    assert choose_split_ms(decoded) == 11_980
    assert (
        choose_split_ms(DecodeHypothesis((DecodedToken("境界。", 0, 12_000),)))
        == 12_000
    )


@pytest.mark.parametrize("end_ms", [9_999, 12_001])
def test_split_ignores_punctuation_outside_permitted_window(end_ms: int) -> None:
    assert (
        choose_split_ms(DecodeHypothesis((DecodedToken("対象外。", 0, end_ms),)))
        == 12_000
    )


def test_split_rejects_non_hypothesis() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        choose_split_ms(cast(DecodeHypothesis, object()))


def test_equal_start_time_orders_me_before_others() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.ME, 100)
    buffer.set_frontier(AudioSource.OTHERS, 100)
    buffer.push(candidate(AudioSource.OTHERS, "他者", start_ms=100))
    buffer.push(candidate(AudioSource.ME, "自分", start_ms=100))

    buffer.set_frontier(AudioSource.ME, None)
    first = buffer.release_ready()
    buffer.set_frontier(AudioSource.OTHERS, None)
    second = buffer.release_ready()

    assert [record.text for record in first] == ["自分"]
    assert [record.text for record in second] == ["他者"]


def test_later_finishing_older_utterance_blocks_newer_commit() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.ME, 100)
    buffer.push(candidate(AudioSource.OTHERS, "新しい", start_ms=200))
    assert buffer.release_ready() == ()

    buffer.push(candidate(AudioSource.ME, "古い", start_ms=100))
    buffer.set_frontier(AudioSource.ME, None)

    assert [record.text for record in buffer.release_ready()] == [
        "古い",
        "新しい",
    ]


def test_released_record_contains_every_exact_field() -> None:
    buffer = make_commit_buffer()
    item = candidate(AudioSource.ME, "内容", start_ms=100)
    buffer.push(item)

    assert buffer.release_ready() == (
        TranscriptRecord(
            schema_version=1,
            segment_id="01J00000000000000000000001",
            sequence=1,
            source=AudioSource.ME,
            text="内容",
            session_start_ms=100,
            session_end_ms=300,
            source_start_sample=1_600,
            source_end_sample=4_800,
            committed_at=NOW,
        ),
    )


def test_default_commit_buffer_uses_timezone_aware_wall_time() -> None:
    buffer = ChronologicalCommitBuffer(
        segment_id_factory=lambda: "01J00000000000000000000001"
    )
    buffer.push(candidate(AudioSource.ME, "内容", start_ms=100))

    (record,) = buffer.release_ready()

    assert record.committed_at.utcoffset() is not None


def test_finalize_releases_all_in_key_order_and_is_idempotent() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.ME, 50)
    buffer.set_frontier(AudioSource.OTHERS, 50)
    buffer.push(candidate(AudioSource.OTHERS, "他者", start_ms=100))
    buffer.push(candidate(AudioSource.ME, "自分", start_ms=100))

    records = buffer.finalize()

    assert [(record.sequence, record.text) for record in records] == [
        (1, "自分"),
        (2, "他者"),
    ]
    assert buffer.finalize() == ()
    assert buffer.release_ready() == ()


def test_sequences_remain_consecutive_across_separate_releases() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.OTHERS, 200)
    buffer.push(candidate(AudioSource.ME, "先", start_ms=100))
    first = buffer.release_ready()
    buffer.push(candidate(AudioSource.OTHERS, "後", start_ms=200))
    buffer.set_frontier(AudioSource.OTHERS, None)
    second = buffer.release_ready()

    assert [record.sequence for record in first + second] == [1, 2]


def test_release_uses_injected_now_instead_of_candidate_timestamp() -> None:
    candidate_time = NOW - timedelta(seconds=5)
    item = CommitCandidate(
        source=AudioSource.ME,
        text="内容",
        session_start_ms=100,
        session_end_ms=300,
        source_start_sample=1_600,
        source_end_sample=4_800,
        committed_at=candidate_time,
    )
    buffer = make_commit_buffer()
    buffer.push(item)

    assert buffer.release_ready()[0].committed_at == NOW


def test_buffer_survives_pickle_with_pending_candidate() -> None:
    buffer = ChronologicalCommitBuffer(now=fixed_now)
    buffer.set_frontier(AudioSource.ME, 100)
    buffer.push(candidate(AudioSource.ME, "保留", start_ms=100))

    restored = pickle.loads(pickle.dumps(buffer))

    assert isinstance(restored, ChronologicalCommitBuffer)
    assert [record.text for record in restored.finalize()] == ["保留"]


def test_buffer_rejects_duplicate_and_retrograde_candidates() -> None:
    buffer = make_commit_buffer()
    buffer.push(candidate(AudioSource.ME, "先", start_ms=100))

    with pytest.raises(ValueError, match="candidate start"):
        buffer.push(candidate(AudioSource.ME, "重複", start_ms=100))
    with pytest.raises(ValueError, match="candidate start"):
        buffer.push(candidate(AudioSource.ME, "逆行", start_ms=99))


def test_buffer_rejects_retrograde_frontier() -> None:
    buffer = make_commit_buffer()
    buffer.set_frontier(AudioSource.ME, 200)

    with pytest.raises(ValueError, match="frontier"):
        buffer.set_frontier(AudioSource.ME, 199)


def test_buffer_rejects_invalid_constructor_and_operation_arguments() -> None:
    with pytest.raises(ValueError, match="segment_id_factory"):
        ChronologicalCommitBuffer(
            segment_id_factory=cast(Callable[[], str], object()),
            now=fixed_now,
        )
    with pytest.raises(ValueError, match="now"):
        ChronologicalCommitBuffer(now=cast(Callable[[], datetime], object()))

    buffer = make_commit_buffer()
    with pytest.raises(ValueError, match="source"):
        buffer.set_frontier(cast(AudioSource, "ME"), 100)
    for invalid_start in (-1, True):
        with pytest.raises(ValueError, match="start_ms"):
            buffer.set_frontier(AudioSource.ME, invalid_start)
    with pytest.raises(ValueError, match="candidate"):
        buffer.push(cast(CommitCandidate, object()))


def test_buffer_rejects_frontier_or_candidate_behind_released_key() -> None:
    buffer = make_commit_buffer()
    buffer.push(candidate(AudioSource.OTHERS, "先", start_ms=100))
    assert buffer.release_ready()[0].text == "先"

    with pytest.raises(ValueError, match="frontier"):
        buffer.set_frontier(AudioSource.ME, 100)
    with pytest.raises(ValueError, match="candidate"):
        buffer.push(candidate(AudioSource.ME, "逆行", start_ms=100))


def test_buffer_rejects_invalid_factory_id_and_naive_now() -> None:
    invalid_id = ChronologicalCommitBuffer(
        segment_id_factory=lambda: "SEG-001",
        now=fixed_now,
    )
    invalid_id.push(candidate(AudioSource.ME, "内容", start_ms=100))
    with pytest.raises(ValueError, match="segment_id"):
        invalid_id.release_ready()

    naive_time = ChronologicalCommitBuffer(
        segment_id_factory=lambda: "01J00000000000000000000001",
        now=lambda: NOW.replace(tzinfo=None),
    )
    naive_time.push(candidate(AudioSource.ME, "内容", start_ms=100))
    with pytest.raises(ValueError, match="committed_at"):
        naive_time.release_ready()


def test_failed_record_validation_keeps_candidate_pending_for_retry() -> None:
    identifiers = iter(("SEG-001", "01J00000000000000000000001"))
    buffer = ChronologicalCommitBuffer(
        segment_id_factory=lambda: next(identifiers),
        now=fixed_now,
    )
    buffer.push(candidate(AudioSource.ME, "内容", start_ms=100))
    with pytest.raises(ValueError, match="segment_id"):
        buffer.release_ready()

    assert [record.text for record in buffer.release_ready()] == ["内容"]


def test_failed_finalize_can_retry_without_losing_pending_candidate() -> None:
    identifiers = iter(("SEG-001", "01J00000000000000000000001"))
    buffer = ChronologicalCommitBuffer(
        segment_id_factory=lambda: next(identifiers),
        now=fixed_now,
    )
    buffer.set_frontier(AudioSource.ME, 100)
    buffer.push(candidate(AudioSource.ME, "内容", start_ms=100))
    with pytest.raises(ValueError, match="segment_id"):
        buffer.finalize()

    assert [record.text for record in buffer.finalize()] == ["内容"]


def test_buffer_rejects_operations_after_finalize() -> None:
    buffer = make_commit_buffer()
    assert buffer.finalize() == ()

    with pytest.raises(ValueError, match="finalize"):
        buffer.push(candidate(AudioSource.ME, "遅い", start_ms=100))
    with pytest.raises(ValueError, match="finalize"):
        buffer.set_frontier(AudioSource.ME, 100)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  ",
        "[music]",
        " (APPLAUSE) ",
        "[laughter]",
        "(silence)",
        "[noise]",
        "[音楽]",
        "(拍手)",
        "[笑い]",
        "(無音)",
        "[雑音]",
    ],
)
def test_non_content_markers_are_filtered(text: str) -> None:
    assert is_transcript_content(text) is False


@pytest.mark.parametrize("text", ["えー", "えっと", "あの", "[music] 続き"])
def test_fillers_and_embedded_markers_remain_content(text: str) -> None:
    assert is_transcript_content(text) is True


def test_content_filter_rejects_non_string_input() -> None:
    with pytest.raises(ValueError, match="text"):
        is_transcript_content(cast(str, 1))
