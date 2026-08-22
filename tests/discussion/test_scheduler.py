"""Behavioral tests for deterministic discussion-request scheduling."""

import pickle
import random
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from flowlens.discussion.contracts import DiscussionRequest
from flowlens.discussion.scheduler import DiscussionScheduler
from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import SessionMode
from flowlens.domain.messages import TranscriptRecord
from tests.discussion.factories import NOW, make_record, make_state


def make_scheduler() -> DiscussionScheduler:
    """Build a scheduler with a deterministic initial state."""

    return DiscussionScheduler(make_state())


def record_with_text(text: str, *, sequence: int = 1) -> TranscriptRecord:
    """Build an adversarial domain instance for unreachable blank ASR text."""

    record = make_record(sequence=sequence)
    object.__setattr__(record, "text", text)
    return record


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "[音楽]",
        "[音楽]。",
        "(無音)",
        "（無音）……",  # noqa: RUF001
        "えー",
        " えっと。 ",
        "あー……",
        "あの!",
    ],
)
def test_non_meaningful_text_does_not_schedule(text: str) -> None:
    scheduler = make_scheduler()

    scheduler.add(record_with_text(text), now_ms=0)

    assert scheduler.has_pending is False


@pytest.mark.parametrize(
    "text",
    [
        "はい",
        "いいえ",
        "そうです",
        "えっと、結論は進めます",
        "あの件",
        "えーと考えています",
    ],
)
def test_short_answers_and_meaningful_sentences_schedule(text: str) -> None:
    scheduler = make_scheduler()

    scheduler.add(make_record(sequence=1, text=text), now_ms=0)

    assert scheduler.has_pending is True


def test_coalesces_from_latest_meaningful_commit_at_exact_boundary() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="方針を確認します"), now_ms=100)
    scheduler.add(make_record(sequence=2, text="はい"), now_ms=450)

    assert scheduler.next_request(now_ms=949, updated_at=NOW) is None
    request = scheduler.next_request(now_ms=950, updated_at=NOW)

    assert request is not None
    assert [record.sequence for record in request.records] == [1, 2]
    assert request.requested_revision == 1
    assert request.current_state == make_state()


def test_meaningless_commit_does_not_move_coalescing_deadline() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="方針です"), now_ms=100)
    scheduler.add(record_with_text("えっと。", sequence=2), now_ms=499)

    request = scheduler.next_request(now_ms=600, updated_at=NOW)

    assert request is not None
    assert [record.sequence for record in request.records] == [1]


def test_allows_only_one_request_and_queues_new_records_for_next_batch() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="一件目"), now_ms=0)
    first = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert first is not None

    scheduler.add(make_record(sequence=2, text="二件目"), now_ms=501)

    assert scheduler.next_request(now_ms=5_000, updated_at=NOW) is None
    scheduler.succeed(first, make_state(revision=1))
    second = scheduler.next_request(now_ms=5_000, updated_at=NOW)
    assert second is not None
    assert [record.sequence for record in second.records] == [2]
    assert second.requested_revision == 2


def test_failure_retains_batch_but_waits_for_next_meaningful_commit() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="論点です"), now_ms=0)
    failed = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert failed is not None

    scheduler.fail(failed)

    assert scheduler.has_pending is True
    assert scheduler.current_state.revision == 0
    assert scheduler.next_request(now_ms=10_000, updated_at=NOW) is None
    scheduler.add(record_with_text("えー", sequence=2), now_ms=10_001)
    assert scheduler.next_request(now_ms=20_000, updated_at=NOW) is None
    scheduler.add(make_record(sequence=3, text="補足です"), now_ms=20_001)
    assert scheduler.next_request(now_ms=20_500, updated_at=NOW) is None
    retry = scheduler.next_request(now_ms=20_501, updated_at=NOW)
    assert retry is not None
    assert [record.sequence for record in retry.records] == [1, 3]


def test_success_replaces_state_and_removes_only_completed_batch() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="一件目"), now_ms=0)
    request = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert request is not None
    scheduler.add(make_record(sequence=2, text="二件目"), now_ms=501)

    replacement = make_state(revision=1)
    scheduler.succeed(request, replacement)

    assert scheduler.current_state is replacement
    assert scheduler.has_pending is True
    next_request = scheduler.next_request(now_ms=1_001, updated_at=NOW)
    assert next_request is not None
    assert [record.sequence for record in next_request.records] == [2]


@pytest.mark.parametrize(
    "replacement",
    [
        make_state(revision=0),
        make_state(revision=2),
        make_state(mode=SessionMode.INTERVIEW, revision=1),
    ],
)
def test_invalid_success_retains_batch_and_requires_new_commit(
    replacement: DiscussionState,
) -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)
    request = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert request is not None

    with pytest.raises(ValueError, match="replacement"):
        scheduler.succeed(request, replacement)

    assert scheduler.current_state.revision == 0
    assert scheduler.has_pending is True
    assert scheduler.next_request(now_ms=5_000, updated_at=NOW) is None
    scheduler.add(make_record(sequence=2), now_ms=5_001)
    assert scheduler.next_request(now_ms=5_501, updated_at=NOW) is not None


@pytest.mark.parametrize(
    ("field_name", "corrupt_value"),
    [
        pytest.param("revision", True, id="boolean-revision"),
        pytest.param("revision", 1.0, id="float-revision"),
        pytest.param("mode", "MEETING", id="string-mode"),
        pytest.param("current_focus", 7, id="non-string-focus"),
        pytest.param("key_points", ["mutable"], id="mutable-key-points"),
        pytest.param("key_points", ("valid", 7), id="invalid-key-point-item"),
        pytest.param(
            "confirmed_outcomes",
            ["mutable"],
            id="mutable-confirmed-outcomes",
        ),
        pytest.param(
            "follow_up_items",
            ("valid", object()),
            id="invalid-follow-up-item",
        ),
        pytest.param(
            "updated_at",
            datetime.fromisoformat("2026-08-19T12:35:02.125"),
            id="naive-timestamp",
        ),
    ],
)
def test_corrupted_replacement_is_revalidated_before_batch_removal(
    field_name: str,
    corrupt_value: object,
) -> None:
    scheduler = make_scheduler()
    initial_state = scheduler.current_state
    scheduler.add(make_record(sequence=1), now_ms=0)
    request = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert request is not None
    replacement = make_state(revision=1)
    object.__setattr__(replacement, field_name, corrupt_value)

    with pytest.raises(ValueError, match="replacement"):
        scheduler.succeed(request, replacement)

    assert scheduler.current_state is initial_state
    assert scheduler.has_pending is True
    assert scheduler.next_request(now_ms=10_000, updated_at=NOW) is None
    scheduler.add(make_record(sequence=2), now_ms=10_001)
    retry = scheduler.next_request(now_ms=10_501, updated_at=NOW)
    assert retry is not None
    assert [record.sequence for record in retry.records] == [1, 2]


def test_foreign_equal_request_cannot_complete_or_fail_in_flight_work() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)
    request = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert request is not None
    foreign = DiscussionRequest(
        current_state=request.current_state,
        records=request.records,
        requested_revision=request.requested_revision,
        updated_at=request.updated_at,
    )
    assert foreign == request
    assert foreign is not request

    with pytest.raises(ValueError, match="in-flight"):
        scheduler.fail(foreign)
    with pytest.raises(ValueError, match="in-flight"):
        scheduler.succeed(foreign, make_state(revision=1))

    assert scheduler.next_request(now_ms=1_000, updated_at=NOW) is None
    scheduler.succeed(request, make_state(revision=1))
    assert scheduler.current_state.revision == 1


def test_completed_request_cannot_be_replayed() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)
    request = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert request is not None
    scheduler.succeed(request, make_state(revision=1))

    with pytest.raises(ValueError, match="in-flight"):
        scheduler.succeed(request, make_state(revision=1))
    with pytest.raises(ValueError, match="in-flight"):
        scheduler.fail(request)


def test_pause_blocks_launch_without_resetting_deadline() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=100)
    scheduler.set_paused(True)

    assert scheduler.next_request(now_ms=10_000, updated_at=NOW) is None
    scheduler.set_paused(True)
    scheduler.set_paused(False)
    scheduler.set_paused(False)

    assert scheduler.next_request(now_ms=10_000, updated_at=NOW) is not None


def test_pause_before_deadline_preserves_remaining_coalescing_time() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=100)
    scheduler.set_paused(True)
    scheduler.set_paused(False)

    assert scheduler.next_request(now_ms=599, updated_at=NOW) is None
    assert scheduler.next_request(now_ms=600, updated_at=NOW) is not None


def test_final_request_bypasses_delay_is_single_flight_and_idempotent() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=100)

    request = scheduler.final_request(updated_at=NOW)

    assert request is not None
    assert [record.sequence for record in request.records] == [1]
    assert scheduler.final_request(updated_at=NOW) is None
    assert scheduler.next_request(now_ms=10_000, updated_at=NOW) is None
    scheduler.succeed(request, make_state(revision=1))
    assert scheduler.final_request(updated_at=NOW) is None


def test_failed_final_request_retains_state_and_needs_new_commit() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)
    request = scheduler.final_request(updated_at=NOW)
    assert request is not None

    scheduler.fail(request)

    assert scheduler.current_state.revision == 0
    assert scheduler.has_pending is True
    assert scheduler.final_request(updated_at=NOW) is None
    scheduler.add(make_record(sequence=2), now_ms=1)
    retry = scheduler.final_request(updated_at=NOW)
    assert retry is not None
    assert [record.sequence for record in retry.records] == [1, 2]


def test_final_request_respects_pause() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)
    scheduler.set_paused(True)

    assert scheduler.final_request(updated_at=NOW) is None
    assert scheduler.has_pending is True


@pytest.mark.parametrize("now_ms", [-1, True, 1.5, "500"])
def test_time_arguments_reject_invalid_values_without_consuming_record(
    now_ms: object,
) -> None:
    scheduler = make_scheduler()
    record = make_record(sequence=1)

    with pytest.raises(ValueError, match="now_ms"):
        scheduler.add(record, now_ms=now_ms)  # type: ignore[arg-type]

    scheduler.add(record, now_ms=0)
    assert scheduler.has_pending is True


def test_clock_rollback_is_rejected_without_mutating_state() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=100)

    with pytest.raises(ValueError, match="monotonic"):
        scheduler.next_request(now_ms=99, updated_at=NOW)

    assert scheduler.next_request(now_ms=600, updated_at=NOW) is not None


def test_duplicate_and_retrograde_sequences_are_rejected_transactionally() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=2), now_ms=100)

    with pytest.raises(ValueError, match="sequence"):
        scheduler.add(make_record(sequence=2), now_ms=101)
    with pytest.raises(ValueError, match="sequence"):
        scheduler.add(make_record(sequence=1), now_ms=102)

    request = scheduler.next_request(now_ms=600, updated_at=NOW)
    assert request is not None
    assert [record.sequence for record in request.records] == [2]


def test_rejected_clock_rollback_does_not_consume_sequence() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=100)

    with pytest.raises(ValueError, match="monotonic"):
        scheduler.add(make_record(sequence=2), now_ms=99)

    scheduler.add(make_record(sequence=2), now_ms=101)
    assert scheduler.next_request(now_ms=601, updated_at=NOW) is not None


def test_arguments_are_strict_and_do_not_accept_bool_or_foreign_values() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        DiscussionScheduler(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coalesce_ms"):
        DiscussionScheduler(make_state(), coalesce_ms=True)
    scheduler = make_scheduler()
    with pytest.raises(ValueError, match="record"):
        scheduler.add(object(), now_ms=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="paused"):
        scheduler.set_paused(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="request"):
        scheduler.fail(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="request"):
        scheduler.succeed(object(), make_state(revision=1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "updated_at",
    [datetime.fromisoformat("2026-08-19T12:35:02.125"), object()],
)
def test_invalid_request_timestamp_leaves_pending_launchable(
    updated_at: object,
) -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)

    with pytest.raises(ValueError, match="updated_at"):
        scheduler.next_request(
            now_ms=500,
            updated_at=updated_at,  # type: ignore[arg-type]
        )

    assert scheduler.next_request(now_ms=500, updated_at=NOW) is not None


def test_request_and_state_are_immutable_and_scheduler_survives_pickle() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1), now_ms=0)
    restored = pickle.loads(pickle.dumps(scheduler))
    request = restored.next_request(now_ms=500, updated_at=NOW)
    assert request is not None

    with pytest.raises(FrozenInstanceError):
        request.requested_revision = 99
    with pytest.raises(FrozenInstanceError):
        restored.current_state.revision = 99

    restored.succeed(request, make_state(revision=1))
    assert restored.has_pending is False


def test_randomized_state_machine_never_loses_or_duplicates_records() -> None:
    rng = random.Random(20260822)
    scheduler = make_scheduler()
    accepted: list[int] = []
    completed: list[int] = []
    sequence = 0
    now_ms = 0
    in_flight: DiscussionRequest | None = None

    for _ in range(2_000):
        action = rng.randrange(6)
        now_ms += rng.randrange(0, 700)
        if action <= 1:
            sequence += 1
            meaningful = rng.choice([True, True, False])
            text = "意味があります" if meaningful else "えっと。"
            scheduler.add(record_with_text(text, sequence=sequence), now_ms=now_ms)
            if meaningful:
                accepted.append(sequence)
        elif action == 2:
            candidate = scheduler.next_request(now_ms=now_ms, updated_at=NOW)
            if in_flight is None and candidate is not None:
                in_flight = candidate
        elif action == 3 and in_flight is not None:
            batch = [record.sequence for record in in_flight.records]
            scheduler.succeed(
                in_flight,
                make_state(revision=scheduler.current_state.revision + 1),
            )
            completed.extend(batch)
            accepted = [item for item in accepted if item not in set(batch)]
            in_flight = None
        elif action == 4 and in_flight is not None:
            scheduler.fail(in_flight)
            in_flight = None
        else:
            scheduler.set_paused(rng.choice([True, False]))

        assert len(completed) == len(set(completed))
        assert set(completed).isdisjoint(accepted)
        assert scheduler.has_pending is bool(accepted)
        assert scheduler.current_state.revision >= 0
