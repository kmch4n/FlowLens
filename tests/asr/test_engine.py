"""Two-source ASR scheduling and transcript-boundary tests."""

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import pytest

from flowlens.asr.engine import AsrBatch, AsrEngine
from flowlens.asr.types import DecodedToken, DecodeHypothesis, PartialTranscript
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource

NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)


def hypothesis(text: str, start_ms: int = 0, end_ms: int = 500) -> DecodeHypothesis:
    """Build a one-token decoder result with literal relative bounds."""

    if not text:
        return DecodeHypothesis(())
    return DecodeHypothesis((DecodedToken(text, start_ms, end_ms),))


class RecordingDecoder:
    """Return deterministic hypotheses and record source-coded PCM calls."""

    def __init__(self, results: Iterable[DecodeHypothesis]) -> None:
        self._results: Iterator[DecodeHypothesis] = iter(results)
        self.decoded_sources: list[AudioSource] = []
        self.call_count = 0
        self.max_concurrent_calls = 0
        self._concurrent_calls = 0
        self._lock = Lock()

    @classmethod
    def repeat(cls, result: DecodeHypothesis) -> "RecordingDecoder":
        """Build a decoder with an effectively unbounded repeated result."""

        def repeated() -> Iterator[DecodeHypothesis]:
            while True:
                yield result

        return cls(repeated())

    def decode(self, pcm_s16le: bytes) -> DecodeHypothesis:
        """Decode source-coded PCM while measuring call serialization."""

        with self._lock:
            self._concurrent_calls += 1
            self.max_concurrent_calls = max(
                self.max_concurrent_calls,
                self._concurrent_calls,
            )
        try:
            self.call_count += 1
            source = (
                AudioSource.ME if pcm_s16le[:2] == b"\x01\x00" else AudioSource.OTHERS
            )
            self.decoded_sources.append(source)
            return next(self._results)
        finally:
            with self._lock:
                self._concurrent_calls -= 1


class PatternSpeechDetector:
    """Classify frames from a fixed pattern or a constant."""

    def __init__(self, pattern: bool | Iterable[bool]) -> None:
        self._constant = pattern if isinstance(pattern, bool) else None
        self._pattern = iter(()) if isinstance(pattern, bool) else iter(pattern)

    def is_speech(self, frame: AudioFrame) -> bool:
        """Return the next fake classification."""

        del frame
        if self._constant is not None:
            return self._constant
        return next(self._pattern)


def make_engine(
    *,
    decoder: RecordingDecoder,
    speech: bool | Iterable[bool],
) -> AsrEngine:
    """Build the production engine through its public constructor."""

    identifiers = (f"01J{'0' * 22}{index}" for index in range(1, 10))
    from flowlens.asr.types import AsrWorkerConfig

    return AsrEngine(
        config=AsrWorkerConfig(
            session_id="01J00000000000000000000000",
            model_path=Path("C:/models/kotoba"),
        ),
        decoder=decoder,
        speech_detector=PatternSpeechDetector(speech),
        segment_id_factory=lambda: next(identifiers),
        now=lambda: NOW,
    )


def frame(
    source: AudioSource,
    *,
    session_ms: int,
    captured_ms: int | None = None,
    source_sample: int | None = None,
) -> AudioFrame:
    """Build one canonical frame with source-coded PCM."""

    start_sample = session_ms * 16 if source_sample is None else source_sample
    sample = b"\x01\x00" if source is AudioSource.ME else b"\x02\x00"
    return AudioFrame(
        source=source,
        pcm_s16le=sample * 320,
        source_start_sample=start_sample,
        source_end_sample=start_sample + 320,
        session_start_ms=session_ms,
        captured_monotonic_ms=session_ms if captured_ms is None else captured_ms,
    )


def feed(
    engine: AsrEngine,
    source: AudioSource,
    *,
    frame_count: int,
    start_ms: int,
    start_sample: int = 0,
    captured_start_ms: int | None = None,
) -> None:
    """Feed consecutive canonical frames through the public API."""

    capture_start = start_ms if captured_start_ms is None else captured_start_ms
    for index in range(frame_count):
        engine.accept(
            frame(
                source,
                session_ms=start_ms + index * 20,
                captured_ms=capture_start + index * 20,
                source_sample=start_sample + index * 320,
            )
        )


def test_oldest_pending_source_is_decoded_first_with_one_shared_model() -> None:
    decoder = RecordingDecoder((hypothesis("先"), hypothesis("後")))
    engine = make_engine(decoder=decoder, speech=True)
    engine.accept(
        frame(AudioSource.OTHERS, session_ms=100, captured_ms=1_100, source_sample=0)
    )
    engine.accept(
        frame(AudioSource.ME, session_ms=80, captured_ms=1_080, source_sample=0)
    )
    engine.process_ready(now_monotonic_ms=1_600)
    assert decoder.decoded_sources == [AudioSource.ME, AudioSource.OTHERS]
    assert decoder.max_concurrent_calls == 1


def test_exact_scheduler_tie_prefers_me_and_does_not_starve_others() -> None:
    decoder = RecordingDecoder.repeat(hypothesis("発言"))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.OTHERS, frame_count=25, start_ms=0)
    feed(engine, AudioSource.ME, frame_count=25, start_ms=0)

    engine.process_ready(500)

    assert decoder.decoded_sources == [AudioSource.ME, AudioSource.OTHERS]


def test_scheduler_advances_each_source_to_its_oldest_undecoded_capture() -> None:
    decoder = RecordingDecoder.repeat(hypothesis("発言"))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=25, start_ms=0)
    engine.process_ready(500)

    feed(engine, AudioSource.OTHERS, frame_count=25, start_ms=100)
    feed(
        engine,
        AudioSource.ME,
        frame_count=25,
        start_ms=500,
        start_sample=8_000,
    )
    engine.process_ready(1_000)

    assert decoder.decoded_sources == [
        AudioSource.ME,
        AudioSource.OTHERS,
        AudioSource.ME,
    ]


def test_active_speech_decodes_at_five_hundred_ms_cadence() -> None:
    decoder = RecordingDecoder.repeat(hypothesis("発言"))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=24, start_ms=0)
    engine.process_ready(499)
    assert decoder.call_count == 0
    feed(engine, AudioSource.ME, frame_count=1, start_ms=480, start_sample=7_680)
    engine.process_ready(500)
    assert decoder.call_count == 1


def test_changed_partial_is_ephemeral_and_duplicate_text_is_suppressed() -> None:
    decoder = RecordingDecoder(
        (hypothesis("確認"), hypothesis("確認"), hypothesis("確認中"))
    )
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=50, start_ms=0)

    first = engine.process_ready(500)
    duplicate = engine.process_ready(1_000)
    changed = engine.process_ready(1_500)

    assert [partial.text for partial in first.partials + changed.partials] == [
        "確認",
        "確認中",
    ]
    assert duplicate.partials == ()
    assert first.committed + duplicate.committed + changed.committed == ()
    assert all(isinstance(item, PartialTranscript) for item in first.partials)


def test_separate_stable_adjacent_tokens_commit_non_overlapping_frame_spans() -> None:
    first = DecodeHypothesis((DecodedToken("A", 0, 13),))
    both = DecodeHypothesis(
        (
            DecodedToken("A", 0, 13),
            DecodedToken("B", 13, 30),
        )
    )
    decoder = RecordingDecoder((first, both, both))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=25, start_ms=0)

    engine.process_ready(500)
    stable_a = engine.process_ready(1_700)
    stable_b = engine.process_ready(2_900)

    records = stable_a.committed + stable_b.committed
    assert [
        (item.text, item.session_start_ms, item.session_end_ms) for item in records
    ] == [("A", 0, 20), ("B", 20, 40)]


def test_stable_token_beyond_buffer_waits_for_additional_audio() -> None:
    first = DecodeHypothesis((DecodedToken("A", 0, 500),))
    both = DecodeHypothesis(
        (
            DecodedToken("A", 0, 500),
            DecodedToken("B", 500, 600),
        )
    )
    decoder = RecordingDecoder((first, first, both, both, both))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=25, start_ms=0)

    engine.process_ready(500)
    stable_a = engine.process_ready(1_700)
    engine.process_ready(2_200)
    no_room = engine.process_ready(3_400)

    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (stable_a.committed)
    ] == [("A", 0, 500)]
    assert no_room.committed == ()

    feed(
        engine,
        AudioSource.ME,
        frame_count=5,
        start_ms=500,
        start_sample=8_000,
    )
    after_audio = engine.process_ready(4_600)

    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (after_audio.committed)
    ] == [("B", 500, 600)]


def test_post_boundary_deferred_token_is_redecoded_from_hard_split_tail() -> None:
    prefix = DecodeHypothesis((DecodedToken("A。", 0, 10_000),))
    with_deferred = DecodeHypothesis(
        (
            DecodedToken("A。", 0, 10_000),
            DecodedToken("B", 10_500, 11_000),
        )
    )
    tail = DecodeHypothesis((DecodedToken("B", 500, 1_000),))
    decoder = RecordingDecoder(
        (prefix, prefix, with_deferred, with_deferred, with_deferred, tail)
    )
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=500, start_ms=0)

    engine.process_ready(500)
    stable_prefix = engine.process_ready(1_700)
    engine.process_ready(2_200)
    engine.process_ready(3_400)
    feed(
        engine,
        AudioSource.ME,
        frame_count=100,
        start_ms=10_000,
        start_sample=160_000,
    )

    split = engine.process_ready(12_000)
    final = engine.finalize(13_000)

    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (stable_prefix.committed)
    ] == [("A。", 0, 10_000)]
    assert split.committed == ()
    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (final.committed)
    ] == [("B", 10_500, 11_000)]


def test_token_crossing_hard_split_boundary_is_clamped_before_tail_redecode() -> None:
    split_hypothesis = DecodeHypothesis(
        (
            DecodedToken("境界。", 0, 10_010),
            DecodedToken("後", 10_010, 12_000),
        )
    )
    tail_hypothesis = DecodeHypothesis((DecodedToken("後", 10, 2_000),))
    engine = make_engine(
        decoder=RecordingDecoder((split_hypothesis, tail_hypothesis)),
        speech=True,
    )
    feed(engine, AudioSource.ME, frame_count=600, start_ms=0)

    split = engine.process_ready(12_000)
    tail = engine.finalize(13_000)

    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (split.committed)
    ] == [("境界。", 0, 10_000)]
    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (tail.committed)
    ] == [("後", 10_000, 12_000)]


def test_hard_split_never_moves_before_an_immutable_commit_frontier() -> None:
    old = DecodeHypothesis((DecodedToken("OLD", 0, 11_000),))
    revised = DecodeHypothesis(
        (
            DecodedToken("P。", 0, 10_000),
            DecodedToken("TAIL", 10_000, 12_000),
        )
    )
    tail = DecodeHypothesis((DecodedToken("TAIL", 0, 1_000),))
    decoder = RecordingDecoder((old, old, revised, tail))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=550, start_ms=0)

    engine.process_ready(500)
    stable_old = engine.process_ready(1_700)
    feed(
        engine,
        AudioSource.ME,
        frame_count=50,
        start_ms=11_000,
        start_sample=176_000,
    )

    split = engine.process_ready(12_000)
    final = engine.finalize(13_000)

    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (stable_old.committed)
    ] == [("OLD", 0, 11_000)]
    assert split.committed == ()
    assert [
        (item.text, item.session_start_ms, item.session_end_ms)
        for item in (final.committed)
    ] == [("TAIL", 11_000, 12_000)]


def test_silence_end_runs_final_decode_commits_and_clears_partial() -> None:
    decoder = RecordingDecoder((hypothesis("確認"), hypothesis("確認します。")))
    engine = make_engine(decoder=decoder, speech=[True] * 10 + [False] * 23)
    feed(engine, AudioSource.ME, frame_count=10, start_ms=0)
    first = engine.process_ready(500)
    feed(engine, AudioSource.ME, frame_count=23, start_ms=200, start_sample=3_200)

    ended = engine.process_ready(660)

    assert [partial.text for partial in first.partials] == ["確認"]
    assert [record.text for record in ended.committed] == ["確認します。"]
    assert [partial.text for partial in ended.partials] == [""]


def test_silence_end_requires_exactly_twenty_three_consecutive_frames() -> None:
    decoder = RecordingDecoder((hypothesis("終了"),))
    engine = make_engine(decoder=decoder, speech=[True] * 10 + [False] * 23)
    feed(engine, AudioSource.ME, frame_count=32, start_ms=0)

    before_boundary = engine.process_ready(440)
    assert before_boundary.committed == ()
    assert decoder.call_count == 0

    engine.accept(
        frame(AudioSource.ME, session_ms=640, captured_ms=640, source_sample=10_240)
    )
    at_boundary = engine.process_ready(660)
    assert [record.text for record in at_boundary.committed] == ["終了"]


@pytest.mark.parametrize("text", ["えっと", "えー"])
def test_filler_only_text_is_committed(text: str) -> None:
    engine = make_engine(decoder=RecordingDecoder((hypothesis(text),)), speech=True)
    feed(engine, AudioSource.ME, frame_count=1, start_ms=0)
    assert [record.text for record in engine.finalize(1_000).committed] == [text]


def test_default_wall_clock_produces_valid_millisecond_commit_time() -> None:
    from flowlens.asr.types import AsrWorkerConfig

    engine = AsrEngine(
        config=AsrWorkerConfig(
            session_id="01J00000000000000000000000",
            model_path=Path("C:/models/kotoba"),
        ),
        decoder=RecordingDecoder((hypothesis("内容"),)),
        speech_detector=PatternSpeechDetector(True),
        segment_id_factory=lambda: "01J00000000000000000000001",
    )
    engine.accept(frame(AudioSource.ME, session_ms=0, source_sample=0))

    record = engine.finalize(1_000).committed[0]

    assert record.committed_at.microsecond % 1_000 == 0


@pytest.mark.parametrize("text", ["[音楽]", " [music] ", ""])
def test_exact_non_speech_or_blank_content_is_discarded(text: str) -> None:
    normalized = text.strip()
    engine = make_engine(
        decoder=RecordingDecoder((hypothesis(normalized),)),
        speech=True,
    )
    feed(engine, AudioSource.ME, frame_count=1, start_ms=0)
    assert engine.finalize(1_000).committed == ()


def test_continuous_speech_splits_at_frame_aligned_hypothesis_boundary() -> None:
    split_hypothesis = DecodeHypothesis(
        (
            DecodedToken("前半。", 0, 10_410),
            DecodedToken("後半", 10_410, 12_000),
        )
    )
    engine = make_engine(decoder=RecordingDecoder((split_hypothesis,)), speech=True)
    feed(engine, AudioSource.ME, frame_count=600, start_ms=0)

    batch = engine.process_ready(12_000)

    assert [record.text for record in batch.committed] == ["前半。"]
    assert batch.committed[0].session_end_ms == 10_400
    assert batch.committed[0].source_end_sample == 166_400


def test_hard_split_requires_six_hundred_frames_not_five_hundred_ninety_nine() -> None:
    decoder = RecordingDecoder((hypothesis("継続。", end_ms=12_000),))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=599, start_ms=0)
    assert engine.process_ready(499).committed == ()
    assert decoder.call_count == 0

    engine.accept(
        frame(
            AudioSource.ME,
            session_ms=11_980,
            captured_ms=11_980,
            source_sample=191_680,
        )
    )
    assert engine.process_ready(12_000).committed
    assert decoder.call_count == 1


def test_overlapping_sources_produce_separate_globally_ordered_records() -> None:
    decoder = RecordingDecoder((hypothesis("自分"), hypothesis("他者")))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.OTHERS, frame_count=1, start_ms=120)
    feed(engine, AudioSource.ME, frame_count=1, start_ms=100)

    records = engine.finalize(2_000).committed

    assert [(record.source, record.text, record.sequence) for record in records] == [
        (AudioSource.ME, "自分", 1),
        (AudioSource.OTHERS, "他者", 2),
    ]


def test_finalize_commits_both_sources_once_and_rejects_later_input() -> None:
    decoder = RecordingDecoder((hypothesis("自分"), hypothesis("他者")))
    engine = make_engine(decoder=decoder, speech=True)
    feed(engine, AudioSource.ME, frame_count=1, start_ms=100)
    feed(engine, AudioSource.OTHERS, frame_count=1, start_ms=120)

    first = engine.finalize(2_000)
    second = engine.finalize(2_100)

    assert len(first.committed) == 2
    assert second == AsrBatch((), ())
    with pytest.raises(ValueError, match="finalized"):
        engine.accept(frame(AudioSource.ME, session_ms=120, source_sample=320))


@pytest.mark.parametrize(
    ("session_ms", "source_sample", "captured_ms"),
    [
        (20, 0, 20),
        (20, 160, 20),
        (20, 640, 20),
        (0, 320, 20),
        (20, 320, 0),
    ],
)
def test_accept_rejects_duplicate_overlap_gap_and_retrograde_timestamps(
    session_ms: int,
    source_sample: int,
    captured_ms: int,
) -> None:
    engine = make_engine(
        decoder=RecordingDecoder.repeat(hypothesis("発言")), speech=True
    )
    engine.accept(frame(AudioSource.ME, session_ms=0, captured_ms=0, source_sample=0))

    with pytest.raises(ValueError):
        engine.accept(
            frame(
                AudioSource.ME,
                session_ms=session_ms,
                captured_ms=captured_ms,
                source_sample=source_sample,
            )
        )


def test_sources_validate_offsets_independently_and_allow_pause_gaps() -> None:
    engine = make_engine(
        decoder=RecordingDecoder.repeat(hypothesis("発言")), speech=True
    )
    engine.accept(frame(AudioSource.ME, session_ms=0, source_sample=0))
    engine.accept(frame(AudioSource.OTHERS, session_ms=0, source_sample=0))
    engine.accept(
        frame(
            AudioSource.ME,
            session_ms=2_020,
            captured_ms=2_020,
            source_sample=320,
        )
    )


def test_first_source_frame_must_begin_at_zero_samples() -> None:
    engine = make_engine(
        decoder=RecordingDecoder.repeat(hypothesis("発言")),
        speech=True,
    )

    with pytest.raises(ValueError, match="source sample offsets"):
        engine.accept(frame(AudioSource.ME, session_ms=100, source_sample=320))


def test_token_times_are_clamped_to_available_frame_boundaries() -> None:
    decoded = hypothesis("境界", start_ms=13, end_ms=999)
    engine = make_engine(decoder=RecordingDecoder((decoded,)), speech=True)
    feed(engine, AudioSource.ME, frame_count=10, start_ms=100, start_sample=0)

    record = engine.finalize(1_000).committed[0]

    assert (record.session_start_ms, record.session_end_ms) == (100, 300)
    assert (record.source_start_sample, record.source_end_sample) == (0, 3_200)


def test_token_session_bounds_follow_frame_timeline_across_pause_gap() -> None:
    decoded = hypothesis("再開", start_ms=20, end_ms=40)
    engine = make_engine(decoder=RecordingDecoder((decoded,)), speech=True)
    engine.accept(frame(AudioSource.ME, session_ms=0, source_sample=0))
    engine.accept(
        frame(
            AudioSource.ME,
            session_ms=2_020,
            captured_ms=2_020,
            source_sample=320,
        )
    )

    record = engine.finalize(3_000).committed[0]

    assert (record.session_start_ms, record.session_end_ms) == (2_020, 2_040)
    assert (record.source_start_sample, record.source_end_sample) == (320, 640)


def test_backlog_is_oldest_pending_age_nonnegative_and_zero_after_finalize() -> None:
    engine = make_engine(
        decoder=RecordingDecoder.repeat(hypothesis("発言")), speech=True
    )
    engine.accept(
        frame(AudioSource.OTHERS, session_ms=0, captured_ms=1_100, source_sample=0)
    )
    engine.accept(
        frame(AudioSource.ME, session_ms=0, captured_ms=1_000, source_sample=0)
    )

    assert engine.backlog_ms(1_250) == 250
    assert engine.backlog_ms(900) == 0
    engine.finalize(1_300)
    assert engine.backlog_ms(2_000) == 0


def test_public_times_reject_boolean_negative_and_retrograde_values() -> None:
    engine = make_engine(
        decoder=RecordingDecoder.repeat(hypothesis("発言")), speech=True
    )
    with pytest.raises(ValueError, match="now_monotonic_ms"):
        engine.process_ready(-1)
    with pytest.raises(ValueError, match="now_monotonic_ms"):
        engine.backlog_ms(True)
    engine.process_ready(100)
    with pytest.raises(ValueError, match="backwards"):
        engine.process_ready(99)


def test_asr_batch_is_immutable_and_rejects_non_tuple_payloads() -> None:
    batch = AsrBatch((), ())
    with pytest.raises(AttributeError):
        batch.partials = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="partials"):
        AsrBatch([], ())  # type: ignore[arg-type]
