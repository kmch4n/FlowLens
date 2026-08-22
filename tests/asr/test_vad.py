"""Contract tests for ASR-local records, ports, and VAD boundaries."""

import pickle
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from flowlens.asr.ports import DecoderPort, SpeechDetectorPort
from flowlens.asr.types import (
    AsrWorkerConfig,
    CommitCandidate,
    DecodedToken,
    DecodeHypothesis,
    PartialTranscript,
)
from flowlens.asr.vad import UtteranceBoundaryTracker, WebRtcSpeechDetector
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource


def make_frame(*, source: AudioSource = AudioSource.ME) -> AudioFrame:
    """Build one valid canonical frame."""

    return AudioFrame(source, bytes(640), 0, 320, 0, 1_000)


def make_config(model_path: Path) -> AsrWorkerConfig:
    """Build one valid ASR worker configuration."""

    return AsrWorkerConfig(
        session_id="01J00000000000000000000000",
        model_path=model_path,
    )


def test_utterance_ends_after_twenty_three_silent_frames() -> None:
    tracker = UtteranceBoundaryTracker(silence_end_ms=450, max_utterance_ms=12_000)
    assert tracker.observe(is_speech=True) == "ACTIVE"
    for _ in range(22):
        assert tracker.observe(is_speech=False) == "ACTIVE"
    assert tracker.observe(is_speech=False) == "END"
    assert tracker.active_duration_ms == 0


def test_continuous_speech_forces_split_at_twelve_seconds() -> None:
    tracker = UtteranceBoundaryTracker(silence_end_ms=450, max_utterance_ms=12_000)
    for frame_number in range(1, 600):
        assert tracker.observe(is_speech=True) == "ACTIVE"
        assert tracker.active_duration_ms == frame_number * 20
    assert tracker.observe(is_speech=True) == "HARD_SPLIT"
    assert tracker.active_duration_ms == 0


def test_leading_silence_is_ignored_and_speech_resets_silence_run() -> None:
    tracker = UtteranceBoundaryTracker(silence_end_ms=450, max_utterance_ms=12_000)
    for _ in range(30):
        assert tracker.observe(is_speech=False) == "INACTIVE"
    assert tracker.active_duration_ms == 0

    assert tracker.observe(is_speech=True) == "ACTIVE"
    for _ in range(22):
        assert tracker.observe(is_speech=False) == "ACTIVE"
    assert tracker.observe(is_speech=True) == "ACTIVE"
    for _ in range(22):
        assert tracker.observe(is_speech=False) == "ACTIVE"
    assert tracker.observe(is_speech=False) == "END"


@pytest.mark.parametrize(
    ("silence_end_ms", "max_utterance_ms", "field_name"),
    [
        (0, 12_000, "silence_end_ms"),
        (True, 12_000, "silence_end_ms"),
        (450, 0, "max_utterance_ms"),
        (450, True, "max_utterance_ms"),
        (12_001, 12_000, "silence_end_ms"),
    ],
)
def test_boundary_tracker_rejects_invalid_thresholds(
    silence_end_ms: object,
    max_utterance_ms: object,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        UtteranceBoundaryTracker(
            silence_end_ms=cast(int, silence_end_ms),
            max_utterance_ms=cast(int, max_utterance_ms),
        )


class FakeVad:
    """Record calls made by the production WebRTC adapter."""

    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls: list[tuple[bytes, int]] = []

    def is_speech(self, pcm_s16le: bytes, sample_rate_hz: int) -> bool:
        self.calls.append((pcm_s16le, sample_rate_hz))
        return self.result


def test_webrtc_detector_constructs_mode_two_and_calls_sixteen_kilohertz() -> None:
    fake = FakeVad(result=True)
    modes: list[int] = []

    def factory(mode: int) -> FakeVad:
        modes.append(mode)
        return fake

    detector = WebRtcSpeechDetector(vad_factory=factory)
    frame = make_frame(source=AudioSource.OTHERS)

    assert detector.is_speech(frame) is True
    assert modes == [2]
    assert fake.calls == [(frame.pcm_s16le, 16_000)]


def test_webrtc_detector_rejects_noncanonical_frames_before_vad_call() -> None:
    fake = FakeVad(result=False)
    detector = WebRtcSpeechDetector(vad_factory=lambda mode: fake)
    malformed = object.__new__(AudioFrame)
    object.__setattr__(malformed, "source", AudioSource.ME)
    object.__setattr__(malformed, "pcm_s16le", bytes(638))
    object.__setattr__(malformed, "source_start_sample", 0)
    object.__setattr__(malformed, "source_end_sample", 319)
    object.__setattr__(malformed, "session_start_ms", 0)
    object.__setattr__(malformed, "captured_monotonic_ms", 0)

    with pytest.raises(ValueError, match="640-byte"):
        detector.is_speech(malformed)
    assert fake.calls == []


@pytest.mark.parametrize("mode", [0, 1, 3, True])
def test_webrtc_detector_rejects_modes_other_than_two(mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        WebRtcSpeechDetector(
            mode=cast(int, mode),
            vad_factory=lambda selected_mode: FakeVad(False),
        )


def test_asr_records_are_exact_frozen_slotted_and_pickle_safe(tmp_path: Path) -> None:
    token = DecodedToken(" 方針", 0, 400)
    hypothesis = DecodeHypothesis((token, DecodedToken("です ", 400, 600)))
    partial = PartialTranscript(AudioSource.ME, "方針です", 10, 610, 0, 9_600)
    committed = CommitCandidate(
        AudioSource.OTHERS,
        "確認します。",
        20,
        620,
        320,
        9_920,
        datetime(2026, 8, 22, 1, 2, 3, 456000, UTC),
    )
    config = make_config(tmp_path.resolve())

    assert hypothesis.text == "方針です"
    assert pickle.loads(pickle.dumps((hypothesis, partial, committed, config))) == (
        hypothesis,
        partial,
        committed,
        config,
    )
    assert [field.name for field in fields(DecodedToken)] == [
        "text",
        "start_ms",
        "end_ms",
    ]
    assert [field.name for field in fields(DecodeHypothesis)] == ["tokens"]
    assert [field.name for field in fields(PartialTranscript)] == [
        "source",
        "text",
        "session_start_ms",
        "session_end_ms",
        "source_start_sample",
        "source_end_sample",
    ]
    assert [field.name for field in fields(CommitCandidate)] == [
        "source",
        "text",
        "session_start_ms",
        "session_end_ms",
        "source_start_sample",
        "source_end_sample",
        "committed_at",
    ]
    with pytest.raises((FrozenInstanceError, TypeError)):
        token.text = "変更"  # type: ignore[misc]


def test_decode_hypothesis_requires_an_immutable_ordered_token_tuple() -> None:
    token = DecodedToken("発言", 0, 200)
    with pytest.raises(ValueError, match="tokens"):
        DecodeHypothesis(cast(tuple[DecodedToken, ...], [token]))
    with pytest.raises(ValueError, match="tokens"):
        DecodeHypothesis((cast(DecodedToken, "token"),))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("text", ""),
        ("text", "   "),
        ("text", 7),
        ("start_ms", -1),
        ("start_ms", True),
        ("end_ms", -1),
        ("end_ms", True),
    ],
)
def test_decoded_token_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {"text": "発言", "start_ms": 0, "end_ms": 200}
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        DecodedToken(**values)  # type: ignore[arg-type]


def test_decoded_token_rejects_reversed_span() -> None:
    with pytest.raises(ValueError, match="end_ms"):
        DecodedToken("発言", 201, 200)


@pytest.mark.parametrize("record_type", [PartialTranscript, CommitCandidate])
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", "ME"),
        ("text", 1),
        ("session_start_ms", -1),
        ("session_start_ms", True),
        ("session_end_ms", -1),
        ("source_start_sample", -1),
        ("source_end_sample", True),
    ],
)
def test_transcript_candidates_reject_invalid_fields(
    record_type: type[PartialTranscript] | type[CommitCandidate],
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "source": AudioSource.ME,
        "text": "発言",
        "session_start_ms": 0,
        "session_end_ms": 200,
        "source_start_sample": 0,
        "source_end_sample": 3_200,
    }
    if record_type is CommitCandidate:
        values["committed_at"] = datetime.now(UTC).replace(microsecond=0)
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        record_type(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("record_type", "text"),
    [
        (PartialTranscript, " 発言"),
        (PartialTranscript, "発言 "),
        (CommitCandidate, ""),
        (CommitCandidate, " 発言"),
    ],
)
def test_transcript_text_boundaries_are_rejected_without_rewriting(
    record_type: type[PartialTranscript] | type[CommitCandidate],
    text: str,
) -> None:
    values: dict[str, object] = {
        "source": AudioSource.ME,
        "text": text,
        "session_start_ms": 0,
        "session_end_ms": 200,
        "source_start_sample": 0,
        "source_end_sample": 3_200,
    }
    if record_type is CommitCandidate:
        values["committed_at"] = datetime.now(UTC).replace(microsecond=0)
    with pytest.raises(ValueError, match="text"):
        record_type(**values)  # type: ignore[arg-type]


def test_empty_partial_is_valid_for_clearing_ephemeral_text() -> None:
    partial = PartialTranscript(AudioSource.ME, "", 0, 200, 0, 3_200)
    assert partial.text == ""


@pytest.mark.parametrize(
    ("session_start_ms", "session_end_ms", "source_start", "source_end"),
    [(200, 200, 0, 3_200), (201, 200, 0, 3_200), (0, 200, 10, 10), (0, 200, 11, 10)],
)
def test_transcript_records_require_forward_spans(
    session_start_ms: int,
    session_end_ms: int,
    source_start: int,
    source_end: int,
) -> None:
    with pytest.raises(ValueError, match="end"):
        PartialTranscript(
            AudioSource.ME,
            "発言",
            session_start_ms,
            session_end_ms,
            source_start,
            source_end,
        )


@pytest.mark.parametrize(
    "committed_at",
    [
        datetime(2026, 8, 22, 1, 2, 3),
        datetime(2026, 8, 22, 1, 2, 3, 456789, UTC),
        "2026-08-22T01:02:03+00:00",
    ],
)
def test_commit_candidate_requires_aware_millisecond_timestamp(
    committed_at: object,
) -> None:
    with pytest.raises(ValueError, match="committed_at"):
        CommitCandidate(
            AudioSource.ME,
            "発言",
            0,
            200,
            0,
            3_200,
            cast(datetime, committed_at),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("session_id", "invalid"),
        ("session_id", 1),
        ("model_path", "C:/models/kotoba"),
        ("partial_interval_ms", 0),
        ("partial_interval_ms", True),
        ("silence_end_ms", 0),
        ("stable_age_ms", 0),
        ("max_utterance_ms", 0),
        ("delayed_threshold_ms", 0),
        ("analysis_pause_threshold_ms", 0),
    ],
)
def test_asr_worker_config_rejects_invalid_fields(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "session_id": "01J00000000000000000000000",
        "model_path": tmp_path.resolve(),
    }
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        AsrWorkerConfig(**values)  # type: ignore[arg-type]


def test_asr_worker_config_rejects_nonlocal_and_crossed_thresholds(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="model_path"):
        make_config(Path("models/kotoba"))
    with pytest.raises(ValueError, match="silence_end_ms"):
        AsrWorkerConfig(
            "01J00000000000000000000000",
            tmp_path.resolve(),
            silence_end_ms=12_001,
            max_utterance_ms=12_000,
        )
    with pytest.raises(ValueError, match="analysis_pause_threshold_ms"):
        AsrWorkerConfig(
            "01J00000000000000000000000",
            tmp_path.resolve(),
            delayed_threshold_ms=5_001,
            analysis_pause_threshold_ms=5_000,
        )


def test_asr_ports_are_importable() -> None:
    assert SpeechDetectorPort is not None
    assert DecoderPort is not None
