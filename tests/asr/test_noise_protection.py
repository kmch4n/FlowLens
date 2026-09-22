"""Silent regression fixtures for false speech and decoder confidence gates."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from flowlens.asr.engine import AsrEngine
from flowlens.asr.types import AsrWorkerConfig
from flowlens.asr.vad import WebRtcSpeechDetector
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource
from tests.asr.test_engine import PatternSpeechDetector, RecordingDecoder, hypothesis
from tests.asr.test_kotoba_whisper import (
    FakeModel,
    create_model_directory,
    make_decoder,
)
from tests.asr.test_vad import FakeVad


def config(**kwargs: Any) -> AsrWorkerConfig:
    return AsrWorkerConfig("01J00000000000000000000000", Path.cwd().resolve(), **kwargs)


@pytest.mark.parametrize(
    "amplitude, expected", [(0, False), (199, False), (200, True), (-32768, True)]
)
def test_energy_gate_rejects_quiet_vad_false_positives(
    amplitude: int, expected: bool
) -> None:
    detector = WebRtcSpeechDetector(
        vad_factory=lambda _: FakeVad(True), min_speech_rms=200
    )
    frame = AudioFrame(
        AudioSource.ME, amplitude.to_bytes(2, "little", signed=True) * 320, 0, 320, 0, 0
    )
    assert detector.is_speech(frame) is expected


@pytest.mark.parametrize("speech_frames", [1, 7, 8])
@pytest.mark.parametrize("finalize", [False, True])
def test_minimum_speech_prevents_blip_decode_without_trimming_onset(
    speech_frames: int, finalize: bool
) -> None:
    decoder = RecordingDecoder.repeat(hypothesis("ごめん", 0, 160))
    engine = AsrEngine(
        config(), decoder, PatternSpeechDetector([True] * speech_frames + [False] * 23)
    )
    count = speech_frames if finalize else speech_frames + 23
    for index in range(count):
        engine.accept(
            AudioFrame(
                AudioSource.ME,
                b"\x01\x00" * 320,
                index * 320,
                (index + 1) * 320,
                index * 20,
                index * 20,
            )
        )
    batch = engine.finalize(1000) if finalize else engine.process_ready(1000)
    if speech_frames < 8:
        assert decoder.call_count == 0
        assert not batch.committed
        assert not batch.partials
    else:
        assert batch.committed[0].text == "ごめん"
        assert batch.committed[0].source_start_sample == 0
        assert batch.committed[0].session_start_ms == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_speech_rms", -1),
        ("min_speech_rms", 2001),
        ("min_speech_rms", True),
        ("min_speech_ms", 0),
        ("min_speech_ms", 1001),
        ("min_speech_ms", 21),
        ("min_speech_ms", True),
    ],
)
def test_noise_configuration_rejects_invalid_thresholds(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        replace(config(), **cast(dict[str, Any], {field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("no_speech_prob", 0.6),
        ("no_speech_prob", -0.1),
        ("avg_logprob", -1.01),
        ("avg_logprob", 0.1),
        ("compression_ratio", 2.41),
        ("compression_ratio", -1.0),
        ("no_speech_prob", float("nan")),
        ("avg_logprob", float("inf")),
        ("compression_ratio", "bad"),
        ("no_speech_prob", None),
        ("avg_logprob", True),
    ],
)
def test_decoder_rejects_unreliable_segments_before_words_or_fallback(
    tmp_path: Path, field: str, value: object
) -> None:
    segment = SimpleNamespace(
        text="ごめん",
        start=0.0,
        end=0.2,
        words=None,
        no_speech_prob=0.1,
        avg_logprob=-0.2,
        compression_ratio=1.0,
    )
    setattr(segment, field, value)
    decoder, _ = make_decoder(
        create_model_directory(tmp_path), FakeModel([cast(Any, segment)])
    )
    assert decoder.decode(bytes(6400)).tokens == ()


def test_confident_repeated_words_are_not_blacklisted(tmp_path: Path) -> None:
    segments = [
        SimpleNamespace(
            text="ごめん",
            start=start,
            end=start + 0.2,
            words=None,
            no_speech_prob=0.59,
            avg_logprob=-1.0,
            compression_ratio=2.4,
        )
        for start in (0.0, 0.2)
    ]
    decoder, _ = make_decoder(
        create_model_directory(tmp_path), FakeModel(cast(Any, segments))
    )
    result = decoder.decode(bytes(12800))
    assert result.text == "ごめんごめん"
    assert [(token.start_ms, token.end_ms) for token in result.tokens] == [
        (0, 200),
        (200, 400),
    ]
