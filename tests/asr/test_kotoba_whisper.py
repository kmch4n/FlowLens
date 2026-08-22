"""Contract tests for the offline-only Kotoba-Whisper adapter."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from flowlens.asr.kotoba_whisper import (
    KotobaWhisperDecoder,
    ModelPathError,
)
from flowlens.asr.types import DecodedToken, DecodeHypothesis


@dataclass(frozen=True, slots=True)
class FakeWord:
    word: object
    start: object
    end: object


@dataclass(frozen=True, slots=True)
class FakeSegment:
    text: object = ""
    start: object = 0.0
    end: object = 0.0
    words: object = None


class FakeModel:
    def __init__(self, segments: Iterable[FakeSegment]) -> None:
        self._segments = segments
        self.audio: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None
        self.kwargs: dict[str, object] = {}

    def transcribe(
        self,
        audio: np.ndarray[tuple[int], np.dtype[np.float32]],
        **kwargs: object,
    ) -> tuple[Iterator[FakeSegment], object]:
        self.audio = audio.copy()
        self.kwargs = kwargs
        return iter(self._segments), object()


def make_decoder(
    model_dir: Path,
    fake: FakeModel,
) -> tuple[KotobaWhisperDecoder, dict[str, object]]:
    constructor: dict[str, object] = {}

    def factory(model_size_or_path: str, **kwargs: object) -> FakeModel:
        constructor.update({"path": model_size_or_path, **kwargs})
        return fake

    return KotobaWhisperDecoder(model_dir, model_factory=factory), constructor


def test_decoder_uses_fixed_offline_settings_and_converts_pcm(tmp_path: Path) -> None:
    model_dir = tmp_path / "kotoba-whisper-v2.0-faster"
    model_dir.mkdir()
    fake = FakeModel(
        [
            FakeSegment(
                words=[FakeWord("\u65b9\u91dd", 0.1004, 0.4006)],
            )
        ]
    )
    decoder, constructor = make_decoder(model_dir, fake)
    pcm = np.array([-32768, -1, 0, 32767], dtype="<i2").tobytes()

    result = decoder.decode(pcm)

    assert constructor == {
        "path": str(model_dir),
        "device": "cuda",
        "compute_type": "float16",
        "local_files_only": True,
    }
    assert fake.kwargs == {
        "language": "ja",
        "task": "transcribe",
        "beam_size": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": False,
    }
    assert fake.audio is not None
    assert fake.audio.dtype == np.float32
    np.testing.assert_array_equal(
        fake.audio,
        np.array([-1.0, -1.0 / 32768.0, 0.0, 32767.0 / 32768.0], np.float32),
    )
    assert result == DecodeHypothesis((DecodedToken("\u65b9\u91dd", 100, 401),))


def test_decoder_rejects_missing_model_before_factory_call(tmp_path: Path) -> None:
    calls = 0

    def factory(model_size_or_path: str, **kwargs: object) -> FakeModel:
        del model_size_or_path, kwargs
        nonlocal calls
        calls += 1
        return FakeModel([])

    with pytest.raises(ModelPathError, match="existing local directory"):
        KotobaWhisperDecoder(tmp_path / "missing", model_factory=factory)

    assert calls == 0


def test_decoder_rejects_file_model_path_before_factory_call(tmp_path: Path) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"model")

    def factory(model_size_or_path: str, **kwargs: object) -> FakeModel:
        raise AssertionError(f"factory called for {model_size_or_path} with {kwargs}")

    with pytest.raises(ModelPathError, match="existing local directory"):
        KotobaWhisperDecoder(model_file, model_factory=factory)


def test_decode_rejects_odd_length_pcm_without_transcribing(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake = FakeModel([])
    decoder, _ = make_decoder(model_dir, fake)

    with pytest.raises(ValueError, match="whole little-endian int16 samples"):
        decoder.decode(b"\x00")

    assert fake.audio is None


def test_decode_consumes_all_segments_and_preserves_input(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    consumed: list[str] = []

    def segments() -> Iterator[FakeSegment]:
        consumed.append("first")
        yield FakeSegment(words=[FakeWord("\u524d", 0.0, 0.1)])
        consumed.append("second")
        yield FakeSegment(words=[FakeWord("\u5f8c", 0.1, 0.2)])
        consumed.append("done")

    pcm = np.array([100, -100], dtype="<i2").tobytes()
    original = pcm
    decoder, _ = make_decoder(model_dir, FakeModel(segments()))

    result = decoder.decode(pcm)

    assert consumed == ["first", "second", "done"]
    assert result.tokens == (
        DecodedToken("\u524d", 0, 100),
        DecodedToken("\u5f8c", 100, 200),
    )
    assert pcm == original


def test_decode_falls_back_for_nonempty_segment_without_usable_words(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake = FakeModel(
        [
            FakeSegment(text="  \u4e00つ目  ", start=0.25, end=0.75, words=None),
            FakeSegment(
                text="\u4e8cつ目",
                start=0.75,
                end=1.25,
                words=[
                    None,
                    FakeWord("   ", 0.8, 0.9),
                    FakeWord("\u58caれた", "bad", 1.0),
                    FakeWord("\u9006転", 1.1, 1.0),
                ],
            ),
            FakeSegment(text="   ", start=1.25, end=1.5, words=None),
        ]
    )
    decoder, _ = make_decoder(model_dir, fake)

    result = decoder.decode(b"")

    assert result.tokens == (
        DecodedToken("  \u4e00つ目  ", 250, 750),
        DecodedToken("\u4e8cつ目", 750, 1250),
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, 1.0),
        (0.0, None),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (0.0, 1e308),
        (0, 10**1_000),
        (-0.1, 1.0),
        (1.0, 0.9),
    ],
)
def test_decode_ignores_segment_with_malformed_fallback_span(
    tmp_path: Path,
    start: object,
    end: object,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    decoder, _ = make_decoder(
        model_dir,
        FakeModel([FakeSegment(text="\u767a言", start=start, end=end)]),
    )

    assert decoder.decode(b"").tokens == ()


def test_decode_ignores_out_of_order_tokens_deterministically(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    decoder, _ = make_decoder(
        model_dir,
        FakeModel(
            [
                FakeSegment(
                    words=[
                        FakeWord("\u5148", 0.5, 0.7),
                        FakeWord("\u623bる", 0.2, 0.4),
                        FakeWord("\u7d9aく", 0.7, 0.9),
                    ]
                )
            ]
        ),
    )

    assert decoder.decode(b"").tokens == (
        DecodedToken("\u5148", 500, 700),
        DecodedToken("\u7d9aく", 700, 900),
    )
