"""Final speech uses a larger search without slowing partial hypotheses."""

from pathlib import Path

from flowlens.asr.engine import AsrEngine
from flowlens.asr.types import AsrWorkerConfig, DecodeHypothesis
from flowlens.audio.types import AudioFrame
from flowlens.domain.enums import AudioSource
from tests.asr.test_engine import PatternSpeechDetector, RecordingDecoder, hypothesis
from tests.asr.test_kotoba_whisper import (
    FakeModel,
    create_model_directory,
    make_decoder,
)


def test_final_decode_expands_search_only_at_end_of_speech(tmp_path: Path) -> None:
    model = FakeModel([])
    decoder, _ = make_decoder(create_model_directory(tmp_path), model)
    decoder.decode(bytes(6400))
    assert model.kwargs["beam_size"] == 1
    decoder.decode_final(bytes(6400))
    assert model.kwargs["beam_size"] == 5


def test_engine_uses_final_hypothesis_before_committing() -> None:
    class Decoder(RecordingDecoder):
        def decode_final(self, pcm_s16le: bytes) -> DecodeHypothesis:
            return hypothesis("Final result", 0, 160)

    engine = AsrEngine(
        AsrWorkerConfig("01J00000000000000000000000", Path.cwd()),
        Decoder([hypothesis("Partial result", 0, 160)]),
        PatternSpeechDetector(True),
    )
    for index in range(8):
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
    assert engine.process_ready(500).partials[0].text == "Partial result"
    assert engine.finalize(600).committed[0].text == "Final result"
