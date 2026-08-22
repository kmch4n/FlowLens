"""Hardware- and model-independent ports for ASR processing."""

from typing import Protocol

from flowlens.asr.types import DecodeHypothesis
from flowlens.audio.types import AudioFrame


class SpeechDetectorPort(Protocol):
    """Classify one canonical audio frame."""

    def is_speech(self, frame: AudioFrame) -> bool: ...


class DecoderPort(Protocol):
    """Decode canonical PCM into one timestamped hypothesis."""

    def decode(self, pcm_s16le: bytes) -> DecodeHypothesis: ...
