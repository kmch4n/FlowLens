"""Offline-only adapter for the fixed Kotoba-Whisper model."""

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from flowlens.asr.types import DecodedToken, DecodeHypothesis
from flowlens.offline_imports import import_local_module


class ModelPathError(ValueError):
    """Raised when the configured local model directory is unavailable."""


class WhisperModelPort(Protocol):
    """Minimal faster-whisper model surface consumed by the adapter."""

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        **kwargs: object,
    ) -> tuple[Iterable[object], object]: ...


class WhisperModelFactory(Protocol):
    """Construct a model from one already-installed local directory."""

    def __call__(
        self,
        model_size_or_path: str,
        **kwargs: object,
    ) -> WhisperModelPort: ...


def _default_model_factory(
    model_size_or_path: str,
    **kwargs: object,
) -> WhisperModelPort:
    module = import_local_module("faster_whisper")
    factory = cast(WhisperModelFactory, vars(module)["WhisperModel"])
    return factory(model_size_or_path, **kwargs)


def _read_attribute(value: object, name: str) -> object:
    return cast(object, getattr(value, name, None))


def _normalized_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value.strip() else None


def _timestamp_ms(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        seconds = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    milliseconds = seconds * 1_000
    if not math.isfinite(milliseconds):
        return None
    return round(milliseconds)


def _token_from_values(
    text_value: object,
    start_value: object,
    end_value: object,
) -> DecodedToken | None:
    text = _normalized_text(text_value)
    start_ms = _timestamp_ms(start_value)
    end_ms = _timestamp_ms(end_value)
    if text is None or start_ms is None or end_ms is None or end_ms < start_ms:
        return None
    return DecodedToken(text=text, start_ms=start_ms, end_ms=end_ms)


class KotobaWhisperDecoder:
    """Decode canonical PCM using fixed local Kotoba-Whisper settings."""

    def __init__(
        self,
        model_path: Path,
        model_factory: WhisperModelFactory = _default_model_factory,
    ) -> None:
        if not isinstance(model_path, Path) or not model_path.is_dir():
            raise ModelPathError("model_path must be an existing local directory")
        self._model = model_factory(
            str(model_path),
            device="cuda",
            compute_type="float16",
            local_files_only=True,
        )

    def decode(self, pcm_s16le: bytes) -> DecodeHypothesis:
        """Decode complete little-endian mono int16 PCM into ordered tokens."""

        if len(pcm_s16le) % 2 != 0:
            raise ValueError("pcm_s16le must contain whole little-endian int16 samples")
        audio = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32)
        audio /= np.float32(32768.0)
        segments, _ = self._model.transcribe(
            audio,
            language="ja",
            task="transcribe",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=False,
        )
        tokens: list[DecodedToken] = []
        for segment in segments:
            self._append_segment_tokens(tokens, segment)
        return DecodeHypothesis(tuple(tokens))

    @staticmethod
    def _append_segment_tokens(tokens: list[DecodedToken], segment: object) -> None:
        accepted_word = False
        words = _read_attribute(segment, "words")
        if isinstance(words, Iterable) and not isinstance(words, str | bytes):
            for word in words:
                if word is None:
                    continue
                token = _token_from_values(
                    _read_attribute(word, "word"),
                    _read_attribute(word, "start"),
                    _read_attribute(word, "end"),
                )
                if token is not None and KotobaWhisperDecoder._append_ordered(
                    tokens,
                    token,
                ):
                    accepted_word = True
        if accepted_word:
            return
        fallback = _token_from_values(
            _read_attribute(segment, "text"),
            _read_attribute(segment, "start"),
            _read_attribute(segment, "end"),
        )
        if fallback is not None:
            KotobaWhisperDecoder._append_ordered(tokens, fallback)

    @staticmethod
    def _append_ordered(tokens: list[DecodedToken], token: DecodedToken) -> bool:
        if tokens and token.start_ms < tokens[-1].end_ms:
            return False
        tokens.append(token)
        return True
