"""Strictly local llama.cpp adapter for Qwen discussion analysis."""

import hashlib
import importlib
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from flowlens.discussion.contracts import (
    ChatMessage,
    DiscussionGenerationError,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_SIZE = 1024 * 1024


class _LlamaClient(Protocol):
    """Small llama.cpp surface used by the discussion backend."""

    def create_chat_completion(self, **kwargs: object) -> object:
        """Generate one chat completion."""

        ...

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool,
        special: bool,
    ) -> Sequence[int]:
        """Tokenize UTF-8 input locally."""

        ...


class _LlamaFactory(Protocol):
    """Injectable llama.cpp constructor."""

    def __call__(self, **kwargs: object) -> _LlamaClient:
        """Construct one already configured local client."""

        ...


def _require_int(value: object, field_name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{field_name} must be a {qualifier} integer")
    return value


def _require_model_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise ValueError("model_path must be a Path")
    if not value.is_absolute():
        raise ValueError("model_path must be absolute")

    canonical = value.resolve(strict=False)
    if os.path.normcase(str(canonical)) != os.path.normcase(str(value)):
        raise ValueError("model_path must be a canonical non-symlink path")
    try:
        status = value.lstat()
    except OSError as error:
        raise ValueError("model_path must be an existing regular local file") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(status, "st_file_attributes", 0)
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or bool(attributes & reparse_flag)
        or status.st_nlink != 1
    ):
        raise ValueError("model_path must be an existing regular local file")
    if value.suffix.lower() != ".gguf":
        raise ValueError("model_path must name a .gguf file")
    try:
        strict_canonical = value.resolve(strict=True)
    except OSError as error:
        raise ValueError("model_path must be an existing regular local file") from error
    if os.path.normcase(str(strict_canonical)) != os.path.normcase(str(value)):
        raise ValueError("model_path must be a canonical non-symlink path")
    return value


@dataclass(frozen=True, slots=True)
class DiscussionModelConfig:
    """Validated immutable local discussion-model configuration."""

    model_path: Path
    sha256: str
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.0
    max_tokens: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", _require_model_path(self.model_path))
        if (
            not isinstance(self.sha256, str)
            or _SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise ValueError("sha256 must be a lowercase 64-hex digest")
        _require_int(self.n_ctx, "n_ctx", minimum=1)
        _require_int(self.n_gpu_layers, "n_gpu_layers", minimum=-1)
        if (
            not isinstance(self.temperature, float)
            or not math.isfinite(self.temperature)
            or self.temperature < 0.0
        ):
            raise ValueError("temperature must be a non-negative finite float")
        _require_int(self.max_tokens, "max_tokens", minimum=1)


class LlamaCppDiscussionBackend:
    """Grammar-constrained backend backed by one injected local llama client."""

    def __init__(self, config: DiscussionModelConfig, cl: _LlamaClient) -> None:
        self._config = config
        self._cl = cl

    def count_tokens(self, text: str) -> int:
        """Count local llama.cpp tokens for UTF-8 text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            tokens = self._cl.tokenize(
                text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
        except DiscussionGenerationError:
            raise
        except Exception:
            raise DiscussionGenerationError("llama.cpp tokenization failed") from None
        try:
            malformed = isinstance(tokens, str | bytes) or not isinstance(
                tokens, Sequence
            )
            if not malformed:
                malformed = not all(
                    isinstance(token, int) and not isinstance(token, bool)
                    for token in tokens
                )
            token_count = len(tokens)
        except Exception:
            raise DiscussionGenerationError(
                "llama.cpp tokenization was malformed"
            ) from None
        if malformed:
            raise DiscussionGenerationError("llama.cpp tokenization was malformed")
        return token_count

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        response_schema: dict[str, object],
    ) -> str:
        """Generate and strictly extract one JSON response string."""

        try:
            response = self._cl.create_chat_completion(
                messages=[
                    {"role": item.role, "content": item.content} for item in messages
                ],
                response_format={"type": "json_object", "schema": response_schema},
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stream=False,
            )
        except DiscussionGenerationError:
            raise
        except Exception:
            raise DiscussionGenerationError("llama.cpp generation failed") from None
        return _extract_content(response)


def _extract_content(response: object) -> str:
    try:
        if not isinstance(response, Mapping):
            raise ValueError
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ValueError
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ValueError
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError
        return content
    except Exception:
        raise DiscussionGenerationError("llama.cpp response was malformed") from None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError:
        raise DiscussionGenerationError(
            "discussion model could not be verified"
        ) from None
    return digest.hexdigest()


def _default_factory(**kwargs: object) -> _LlamaClient:
    try:
        module = importlib.import_module("llama_cpp")
        constructor = module.Llama
        return cast(_LlamaClient, constructor(**kwargs))
    except Exception:
        raise DiscussionGenerationError("llama.cpp model loading failed") from None


def load_llama_cpp_backend(
    config: DiscussionModelConfig,
    *,
    factory: _LlamaFactory | None = None,
) -> LlamaCppDiscussionBackend:
    """Verify the exact local file before constructing llama.cpp."""

    if not isinstance(config, DiscussionModelConfig):
        raise TypeError("config must be a DiscussionModelConfig")
    if _hash_file(config.model_path) != config.sha256:
        raise DiscussionGenerationError("discussion model checksum mismatch")
    selected_factory = _default_factory if factory is None else factory
    try:
        cl = selected_factory(
            model_path=str(config.model_path),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
            use_mmap=True,
        )
    except DiscussionGenerationError:
        raise
    except Exception:
        raise DiscussionGenerationError("llama.cpp model loading failed") from None
    return LlamaCppDiscussionBackend(config, cl)
