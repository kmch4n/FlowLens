"""Fake-only tests for the strictly local llama.cpp discussion adapter."""

import pickle
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import overload

import pytest

from flowlens.discussion.contracts import (
    ChatMessage,
    DiscussionGenerationError,
)
from flowlens.discussion.llama_cpp_adapter import (
    DiscussionModelConfig,
    LlamaCppDiscussionBackend,
    load_llama_cpp_backend,
)


class SpyLlama:
    """Record tokenizer and generation calls without loading a model."""

    def __init__(self, response: object | None = None) -> None:
        self.response = response or {
            "choices": [{"message": {"content": '{"revision":1}'}}]
        }
        self.call_kwargs: dict[str, object] | None = None
        self.tokenize_args: tuple[bytes, bool, bool] | None = None

    def create_chat_completion(self, **kwargs: object) -> object:
        self.call_kwargs = dict(kwargs)
        return self.response

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool,
        special: bool,
    ) -> list[int]:
        self.tokenize_args = (text, add_bos, special)
        return [11, 22, 33]


class RaisingLlama(SpyLlama):
    """Raise dependency errors containing data that must not escape."""

    def create_chat_completion(self, **kwargs: object) -> object:
        raise RuntimeError("secret prompt at C:/private/model.gguf")

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool,
        special: bool,
    ) -> list[int]:
        raise RuntimeError("secret transcript at C:/private/model.gguf")


class InterruptingLlama(SpyLlama):
    """Prove process-control exceptions are not converted."""

    def create_chat_completion(self, **kwargs: object) -> object:
        raise KeyboardInterrupt


class HostileMapping(Mapping[str, object]):
    """Raise sensitive dependency errors during response extraction."""

    def __getitem__(self, key: str) -> object:
        raise RuntimeError("secret transcript at C:/private/model.gguf")

    def __iter__(self) -> Iterator[str]:
        return iter(("choices",))

    def __len__(self) -> int:
        return 1


class HostileSequence(Sequence[int]):
    """Raise sensitive dependency errors during token inspection."""

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        raise RuntimeError("secret transcript at C:/private/model.gguf")

    def __len__(self) -> int:
        raise RuntimeError("secret transcript at C:/private/model.gguf")


class HostileTokenLlama:
    """Return a hostile dependency-owned token sequence."""

    def create_chat_completion(self, **kwargs: object) -> object:
        return {"choices": [{"message": {"content": "{}"}}]}

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool,
        special: bool,
    ) -> Sequence[int]:
        return HostileSequence()


class DiscussionErrorLlama(SpyLlama):
    """Raise an already sanitized boundary error."""

    def __init__(self, error: DiscussionGenerationError) -> None:
        super().__init__()
        self.error = error

    def create_chat_completion(self, **kwargs: object) -> object:
        raise self.error

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool,
        special: bool,
    ) -> list[int]:
        raise self.error


class SpyLlamaFactory:
    """Capture exact model-loading arguments."""

    def __init__(self, cl: SpyLlama | None = None) -> None:
        self.cl = cl or SpyLlama()
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> SpyLlama:
        self.kwargs = dict(kwargs)
        return self.cl


class RaisingFactory:
    """Fail while carrying a sensitive dependency message."""

    def __call__(self, **kwargs: object) -> SpyLlama:
        raise RuntimeError("cannot load C:/private/model.gguf with token=secret")


def _write_model(tmp_path: Path, content: bytes = b"local gguf") -> Path:
    model_path = tmp_path / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    model_path.write_bytes(content)
    return model_path


def _digest(content: bytes = b"local gguf") -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def _config(model_path: Path, content: bytes = b"local gguf") -> DiscussionModelConfig:
    return DiscussionModelConfig(model_path=model_path, sha256=_digest(content))


def test_config_is_pickle_safe_and_keeps_exact_defaults(tmp_path: Path) -> None:
    config = _config(_write_model(tmp_path))

    assert pickle.loads(pickle.dumps(config)) == config
    assert config.n_ctx == 8192
    assert config.n_gpu_layers == -1
    assert config.temperature == 0.0
    assert config.max_tokens == 512


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, ""])
def test_config_rejects_noncanonical_sha256(tmp_path: Path, digest: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        DiscussionModelConfig(_write_model(tmp_path), digest)


def test_config_rejects_bool_as_integer_values(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    with pytest.raises(ValueError, match="n_ctx"):
        DiscussionModelConfig(model_path, _digest(), n_ctx=True)
    with pytest.raises(ValueError, match="n_gpu_layers"):
        DiscussionModelConfig(
            model_path,
            _digest(),
            n_gpu_layers=True,
        )
    with pytest.raises(ValueError, match="max_tokens"):
        DiscussionModelConfig(
            model_path,
            _digest(),
            max_tokens=True,
        )


def test_config_rejects_invalid_numeric_boundaries(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    with pytest.raises(ValueError, match="n_ctx"):
        DiscussionModelConfig(model_path, _digest(), n_ctx=0)
    with pytest.raises(ValueError, match="n_gpu_layers"):
        DiscussionModelConfig(model_path, _digest(), n_gpu_layers=-2)
    with pytest.raises(ValueError, match="temperature"):
        DiscussionModelConfig(model_path, _digest(), temperature=-0.1)
    with pytest.raises(ValueError, match="max_tokens"):
        DiscussionModelConfig(model_path, _digest(), max_tokens=0)


@pytest.mark.parametrize("value", [0, True, "0.0"])
def test_config_requires_temperature_to_be_float(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        DiscussionModelConfig(
            _write_model(tmp_path),
            _digest(),
            temperature=value,  # type: ignore[arg-type]
        )


def test_config_rejects_relative_missing_directory_and_wrong_suffix(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        DiscussionModelConfig(Path("model.gguf"), _digest())
    with pytest.raises(ValueError, match="regular local"):
        DiscussionModelConfig(tmp_path / "missing.gguf", _digest())
    with pytest.raises(ValueError, match="regular local"):
        DiscussionModelConfig(tmp_path, _digest())
    wrong_suffix = tmp_path / "model.bin"
    wrong_suffix.write_bytes(b"local gguf")
    with pytest.raises(ValueError, match=".gguf"):
        DiscussionModelConfig(wrong_suffix, _digest())


def test_config_rejects_noncanonical_or_symlinked_model_path(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        DiscussionModelConfig(tmp_path / "child" / ".." / model_path.name, _digest())

    link = tmp_path / "linked.gguf"
    try:
        link.symlink_to(model_path)
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows account")
    with pytest.raises(ValueError, match="canonical"):
        DiscussionModelConfig(link, _digest())


def test_load_uses_verified_local_path_full_gpu_and_fixed_context(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path)
    factory = SpyLlamaFactory()

    backend = load_llama_cpp_backend(_config(model_path), factory=factory)

    assert isinstance(backend, LlamaCppDiscussionBackend)
    assert factory.kwargs == {
        "model_path": str(model_path),
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_batch": 128,
        "n_ubatch": 128,
        "offload_kqv": False,
        "verbose": False,
        "use_mmap": True,
    }


def test_checksum_mismatch_never_calls_factory(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config = DiscussionModelConfig(model_path, "0" * 64)
    factory = SpyLlamaFactory()

    with pytest.raises(DiscussionGenerationError, match="checksum"):
        load_llama_cpp_backend(config, factory=factory)

    assert factory.kwargs is None


def test_factory_failure_is_sanitized(tmp_path: Path) -> None:
    config = _config(_write_model(tmp_path))

    with pytest.raises(DiscussionGenerationError) as caught:
        load_llama_cpp_backend(config, factory=RaisingFactory())

    message = str(caught.value)
    assert "private" not in message
    assert "secret" not in message
    assert str(config.model_path) not in message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_generate_uses_schema_grammar_and_fixed_sampling(tmp_path: Path) -> None:
    cl = SpyLlama()
    backend = LlamaCppDiscussionBackend(_config(_write_model(tmp_path)), cl)
    messages = (
        ChatMessage("system", "日本語の指示"),
        ChatMessage("user", "議論の入力"),
    )
    schema: dict[str, object] = {"type": "object"}

    raw = backend.generate(messages, schema)

    assert raw == '{"revision":1}'
    assert cl.call_kwargs == {
        "messages": [
            {"role": "system", "content": "日本語の指示"},
            {"role": "user", "content": "議論の入力"},
        ],
        "response_format": {"type": "json_object", "schema": schema},
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"choices": []},
        {"choices": "not-a-list"},
        {"choices": [{}]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": 7}}]},
    ],
)
def test_generate_rejects_every_malformed_response(
    tmp_path: Path,
    response: object,
) -> None:
    cl = SpyLlama(response={"unused": response})
    cl.response = response
    backend = LlamaCppDiscussionBackend(_config(_write_model(tmp_path)), cl)

    with pytest.raises(DiscussionGenerationError, match="malformed"):
        backend.generate((ChatMessage("user", "do not leak me"),), {})


def test_generate_dependency_failure_does_not_leak_input_or_path(
    tmp_path: Path,
) -> None:
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        RaisingLlama(),
    )

    with pytest.raises(DiscussionGenerationError) as caught:
        backend.generate((ChatMessage("user", "highly secret transcript"),), {})

    message = str(caught.value)
    assert "secret" not in message
    assert "private" not in message
    assert str(tmp_path) not in message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_generate_sanitizes_hostile_mapping_access(tmp_path: Path) -> None:
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        SpyLlama(HostileMapping()),
    )

    with pytest.raises(DiscussionGenerationError) as caught:
        backend.generate((ChatMessage("user", "highly secret transcript"),), {})

    assert str(caught.value) == "llama.cpp response was malformed"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_generate_does_not_catch_keyboard_interrupt(tmp_path: Path) -> None:
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        InterruptingLlama(),
    )

    with pytest.raises(KeyboardInterrupt):
        backend.generate((ChatMessage("user", "input"),), {})


def test_generate_preserves_existing_generation_error(tmp_path: Path) -> None:
    emitted = DiscussionGenerationError("already sanitized")
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        DiscussionErrorLlama(emitted),
    )

    with pytest.raises(DiscussionGenerationError) as caught:
        backend.generate((ChatMessage("user", "input"),), {})

    assert caught.value is emitted


def test_count_tokens_uses_utf8_without_bos_and_with_special_tokens(
    tmp_path: Path,
) -> None:
    cl = SpyLlama()
    backend = LlamaCppDiscussionBackend(_config(_write_model(tmp_path)), cl)

    assert backend.count_tokens("日本語🙂") == 3
    assert cl.tokenize_args == ("日本語🙂".encode(), False, True)


def test_count_tokens_sanitizes_dependency_failure(tmp_path: Path) -> None:
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        RaisingLlama(),
    )

    with pytest.raises(DiscussionGenerationError) as caught:
        backend.count_tokens("highly secret transcript")

    assert "secret" not in str(caught.value)
    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_count_tokens_sanitizes_hostile_sequence_iteration(tmp_path: Path) -> None:
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        HostileTokenLlama(),
    )

    with pytest.raises(DiscussionGenerationError) as caught:
        backend.count_tokens("highly secret transcript")

    assert str(caught.value) == "llama.cpp tokenization was malformed"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_count_tokens_preserves_existing_generation_error(tmp_path: Path) -> None:
    emitted = DiscussionGenerationError("already sanitized")
    backend = LlamaCppDiscussionBackend(
        _config(_write_model(tmp_path)),
        DiscussionErrorLlama(emitted),
    )

    with pytest.raises(DiscussionGenerationError) as caught:
        backend.count_tokens("input")

    assert caught.value is emitted


def test_response_accepts_mapping_implementations(tmp_path: Path) -> None:
    class CustomMapping(dict[str, object]):
        pass

    response: Mapping[str, object] = CustomMapping(
        choices=[CustomMapping(message=CustomMapping(content="日本語の結果"))]
    )
    cl = SpyLlama(response)
    backend = LlamaCppDiscussionBackend(_config(_write_model(tmp_path)), cl)

    assert backend.generate((ChatMessage("user", "input"),), {}) == "日本語の結果"
