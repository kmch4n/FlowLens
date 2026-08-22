"""Pure behavioral tests for deterministic local model manifests."""

import copy
import random

import pytest

from flowlens.discussion.model_manifest import (
    ModelManifestError,
    empty_manifest,
    encode_manifest,
    parse_manifest_bytes,
    update_qwen_manifest,
    validate_manifest,
)


def _entry(seed: str, *, converter: bool = False) -> dict[str, str]:
    entry = {
        "repository": f"local/{seed}",
        "source_revision": f"revision-{seed}",
        "runtime_format": f"format-{seed}",
        "relative_path": f"{seed}/model.bin",
        "sha256": (seed.encode().hex() + "0" * 64)[:64],
        "license": "Apache-2.0",
    }
    if converter:
        entry["converter_revision"] = f"converter-{seed}"
    return entry


def _qwen_entry(seed: str = "a") -> dict[str, str]:
    return {
        "repository": "Qwen/Qwen3-4B-Instruct-2507",
        "source_revision": f"source-{seed}",
        "converter_revision": f"converter-{seed}",
        "runtime_format": "GGUF Q4_K_M",
        "relative_path": (
            "qwen3-4b-instruct-2507/" "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        ),
        "sha256": seed * 64,
        "license": "Apache-2.0",
    }


def test_update_preserves_asr_and_unrelated_entries_without_mutating_input() -> None:
    existing: dict[str, object] = {
        "schema_version": 1,
        "models": {
            "unrelated-model": _entry("c"),
            "kotoba-whisper-v2.0-faster": _entry("b"),
            "qwen3-4b-instruct-2507": _qwen_entry("d"),
        },
    }
    before = copy.deepcopy(existing)

    updated = update_qwen_manifest(existing, _qwen_entry("e"))

    assert existing == before
    assert updated["models"]["kotoba-whisper-v2.0-faster"] == _entry("b")
    assert updated["models"]["unrelated-model"] == _entry("c")
    assert updated["models"]["qwen3-4b-instruct-2507"] == _qwen_entry("e")


def test_update_is_deterministic_for_randomized_valid_manifest_order() -> None:
    rng = random.Random(20260822)
    model_items = [(f"model-{index:03}", _entry(f"{index:02x}")) for index in range(96)]
    expected: bytes | None = None

    for _ in range(64):
        rng.shuffle(model_items)
        existing: dict[str, object] = {
            "schema_version": 1,
            "models": dict(model_items),
        }
        encoded = encode_manifest(update_qwen_manifest(existing, _qwen_entry()))
        if expected is None:
            expected = encoded
        assert encoded == expected
        assert (
            parse_manifest_bytes(encoded)["models"]["qwen3-4b-instruct-2507"]
            == _qwen_entry()
        )


def test_empty_manifest_has_the_exact_wire_shape() -> None:
    assert empty_manifest() == {"schema_version": 1, "models": {}}


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"schema_version": True, "models": {}},
        {"schema_version": 1, "models": [], "extra": None},
        {"schema_version": 1, "models": {"model": "not-an-object"}},
        {
            "schema_version": 1,
            "models": {"model": {"sha256": "0" * 64}},
        },
        {
            "schema_version": 1,
            "models": {"model": {**_entry("f"), "unknown": "value"}},
        },
        {
            "schema_version": 1,
            "models": {"model": {**_entry("f"), "sha256": "F" * 64}},
        },
    ],
)
def test_validation_fails_closed_for_malformed_shapes(manifest: object) -> None:
    with pytest.raises(ModelManifestError):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "encoded",
    [
        b"\xef\xbb\xbf{}",
        b"\xff",
        b'{"schema_version":1,"schema_version":1,"models":{}}',
        b'{"schema_version":1,"models":{"A":{},"a":{}}}',
        b'{"schema_version":1,"models":{},"value":NaN}',
    ],
)
def test_parser_rejects_bom_invalid_utf8_duplicates_and_constants(
    encoded: bytes,
) -> None:
    with pytest.raises(ModelManifestError):
        parse_manifest_bytes(encoded)


def test_encoding_is_utf8_no_bom_lf_and_round_trips_unicode() -> None:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "models": {"日本語-model": _entry("a")},
    }

    encoded = encode_manifest(manifest)

    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in encoded
    assert encoded.endswith(b"\n")
    assert "日本語-model" in encoded.decode("utf-8")
    assert parse_manifest_bytes(encoded) == validate_manifest(manifest)
