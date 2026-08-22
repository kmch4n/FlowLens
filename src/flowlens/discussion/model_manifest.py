"""Strict deterministic model-manifest transformation for initial setup."""

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict, cast

_QWEN_MODEL_ID = "qwen3-4b-instruct-2507"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_ENTRY_FIELDS = (
    "repository",
    "source_revision",
    "runtime_format",
    "relative_path",
    "sha256",
    "license",
)


class ModelManifestError(ValueError):
    """Raised when a local model manifest is not strictly valid."""


class ModelEntry(TypedDict):
    """One canonical model identity and runtime artifact."""

    repository: str
    source_revision: str
    converter_revision: NotRequired[str]
    runtime_format: str
    relative_path: str
    sha256: str
    license: str


class ModelManifest(TypedDict):
    """Versioned local model-manifest wire shape."""

    schema_version: int
    models: dict[str, ModelEntry]


def empty_manifest() -> ModelManifest:
    """Return a fresh empty schema-version-one manifest."""

    return {"schema_version": 1, "models": {}}


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ModelManifestError(f"{field_name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    field_name: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ModelManifestError(f"{field_name} has missing or unknown fields")


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"{field_name} must be a non-blank string")
    return value


def _validate_entry(value: object, field_name: str) -> ModelEntry:
    mapping = _require_mapping(value, field_name)
    required = frozenset(_REQUIRED_ENTRY_FIELDS)
    actual = frozenset(mapping)
    if actual not in (required, required | {"converter_revision"}):
        raise ModelManifestError(f"{field_name} has missing or unknown fields")

    normalized: ModelEntry = {
        "repository": _require_string(mapping["repository"], "repository"),
        "source_revision": _require_string(
            mapping["source_revision"],
            "source_revision",
        ),
        "runtime_format": _require_string(
            mapping["runtime_format"],
            "runtime_format",
        ),
        "relative_path": _require_string(
            mapping["relative_path"],
            "relative_path",
        ),
        "sha256": _require_string(mapping["sha256"], "sha256"),
        "license": _require_string(mapping["license"], "license"),
    }
    if "converter_revision" in mapping:
        normalized["converter_revision"] = _require_string(
            mapping["converter_revision"],
            "converter_revision",
        )
    if _SHA256_PATTERN.fullmatch(normalized["sha256"]) is None:
        raise ModelManifestError("sha256 must be lowercase 64-hex")
    return _canonical_entry(normalized)


def _canonical_entry(entry: ModelEntry) -> ModelEntry:
    converter_revision = entry.get("converter_revision")
    if converter_revision is not None:
        return {
            "repository": entry["repository"],
            "source_revision": entry["source_revision"],
            "converter_revision": converter_revision,
            "runtime_format": entry["runtime_format"],
            "relative_path": entry["relative_path"],
            "sha256": entry["sha256"],
            "license": entry["license"],
        }
    return {
        "repository": entry["repository"],
        "source_revision": entry["source_revision"],
        "runtime_format": entry["runtime_format"],
        "relative_path": entry["relative_path"],
        "sha256": entry["sha256"],
        "license": entry["license"],
    }


def validate_manifest(value: object) -> ModelManifest:
    """Validate and defensively normalize one decoded manifest."""

    mapping = _require_mapping(value, "manifest")
    _require_exact_keys(mapping, frozenset(("schema_version", "models")), "manifest")
    schema_version = mapping["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ModelManifestError("schema_version must be integer 1")
    models = _require_mapping(mapping["models"], "models")
    normalized_models: dict[str, ModelEntry] = {}
    folded_ids: set[str] = set()
    for model_id in sorted(models):
        normalized_id = _require_string(model_id, "model_id")
        folded = normalized_id.casefold()
        if folded in folded_ids:
            raise ModelManifestError("model IDs must be case-insensitively unique")
        folded_ids.add(folded)
        normalized_models[normalized_id] = _validate_entry(
            models[model_id],
            f"models.{normalized_id}",
        )
    return {"schema_version": 1, "models": normalized_models}


def update_qwen_manifest(existing: object, qwen_entry: object) -> ModelManifest:
    """Preserve all unrelated models and replace only the fixed Qwen entry."""

    manifest = validate_manifest(existing)
    normalized_qwen = _validate_entry(qwen_entry, _QWEN_MODEL_ID)
    if "converter_revision" not in normalized_qwen:
        raise ModelManifestError("Qwen entry requires converter_revision")
    models = dict(manifest["models"])
    models[_QWEN_MODEL_ID] = normalized_qwen
    return {
        "schema_version": 1,
        "models": {model_id: models[model_id] for model_id in sorted(models)},
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    folded: set[str] = set()
    for key, value in pairs:
        normalized = key.casefold()
        if normalized in folded:
            raise ModelManifestError("duplicate JSON key")
        folded.add(normalized)
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ModelManifestError(f"invalid JSON constant: {value}")


def parse_manifest_bytes(encoded: bytes) -> ModelManifest:
    """Parse strict UTF-8 JSON without BOM or duplicate keys."""

    if not isinstance(encoded, bytes):
        raise ModelManifestError("manifest content must be bytes")
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise ModelManifestError("manifest must be UTF-8 without BOM")
    try:
        decoded = encoded.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ModelManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ModelManifestError("manifest must be valid UTF-8 JSON") from error
    return validate_manifest(value)


def encode_manifest(value: object) -> bytes:
    """Encode one validated manifest deterministically with final LF."""

    manifest = validate_manifest(value)
    return (
        json.dumps(manifest, ensure_ascii=False, indent=4, allow_nan=False) + "\n"
    ).encode()


def _load_manifest(path: Path) -> ModelManifest:
    if not path.exists():
        return empty_manifest()
    if path.is_symlink() or not path.is_file():
        raise ModelManifestError("manifest path must be a regular local file")
    return parse_manifest_bytes(path.read_bytes())


def _write_new_manifest(path: Path, value: object) -> None:
    encoded = encode_manifest(value)
    with path.open("xb") as file:
        remaining = memoryview(encoded)
        while remaining:
            written = file.write(remaining)
            if written is None or written <= 0:
                raise OSError("manifest write made no progress")
            remaining = remaining[written:]
        file.flush()
        os.fsync(file.fileno())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or update model manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    update = subparsers.add_parser("update")
    update.add_argument("--manifest", required=True)
    update.add_argument("--output", required=True)
    for argument in (
        "repository",
        "source-revision",
        "converter-revision",
        "runtime-format",
        "relative-path",
        "sha256",
        "license",
    ):
        update.add_argument(f"--{argument}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local-only manifest helper CLI."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    manifest_path = Path(cast(str, arguments.manifest))
    try:
        existing = _load_manifest(manifest_path)
        if cast(str, arguments.command) == "validate":
            return 0
        entry = {
            "repository": cast(str, arguments.repository),
            "source_revision": cast(str, arguments.source_revision),
            "converter_revision": cast(str, arguments.converter_revision),
            "runtime_format": cast(str, arguments.runtime_format),
            "relative_path": cast(str, arguments.relative_path),
            "sha256": cast(str, arguments.sha256),
            "license": cast(str, arguments.license),
        }
        updated = update_qwen_manifest(existing, entry)
        _write_new_manifest(Path(cast(str, arguments.output)), updated)
    except (ModelManifestError, OSError) as error:
        parser.exit(1, f"model manifest error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
