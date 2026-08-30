"""Strictly local model readiness checks with no download fallback."""

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path

from flowlens.controller.models import ModelCheck
from flowlens.discussion.model_manifest import ModelManifestError, parse_manifest_bytes

_ASR_ID = "kotoba-whisper-v2.0-faster"
_DISCUSSION_ID = "qwen3-4b-instruct-2507"
_QWEN_REPOSITORY = "Qwen/Qwen3-4B-Instruct-2507"
_QWEN_SOURCE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
_QWEN_CONVERTER_REVISION = "2e92ecd0247d25f09797f8fdb044a166522fc05d"
_QWEN_RUNTIME_PATH = "qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
_ASR_REPOSITORY = "kotoba-tech/kotoba-whisper-v2.0-faster"
_ASR_RUNTIME_PATH = "kotoba-whisper-v2.0-faster/model.bin"
_ASR_SOURCE_REVISION = "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"
_ASR_LICENSE = "MIT"
_ASR_CONFIG_NAME = "config.json"
_ASR_ALIGNMENT_HEADS = [[1, head] for head in range(20)]
_ASR_ARTIFACT_SHA256 = {
    "config.json": "90c55f775cc4e0bb17293d0bf12f96557a486f20dea886fabd8e6075a3588b21",
    "preprocessor_config.json": (
        "902ea83da8365e6dab586b37f0e8deb8b925d323f726c2e8972c176f5dbfa201"
    ),
    "tokenizer.json": (
        "79405583bbc27b16f69416241d4db38983d83dabf4c90811096528bf00f39101"
    ),
    "vocabulary.json": (
        "697ee5a65726e46b4d294d1a243d98e1878f5d81caba17891b1c5ebab7a912a9"
    ),
}


class LocalModelReadiness:
    """Validate the required models entirely under one configured local root."""

    def __init__(
        self,
        model_root: Path,
        manifest_path: Path,
        *,
        hash_chunk_size: int = 1024 * 1024,
    ) -> None:
        if not isinstance(model_root, Path) or not model_root.is_absolute():
            raise ValueError("model_root must be an absolute Path")
        if model_root.resolve(strict=False) != model_root:
            raise ValueError("model_root must be normalized")
        if not isinstance(manifest_path, Path) or not manifest_path.is_absolute():
            raise ValueError("manifest_path must be an absolute Path")
        if manifest_path.resolve(strict=False) != manifest_path:
            raise ValueError("manifest_path must be normalized")
        if manifest_path.parent != model_root:
            raise ValueError("manifest_path must be directly inside model_root")
        if type(hash_chunk_size) is not int or hash_chunk_size <= 0:
            raise ValueError("hash_chunk_size must be a positive integer")
        self._root = model_root
        self._manifest = manifest_path
        self._chunk_size = hash_chunk_size

    def check_required(self) -> dict[str, ModelCheck]:
        """Return a structured result for both models on every ordinary failure."""

        fallback = {
            "asr": ModelCheck(_ASR_ID, None, False, "missing"),
            "discussion": ModelCheck(_DISCUSSION_ID, None, False, "missing"),
        }
        try:
            if not _path_exists(self._manifest):
                return fallback
            if not _safe_regular_file(self._manifest):
                raise ModelManifestError("manifest path is not a regular local file")
            manifest = parse_manifest_bytes(self._manifest.read_bytes())
        except (ModelManifestError, OSError, ValueError):
            return {
                "asr": ModelCheck(_ASR_ID, None, False, "invalid"),
                "discussion": ModelCheck(_DISCUSSION_ID, None, False, "invalid"),
            }
        models = manifest["models"]
        return {
            "asr": self._check_entry(models, _ASR_ID),
            "discussion": self._check_entry(models, _DISCUSSION_ID),
        }

    def _check_entry(
        self,
        models: Mapping[str, object],
        model_id: str,
    ) -> ModelCheck:
        value = models.get(model_id)
        if not isinstance(value, Mapping):
            return ModelCheck(model_id, None, False, "missing")
        relative_path = value.get("relative_path")
        if not isinstance(relative_path, str):
            return ModelCheck(model_id, None, False, "invalid")
        try:
            target = self._resolve_target(relative_path)
        except ValueError:
            return ModelCheck(model_id, None, False, "invalid")
        if not _metadata_matches(model_id, value, relative_path):
            return ModelCheck(model_id, target, False, "invalid")
        if not _path_exists(target):
            return ModelCheck(model_id, target, False, "missing")
        if not _safe_regular_file(target):
            return ModelCheck(model_id, target, False, "invalid")
        try:
            if target.stat().st_size <= 0:
                return ModelCheck(model_id, target, False, "invalid")
            actual = _hash_file(target, self._chunk_size)
        except OSError:
            return ModelCheck(model_id, target, False, "invalid")
        if actual != value["sha256"]:
            return ModelCheck(model_id, target, False, "checksum")
        if model_id == _ASR_ID and not _valid_asr_artifacts(
            target.parent,
            self._chunk_size,
        ):
            return ModelCheck(model_id, target, False, "invalid")
        return ModelCheck(model_id, target, True, None)

    def _resolve_target(self, relative_path: str) -> Path:
        candidate_part = Path(relative_path)
        if candidate_part.is_absolute() or not candidate_part.parts:
            raise ValueError("model path must be relative")
        if any(part in ("", ".", "..") for part in candidate_part.parts):
            raise ValueError("model path must not traverse")
        candidate = self._root.joinpath(candidate_part)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._root)
        except ValueError as error:
            raise ValueError("model path escapes model root") from error
        if resolved != candidate:
            raise ValueError("model path must be canonical")
        return candidate


def _metadata_matches(
    model_id: str,
    entry: Mapping[str, object],
    relative_path: str,
) -> bool:
    if model_id == _DISCUSSION_ID:
        return (
            entry.get("repository") == _QWEN_REPOSITORY
            and entry.get("source_revision") == _QWEN_SOURCE_REVISION
            and entry.get("converter_revision") == _QWEN_CONVERTER_REVISION
            and entry.get("runtime_format") == "GGUF Q4_K_M"
            and relative_path == _QWEN_RUNTIME_PATH
            and entry.get("license") == "Apache-2.0"
        )
    return (
        entry.get("repository") == _ASR_REPOSITORY
        and entry.get("source_revision") == _ASR_SOURCE_REVISION
        and entry.get("runtime_format") == "CTranslate2"
        and relative_path == _ASR_RUNTIME_PATH
        and entry.get("license") == _ASR_LICENSE
        and "converter_revision" not in entry
    )


def _safe_regular_file(path: Path) -> bool:
    try:
        status = path.lstat()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        attributes = getattr(status, "st_file_attributes", 0)
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or bool(attributes & reparse)
            or status.st_nlink != 1
        ):
            return False
        return path.resolve(strict=True) == path
    except OSError:
        return False


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _hash_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _valid_asr_artifacts(model_directory: Path, chunk_size: int) -> bool:
    for filename, expected_hash in _ASR_ARTIFACT_SHA256.items():
        artifact = model_directory / filename
        if not _safe_regular_file(artifact):
            return False
        try:
            if artifact.stat().st_size <= 0:
                return False
            if _hash_file(artifact, chunk_size) != expected_hash:
                return False
        except OSError:
            return False
    return _valid_asr_config(model_directory)


def _valid_asr_config(model_directory: Path) -> bool:
    config_path = model_directory / _ASR_CONFIG_NAME
    if not _safe_regular_file(config_path):
        return False
    try:
        payload = config_path.read_bytes()
        if payload.startswith(b"\xef\xbb\xbf"):
            return False
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    return isinstance(value, Mapping) and value.get("alignment_heads") == (
        _ASR_ALIGNMENT_HEADS
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")
