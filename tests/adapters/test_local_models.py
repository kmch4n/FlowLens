import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from flowlens.adapters.local_models import LocalModelReadiness
from flowlens.adapters.storage import LocalStorageReadiness
from flowlens.adapters.windows_devices import WindowsDeviceCatalog
from flowlens.audio.types import CaptureDevice
from flowlens.domain.enums import AudioSource

QWEN_PATH = "qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
ASR_PATH = "kotoba-whisper-v2.0-faster/model.bin"
ASR_REVISION = "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"
ASR_ALIGNMENT_HEADS = [[1, head] for head in range(20)]
ASR_CONFIG = json.dumps({"alignment_heads": ASR_ALIGNMENT_HEADS}).encode("utf-8")
ASR_SIDECARS = {
    "config.json": ASR_CONFIG,
    "preprocessor_config.json": b'{"sampling_rate":16000}',
    "tokenizer.json": b'{"version":"1.0"}',
    "vocabulary.json": b'{"<pad>":0}',
}


@pytest.fixture(autouse=True)
def approved_asr_sidecars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flowlens.adapters.local_models._ASR_ARTIFACT_SHA256",
        {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in ASR_SIDECARS.items()
        },
    )


def manifest_entry(
    *, repository: str, runtime_format: str, relative_path: str, payload: bytes
) -> dict[str, str]:
    return {
        "repository": repository,
        "source_revision": ASR_REVISION,
        "runtime_format": runtime_format,
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "license": "MIT",
    }


def write_ready_manifest(root: Path) -> Path:
    qwen = root / Path(QWEN_PATH)
    asr = root / Path(ASR_PATH)
    qwen.parent.mkdir(parents=True)
    asr.parent.mkdir(parents=True)
    qwen.write_bytes(b"qwen")
    asr.write_bytes(b"asr")
    for name, payload in ASR_SIDECARS.items():
        (asr.parent / name).write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "models": {
            "kotoba-whisper-v2.0-faster": manifest_entry(
                repository="kotoba-tech/kotoba-whisper-v2.0-faster",
                runtime_format="CTranslate2",
                relative_path=ASR_PATH,
                payload=b"asr",
            ),
            "qwen3-4b-instruct-2507": {
                **manifest_entry(
                    repository="Qwen/Qwen3-4B-Instruct-2507",
                    runtime_format="GGUF Q4_K_M",
                    relative_path=QWEN_PATH,
                    payload=b"qwen",
                ),
                "source_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                "converter_revision": "2e92ecd0247d25f09797f8fdb044a166522fc05d",
                "license": "Apache-2.0",
            },
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    return path


def test_model_probe_hashes_only_manifested_local_files(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)

    result = LocalModelReadiness(root, manifest, hash_chunk_size=2).check_required()

    assert result["asr"].ready is True
    assert result["discussion"].ready is True
    assert result["discussion"].path == (root / QWEN_PATH).resolve()


def test_model_probe_rejects_wrong_asr_runtime_path(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    wrong = root / "kotoba-whisper-v2.0-faster" / "other.bin"
    wrong.write_bytes(b"other")
    asr = data["models"]["kotoba-whisper-v2.0-faster"]
    asr["relative_path"] = "kotoba-whisper-v2.0-faster/other.bin"
    asr["sha256"] = hashlib.sha256(b"other").hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["asr"].ready is False
    assert result["asr"].reason == "invalid"


@pytest.mark.parametrize(
    "config_payload",
    [
        b"{}",
        b'{"alignment_heads":[[0,0]]}',
        b'{"alignment_heads":[[1,0]],"alignment_heads":[[1,1]]}',
        b'\xef\xbb\xbf{"alignment_heads":[]}',
        b"not-json",
    ],
)
def test_model_probe_requires_corrected_kotoba_alignment_heads(
    tmp_path: Path,
    config_payload: bytes,
) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    (root / "kotoba-whisper-v2.0-faster" / "config.json").write_bytes(config_payload)

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["asr"].ready is False
    assert result["asr"].reason == "invalid"


def test_model_probe_requires_regular_kotoba_config_file(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    config = root / "kotoba-whisper-v2.0-faster" / "config.json"
    config.unlink()
    config.mkdir()

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["asr"].ready is False
    assert result["asr"].reason == "invalid"


@pytest.mark.parametrize(
    "sidecar_name",
    ["preprocessor_config.json", "tokenizer.json", "vocabulary.json"],
)
def test_model_probe_requires_every_pinned_kotoba_sidecar(
    tmp_path: Path,
    sidecar_name: str,
) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    (root / "kotoba-whisper-v2.0-faster" / sidecar_name).unlink()

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["asr"].ready is False
    assert result["asr"].reason == "invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "wrong/repository"),
        ("source_revision", "wrong-revision"),
        ("runtime_format", "wrong-runtime"),
        ("relative_path", "kotoba-whisper-v2.0-faster/other.bin"),
        ("license", "WRONG"),
        ("converter_revision", "unexpected"),
    ],
)
def test_model_probe_requires_exact_approved_asr_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entry = data["models"]["kotoba-whisper-v2.0-faster"]
    entry[field] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["asr"].ready is False
    assert result["asr"].reason == "invalid"


def test_model_probe_distinguishes_wrong_file_type_from_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    qwen = root / QWEN_PATH
    qwen.unlink()
    qwen.mkdir()

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["discussion"].ready is False
    assert result["discussion"].reason == "invalid"


@pytest.mark.parametrize(
    "mutation",
    ["bom", "duplicate", "traversal", "wrong_path", "empty", "checksum"],
)
def test_model_probe_fails_closed_for_untrusted_manifest_and_files(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path.resolve()
    manifest = write_ready_manifest(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "bom":
        manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())
    elif mutation == "duplicate":
        manifest.write_text('{"schema_version":1,"schema_version":1,"models":{}}')
    elif mutation == "traversal":
        data["models"]["qwen3-4b-instruct-2507"]["relative_path"] = "../escape.gguf"
        manifest.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "wrong_path":
        data["models"]["qwen3-4b-instruct-2507"]["relative_path"] = "qwen/model.gguf"
        manifest.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "empty":
        (root / QWEN_PATH).write_bytes(b"")
    else:
        (root / QWEN_PATH).write_bytes(b"wrong")

    result = LocalModelReadiness(root, manifest).check_required()

    assert result["asr"].ready is False or result["discussion"].ready is False
    assert set(result) == {"asr", "discussion"}


def test_storage_probe_flushes_and_removes_unique_probe(tmp_path: Path) -> None:
    root = (tmp_path / "sessions").resolve()

    result = LocalStorageReadiness().check(root, 1)

    assert result.root == root
    assert result.writable is True
    assert not list(root.glob(".flowlens-write-probe-*"))


class FakeBackend:
    def list_microphones(self) -> tuple[CaptureDevice, ...]:
        return (
            CaptureDevice("input:2", "Zulu", 2, 48_000, 1, False),
            CaptureDevice("input:1", "Alpha", 1, 48_000, 1, False),
        )

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
        return (CaptureDevice("wasapi-output:3", "Speakers", 4, 48_000, 2, True),)


def test_windows_device_adapter_emits_values_only_in_stable_order() -> None:
    catalog = WindowsDeviceCatalog(FakeBackend(), lambda source, device_id: 0.5)

    assert [item.id for item in catalog.list_microphones()] == ["input:1", "input:2"]
    assert catalog.list_loopback_outputs()[0].loopback_capable is True
    assert catalog.read_level(AudioSource.ME, "input:1") == 0.5


class HostileVendorError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("do not inspect vendor exception text")


class HostileBackend:
    def list_microphones(self) -> tuple[CaptureDevice, ...]:
        raise HostileVendorError()

    def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
        raise HostileVendorError()


def test_windows_device_adapter_sanitizes_generic_discovery_and_meter_errors() -> None:
    def fail_level(source: AudioSource, device_id: str) -> float:
        del source, device_id
        raise HostileVendorError()

    catalog = WindowsDeviceCatalog(HostileBackend(), fail_level)

    assert catalog.list_microphones() == ()
    assert catalog.list_loopback_outputs() == ()
    assert catalog.read_level(AudioSource.ME, "input:1") == 0.0


def test_windows_device_adapter_does_not_swallow_base_exceptions() -> None:
    class InterruptedBackend:
        def list_microphones(self) -> tuple[CaptureDevice, ...]:
            raise KeyboardInterrupt()

        def list_loopback_outputs(self) -> tuple[CaptureDevice, ...]:
            return ()

    catalog = WindowsDeviceCatalog(InterruptedBackend(), lambda source, device: 0.0)
    with pytest.raises(KeyboardInterrupt):
        catalog.list_microphones()


def test_storage_probe_rejects_noncanonical_root(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "missing" / ".." / "sessions"
    result = LocalStorageReadiness().check(root, 1)
    assert result.writable is False


def test_storage_probe_never_removes_a_preexisting_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    collision = root / ".flowlens-write-probe-fixed"
    collision.write_bytes(b"keep")
    monkeypatch.setattr(
        "flowlens.adapters.storage.uuid.uuid4",
        lambda: SimpleNamespace(hex="fixed"),
    )

    result = LocalStorageReadiness().check(root, 1)

    assert result.writable is False
    assert collision.read_bytes() == b"keep"


def test_storage_probe_surfaces_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    original_unlink = Path.unlink

    def fail_probe_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.name.startswith(".flowlens-write-probe-"):
            raise OSError("locked")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_probe_unlink)
    result = LocalStorageReadiness().check(root, 1)

    assert result.writable is False
    assert result.reason == "cleanup_failed"


@pytest.mark.parametrize("operation", ["write", "fsync", "close"])
def test_storage_probe_contains_file_operation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    root = tmp_path.resolve()
    original = getattr(os, operation)
    failed = False

    def fail_once(*args: object) -> object:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("probe failure")
        return original(*args)

    monkeypatch.setattr(os, operation, fail_once)
    result = LocalStorageReadiness().check(root, 1)

    assert result.writable is False
    assert not list(root.glob(".flowlens-write-probe-*"))
