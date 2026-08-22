"""Static and AST checks for the explicit Qwen preparation script."""

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path("scripts/prepare_qwen_model.ps1")


def _script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_qwen_preparation_pins_source_converter_and_quantization() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    script = _script()

    assert requirements.count("llama-cpp-python==0.3.35") == 1
    assert "cdbee75f17c01a7cc42f958dc650907174af0554" in script
    assert "2e92ecd0247d25f09797f8fdb044a166522fc05d" in script
    assert "Qwen/Qwen3-4B-Instruct-2507" in script
    assert "ggml-org/llama.cpp" in script
    assert "Q4_K_M" in script
    assert "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" in script
    assert "Apache-2.0" in script


def test_qwen_preparation_is_explicit_local_setup_without_runtime_fallback() -> None:
    script = _script()

    assert "#requires -Version 7.0" in script
    assert "[CmdletBinding()]" in script
    assert "Assert-PreparationPrerequisites" in script
    assert "git lfs" in script.lower()
    assert "nvcc" in script
    assert "cl.exe" in script
    assert "GGML_CUDA=ON" in script
    assert "convert_hf_to_gguf.py" in script
    assert "Get-FileHash" in script
    assert "from_pretrained" not in script


def test_qwen_preparation_uses_verified_pid_scoped_staging() -> None:
    script = _script()

    assert '".staging-qwen-$PID"' in script
    assert "Assert-DirectChildPath" in script
    assert "[System.IO.Path]::GetFullPath" in script
    assert "Remove-Item -LiteralPath $StagingRoot -Recurse" in script
    assert "Move-Item -LiteralPath" in script
    assert "Remove-Item -Recurse $env:LOCALAPPDATA" not in script


def test_qwen_preparation_updates_manifest_last_and_has_rollback() -> None:
    script = _script()

    model_publish = script.index("$publication = Publish-ModelArtifact")
    manifest_publish = script.index("$manifestPublication = Publish-ManifestArtifact")
    assert model_publish < manifest_publish
    assert "Restore-PreviousModelArtifact" in script
    assert "model_manifest.py" in script
    assert '"validate", "--manifest"' in script
    assert '"update", "--manifest"' in script
    assert "Get-UpdatedManifest" not in script
    assert "Write-DeterministicJson" not in script
    assert "manifest.json" in script
    assert "kotoba-whisper-v2.0-faster" in script
    assert "qwen3-4b-instruct-2507" in script


def test_qwen_preparation_retains_backups_when_rollback_cannot_be_verified() -> None:
    script = _script()

    retain_on = script.index("$retainStagingForRollback = $true")
    rollback_check = script.index("Assert-RollbackState", retain_on)
    retain_off = script.index("$retainStagingForRollback = $false", retain_on)
    cleanup_guard = script.index("if ($retainStagingForRollback)", retain_off)
    cleanup = script.index("Remove-VerifiedStagingDirectory", cleanup_guard)

    assert retain_on < rollback_check < retain_off < cleanup_guard < cleanup
    assert "[System.IO.File]::Copy($BackupPath, $restorePath, $false)" in script
    assert "[System.IO.File]::Replace($restorePath, $TargetPath" in script
    assert "Publication rollback failed; staging and backups were retained" in script
    assert "Restore the prior model and manifest from this directory" in script


def test_qwen_preparation_is_utf8_without_bom_and_lf_only() -> None:
    encoded = SCRIPT_PATH.read_bytes()

    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in encoded
    encoded.decode("utf-8")


def test_qwen_preparation_has_valid_powershell_ast() -> None:
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:FLOWLENS_AST_PATH, [ref]$tokens, [ref]$errors) "
        "> $null; if ($errors.Count -ne 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    environment = os.environ.copy()
    environment["FLOWLENS_AST_PATH"] = str(SCRIPT_PATH.resolve())
    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
