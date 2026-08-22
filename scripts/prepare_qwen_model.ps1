#requires -Version 7.0

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$QwenRepository = "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"
$QwenRepositoryId = "Qwen/Qwen3-4B-Instruct-2507"
$QwenRevision = "cdbee75f17c01a7cc42f958dc650907174af0554"
$LlamaRepository = "https://github.com/ggml-org/llama.cpp.git"
$LlamaRevision = "2e92ecd0247d25f09797f8fdb044a166522fc05d"
$QwenFileName = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$QwenModelId = "qwen3-4b-instruct-2507"
$AsrModelId = "kotoba-whisper-v2.0-faster"
$QwenRelativePath = "qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

function Get-FullPath {
    param([Parameter(Mandatory)][string] $LiteralPath)

    return [System.IO.Path]::GetFullPath($LiteralPath)
}

function Assert-DirectChildPath {
    param(
        [Parameter(Mandatory)][string] $Root,
        [Parameter(Mandatory)][string] $Candidate,
        [Parameter(Mandatory)][string] $ExpectedName
    )

    $resolvedRoot = Get-FullPath -LiteralPath $Root
    $resolvedCandidate = Get-FullPath -LiteralPath $Candidate
    $parent = [System.IO.Directory]::GetParent($resolvedCandidate)
    if ($null -eq $parent -or
        $parent.FullName -cne $resolvedRoot -or
        [System.IO.Path]::GetFileName($resolvedCandidate) -cne $ExpectedName) {
        throw "The staging path is not the expected direct child of the model root."
    }
    return $resolvedCandidate
}

function Assert-SafeDirectory {
    param([Parameter(Mandatory)][string] $LiteralPath)

    $item = Get-Item -LiteralPath $LiteralPath -Force
    if (-not $item.PSIsContainer -or
        $null -ne $item.LinkType -or
        ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "A required directory is not a real local directory."
    }
}

function Assert-SafeRegularFile {
    param([Parameter(Mandatory)][string] $LiteralPath)

    $item = Get-Item -LiteralPath $LiteralPath -Force
    if ($item.PSIsContainer -or
        $null -ne $item.LinkType -or
        ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "A required artifact is not a regular local file."
    }
}

function Get-RequiredCommandPath {
    param([Parameter(Mandatory)][string] $Name)

    $command = Get-Command -Name $Name -CommandType Application -ErrorAction Stop |
        Select-Object -First 1
    if ($null -eq $command -or [string]::IsNullOrWhiteSpace($command.Source)) {
        throw "Required tool is unavailable: $Name"
    }
    return $command.Source
}

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory)][string] $Command,
        [Parameter(Mandatory)][string[]] $Arguments,
        [string] $WorkingDirectory
    )

    $pushed = $false
    try {
        if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
            Push-Location -LiteralPath $WorkingDirectory
            $pushed = $true
        }
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "A required preparation command failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        if ($pushed) {
            Pop-Location
        }
    }
}

function Assert-PreparationPrerequisites {
    if ($PSVersionTable.PSVersion.Major -lt 7) {
        throw "PowerShell 7 or newer is required."
    }
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        throw "LOCALAPPDATA is required."
    }

    $git = Get-RequiredCommandPath -Name "git.exe"
    $null = Get-RequiredCommandPath -Name "cmake.exe"
    $null = Get-RequiredCommandPath -Name "cl.exe"
    $null = Get-RequiredCommandPath -Name "nvcc.exe"
    $null = Get-RequiredCommandPath -Name "python.exe"

    # git lfs must already be installed; this command performs no download.
    & $git lfs version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Git LFS is required."
    }
}

function Invoke-ManifestHelper {
    param(
        [Parameter(Mandatory)][string] $HelperPath,
        [Parameter(Mandatory)][string[]] $Arguments
    )

    Assert-SafeRegularFile -LiteralPath $HelperPath
    $allArguments = @($HelperPath) + $Arguments
    Invoke-CheckedNative -Command "python.exe" -Arguments $allArguments
}

function Read-ValidatedManifest {
    param([Parameter(Mandatory)][string] $ManifestPath)

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        return [ordered]@{ schema_version = 1; models = [ordered]@{} }
    }
    Assert-SafeRegularFile -LiteralPath $ManifestPath
    try {
        return Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 |
            ConvertFrom-Json -AsHashtable -Depth 20
    }
    catch {
        throw "Validated model manifest changed before it could be read."
    }
}

function Assert-ExistingQwenInstallation {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary] $Manifest,
        [Parameter(Mandatory)][string] $TargetModelPath
    )

    $models = $Manifest["models"]
    $hasEntry = $models.Contains($QwenModelId)
    $hasFile = Test-Path -LiteralPath $TargetModelPath -PathType Leaf
    if ($hasEntry -ne $hasFile) {
        throw "Existing Qwen model and manifest entry are inconsistent."
    }
    if (-not $hasEntry) {
        return
    }
    Assert-SafeRegularFile -LiteralPath $TargetModelPath
    $entry = $models[$QwenModelId]
    if ([string]$entry["relative_path"] -cne $QwenRelativePath) {
        throw "Existing Qwen manifest path is invalid."
    }
    $installedHash = (Get-FileHash -LiteralPath $TargetModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($installedHash -cne [string]$entry["sha256"]) {
        throw "Existing Qwen model checksum does not match its manifest."
    }
}

function Publish-ModelArtifact {
    param(
        [Parameter(Mandatory)][string] $PreparedPath,
        [Parameter(Mandatory)][string] $TargetPath,
        [Parameter(Mandatory)][string] $BackupPath
    )

    Assert-SafeRegularFile -LiteralPath $PreparedPath
    if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
        Assert-SafeRegularFile -LiteralPath $TargetPath
        [System.IO.File]::Replace($PreparedPath, $TargetPath, $BackupPath, $true)
        return "replaced"
    }
    Move-Item -LiteralPath $PreparedPath -Destination $TargetPath
    return "created"
}

function Restore-PreviousModelArtifact {
    param(
        [Parameter(Mandatory)][string] $Publication,
        [Parameter(Mandatory)][string] $TargetPath,
        [Parameter(Mandatory)][string] $BackupPath
    )

    if ($Publication -ceq "replaced") {
        Restore-ReplacedArtifactFromBackup -TargetPath $TargetPath -BackupPath $BackupPath
        return
    }
    if ($Publication -ceq "created" -and
        (Test-Path -LiteralPath $TargetPath -PathType Leaf)) {
        Assert-SafeRegularFile -LiteralPath $TargetPath
        Remove-Item -LiteralPath $TargetPath -Force
    }
}

function Restore-ReplacedArtifactFromBackup {
    param(
        [Parameter(Mandatory)][string] $TargetPath,
        [Parameter(Mandatory)][string] $BackupPath
    )

    Assert-SafeRegularFile -LiteralPath $TargetPath
    Assert-SafeRegularFile -LiteralPath $BackupPath
    $restorePath = "$BackupPath.restore"
    if (Test-Path -LiteralPath $restorePath) {
        throw "Rollback restore staging artifact already exists."
    }
    [System.IO.File]::Copy($BackupPath, $restorePath, $false)
    try {
        Assert-SafeRegularFile -LiteralPath $restorePath
        [System.IO.File]::Replace($restorePath, $TargetPath, $null, $true)
    }
    finally {
        if (Test-Path -LiteralPath $restorePath -PathType Leaf) {
            Assert-SafeRegularFile -LiteralPath $restorePath
            Remove-Item -LiteralPath $restorePath -Force
        }
    }
}

function Publish-ManifestArtifact {
    param(
        [Parameter(Mandatory)][string] $PreparedPath,
        [Parameter(Mandatory)][string] $TargetPath,
        [Parameter(Mandatory)][string] $BackupPath
    )

    Assert-SafeRegularFile -LiteralPath $PreparedPath
    if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
        Assert-SafeRegularFile -LiteralPath $TargetPath
        [System.IO.File]::Replace($PreparedPath, $TargetPath, $BackupPath, $true)
        return "replaced"
    }
    Move-Item -LiteralPath $PreparedPath -Destination $TargetPath
    return "created"
}

function Restore-PreviousManifestArtifact {
    param(
        [Parameter(Mandatory)][string] $Publication,
        [Parameter(Mandatory)][string] $TargetPath,
        [Parameter(Mandatory)][string] $BackupPath
    )

    if ($Publication -ceq "replaced") {
        Restore-ReplacedArtifactFromBackup -TargetPath $TargetPath -BackupPath $BackupPath
        return
    }
    if ($Publication -ceq "created" -and
        (Test-Path -LiteralPath $TargetPath -PathType Leaf)) {
        Assert-SafeRegularFile -LiteralPath $TargetPath
        Remove-Item -LiteralPath $TargetPath -Force
    }
}

function Assert-PublishedInstallation {
    param(
        [Parameter(Mandatory)][string] $TargetModelPath,
        [Parameter(Mandatory)][string] $ManifestPath,
        [Parameter(Mandatory)][string] $ExpectedSha256,
        [Parameter(Mandatory)][string] $ManifestHelperPath
    )

    Assert-SafeRegularFile -LiteralPath $TargetModelPath
    $publishedHash = (Get-FileHash -LiteralPath $TargetModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($publishedHash -cne $ExpectedSha256) {
        throw "Published Qwen model checksum validation failed."
    }
    Invoke-ManifestHelper -HelperPath $ManifestHelperPath -Arguments @("validate", "--manifest", $ManifestPath)
    $publishedManifest = Read-ValidatedManifest -ManifestPath $ManifestPath
    $publishedEntry = $publishedManifest["models"][$QwenModelId]
    if ($null -eq $publishedEntry -or
        [string]$publishedEntry["relative_path"] -cne $QwenRelativePath -or
        [string]$publishedEntry["sha256"] -cne $ExpectedSha256) {
        throw "Published Qwen manifest validation failed."
    }
}

function Assert-RollbackState {
    param(
        [Parameter(Mandatory)][string] $TargetModelPath,
        [Parameter(Mandatory)][bool] $ModelExisted,
        [AllowNull()][string] $ExpectedModelSha256,
        [Parameter(Mandatory)][string] $ManifestPath,
        [Parameter(Mandatory)][bool] $ManifestExisted,
        [AllowNull()][string] $ExpectedManifestSha256
    )

    if ($ModelExisted) {
        Assert-SafeRegularFile -LiteralPath $TargetModelPath
        $modelHash = (Get-FileHash -LiteralPath $TargetModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($modelHash -cne $ExpectedModelSha256) {
            throw "Restored Qwen model does not match the prior artifact."
        }
    }
    elseif (Test-Path -LiteralPath $TargetModelPath) {
        throw "Rollback left a Qwen model that was not previously installed."
    }

    if ($ManifestExisted) {
        Assert-SafeRegularFile -LiteralPath $ManifestPath
        $manifestHash = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($manifestHash -cne $ExpectedManifestSha256) {
            throw "Restored manifest does not match the prior artifact."
        }
    }
    elseif (Test-Path -LiteralPath $ManifestPath) {
        throw "Rollback left a manifest that did not previously exist."
    }
}

function Remove-VerifiedStagingDirectory {
    param(
        [Parameter(Mandatory)][string] $ModelRoot,
        [Parameter(Mandatory)][string] $StagingRoot
    )

    $expectedName = ".staging-qwen-$PID"
    $verified = Assert-DirectChildPath -Root $ModelRoot -Candidate $StagingRoot -ExpectedName $expectedName
    if (Test-Path -LiteralPath $verified) {
        Assert-SafeDirectory -LiteralPath $verified
        Remove-Item -LiteralPath $StagingRoot -Recurse -Force
    }
}

function Invoke-QwenModelPreparation {
    Assert-PreparationPrerequisites

    $localAppData = Get-FullPath -LiteralPath $env:LOCALAPPDATA
    $flowLensRoot = Get-FullPath -LiteralPath (Join-Path $localAppData "FlowLens")
    $modelRoot = Get-FullPath -LiteralPath (Join-Path $flowLensRoot "models")
    $stagingName = ".staging-qwen-$PID"
    $stagingRoot = Assert-DirectChildPath -Root $modelRoot -Candidate (Join-Path $modelRoot $stagingName) -ExpectedName $stagingName
    $targetDirectory = Join-Path $modelRoot $QwenModelId
    $targetModel = Join-Path $targetDirectory $QwenFileName
    $manifestPath = Join-Path $modelRoot "manifest.json"
    $manifestHelperPath = Get-FullPath -LiteralPath (Join-Path $PSScriptRoot "../src/flowlens/discussion/model_manifest.py")
    Assert-SafeRegularFile -LiteralPath $manifestHelperPath

    if (Test-Path -LiteralPath $flowLensRoot) {
        Assert-SafeDirectory -LiteralPath $flowLensRoot
    }
    else {
        New-Item -ItemType Directory -Path $flowLensRoot | Out-Null
    }
    if (Test-Path -LiteralPath $modelRoot) {
        Assert-SafeDirectory -LiteralPath $modelRoot
    }
    else {
        New-Item -ItemType Directory -Path $modelRoot | Out-Null
    }
    Assert-SafeDirectory -LiteralPath $flowLensRoot
    Assert-SafeDirectory -LiteralPath $modelRoot
    if (Test-Path -LiteralPath $stagingRoot) {
        throw "The PID-scoped staging directory already exists."
    }

    Invoke-ManifestHelper -HelperPath $manifestHelperPath -Arguments @("validate", "--manifest", $manifestPath)
    $existingManifest = Read-ValidatedManifest -ManifestPath $manifestPath
    $null = Assert-DirectChildPath -Root $modelRoot -Candidate $targetDirectory -ExpectedName $QwenModelId
    if (Test-Path -LiteralPath $targetDirectory) {
        Assert-SafeDirectory -LiteralPath $targetDirectory
    }
    Assert-ExistingQwenInstallation -Manifest $existingManifest -TargetModelPath $targetModel
    $manifestExistedBefore = Test-Path -LiteralPath $manifestPath -PathType Leaf
    $manifestHashBefore = if ($manifestExistedBefore) {
        (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    else {
        $null
    }
    $modelExistedBefore = Test-Path -LiteralPath $targetModel -PathType Leaf
    $modelHashBefore = if ($modelExistedBefore) {
        (Get-FileHash -LiteralPath $targetModel -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    else {
        $null
    }

    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    Assert-SafeDirectory -LiteralPath $stagingRoot
    $publication = $null
    $manifestPublication = $null
    $createdTargetDirectory = $false
    $retainStagingForRollback = $false
    try {
        $qwenSource = Join-Path $stagingRoot "qwen-source"
        $llamaSource = Join-Path $stagingRoot "llama.cpp"
        $llamaBuild = Join-Path $stagingRoot "llama-build"
        $f16Path = Join-Path $stagingRoot "Qwen3-4B-Instruct-2507-F16.gguf"
        $preparedPath = Join-Path $stagingRoot $QwenFileName
        $preparedManifest = Join-Path $stagingRoot "manifest.json"
        $modelBackup = Join-Path $stagingRoot "previous-model.gguf"
        $manifestBackup = Join-Path $stagingRoot "previous-manifest.json"

        Invoke-CheckedNative -Command "git.exe" -Arguments @("clone", "--no-checkout", $QwenRepository, $qwenSource)
        Invoke-CheckedNative -Command "git.exe" -Arguments @("lfs", "install", "--local") -WorkingDirectory $qwenSource
        Invoke-CheckedNative -Command "git.exe" -Arguments @("checkout", "--detach", $QwenRevision) -WorkingDirectory $qwenSource
        Invoke-CheckedNative -Command "git.exe" -Arguments @("lfs", "pull") -WorkingDirectory $qwenSource
        $actualQwenRevision = (& git.exe -C $qwenSource rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $actualQwenRevision -cne $QwenRevision) {
            throw "Qwen checkout revision does not match the pinned revision."
        }

        Invoke-CheckedNative -Command "git.exe" -Arguments @("clone", "--no-checkout", $LlamaRepository, $llamaSource)
        Invoke-CheckedNative -Command "git.exe" -Arguments @("checkout", "--detach", $LlamaRevision) -WorkingDirectory $llamaSource
        $actualLlamaRevision = (& git.exe -C $llamaSource rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $actualLlamaRevision -cne $LlamaRevision) {
            throw "llama.cpp checkout revision does not match the pinned revision."
        }

        Invoke-CheckedNative -Command "cmake.exe" -Arguments @(
            "-S", $llamaSource,
            "-B", $llamaBuild,
            "-DGGML_CUDA=ON",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release"
        )
        Invoke-CheckedNative -Command "cmake.exe" -Arguments @(
            "--build", $llamaBuild,
            "--config", "Release",
            "--target", "llama-quantize"
        )

        $converter = Join-Path $llamaSource "convert_hf_to_gguf.py"
        Assert-SafeRegularFile -LiteralPath $converter
        Invoke-CheckedNative -Command "python.exe" -Arguments @(
            $converter,
            $qwenSource,
            "--outfile", $f16Path,
            "--outtype", "f16"
        )
        Assert-SafeRegularFile -LiteralPath $f16Path

        $quantizerCandidates = @(
            (Join-Path $llamaBuild "bin/Release/llama-quantize.exe"),
            (Join-Path $llamaBuild "bin/llama-quantize.exe")
        )
        $quantizer = $quantizerCandidates |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
            Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($quantizer)) {
            throw "Pinned llama.cpp quantizer was not produced."
        }
        Assert-SafeRegularFile -LiteralPath $quantizer
        Invoke-CheckedNative -Command $quantizer -Arguments @($f16Path, $preparedPath, "Q4_K_M")
        Assert-SafeRegularFile -LiteralPath $preparedPath

        $sha256 = (Get-FileHash -LiteralPath $preparedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sha256 -cnotmatch "^[0-9a-f]{64}$") {
            throw "Prepared Qwen SHA-256 is invalid."
        }
        Invoke-ManifestHelper -HelperPath $manifestHelperPath -Arguments @(
            "update", "--manifest", $manifestPath,
            "--output", $preparedManifest,
            "--repository", $QwenRepositoryId,
            "--source-revision", $QwenRevision,
            "--converter-revision", $LlamaRevision,
            "--runtime-format", "GGUF Q4_K_M",
            "--relative-path", $QwenRelativePath,
            "--sha256", $sha256,
            "--license", "Apache-2.0"
        )
        Invoke-ManifestHelper -HelperPath $manifestHelperPath -Arguments @("validate", "--manifest", $preparedManifest)
        $roundTrip = Read-ValidatedManifest -ManifestPath $preparedManifest
        if ([string]$roundTrip["models"][$QwenModelId]["sha256"] -cne $sha256) {
            throw "Prepared manifest validation failed."
        }

        if (-not (Test-Path -LiteralPath $targetDirectory)) {
            New-Item -ItemType Directory -Path $targetDirectory | Out-Null
            $createdTargetDirectory = $true
        }
        Assert-SafeDirectory -LiteralPath $targetDirectory
        $publication = Publish-ModelArtifact -PreparedPath $preparedPath -TargetPath $targetModel -BackupPath $modelBackup
        try {
            $manifestPublication = Publish-ManifestArtifact -PreparedPath $preparedManifest -TargetPath $manifestPath -BackupPath $manifestBackup
            Assert-PublishedInstallation -TargetModelPath $targetModel -ManifestPath $manifestPath -ExpectedSha256 $sha256 -ManifestHelperPath $manifestHelperPath
        }
        catch {
            $publicationError = $_
            $retainStagingForRollback = $true
            try {
                if ($null -ne $manifestPublication) {
                    Restore-PreviousManifestArtifact -Publication $manifestPublication -TargetPath $manifestPath -BackupPath $manifestBackup
                }
                Restore-PreviousModelArtifact -Publication $publication -TargetPath $targetModel -BackupPath $modelBackup
                Assert-RollbackState `
                    -TargetModelPath $targetModel `
                    -ModelExisted $modelExistedBefore `
                    -ExpectedModelSha256 $modelHashBefore `
                    -ManifestPath $manifestPath `
                    -ManifestExisted $manifestExistedBefore `
                    -ExpectedManifestSha256 $manifestHashBefore
                $retainStagingForRollback = $false
            }
            catch {
                throw "Publication rollback failed; staging and backups were retained at $stagingRoot. Restore the prior model and manifest from this directory before retrying. Rollback error: $($_.Exception.Message)"
            }
            throw $publicationError
        }
    }
    finally {
        if ($retainStagingForRollback) {
            Write-Warning "Rollback is incomplete. The verified PID-scoped staging directory was retained: $stagingRoot"
        }
        else {
            Remove-VerifiedStagingDirectory -ModelRoot $modelRoot -StagingRoot $stagingRoot
            if ($createdTargetDirectory -and
                (Test-Path -LiteralPath $targetDirectory -PathType Container)) {
                $null = Assert-DirectChildPath -Root $modelRoot -Candidate $targetDirectory -ExpectedName $QwenModelId
                Assert-SafeDirectory -LiteralPath $targetDirectory
                if (@(Get-ChildItem -LiteralPath $targetDirectory -Force).Count -eq 0) {
                    Remove-Item -LiteralPath $targetDirectory
                }
            }
        }
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-QwenModelPreparation
}
