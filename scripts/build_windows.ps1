Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ValidatedRepoChild {
    param(
        [Parameter(Mandatory = $true)][string] $RepositoryRoot,
        [Parameter(Mandatory = $true)][string] $RelativePath
    )

    $root = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd(
        [char[]]@("\", "/")
    )
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    $prefix = $root + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes repository root: $RelativePath"
    }
    Assert-NoReparsePathComponents -RepositoryRoot $root -TargetPath $candidate
    return $candidate
}

function Assert-NoReparsePathComponents {
    param(
        [Parameter(Mandatory = $true)][string] $RepositoryRoot,
        [Parameter(Mandatory = $true)][string] $TargetPath
    )

    $rootItem = Get-Item -LiteralPath $RepositoryRoot -Force
    if (($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing repository root reparse point: $RepositoryRoot"
    }

    $relative = $TargetPath.Substring($RepositoryRoot.Length).TrimStart(
        [char[]]@("\\", "/")
    )
    if ([string]::IsNullOrEmpty($relative)) {
        return
    }
    $current = $RepositoryRoot
    foreach ($component in ($relative -split "[\\/]")) {
        if ([string]::IsNullOrEmpty($component)) {
            continue
        }
        $current = Join-Path $current $component
        if (-not (Test-Path -LiteralPath $current)) {
            continue
        }
        $item = Get-Item -LiteralPath $current -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing reparse-point path component: $current"
        }
    }
}

function Remove-ValidatedPackageTarget {
    param(
        [Parameter(Mandatory = $true)][string] $RepositoryRoot,
        [Parameter(Mandatory = $true)][string] $RelativePath
    )

    $target = Get-ValidatedRepoChild -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
    $root = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd(
        [char[]]@("\\", "/")
    )
    $allowedTargets = @(
        [System.IO.Path]::GetFullPath((Join-Path $root "build\\FlowLens")),
        [System.IO.Path]::GetFullPath((Join-Path $root "dist\\FlowLens"))
    )
    if (-not ($allowedTargets -contains $target)) {
        throw "Refusing non-package cleanup target: $RelativePath"
    }
    if (-not (Test-Path -LiteralPath $target)) {
        return
    }
    $item = Get-Item -LiteralPath $target -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to remove reparse-point target: $target"
    }
    Remove-Item -LiteralPath $target -Recurse -Force
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repositoryRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Expected virtual-environment Python is missing: $python"
}

$preflight = @"
import importlib.util
import sys
missing = [name for name in ('PyInstaller', 'PySide6', 'llama_cpp', 'ctranslate2', 'pyaudiowpatch') if importlib.util.find_spec(name) is None]
if missing:
    print('Missing package-build dependencies: ' + ', '.join(missing), file=sys.stderr)
    raise SystemExit(1)
"@
& $python -c $preflight
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Remove-ValidatedPackageTarget -RepositoryRoot $repositoryRoot -RelativePath "build\\FlowLens"
Remove-ValidatedPackageTarget -RepositoryRoot $repositoryRoot -RelativePath "dist\\FlowLens"

Push-Location -LiteralPath $repositoryRoot
try {
    & $python -m PyInstaller --clean --noconfirm --additional-hooks-dir packaging/hooks packaging/FlowLens.spec
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $licenseSource = Get-ValidatedRepoChild -RepositoryRoot $repositoryRoot -RelativePath "licenses"
    $licenseDestination = Get-ValidatedRepoChild -RepositoryRoot $repositoryRoot -RelativePath "dist\\FlowLens\\licenses"
    if (Test-Path -LiteralPath $licenseDestination) {
        throw "PyInstaller unexpectedly created the licenses target: $licenseDestination"
    }
    New-Item -ItemType Directory -LiteralPath $licenseDestination | Out-Null
    Get-ChildItem -LiteralPath $licenseSource -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $licenseDestination
    }

    & $python scripts/check_package.py --package dist/FlowLens
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
