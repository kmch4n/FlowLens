param(
    [Parameter(Mandatory = $true)][string]$MicrophoneId,
    [Parameter(Mandatory = $true)][string]$LoopbackOutputId,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [ValidateSet(120)][int]$DurationSeconds = 120
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Arguments = @(
    "-m",
    "flowlens.smoke.asr",
    "--microphone-id",
    $MicrophoneId,
    "--loopback-output-id",
    $LoopbackOutputId,
    "--model-path",
    $ModelPath,
    "--output-directory",
    $OutputDirectory,
    "--duration-seconds",
    $DurationSeconds
)

& $Python @Arguments
exit $LASTEXITCODE
