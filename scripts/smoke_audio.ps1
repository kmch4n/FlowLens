param(
    [Parameter(Mandatory = $true)][string]$MicrophoneId,
    [Parameter(Mandatory = $true)][string]$LoopbackOutputId,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [ValidateSet(60)][int]$DurationSeconds = 60
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Arguments = @(
    "-m",
    "flowlens.smoke.audio",
    "--microphone-id",
    $MicrophoneId,
    "--loopback-output-id",
    $LoopbackOutputId,
    "--output-directory",
    $OutputDirectory,
    "--duration-seconds",
    $DurationSeconds
)

& $Python @Arguments
exit $LASTEXITCODE
