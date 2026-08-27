param(
    [Parameter(Mandatory = $true)][string] $Executable,
    [Parameter(Mandatory = $true)][string] $Report,
    [ValidateRange(1, 1440)][int] $MinimumActiveMinutes = 30,
    [switch] $RecoveryCheck,
    [ValidateRange(5, 600)][int] $RecoveryCaptureSeconds = 30,
    [ValidateRange(10, 600)][int] $RecoveryTimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Utf8Json {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][object] $Value
    )

    $json = $Value | ConvertTo-Json -Depth 12
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json + "`n", $encoding)
}

function Assert-RegularLocalFile {
    param([Parameter(Mandatory = $true)][string] $Path)

    $resolved = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).Path)
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer) {
        throw "Expected a file: $resolved"
    }
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing a reparse-point file: $resolved"
    }
    return $resolved
}

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Acceptance harness requires an elevated PowerShell."
    }
}

function Get-FlowLensSessionNames {
    param([Parameter(Mandatory = $true)][string] $SessionsRoot)

    if (-not (Test-Path -LiteralPath $SessionsRoot -PathType Container)) {
        return @()
    }
    return @(
        Get-ChildItem -LiteralPath $SessionsRoot -Directory -Force |
            ForEach-Object { $_.Name }
    )
}

function Get-NewSessionDirectory {
    param(
        [Parameter(Mandatory = $true)][string] $SessionsRoot,
        [Parameter(Mandatory = $true)][string[]] $Before
    )

    $newSessions = @(
        Get-ChildItem -LiteralPath $SessionsRoot -Directory -Force |
            Where-Object { $Before -notcontains $_.Name }
    )
    if ($newSessions.Count -ne 1) {
        throw "Expected exactly one new session, found $($newSessions.Count)."
    }
    $candidate = $newSessions[0]
    if (($candidate.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing a reparse-point session directory."
    }
    return [System.IO.Path]::GetFullPath($candidate.FullName)
}

function Get-LiveProcessIdentity {
    param([Parameter(Mandatory = $true)][int] $ProcessId)

    $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" `
        -ErrorAction SilentlyContinue
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $cim -or $null -eq $process) {
        return $null
    }
    try {
        $path = [System.IO.Path]::GetFullPath($process.Path)
        $startTicks = $process.StartTime.ToUniversalTime().Ticks
    }
    catch {
        throw "Could not establish process ownership for PID $ProcessId."
    }
    return [pscustomobject][ordered]@{
        process_id = [int] $ProcessId
        parent_process_id = [int] $cim.ParentProcessId
        start_time_utc_ticks = [long] $startTicks
        executable_path = $path
    }
}

function Test-LiveProcessIdentity {
    param([Parameter(Mandatory = $true)][object] $Identity)

    $process = Get-Process -Id $Identity.process_id -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $false
    }
    try {
        return (
            $process.StartTime.ToUniversalTime().Ticks -eq
                $Identity.start_time_utc_ticks -and
            [System.IO.Path]::GetFullPath($process.Path) -ieq
                $Identity.executable_path
        )
    }
    catch {
        return $false
    }
}

function Register-OwnedProcessTree {
    param(
        [Parameter(Mandatory = $true)][hashtable] $OwnedProcesses,
        [Parameter(Mandatory = $true)][int] $RootProcessId,
        [Parameter(Mandatory = $true)][string] $RootExecutable
    )

    if ($OwnedProcesses.Count -eq 0) {
        $rootIdentity = Get-LiveProcessIdentity -ProcessId $RootProcessId
        if ($null -eq $rootIdentity -or
            $rootIdentity.executable_path -ine $RootExecutable) {
            throw "FlowLens root process identity could not be established."
        }
        $OwnedProcesses[$RootProcessId] = $rootIdentity
    }

    $liveOwned = New-Object System.Collections.Generic.HashSet[int]
    foreach ($identity in $OwnedProcesses.Values) {
        if (Test-LiveProcessIdentity -Identity $identity) {
            [void] $liveOwned.Add([int] $identity.process_id)
        }
    }
    $snapshot = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    do {
        $changed = $false
        foreach ($candidate in $snapshot) {
            $candidateId = [int] $candidate.ProcessId
            $parentId = [int] $candidate.ParentProcessId
            if (-not $liveOwned.Contains($parentId) -or
                $OwnedProcesses.ContainsKey($candidateId)) {
                continue
            }
            $identity = Get-LiveProcessIdentity -ProcessId $candidateId
            if ($null -ne $identity -and
                $identity.parent_process_id -eq $parentId) {
                $OwnedProcesses[$candidateId] = $identity
                [void] $liveOwned.Add($candidateId)
                $changed = $true
            }
        }
    } while ($changed)
}

function Stop-OwnedProcessTree {
    param(
        [Parameter(Mandatory = $true)][hashtable] $OwnedProcesses,
        [Parameter(Mandatory = $true)][int] $RootProcessId
    )

    $live = @{}
    $mismatches = New-Object System.Collections.Generic.List[int]
    foreach ($identity in $OwnedProcesses.Values) {
        $process = Get-Process -Id $identity.process_id -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            continue
        }
        try {
            $matches = (
                $process.StartTime.ToUniversalTime().Ticks -eq
                    $identity.start_time_utc_ticks -and
                [System.IO.Path]::GetFullPath($process.Path) -ieq
                    $identity.executable_path
            )
        }
        catch {
            $matches = $false
        }
        if (-not $matches) {
            $mismatches.Add([int] $identity.process_id)
        }
        else {
            $live[[int] $identity.process_id] = $process
        }
    }
    if ($mismatches.Count -gt 0) {
        throw "Process ownership changed before cleanup: $($mismatches -join ', ')."
    }

    $orderedIds = @(
        $live.Keys | Sort-Object { if ($_ -eq $RootProcessId) { 1 } else { 0 } }
    )
    foreach ($processId in $orderedIds) {
        $process = $live[[int] $processId]
        if (-not $process.HasExited) {
            $process.Kill()
        }
    }
}

function Test-OwnedFirewallRule {
    param(
        [Parameter(Mandatory = $true)][string] $RuleName,
        [Parameter(Mandatory = $true)][string] $Program
    )

    $rules = @(Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)
    if ($rules.Count -ne 1) {
        return $false
    }
    $rule = $rules[0]
    if ($rule.Enabled -ne "True" -or $rule.Direction -ne "Outbound" -or $rule.Action -ne "Block") {
        return $false
    }
    $filters = @($rule | Get-NetFirewallApplicationFilter)
    return $filters.Count -eq 1 -and (
        [System.IO.Path]::GetFullPath($filters[0].Program) -eq $Program
    )
}

function Assert-CompletedApplicationReport {
    param([Parameter(Mandatory = $true)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "FlowLens did not write its application acceptance report."
    }
    $applicationReport = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if ($applicationReport.schema_version -ne 1 -or
        $applicationReport.exit_code -ne 0 -or
        $null -eq $applicationReport.controller -or
        $applicationReport.controller.state -cne "COMPLETED" -or
        $applicationReport.controller.completion_available -ne $true) {
        throw "FlowLens application report does not prove normal completion."
    }
}

function Start-FlowLensProcess {
    param(
        [Parameter(Mandatory = $true)][string] $Program,
        [Parameter(Mandatory = $true)][string] $AcceptanceReport,
        [Parameter(Mandatory = $true)][string] $Stdout,
        [Parameter(Mandatory = $true)][string] $Stderr
    )

    $quotedReport = '"' + $AcceptanceReport + '"'
    return Start-Process -FilePath $Program `
        -ArgumentList @("--acceptance-report", $quotedReport) `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru
}

function Wait-ForRecoveryArtifacts {
    param(
        [Parameter(Mandatory = $true)][string] $SessionsRoot,
        [Parameter(Mandatory = $true)][string[]] $Before,
        [Parameter(Mandatory = $true)][int] $TimeoutSeconds
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $session = Get-NewSessionDirectory -SessionsRoot $SessionsRoot -Before $Before
            $required = @("session.json", "mic.wav", "loopback.wav")
            if (($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $session $_) -PathType Leaf) }).Count -eq 0) {
                return $session
            }
        }
        catch {
            # A session may not exist yet; ambiguity remains fatal at the deadline.
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Timed out waiting for the new session artifacts."
}

function Wait-ForRecoveredStatus {
    param(
        [Parameter(Mandatory = $true)][string] $Session,
        [Parameter(Mandatory = $true)][int] $TimeoutSeconds
    )

    $manifestPath = Join-Path $Session "session.json"
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($manifest.status -ceq "recovered") {
                return
            }
            if ($manifest.status -ceq "completed") {
                throw "Recovery session was incorrectly marked completed."
            }
        }
        catch [System.Management.Automation.ItemNotFoundException] {
            # The Writer may be atomically replacing the manifest.
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Relaunched FlowLens did not recover the interrupted session."
}

function Add-AcceptanceSample {
    param(
        [Parameter(Mandatory = $true)][System.IO.StreamWriter] $Writer,
        [Parameter(Mandatory = $true)][hashtable] $OwnedProcesses,
        [Parameter(Mandatory = $true)][int] $ElapsedSeconds
    )

    [long] $rss = 0
    foreach ($identity in $OwnedProcesses.Values) {
        $process = Get-Process -Id $identity.process_id -ErrorAction SilentlyContinue
        if ($null -ne $process -and
            (Test-LiveProcessIdentity -Identity $identity)) {
            $rss += [long] $process.WorkingSet64
        }
    }
    $nvidiaSmiAvailable = $null -ne (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
    if ($nvidiaSmiAvailable) {
        & nvidia-smi.exe --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null | Out-Null
    }
    $sample = [ordered]@{
        elapsed_seconds = $ElapsedSeconds
        rss_bytes = $rss
        nvidia_smi_available = [bool] $nvidiaSmiAvailable
    }
    $Writer.WriteLine(($sample | ConvertTo-Json -Compress))
    $Writer.Flush()
}

if ($env:OS -ne "Windows_NT") {
    throw "Acceptance harness is Windows-only."
}
Assert-Administrator

$resolvedExecutable = Assert-RegularLocalFile -Path $Executable
if ([System.IO.Path]::GetFileName($resolvedExecutable) -cne "FlowLens.exe") {
    throw "Executable must be the packaged FlowLens.exe."
}
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Assert-RegularLocalFile -Path (Join-Path $repositoryRoot ".venv\Scripts\python.exe")
$reportPath = [System.IO.Path]::GetFullPath($Report)
$reportParent = Split-Path -Parent $reportPath
if (-not (Test-Path -LiteralPath $reportParent -PathType Container)) {
    New-Item -ItemType Directory -Path $reportParent | Out-Null
}
$reportParent = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $reportParent).Path)
$reportPath = Join-Path $reportParent ([System.IO.Path]::GetFileName($reportPath))
$minimumActiveSeconds = $MinimumActiveMinutes * 60

& $python (Join-Path $repositoryRoot "scripts\check_package.py") `
    --package (Split-Path -Parent $resolvedExecutable)
if ($LASTEXITCODE -ne 0) {
    throw "Packaged executable preflight failed."
}

$localAppData = [Environment]::GetFolderPath("LocalApplicationData")
$sessionsRoot = Join-Path $localAppData "FlowLens\sessions"
$before = @(Get-FlowLensSessionNames -SessionsRoot $sessionsRoot)
$ruleName = "FlowLens-Acceptance-$PID"
$existing = @(Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)
if ($existing.Count -ne 0) {
    throw "Refusing to alter an existing firewall rule: $ruleName"
}

$firewallEvidencePath = Join-Path $reportParent "firewall-evidence.json"
$samplesPath = Join-Path $reportParent "acceptance-samples.jsonl"
$appReportPath = Join-Path $reportParent "application-acceptance.json"
$stdoutPath = Join-Path $reportParent "application-stdout.log"
$stderrPath = Join-Path $reportParent "application-stderr.log"
$ruleCreated = $false
$activeThroughout = $true
$rootProcess = $null
$ownedProcesses = @{}
$sampleWriter = $null
$failure = $null
$cleanupFailure = $null

try {
    New-NetFirewallRule -DisplayName $ruleName -Direction Outbound -Action Block `
        -Program $resolvedExecutable -Profile Any -Enabled True | Out-Null
    $ruleCreated = $true
    if (-not (Test-OwnedFirewallRule -RuleName $ruleName -Program $resolvedExecutable)) {
        throw "Owned outbound firewall rule failed verification."
    }

    $rootProcess = Start-FlowLensProcess -Program $resolvedExecutable `
        -AcceptanceReport $appReportPath -Stdout $stdoutPath -Stderr $stderrPath
    Register-OwnedProcessTree -OwnedProcesses $ownedProcesses `
        -RootProcessId $rootProcess.Id -RootExecutable $resolvedExecutable

    if ($RecoveryCheck) {
        $session = Wait-ForRecoveryArtifacts -SessionsRoot $sessionsRoot `
            -Before $before -TimeoutSeconds $RecoveryTimeoutSeconds
        Start-Sleep -Seconds $RecoveryCaptureSeconds
        Register-OwnedProcessTree -OwnedProcesses $ownedProcesses `
            -RootProcessId $rootProcess.Id -RootExecutable $resolvedExecutable
        Stop-OwnedProcessTree -OwnedProcesses $ownedProcesses `
            -RootProcessId $rootProcess.Id
        $rootProcess.WaitForExit()
        if (-not (Test-OwnedFirewallRule -RuleName $ruleName -Program $resolvedExecutable)) {
            $activeThroughout = $false
            throw "Outbound firewall block was not active at forced termination."
        }

        $recoveryAppReport = Join-Path $reportParent "application-recovery.json"
        $rootProcess = Start-FlowLensProcess -Program $resolvedExecutable `
            -AcceptanceReport $recoveryAppReport -Stdout $stdoutPath -Stderr $stderrPath
        $ownedProcesses = @{}
        Register-OwnedProcessTree -OwnedProcesses $ownedProcesses `
            -RootProcessId $rootProcess.Id -RootExecutable $resolvedExecutable
        Wait-ForRecoveredStatus -Session $session -TimeoutSeconds $RecoveryTimeoutSeconds
        Register-OwnedProcessTree -OwnedProcesses $ownedProcesses `
            -RootProcessId $rootProcess.Id -RootExecutable $resolvedExecutable
        Stop-OwnedProcessTree -OwnedProcesses $ownedProcesses `
            -RootProcessId $rootProcess.Id
        $rootProcess.WaitForExit()
        & $python (Join-Path $repositoryRoot "scripts\validate_session.py") $session `
            --minimum-active-seconds 0 --require-recovered | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Recovered session artifact validation failed."
        }
        $activeThroughout = $activeThroughout -and (
            Test-OwnedFirewallRule -RuleName $ruleName -Program $resolvedExecutable
        )
        if (-not $activeThroughout) {
            throw "Outbound firewall block did not remain active during recovery."
        }
        $recoveryReport = [ordered]@{
            schema_version = 1
            passed = [bool] $activeThroughout
            errors = @()
            local_only = $true
            recovered_session = $session
            firewall_rule = $ruleName
        }
        Write-Utf8Json -Path $reportPath -Value $recoveryReport
    }
    else {
        $encoding = New-Object System.Text.UTF8Encoding($false)
        $sampleWriter = New-Object System.IO.StreamWriter($samplesPath, $false, $encoding)
        $started = [DateTime]::UtcNow
        $lastSample = -5
        while (-not $rootProcess.HasExited) {
            Register-OwnedProcessTree -OwnedProcesses $ownedProcesses `
                -RootProcessId $rootProcess.Id -RootExecutable $resolvedExecutable
            $elapsed = [int] [Math]::Floor(([DateTime]::UtcNow - $started).TotalSeconds)
            if ($elapsed -ge ($lastSample + 5)) {
                Add-AcceptanceSample -Writer $sampleWriter `
                    -OwnedProcesses $ownedProcesses -ElapsedSeconds $elapsed
                $lastSample = $elapsed
            }
            if (-not (Test-OwnedFirewallRule -RuleName $ruleName -Program $resolvedExecutable)) {
                $activeThroughout = $false
                throw "Outbound firewall block did not remain active."
            }
            Start-Sleep -Milliseconds 250
        }
        $sampleWriter.Dispose()
        $sampleWriter = $null
        if ($rootProcess.ExitCode -ne 0) {
            throw "FlowLens exited with code $($rootProcess.ExitCode)."
        }
        Assert-CompletedApplicationReport -Path $appReportPath
        $session = Get-NewSessionDirectory -SessionsRoot $sessionsRoot -Before $before
        $evidence = [ordered]@{
            schema_version = 1
            program = $resolvedExecutable
            rule_name = $ruleName
            outbound_blocked = $true
            active_throughout = [bool] $activeThroughout
        }
        Write-Utf8Json -Path $firewallEvidencePath -Value $evidence
        & $python (Join-Path $repositoryRoot "scripts\collect_acceptance.py") `
            --session $session --samples $samplesPath `
            --offline-evidence $firewallEvidencePath `
            --application-report $appReportPath --output $reportPath `
            --minimum-active-seconds $minimumActiveSeconds --require-pause
        if ($LASTEXITCODE -ne 0) {
            throw "Acceptance thresholds or artifact validation failed."
        }
    }
}
catch {
    $failure = $_
    $failedReport = [ordered]@{
        schema_version = 1
        passed = $false
        errors = @($_.Exception.Message)
        local_only = $true
        deferred = $false
    }
    Write-Utf8Json -Path $reportPath -Value $failedReport
}
finally {
    try {
        if ($null -ne $sampleWriter) {
            $sampleWriter.Dispose()
        }
        if ($ownedProcesses.Count -gt 0) {
            if ($null -ne $rootProcess) {
                Register-OwnedProcessTree -OwnedProcesses $ownedProcesses `
                    -RootProcessId $rootProcess.Id -RootExecutable $resolvedExecutable
            }
            Stop-OwnedProcessTree -OwnedProcesses $ownedProcesses `
                -RootProcessId $rootProcess.Id
        }
        if ($ruleCreated) {
            $owned = @(Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)
            if ($owned.Count -eq 0) {
                throw "Owned firewall rule disappeared before cleanup."
            }
            if (-not (Test-OwnedFirewallRule -RuleName $ruleName `
                    -Program $resolvedExecutable)) {
                throw "Refusing to remove a firewall rule that is no longer owned."
            }
            Remove-NetFirewallRule -DisplayName $ruleName
        }
    }
    catch {
        $cleanupFailure = $_
    }
}

if ($null -ne $cleanupFailure) {
    $cleanupReport = [ordered]@{
        schema_version = 1
        passed = $false
        errors = @($cleanupFailure.Exception.Message)
        local_only = $true
        deferred = $false
    }
    Write-Utf8Json -Path $reportPath -Value $cleanupReport
    throw $cleanupFailure
}
if ($null -ne $failure) {
    throw $failure
}
