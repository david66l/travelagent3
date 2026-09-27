#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9][a-z0-9._-]{7,95}$")]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{64}$")]
    [string]$ExpectedManifestSha256,

    [string]$RemoteHost = "connect.nmb1.seetacloud.com",

    [int]$RemotePort = 13793,

    [string]$RemoteUser = "root",

    [ValidateRange(60, 7200)]
    [int]$StageTimeoutSeconds = 1800,

    [ValidateRange(10, 300)]
    [int]$MonitorIntervalSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$expectedRunId = "h006-full-agent-loop-v1-20260904-r2"
if ($RunId -ne $expectedRunId) {
    throw "This reviewed wrapper is locked to RunId $expectedRunId"
}

$repoRoot = (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $repoRoot "backend\.venv\Scripts\python.exe"
$auditScript = Join-Path $repoRoot "scripts\audit_h006_full_agent_loop_run.py"
$serverHelper = Join-Path $repoRoot "scripts\start_h006_checkpoint32_server.ps1"
$runtimeManifest = Join-Path $repoRoot "experiments\H006-FULL-AGENT-LOOP-RUNTIME-MANIFEST.json"
$sshExe = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$policyModel = "travel-h006-checkpoint32"
$policyBackend = "http://127.0.0.1:18000/v1"
$intentModel = "deepseek-v4-flash"
$intentBackend = "cloud-openai-compatible"
$outputRoot = Join-Path $repoRoot "artifacts\native-react-posttraining"
$runDir = Join-Path $outputRoot $RunId
$monitorLog = Join-Path $runDir "monitor.jsonl"
$controllerLog = Join-Path $runDir "controller.log"
$stageRows = [System.Collections.Generic.List[object]]::new()
$serverProcess = $null
$serverStartedByThisRun = $false
$remoteServerPid = $null
$remoteServerPgid = $null
$remoteServerStartTicks = $null
$gpuBaselineUsedMiB = $null
$transcriptStarted = $false
$evaluationEnvNames = @(
    "LOCAL_LLM_ENABLED",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY"
)
$evaluationEnvSnapshot = @{}
foreach ($name in $evaluationEnvNames) {
    $evaluationEnvSnapshot[$name] = [pscustomobject]@{
        present = Test-Path -LiteralPath "Env:$name"
        value = [Environment]::GetEnvironmentVariable($name, "Process")
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [object]$Value
    )
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Set-EvaluationEnvironment {
    [Environment]::SetEnvironmentVariable("LOCAL_LLM_ENABLED", "false", "Process")
    foreach ($name in @(
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2"
    )) {
        [Environment]::SetEnvironmentVariable($name, "false", "Process")
    }
    foreach ($name in @("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")) {
        [Environment]::SetEnvironmentVariable($name, "", "Process")
    }
}

function Restore-EvaluationEnvironment {
    foreach ($name in $evaluationEnvNames) {
        $original = $evaluationEnvSnapshot[$name]
        if ($original.present) {
            [Environment]::SetEnvironmentVariable($name, $original.value, "Process")
        } else {
            [Environment]::SetEnvironmentVariable($name, $null, "Process")
        }
    }
}

function Write-MonitorEvent {
    param(
        [Parameter(Mandatory = $true)] [string]$Event,
        [Parameter(Mandatory = $true)] [hashtable]$Data
    )
    $row = [ordered]@{
        timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        event = $Event
    }
    foreach ($key in $Data.Keys) {
        $row[$key] = $Data[$key]
    }
    ($row | ConvertTo-Json -Compress -Depth 12) | Add-Content -LiteralPath $monitorLog -Encoding utf8
}

function Get-OptionalProperty {
    param(
        [Parameter(Mandatory = $true)] [object]$Object,
        [Parameter(Mandatory = $true)] [string]$Name
    )
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Stop-ProcessTree {
    param([Parameter(Mandatory = $true)] [int]$ProcessId)
    $output = & "$env:WINDIR\System32\taskkill.exe" /PID $ProcessId /T /F 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0 -and (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        throw "taskkill failed for PID $ProcessId with exit code $exitCode"
    }
}

function Invoke-MonitoredProcess {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [string]$FilePath,
        [Parameter(Mandatory = $true)] [string[]]$Arguments,
        [Parameter(Mandatory = $true)] [string]$StdoutPath,
        [Parameter(Mandatory = $true)] [string]$StderrPath,
        [Parameter(Mandatory = $true)] [int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)] [int]$CheckSeconds
    )
    $started = [DateTimeOffset]::UtcNow
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru
    Start-Sleep -Seconds 1
    $process.Refresh()
    $initialRss = if ($process.HasExited) { 0L } else { [int64]$process.WorkingSet64 }
    $peakRss = $initialRss
    $initialStdoutBytes = if (Test-Path -LiteralPath $StdoutPath) {
        (Get-Item -LiteralPath $StdoutPath).Length
    } else { 0L }
    $initialStderrBytes = if (Test-Path -LiteralPath $StderrPath) {
        (Get-Item -LiteralPath $StderrPath).Length
    } else { 0L }
    $lastBytes = [int64]$initialStdoutBytes + [int64]$initialStderrBytes
    $lastOutputChange = [DateTimeOffset]::UtcNow
    $stallAdvisoryEmitted = $false
    $stallAlerts = 0
    $resourceAlerted = $false
    $timedOut = $false
    Write-MonitorEvent -Event "PROCESS_STARTED" -Data @{
        name = $Name
        pid = $process.Id
        timeout_seconds = $TimeoutSeconds
        initial_rss_bytes = $initialRss
    }

    while (-not $process.HasExited) {
        $beforeSleepElapsed = ([DateTimeOffset]::UtcNow - $started).TotalSeconds
        if ($beforeSleepElapsed -ge $TimeoutSeconds) {
            $timedOut = $true
            Write-MonitorEvent -Event "HARD_TIMEOUT" -Data @{
                name = $Name
                pid = $process.Id
                elapsed_seconds = [Math]::Round($beforeSleepElapsed, 3)
            }
            Stop-ProcessTree -ProcessId $process.Id
            break
        }
        $remainingSeconds = $TimeoutSeconds - $beforeSleepElapsed
        $sleepSeconds = [Math]::Max(1, [Math]::Min($CheckSeconds, [Math]::Ceiling($remainingSeconds)))
        Start-Sleep -Seconds $sleepSeconds
        $process.Refresh()
        $elapsed = ([DateTimeOffset]::UtcNow - $started).TotalSeconds
        if (-not $process.HasExited -and $elapsed -ge $TimeoutSeconds) {
            $timedOut = $true
            Write-MonitorEvent -Event "HARD_TIMEOUT" -Data @{
                name = $Name
                pid = $process.Id
                elapsed_seconds = [Math]::Round($elapsed, 3)
            }
            Stop-ProcessTree -ProcessId $process.Id
            break
        }
        $stdoutBytes = if (Test-Path -LiteralPath $StdoutPath) {
            (Get-Item -LiteralPath $StdoutPath).Length
        } else { 0L }
        $stderrBytes = if (Test-Path -LiteralPath $StderrPath) {
            (Get-Item -LiteralPath $StderrPath).Length
        } else { 0L }
        $totalBytes = [int64]$stdoutBytes + [int64]$stderrBytes
        if ($totalBytes -ne $lastBytes) {
            $lastOutputChange = [DateTimeOffset]::UtcNow
            $stallAdvisoryEmitted = $false
        }
        $lastBytes = $totalBytes
        if (-not $process.HasExited) {
            $peakRss = [Math]::Max($peakRss, [int64]$process.WorkingSet64)
        }
        Write-MonitorEvent -Event "PROCESS_CHECK" -Data @{
            name = $Name
            pid = $process.Id
            elapsed_seconds = [Math]::Round($elapsed, 3)
            stdout_bytes = $stdoutBytes
            stderr_bytes = $stderrBytes
            rss_bytes = if ($process.HasExited) { 0L } else { [int64]$process.WorkingSet64 }
        }
        $secondsSinceOutput = ([DateTimeOffset]::UtcNow - $lastOutputChange).TotalSeconds
        if (-not $stallAdvisoryEmitted -and $secondsSinceOutput -ge 90) {
            $stallAdvisoryEmitted = $true
            $stallAlerts += 1
            Write-MonitorEvent -Event "OUTPUT_STALL_ADVISORY" -Data @{
                name = $Name
                pid = $process.Id
                seconds_without_output = [Math]::Round($secondsSinceOutput, 3)
            }
        }
        if (-not $resourceAlerted -and $initialRss -gt 0 -and $peakRss -gt (3 * $initialRss)) {
            $resourceAlerted = $true
            Write-MonitorEvent -Event "RESOURCE_ALERT_ADVISORY" -Data @{
                name = $Name
                pid = $process.Id
                initial_rss_bytes = $initialRss
                peak_rss_bytes = $peakRss
            }
        }
    }
    if (-not $process.HasExited -and -not $process.WaitForExit(10000)) {
        throw "$Name PID $($process.Id) did not exit within 10 seconds after termination"
    }
    $process.WaitForExit()
    $process.Refresh()
    $ended = [DateTimeOffset]::UtcNow
    $exitCode = if ($timedOut) { 124 } else { $process.ExitCode }
    $result = [ordered]@{
        name = $Name
        pid = $process.Id
        started_at = $started.ToString("o")
        ended_at = $ended.ToString("o")
        elapsed_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
        exit_code = $exitCode
        timed_out = $timedOut
        initial_rss_bytes = $initialRss
        peak_rss_bytes = $peakRss
        output_stall_alerts = $stallAlerts
        stdout = (Split-Path -Leaf $StdoutPath)
        stderr = (Split-Path -Leaf $StderrPath)
    }
    Write-MonitorEvent -Event "PROCESS_FINISHED" -Data @{
        name = $Name
        pid = $process.Id
        exit_code = $exitCode
        timed_out = $timedOut
    }
    return [pscustomobject]$result
}

function Assert-StageInfrastructure {
    param(
        [Parameter(Mandatory = $true)] [string]$ReportPath,
        [Parameter(Mandatory = $true)] [int]$ExpectedTotal,
        [Parameter(Mandatory = $true)] [string]$StageName
    )
    if (-not (Test-Path -LiteralPath $ReportPath)) {
        throw "$StageName did not create $ReportPath"
    }
    $report = Get-Content -Raw -LiteralPath $ReportPath | ConvertFrom-Json
    if ([int]$report.summary.total -ne $ExpectedTotal -or @($report.records).Count -ne $ExpectedTotal) {
        throw "$StageName produced an incomplete report"
    }
    if ([string]$report.policy_model -ne $policyModel -or [string]$report.policy_backend -ne $policyBackend) {
        throw "$StageName model/backend identity mismatch"
    }
    if ([string]$report.intent_model -ne $intentModel) {
        throw "$StageName intent model identity mismatch"
    }
    foreach ($record in @($report.records)) {
        $recordError = Get-OptionalProperty -Object $record -Name "error"
        $agentStatus = Get-OptionalProperty -Object $record -Name "agent_status"
        if ($recordError -or [string]$agentStatus -eq "exception") {
            throw "$StageName contains an infrastructure exception in $($record.case_id)"
        }
        $failures = @(Get-OptionalProperty -Object $record -Name "failures")
        if (@($failures | Where-Object { [string]$_ -like "EXCEPTION:*" }).Count -gt 0) {
            throw "$StageName contains a caught exception in $($record.case_id)"
        }
        $intent = Get-OptionalProperty -Object $record -Name "intent"
        $intentInference = Get-OptionalProperty -Object $record -Name "intent_inference"
        $parseSource = if ($null -eq $intent) {
            $null
        } else {
            Get-OptionalProperty -Object $intent -Name "parse_source"
        }
        $actualIntentModel = if ($null -eq $intentInference) {
            $null
        } else {
            Get-OptionalProperty -Object $intentInference -Name "model"
        }
        $actualIntentBackend = if ($null -eq $intentInference) {
            $null
        } else {
            Get-OptionalProperty -Object $intentInference -Name "backend"
        }
        if ([string]$parseSource -ne "llm") {
            throw "$StageName used intent fallback in $($record.case_id)"
        }
        if ([string]$actualIntentModel -ne $intentModel -or
            [string]$actualIntentBackend -ne $intentBackend) {
            throw "$StageName intent model/backend changed in $($record.case_id)"
        }
        $runtimeError = Get-OptionalProperty -Object $record -Name "runtime_error"
        if ($runtimeError) {
            throw "$StageName contains runtime error in $($record.case_id)"
        }
        $runtimeErrors = Get-OptionalProperty -Object $record -Name "runtime_errors"
        if ($runtimeErrors) {
            foreach ($property in $runtimeErrors.PSObject.Properties) {
                if ($property.Value) {
                    throw "$StageName contains runtime error in $($record.case_id): $($property.Name)"
                }
            }
        }
    }
}

function Invoke-RemoteCapture {
    param(
        [Parameter(Mandatory = $true)] [string]$Command,
        [ValidateRange(5, 600)] [int]$TimeoutSeconds = 15
    )
    $arguments = @(
        "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2",
        "-p", $RemotePort.ToString(),
        "$RemoteUser@$RemoteHost",
        $Command
    )
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $sshExe
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $arguments) {
        [void]$startInfo.ArgumentList.Add([string]$argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { throw "could not start SSH process" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill($true)
        if (-not $process.WaitForExit(10000)) {
            throw "SSH process did not exit within 10 seconds after forced termination"
        }
        throw "remote command exceeded local ${TimeoutSeconds}s deadline"
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) {
        throw "remote command failed with exit code $($process.ExitCode): $($stderr.Trim())"
    }
    return @(($stdout -split "`r?`n") | Where-Object { $_ -ne "" })
}

function Stop-H006Server {
    param([switch]$RequireVerified)
    if (-not $serverStartedByThisRun) {
        return
    }
    $cleanupPassed = $false
    $remotePortReleased = $false
    $processGroupEmpty = $false
    $gpuMemoryRecovered = $false
    $gpuUsedAfterMiB = $null
    $cleanupOutput = @()
    $cleanupError = $null
    try {
        if ($null -eq $remoteServerPid -or $null -eq $remoteServerPgid -or
            $null -eq $remoteServerStartTicks -or $null -eq $gpuBaselineUsedMiB) {
            throw "remote process identity was not captured"
        }
        $cleanupCommand = 'set -e; pid=' + [string]$remoteServerPid +
            '; pgid=' + [string]$remoteServerPgid +
            '; expected=' + [string]$remoteServerStartTicks +
            '; gpu_baseline=' + [string]$gpuBaselineUsedMiB +
            '; if [ "$pid" != "$pgid" ]; then echo REMOTE_GROUP_NOT_DEDICATED; exit 41; fi' +
            '; status=already_stopped; leader_present=0; if [ -r "/proc/$pid/stat" ]; then leader_present=1; ' +
            'actual=$(/root/miniconda3/bin/python -c "import sys; d=open(\"/proc/\"+sys.argv[1]+\"/stat\").read(); print(d[d.rfind(\")\")+2:].split()[19])" "$pid"); ' +
            'actual_pgid=$(ps -o pgid= -p "$pid" | tr -d '' ''); ' +
            'cmd=$(tr ''\0'' '' '' < "/proc/$pid/cmdline"); ' +
            'if [ "$actual" != "$expected" ]; then echo REMOTE_IDENTITY_MISMATCH; exit 42; fi; ' +
            'if [ "$actual_pgid" != "$pgid" ]; then echo REMOTE_PGID_MISMATCH; exit 43; fi; ' +
            'case "$cmd" in *"--port 8000"*"--lora-modules travel-h006-checkpoint32="*) ;; ' +
            '*) echo REMOTE_COMMAND_MISMATCH; exit 44;; esac; ' +
            'elif pgrep -g "$pgid" >/dev/null; then echo LEADER_GONE_GROUP_PRESENT; exit 48; fi; ' +
            'if [ "$leader_present" -eq 1 ] && pgrep -g "$pgid" >/dev/null; then kill -TERM -- "-$pgid"; status=term_sent; fi; ' +
            'for i in $(seq 1 20); do if ! pgrep -g "$pgid" >/dev/null; then status=stopped; break; fi; sleep 0.5; done; ' +
            'if [ "$leader_present" -eq 1 ] && pgrep -g "$pgid" >/dev/null; then kill -KILL -- "-$pgid"; status=killed; sleep 1; fi; ' +
            'if pgrep -g "$pgid" >/dev/null; then echo PROCESS_GROUP_REMAINS; exit 45; fi; ' +
            'timeout 5s /root/miniconda3/bin/python -c "import socket; s=socket.socket(); s.bind((\"127.0.0.1\",8000)); s.close()"; ' +
            'gpu_used=$(timeout 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d '' ''); ' +
            'case "$gpu_used" in ''''|*[!0-9]*) echo GPU_QUERY_INVALID; exit 46;; esac; ' +
            'if [ "$gpu_used" -gt $((gpu_baseline + 1024)) ]; then echo GPU_MEMORY_NOT_RECOVERED=$gpu_used; exit 47; fi; ' +
            'echo PROCESS_GROUP_EMPTY=true; echo REMOTE_PORT_RELEASED=true; echo GPU_MEMORY_RECOVERED=true; echo GPU_USED_AFTER_MIB=$gpu_used; ' +
            'echo CLEANUP_OK=$status'
        $cleanupOutput = @(Invoke-RemoteCapture -Command $cleanupCommand -TimeoutSeconds 30)
        $processGroupEmpty = @($cleanupOutput | Where-Object { $_ -eq 'PROCESS_GROUP_EMPTY=true' }).Count -eq 1
        $remotePortReleased = @($cleanupOutput | Where-Object { $_ -eq 'REMOTE_PORT_RELEASED=true' }).Count -eq 1
        $gpuMemoryRecovered = @($cleanupOutput | Where-Object { $_ -eq 'GPU_MEMORY_RECOVERED=true' }).Count -eq 1
        $gpuAfterLine = $cleanupOutput | Where-Object { $_ -match '^GPU_USED_AFTER_MIB=\d+$' } | Select-Object -First 1
        if ($gpuAfterLine) {
            $gpuUsedAfterMiB = [int]([regex]::Match($gpuAfterLine, '\d+$').Value)
        }
        $cleanupPassed = (
            @($cleanupOutput | Where-Object { $_ -match '^CLEANUP_OK=' }).Count -eq 1 -and
            $processGroupEmpty -and $remotePortReleased -and $gpuMemoryRecovered
        )
    } catch {
        $cleanupError = [string]$_
    }
    try {
        if ($null -ne $serverProcess) {
            $serverProcess.Refresh()
            if (-not $serverProcess.HasExited) {
                Stop-ProcessTree -ProcessId $serverProcess.Id
            }
            if (-not $serverProcess.HasExited -and -not $serverProcess.WaitForExit(10000)) {
                throw "local SSH process did not exit after cleanup"
            }
        }
    } catch {
        $cleanupPassed = $false
        $cleanupError = ((@($cleanupError, [string]$_) | Where-Object { $_ }) -join "; ")
    }
    Write-JsonFile -Path (Join-Path $runDir "server-cleanup.json") -Value ([ordered]@{
        checked_at = [DateTimeOffset]::UtcNow.ToString("o")
        remote_pid = $remoteServerPid
        remote_pgid = $remoteServerPgid
        remote_start_ticks = $remoteServerStartTicks
        cleanup_passed = $cleanupPassed
        process_group_empty = $processGroupEmpty
        remote_port_released = $remotePortReleased
        gpu_memory_recovered = $gpuMemoryRecovered
        gpu_baseline_used_mib = $gpuBaselineUsedMiB
        gpu_used_after_mib = $gpuUsedAfterMiB
        output = $cleanupOutput
        error = $cleanupError
    })
    $script:serverStartedByThisRun = $false
    $script:serverProcess = $null
    if ($RequireVerified -and -not $cleanupPassed) {
        throw "H006 server cleanup could not be verified: $cleanupError"
    }
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment is missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath $runtimeManifest)) {
    throw "Reviewed runtime manifest is missing: $runtimeManifest"
}
$actualManifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
if ($actualManifestSha256 -ne $ExpectedManifestSha256) {
    throw "Reviewed runtime manifest hash mismatch"
}
try {
    Set-EvaluationEnvironment
    & $pythonExe $auditScript verify `
        --repo-root $repoRoot `
        --expected-manifest $runtimeManifest `
        --run-id $RunId
    if ($LASTEXITCODE -ne 0) { throw "reviewed runtime verification failed" }
} finally {
    Restore-EvaluationEnvironment
}
if (Test-Path -LiteralPath $runDir) {
    throw "Run directory already exists; refusing to mix results: $runDir"
}
$reachability = @(
    Invoke-RemoteCapture -Command "printf 'H006_SSH_READY\n'" -TimeoutSeconds 15
)
if ($reachability.Count -ne 1 -or $reachability[0] -ne "H006_SSH_READY") {
    throw "remote SSH reachability preflight failed"
}
New-Item -ItemType Directory -Path $runDir | Out-Null
Start-Transcript -LiteralPath $controllerLog | Out-Null
$transcriptStarted = $true
Set-EvaluationEnvironment

try {
    Write-Host "Run ID: $RunId"
    Write-Host "Scope: internal smoke/development diagnostic only"
    Write-JsonFile -Path (Join-Path $runDir "run-contract.json") -Value ([ordered]@{
        schema_version = "h006-full-agent-loop-run-contract.v1"
        run_id = $RunId
        scope = "internal full-Agent-Loop smoke/development diagnostic only"
        policy_model = $policyModel
        policy_backend = $policyBackend
        intent_model = $intentModel
        intent_backend = $intentBackend
        stage_timeout_seconds = $StageTimeoutSeconds
        monitor_interval_seconds = $MonitorIntervalSeconds
        auto_retry = $false
        external_tracing_enabled = $false
        expected_manifest = $runtimeManifest
        expected_manifest_sha256 = $ExpectedManifestSha256
        promotion_val_opened = $false
        sealed_160_opened = $false
        production_promotion_eligible = $false
    })

    & $pythonExe $auditScript snapshot `
        --repo-root $repoRoot `
        --output (Join-Path $runDir "runtime-pre.json") `
        --phase pre `
        --run-id $RunId `
        --expected-manifest $runtimeManifest
    if ($LASTEXITCODE -ne 0) { throw "pre-run snapshot failed" }

    $testArgs = @(
        "-m", "pytest", "-p", "no:cacheprovider", "-W", "error",
        "backend/tests/unit/evaluation/test_full_agent_loop_benchmark.py",
        "backend/tests/unit/evaluation/test_full_agent_loop_recovery.py",
        "backend/tests/unit/evaluation/test_full_agent_loop_report.py",
        "backend/tests/unit/evaluation/test_h006_full_agent_loop_run_audit.py",
        "backend/tests/unit/evaluation/test_posttraining_promotion_protocol.py",
        "backend/tests/unit/agentic/test_loop.py",
        "backend/tests/unit/core/test_schemas.py",
        "backend/tests/unit/core/test_conversation_turn.py",
        "backend/tests/unit/core/test_langsmith_trace.py",
        "-q"
    )
    $testRun = Invoke-MonitoredProcess `
        -Name "preflight-tests" `
        -FilePath $pythonExe `
        -Arguments $testArgs `
        -StdoutPath (Join-Path $runDir "preflight-tests.stdout.log") `
        -StderrPath (Join-Path $runDir "preflight-tests.stderr.log") `
        -TimeoutSeconds 300 `
        -CheckSeconds 10
    Write-JsonFile -Path (Join-Path $runDir "test-result.json") -Value ([ordered]@{
        command = "$pythonExe $($testArgs -join ' ')"
        exit_code = $testRun.exit_code
        timed_out = $testRun.timed_out
        process = $testRun
    })
    if ($testRun.exit_code -ne 0) { throw "preflight tests failed" }

    if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 18000 -State Listen -ErrorAction SilentlyContinue) {
        throw "local port 18000 is already in use"
    }
    $remotePreflight = @'
set -e
adapter=/root/autodl-tmp/TravelAgent2-h005-eval-20260903/artifacts/native-react-posttraining/h006-internal-corrective-sft-v2-seed20260930/checkpoint-32
base=/root/autodl-tmp/models/Qwen3-1.7B
timeout 5s /root/miniconda3/bin/python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',8000)); s.close()"
adapter_manifest=$(cd "$adapter" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum)
base_manifest=$(cd "$base" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum)
printf '%s\n' "$adapter_manifest" | sed 's/^/ADAPTER /'
printf '%s\n' "$adapter_manifest" | sha256sum | sed 's/^/ADAPTER_COMBINED /'
printf '%s\n' "$base_manifest" | sed 's/^/BASE /'
printf '%s\n' "$base_manifest" | sha256sum | sed 's/^/BASE_COMBINED /'
cd /root/autodl-tmp/TravelAgent2-h005-eval-20260903
sha256sum backend/src/agentic/templates/qwen3_agent_prefix_preserving_v1.jinja | sed 's/^/TEMPLATE /'
/root/miniconda3/bin/python -c "import json,importlib.metadata as m; names=['vllm','torch','transformers','peft']; print('VERSIONS '+json.dumps({n:m.version(n) for n in names},sort_keys=True))"
timeout 10s nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | sed 's/^/GPU /'
'@
    $remoteLines = Invoke-RemoteCapture -Command $remotePreflight -TimeoutSeconds 300
    $remoteLines | Set-Content -LiteralPath (Join-Path $runDir "remote-model-manifest.txt") -Encoding utf8
    $adapterCombinedLine = $remoteLines | Where-Object { $_ -match '^ADAPTER_COMBINED ' } | Select-Object -First 1
    $baseCombinedLine = $remoteLines | Where-Object { $_ -match '^BASE_COMBINED ' } | Select-Object -First 1
    $templateLine = $remoteLines | Where-Object { $_ -match '^TEMPLATE ([0-9a-f]{64})' } | Select-Object -First 1
    $versionsLine = $remoteLines | Where-Object { $_ -match '^VERSIONS ' } | Select-Object -First 1
    $gpuLine = $remoteLines | Where-Object { $_ -match '^GPU ' } | Select-Object -First 1
    if ($adapterCombinedLine -notmatch '^ADAPTER_COMBINED 704ef680e4a4d56229f118fabeaf4f7f666da604d4d05cdda926c6545cf1f122') {
        throw "checkpoint-32 complete directory manifest mismatch"
    }
    if ($baseCombinedLine -notmatch '^BASE_COMBINED aa81aeaaee547ef68a6a993763cb3ef38a8223eafe2f2eeef139a47f2715110d') {
        throw "Qwen3-1.7B complete base-model manifest mismatch"
    }
    if (-not $templateLine -or $templateLine -notmatch '^TEMPLATE dfe4e379b6439a9f01e881660c5f1ea57cd8026a8831553492723b38d15c9e63') {
        throw "chat template hash mismatch"
    }
    $expectedVersions = 'VERSIONS {"peft": "0.20.0", "torch": "2.6.0", "transformers": "4.57.6", "vllm": "0.8.5.post1"}'
    if ([string]$versionsLine -ne $expectedVersions) {
        throw "remote serving package versions mismatch"
    }
    if (-not $gpuLine) { throw "remote GPU preflight output is missing" }
    $gpuParts = @((($gpuLine -replace '^GPU\s+', '') -split ',') | ForEach-Object { $_.Trim() })
    if ($gpuParts.Count -ne 5 -or $gpuParts[0] -ne "NVIDIA GeForce RTX 4080 SUPER" -or
        [int]$gpuParts[1] -lt 30000 -or [int]$gpuParts[3] -lt 30000) {
        throw "remote GPU identity or free-memory preflight failed: $gpuLine"
    }
    $gpuBaselineUsedMiB = [int]$gpuParts[2]
    $remoteManifestVerified = $true

    $hostExe = (Get-Process -Id $PID).Path
    $vllmStdoutPath = Join-Path $runDir "vllm.stdout.log"
    $serverProcess = Start-Process `
        -FilePath $hostExe `
        -ArgumentList @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", $serverHelper,
            "-RemoteHost", $RemoteHost,
            "-RemotePort", $RemotePort.ToString(),
            "-RemoteUser", $RemoteUser
        ) `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $vllmStdoutPath `
        -RedirectStandardError (Join-Path $runDir "vllm.stderr.log") `
        -PassThru
    $serverStartedByThisRun = $true
    Write-MonitorEvent -Event "SERVER_STARTED" -Data @{ pid = $serverProcess.Id }

    $remotePidDeadline = [DateTimeOffset]::UtcNow.AddSeconds(30)
    while ([DateTimeOffset]::UtcNow -lt $remotePidDeadline -and $null -eq $remoteServerPid) {
        $serverProcess.Refresh()
        if ($serverProcess.HasExited) {
            throw "vLLM SSH service exited before remote PID capture; exit=$($serverProcess.ExitCode)"
        }
        if (Test-Path -LiteralPath $vllmStdoutPath) {
            $serverLog = Get-Content -Raw -LiteralPath $vllmStdoutPath -ErrorAction SilentlyContinue
            if ($serverLog -match 'H006_REMOTE_PID=(\d+)') {
                $remoteServerPid = [int]$Matches[1]
            }
        }
        if ($null -eq $remoteServerPid) { Start-Sleep -Seconds 1 }
    }
    if ($null -eq $remoteServerPid) { throw "remote vLLM PID marker was not captured" }

    $identityLines = @()
    $identityDeadline = [DateTimeOffset]::UtcNow.AddSeconds(30)
    while ([DateTimeOffset]::UtcNow -lt $identityDeadline) {
        try {
            $identityCommand = 'pid=' + [string]$remoteServerPid +
                '; test -r "/proc/$pid/stat"; ' +
                'printf "REMOTE_START_TICKS="; /root/miniconda3/bin/python -c "import sys; d=open(\"/proc/\"+sys.argv[1]+\"/stat\").read(); print(d[d.rfind(\")\")+2:].split()[19])" "$pid"; ' +
                'printf "REMOTE_PGID="; ps -o pgid= -p "$pid" | tr -d '' ''; echo; ' +
                'printf "REMOTE_CMDLINE="; tr ''\0'' '' '' < "/proc/$pid/cmdline"; echo'
            $identityLines = @(
                Invoke-RemoteCapture -Command $identityCommand -TimeoutSeconds 15
            )
            $startLine = $identityLines | Where-Object { $_ -match '^REMOTE_START_TICKS=(\d+)$' } | Select-Object -First 1
            $pgidLine = $identityLines | Where-Object { $_ -match '^REMOTE_PGID=(\d+)$' } | Select-Object -First 1
            $cmdline = $identityLines | Where-Object { $_ -match '^REMOTE_CMDLINE=' } | Select-Object -First 1
            if ($startLine -and $pgidLine -and $cmdline -match 'vllm\.entrypoints\.openai\.api_server' -and
                $cmdline -match '--port 8000' -and
                $cmdline -match '--lora-modules travel-h006-checkpoint32=') {
                $remoteServerStartTicks = [int64]([regex]::Match($startLine, '\d+$').Value)
                $remoteServerPgid = [int]([regex]::Match($pgidLine, '\d+$').Value)
                if ($remoteServerPgid -ne $remoteServerPid) {
                    $remoteServerStartTicks = $null
                    throw "vLLM did not start in a dedicated process group"
                }
                break
            }
        } catch {
            $identityLines = @([string]$_)
        }
        Start-Sleep -Seconds 1
    }
    if ($null -eq $remoteServerStartTicks) {
        throw "remote vLLM process identity could not be verified"
    }
    Write-JsonFile -Path (Join-Path $runDir "remote-process.json") -Value ([ordered]@{
        captured_at = [DateTimeOffset]::UtcNow.ToString("o")
        pid = $remoteServerPid
        pgid = $remoteServerPgid
        start_ticks = $remoteServerStartTicks
        identity_verified = $true
        command_line = $cmdline
    })

    $serverDeadline = [DateTimeOffset]::UtcNow.AddMinutes(10)
    $healthy = $false
    while ([DateTimeOffset]::UtcNow -lt $serverDeadline) {
        $serverProcess.Refresh()
        if ($serverProcess.HasExited) {
            throw "vLLM SSH service exited before health check; exit=$($serverProcess.ExitCode)"
        }
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:18000/health" -TimeoutSec 5 | Out-Null
            $healthy = $true
            break
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    if (-not $healthy) { throw "vLLM health check timed out after 10 minutes" }

    $models = Invoke-RestMethod -Uri "http://127.0.0.1:18000/v1/models" -TimeoutSec 10
    $modelIds = @($models.data | ForEach-Object { [string]$_.id })
    $modelAliasPresent = $modelIds -contains $policyModel
    if (-not $modelAliasPresent) { throw "LoRA model alias is absent from /v1/models" }

    $smokeBody = [ordered]@{
        model = $policyModel
        messages = @(@{ role = "user"; content = "请查询北京的历史文化景点。" })
        tools = @(@{
            type = "function"
            function = @{
                name = "search_pois"
                description = "Search points of interest"
                parameters = @{
                    type = "object"
                    properties = @{ keywords = @{ type = "array"; items = @{ type = "string" } } }
                    required = @("keywords")
                    additionalProperties = $false
                }
            }
        })
        tool_choice = "required"
        temperature = 0.0
        max_tokens = 128
        chat_template_kwargs = @{ enable_thinking = $false }
    }
    $smoke = Invoke-RestMethod `
        -Uri "http://127.0.0.1:18000/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json" `
        -Body ($smokeBody | ConvertTo-Json -Depth 12 -Compress) `
        -TimeoutSec 60
    $toolCalls = @($smoke.choices[0].message.tool_calls)
    $toolSmokePassed = $toolCalls.Count -eq 1 -and [string]$toolCalls[0].function.name -eq "search_pois"
    if (-not $toolSmokePassed) { throw "native tool-call smoke test failed" }
    Write-JsonFile -Path (Join-Path $runDir "server-health.json") -Value ([ordered]@{
        checked_at = [DateTimeOffset]::UtcNow.ToString("o")
        health_passed = $healthy
        remote_manifest_verified = $remoteManifestVerified
        adapter_manifest_sha256 = "704ef680e4a4d56229f118fabeaf4f7f666da604d4d05cdda926c6545cf1f122"
        base_manifest_sha256 = "aa81aeaaee547ef68a6a993763cb3ef38a8223eafe2f2eeef139a47f2715110d"
        remote_versions = $versionsLine
        gpu_preflight = $gpuLine
        remote_pid = $remoteServerPid
        remote_start_ticks = $remoteServerStartTicks
        model_ids = $modelIds
        model_alias_present = $modelAliasPresent
        tool_smoke_passed = $toolSmokePassed
        smoke_response_model = $smoke.model
        smoke_tool_name = $toolCalls[0].function.name
        max_model_len = 6144
        chat_template_kwargs = @{ enable_thinking = $false }
    })

    $stages = @(
        [ordered]@{
            name = "core-10"
            script = "scripts/evaluate_full_agent_loop.py"
            expected_total = 10
            arguments = @(
                "scripts/evaluate_full_agent_loop.py",
                "--output", (Join-Path $runDir "core-10.json"),
                "--episode-output", (Join-Path $runDir "core-10-episodes.jsonl"),
                "--rollout-id", "$RunId-core",
                "--suite", "core",
                "--policy-model", $policyModel,
                "--policy-base-url", $policyBackend,
                "--policy-temperature", "0.0"
            )
        },
        [ordered]@{
            name = "expanded-20"
            script = "scripts/evaluate_full_agent_loop.py"
            expected_total = 20
            arguments = @(
                "scripts/evaluate_full_agent_loop.py",
                "--output", (Join-Path $runDir "expanded-20.json"),
                "--episode-output", (Join-Path $runDir "expanded-20-episodes.jsonl"),
                "--rollout-id", "$RunId-expanded",
                "--suite", "expanded",
                "--policy-model", $policyModel,
                "--policy-base-url", $policyBackend,
                "--policy-temperature", "0.0"
            )
        },
        [ordered]@{
            name = "recovery-8x4"
            script = "scripts/evaluate_full_agent_loop_recovery.py"
            expected_total = 32
            arguments = @(
                "scripts/evaluate_full_agent_loop_recovery.py",
                "--output", (Join-Path $runDir "recovery-8x4.json"),
                "--episode-output", (Join-Path $runDir "recovery-8x4-episodes.jsonl"),
                "--group-size", "4",
                "--seed", "421",
                "--rollout-id", $RunId,
                "--policy-model", $policyModel,
                "--policy-base-url", $policyBackend,
                "--policy-temperature", "0.8"
            )
        }
    )

    foreach ($stage in $stages) {
        $stdoutPath = Join-Path $runDir "$($stage.name).stdout.log"
        $stderrPath = Join-Path $runDir "$($stage.name).stderr.log"
        $result = Invoke-MonitoredProcess `
            -Name $stage.name `
            -FilePath $pythonExe `
            -Arguments $stage.arguments `
            -StdoutPath $stdoutPath `
            -StderrPath $stderrPath `
            -TimeoutSeconds $StageTimeoutSeconds `
            -CheckSeconds $MonitorIntervalSeconds
        $stageRows.Add($result)
        Write-JsonFile -Path (Join-Path $runDir "stage-processes.json") -Value ([ordered]@{
            stages = @($stageRows)
        })
        if ($result.exit_code -ne 0) {
            throw "$($stage.name) exited with code $($result.exit_code); no retry will be attempted"
        }
        Assert-StageInfrastructure `
            -ReportPath (Join-Path $runDir "$($stage.name).json") `
            -ExpectedTotal $stage.expected_total `
            -StageName $stage.name
    }

    & $pythonExe $auditScript snapshot `
        --repo-root $repoRoot `
        --output (Join-Path $runDir "runtime-post.json") `
        --phase post `
        --run-id $RunId `
        --expected-manifest $runtimeManifest
    if ($LASTEXITCODE -ne 0) { throw "post-run snapshot failed" }

    # Freeze all redirected service/controller logs before hashing the run artifacts.
    Stop-H006Server -RequireVerified
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
        $transcriptStarted = $false
    }

    & $pythonExe $auditScript postflight `
        --run-dir $runDir `
        --output (Join-Path $runDir "postflight.json") `
        --run-id $RunId `
        --policy-model $policyModel `
        --policy-backend $policyBackend
    if ($LASTEXITCODE -ne 0) { throw "postflight audit failed" }

    $postflight = Get-Content -Raw -LiteralPath (Join-Path $runDir "postflight.json") | ConvertFrom-Json
    if (-not $postflight.infrastructure_pass) {
        throw "postflight marked the run infrastructure-invalid"
    }
    Write-Host "Internal diagnostic completed: $runDir"
} catch {
    Write-JsonFile -Path (Join-Path $runDir "controller-error.json") -Value ([ordered]@{
        failed_at = [DateTimeOffset]::UtcNow.ToString("o")
        error = [string]$_
        stages = @($stageRows)
        auto_retry_attempted = $false
    })
    throw
} finally {
    try {
        Stop-H006Server
    } catch {
        Write-Warning "Best-effort H006 cleanup failed: $_"
    }
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
    Restore-EvaluationEnvironment
}
