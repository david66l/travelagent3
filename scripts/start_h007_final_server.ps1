#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$RemotePort,

    [string]$RemoteUser = "root"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$sshExe = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$serverCommand = @(
    'echo H007_REMOTE_PID=$$ &&'
    "exec /root/miniconda3/bin/python -m vllm.entrypoints.openai.api_server"
    "--model /root/autodl-tmp/models/Qwen3-1.7B"
    "--served-model-name travel-qwen3-1.7b-base"
    "--host 127.0.0.1"
    "--port 8000"
    "--chat-template backend/src/agentic/templates/qwen3_agent_prefix_preserving_v1.jinja"
    "--dtype auto"
    "--max-model-len 6144"
    "--gpu-memory-utilization 0.90"
    "--enable-auto-tool-choice"
    "--tool-call-parser hermes"
    "--enable-lora"
    "--lora-modules travel-h007-final=artifacts/native-react-posttraining/h007-targeted-corrective-sft-v1-seed20260904"
    "--max-lora-rank 16"
    "--max-loras 1"
    "--max-cpu-loras 1"
    "--enable-prefix-caching"
    "--enforce-eager"
    "--disable-log-requests"
) -join " "
$remoteCommand = (
    "cd /root/autodl-tmp/TravelAgent2-h005-eval-20260903 && " +
    "exec setsid --wait sh -c '$serverCommand'"
)

$sshArguments = @(
    "-T"
    "-L", "127.0.0.1:18000:127.0.0.1:8000"
    "-o", "BatchMode=yes"
    "-o", "ConnectTimeout=10"
    "-o", "ConnectionAttempts=1"
    "-o", "ExitOnForwardFailure=yes"
    "-o", "ServerAliveInterval=15"
    "-o", "ServerAliveCountMax=3"
    "-o", "TCPKeepAlive=yes"
    "-p", $RemotePort.ToString()
    "$RemoteUser@$RemoteHost"
    $remoteCommand
)

& $sshExe @sshArguments
exit $LASTEXITCODE

