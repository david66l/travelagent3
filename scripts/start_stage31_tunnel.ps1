[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$RemotePort,

    [string]$RemoteUser = "root",
    [string]$IdentityFile = "",
    [switch]$Reconnect
)

$ErrorActionPreference = "Stop"

$sshArguments = @(
    "-N",
    "-T",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-o", "TCPKeepAlive=yes",
    "-L", "127.0.0.1:18000:127.0.0.1:8000",
    "-L", "127.0.0.1:18002:127.0.0.1:8002",
    "-p", $RemotePort.ToString()
)

if ($IdentityFile) {
    $resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path
    $sshArguments += @("-i", $resolvedIdentity)
}

$sshArguments += "$RemoteUser@$RemoteHost"

do {
    & "$env:WINDIR\System32\OpenSSH\ssh.exe" @sshArguments
    $sshExitCode = $LASTEXITCODE
    if (-not $Reconnect) {
        exit $sshExitCode
    }
    Write-Warning "Stage31 SSH tunnel exited with code $sshExitCode; retrying in 5 seconds."
    Start-Sleep -Seconds 5
} while ($true)
