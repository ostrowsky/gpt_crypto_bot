param(
    [switch]$FailIfNotRunning
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "v2_shadow_bg.json"
$statusFile = Join-Path $root "files\.runtime\v2_shadow_status.json"

$pidState = $null
if (Test-Path $pidFile) {
    try { $pidState = Get-Content $pidFile -Raw | ConvertFrom-Json } catch {}
}
$proc = $null
if ($pidState -and $pidState.python_pid) {
    $proc = Get-Process -Id $pidState.python_pid -ErrorAction SilentlyContinue
}
$status = $null
if (Test-Path $statusFile) {
    try { $status = Get-Content $statusFile -Raw | ConvertFrom-Json } catch {}
}
$fresh = $false
if ($status -and $status.worker -and $status.worker.last_heartbeat) {
    try {
        $hb = [datetime]::Parse([string]$status.worker.last_heartbeat).ToUniversalTime()
        $age = ([datetime]::UtcNow - $hb).TotalSeconds
        # One full watchlist scan can take several minutes, so freshness should
        # reflect a missed worker cycle rather than an arbitrary short wall clock.
        $fresh = ($age -ge 0 -and $age -le 600)
    } catch {}
}

$obj = [pscustomobject]@{
    Running = [bool]($proc -or $fresh)
    PythonPid = if ($proc) { $proc.Id } else { $null }
    StatusFile = $statusFile
    FreshHeartbeat = $fresh
}
$obj | Format-List
if ($status) {
    Write-Host ""
    Write-Host "v2 shadow status:"
    $status | ConvertTo-Json -Depth 8
}
if ($FailIfNotRunning -and -not $obj.Running) { exit 1 }
