$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$runtimeDir = Join-Path $root ".runtime"
$stdout = Join-Path $runtimeDir "rl_worker_wrapper_stdout.log"
$stderr = Join-Path $runtimeDir "rl_worker_runtime.log"
$pidFile = Join-Path $runtimeDir "rl_worker_bg.json"
$statusFile = Join-Path $runtimeDir "rl_worker_status.json"

$pidState = $null
if (Test-Path $pidFile) {
    try {
        $pidState = Get-Content $pidFile -Raw | ConvertFrom-Json
    } catch {
    }
}

$worker = $null
if ($pidState -and $pidState.python_pid) {
    try {
        $worker = Get-Process -Id $pidState.python_pid -ErrorAction Stop
    } catch {
        $worker = $null
    }
}
if (-not $worker -and $pidState -and $pidState.wrapper_pid) {
    try {
        $worker = Get-Process -Id $pidState.wrapper_pid -ErrorAction Stop
    } catch {
        $worker = $null
    }
}
if (-not $worker) {
    try {
        $candidates = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'py.exe'"
        foreach ($proc in $candidates) {
            $cmd = [string]$proc.CommandLine
            if ($cmd -and $cmd -match "rl_headless_worker\.py") {
                $worker = Get-Process -Id $proc.ProcessId -ErrorAction SilentlyContinue
                if ($worker) {
                    break
                }
            }
        }
    } catch {
        $worker = $null
    }
}

$status = $null
if (Test-Path $statusFile) {
    try {
        $status = Get-Content $statusFile -Raw | ConvertFrom-Json
    } catch {
    }
}

[pscustomobject]@{
    Running = [bool]$worker
    PythonPid = if ($worker) { $worker.Id } else { $null }
    Started = if ($worker) { $worker.StartTime } else { $null }
    WrapperPid = if ($pidState) { $pidState.wrapper_pid } else { $null }
    Stdout = $stdout
    Stderr = $stderr
    StatusFile = $statusFile
} | Format-List

if ($status) {
    Write-Host ""
    Write-Host "worker status:"
    $status | ConvertTo-Json -Depth 8
}

if (Test-Path $stderr) {
    Write-Host ""
    Write-Host "stderr tail:"
    Get-Content $stderr -Tail 20
}
