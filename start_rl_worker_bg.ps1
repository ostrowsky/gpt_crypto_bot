param(
    [switch]$ForceRestart = $true,
    [switch]$EnableCollector = $false
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$script = Join-Path $workdir "rl_headless_worker.py"
$loopScript = Join-Path $root "headless_loop.ps1"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "rl_worker_bg.json"
$heartbeatFile = Join-Path $runtimeDir "rl_worker_wrapper_heartbeat.json"
$stdout = Join-Path $runtimeDir "rl_worker_wrapper_stdout.log"
$stderr = Join-Path $runtimeDir "rl_worker_wrapper_stderr.log"
$trainLockFile = Join-Path $runtimeDir "rl_worker_train.lock"
$stopFile = Join-Path $runtimeDir "rl_worker.stop"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

function Get-RLWorkerProcesses {
    if (-not (Test-Path $pidFile)) {
        return @()
    }
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        $out = @()
        foreach ($pid in @($state.wrapper_pid, $state.python_pid)) {
            if ($pid) {
                $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
                if ($proc) {
                    $out += $proc
                }
            }
        }
        return $out
    } catch {
        return @()
    }
}

function Stop-StaleWorker {
    if (-not (Test-Path $pidFile)) {
        return
    }
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($pid in @($state.wrapper_pid, $state.python_pid)) {
            if ($pid) {
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item $heartbeatFile -Force -ErrorAction SilentlyContinue
}

function Stop-ExistingRLWorkers {
    try {
        $workers = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq "python.exe" -and
            $_.ExecutablePath -eq $python -and
            $_.CommandLine -like "*rl_headless_worker.py*"
        }
        foreach ($worker in $workers) {
            Stop-Process -Id $worker.ProcessId -Force -ErrorAction SilentlyContinue
        }
    } catch {
    }
}

function Stop-ExistingRLWrappers {
    try {
        $escapedLoopScript = $loopScript.Replace("\", "\\")
        $wrappers = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq "powershell.exe" -and
            (
                $_.CommandLine -like "*$loopScript*" -or
                $_.CommandLine -like "*$escapedLoopScript*"
            )
        }
        foreach ($wrapper in $wrappers) {
            Stop-Process -Id $wrapper.ProcessId -Force -ErrorAction SilentlyContinue
        }
    } catch {
    }
}

function Clear-OrphanedTrainLock {
    if (-not (Test-Path $trainLockFile)) {
        return
    }
    try {
        $lock = Get-Content $trainLockFile -Raw | ConvertFrom-Json
        $ownerPid = [int]$lock.pid
        if ($ownerPid -and (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue)) {
            return
        }
    } catch {
    }
    Remove-Item $trainLockFile -Force -ErrorAction SilentlyContinue
}

Stop-StaleWorker
Stop-ExistingRLWrappers
Stop-ExistingRLWorkers
Start-Sleep -Seconds 1
Clear-OrphanedTrainLock

Remove-Item $stopFile -Force -ErrorAction SilentlyContinue
Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue
Remove-Item $heartbeatFile -Force -ErrorAction SilentlyContinue

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

$wrapperArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", $loopScript)
if ($EnableCollector) {
    $wrapperArgs += "--enable-collector"
}

$wrapperProc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $wrapperArgs `
    -WindowStyle Hidden `
    -PassThru

if (-not $wrapperProc -or -not $wrapperProc.Id) {
    throw "Detached RL worker process did not start."
}

$wrapperPid = $wrapperProc.Id
$deadline = (Get-Date).AddSeconds(12)
$readyState = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if (-not (Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue)) {
        break
    }
    if (Test-Path $heartbeatFile) {
        try {
            $state = Get-Content $heartbeatFile -Raw | ConvertFrom-Json
            if ($state.wrapper_pid -eq $wrapperPid) {
                $readyState = $state
                break
            }
        } catch {
        }
    }
}

if (-not $readyState) {
    throw "Detached RL wrapper did not initialize heartbeat: pid=$wrapperPid"
}

$payload = [ordered]@{
    wrapper_pid = $readyState.wrapper_pid
    python_pid = $readyState.python_pid
    state = $readyState.state
    started_at = $readyState.started_at
    updated_at = $readyState.updated_at
    stdout = $stdout
    stderr = $stderr
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -Path $pidFile -Encoding UTF8

[pscustomobject]@{
    WrapperPid = $wrapperPid
    PythonPid = $readyState.python_pid
    Stdout = $stdout
    Stderr = $stderr
} | Format-List
