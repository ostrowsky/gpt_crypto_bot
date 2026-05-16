param(
    [switch]$FailIfNotRunning
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stdout = Join-Path $root "bot_stdout.log"
$stderr = Join-Path $root "bot_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "bot_bg.json"
$launcherLog = Join-Path $runtimeDir "start_bot_bg.log"
$lockFile = Join-Path $root "files\bot.lock"
$python = Join-Path $root "pyembed\python.exe"
$botScript = Join-Path $root "files\bot.py"

$state = $null
if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
    } catch {
    }
}

$lockPid = $null
$lockProc = $null
if (Test-Path $lockFile) {
    try {
        $lockPid = [int](Get-Content $lockFile -Raw).Trim()
        if ($lockPid -gt 0) {
            $lockProc = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
        }
    } catch {
        $lockPid = $null
        $lockProc = $null
    }
}

function Test-IsBotPid {
    param([int]$TargetPid)
    if (-not $TargetPid -or $TargetPid -le 0) {
        return $false
    }
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $TargetPid"
        if (-not $proc) {
            return $false
        }
        if ($proc.ExecutablePath -and ($proc.ExecutablePath -ne $python)) {
            return $false
        }
        if ($proc.CommandLine -and ($proc.CommandLine -notmatch [regex]::Escape($botScript))) {
            return $false
        }
        return $true
    } catch {
        return $false
    }
}

$bot = $null
if ($state -and $state.python_pid) {
    try {
        $bot = Get-Process -Id $state.python_pid -ErrorAction Stop
    } catch {
        $bot = $null
    }
}

if (-not $bot -and $lockProc -and (Test-IsBotPid -TargetPid $lockPid)) {
    $bot = $lockProc
}

$wrapper = $null
if ($state -and $state.wrapper_pid) {
    try {
        $wrapper = Get-Process -Id $state.wrapper_pid -ErrorAction Stop
    } catch {
        $wrapper = $null
    }
}

$stderrFresh = $false
$stderrSeenAt = $null
if (Test-Path $stderr) {
    try {
        $stderrInfo = Get-Item $stderr -ErrorAction Stop
        $stderrSeenAt = $stderrInfo.LastWriteTime
        $stderrFresh = ((Get-Date) - $stderrInfo.LastWriteTime).TotalSeconds -le 30
    } catch {
        $stderrFresh = $false
    }
}

$launcherFresh = $false
if (Test-Path $launcherLog) {
    try {
        $launcherInfo = Get-Item $launcherLog -ErrorAction Stop
        $launcherFresh = ((Get-Date) - $launcherInfo.LastWriteTime).TotalSeconds -le 30
    } catch {
        $launcherFresh = $false
    }
}

$running = [bool]$bot

$statusObj = [pscustomobject]@{
    Running = $running
    PythonPid = if ($bot) { $bot.Id } elseif ($lockPid) { $lockPid } elseif ($state) { $state.python_pid } else { $null }
    Started = if ($bot) { $bot.StartTime } elseif ($wrapper) { $wrapper.StartTime } elseif ($state) { $state.started_at } elseif ($stderrSeenAt) { $stderrSeenAt } else { $null }
    WrapperPid = if ($wrapper) { $wrapper.Id } elseif ($state) { $state.wrapper_pid } else { $null }
    Stdout = $stdout
    Stderr = $stderr
}
$statusObj | Format-List

if (Test-Path $stderr) {
    Write-Host ""
    Write-Host "stderr tail:"
    Get-Content $stderr -Tail 20
}

if ($FailIfNotRunning -and -not $statusObj.Running) {
    exit 1
}
