$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$botScript = Join-Path $root "files\bot.py"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "bot_bg.json"
$lockFile = Join-Path $root "files\bot.lock"

$stopped = @()

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

if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        if ($state.python_pid -and (Test-IsBotPid -TargetPid ([int]$state.python_pid))) {
            Stop-Process -Id ([int]$state.python_pid) -Force -ErrorAction SilentlyContinue
            $stopped += [int]$state.python_pid
        }
        if ($state.wrapper_pid) {
            $wrapperPid = [int]$state.wrapper_pid
            $wrapper = Get-CimInstance Win32_Process -Filter "ProcessId = $wrapperPid" -ErrorAction SilentlyContinue
            if ($wrapper -and $wrapper.CommandLine -and $wrapper.CommandLine -like "*bot_bg_runner.cmd*") {
                Stop-Process -Id $wrapperPid -Force -ErrorAction SilentlyContinue
                $stopped += $wrapperPid
            }
        }
    } catch {
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'py.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.ExecutablePath -eq $python -and
        [string]$_.CommandLine -match [regex]::Escape($botScript)
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped += $_.ProcessId
    }

$stopped = $stopped | Sort-Object -Unique
Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
[pscustomobject]@{
    StoppedPids = @($stopped)
} | Format-List
