param(
    [string]$Token = $env:TELEGRAM_BOT_TOKEN,
    [switch]$ForceRestart = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$agentScript = Join-Path $workdir "market_signal_agent.py"
$stdout = Join-Path $root "agent_stdout.log"
$stderr = Join-Path $root "agent_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "market_agent_bg.json"
$runnerFile = Join-Path $runtimeDir "market_agent_bg_runner.cmd"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

function Stop-StaleWrapper {
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
}

function Stop-ExistingMarketAgent {
    try {
        $agents = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq "python.exe" -and
            $_.ExecutablePath -eq $python -and
            $_.CommandLine -like "*market_signal_agent.py*"
        }
        foreach ($agent in $agents) {
            Stop-Process -Id $agent.ProcessId -Force -ErrorAction SilentlyContinue
        }
    } catch {
    }
}

function Get-MarketAgentPythonProcesses {
    try {
        $agentScriptEscaped = [regex]::Escape($agentScript)
        $matches = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'py.exe'" -ErrorAction Stop |
            Where-Object {
                $_.ExecutablePath -eq $python -and
                [string]$_.CommandLine -match $agentScriptEscaped
            }
        $out = @()
        foreach ($match in $matches) {
            $proc = Get-Process -Id $match.ProcessId -ErrorAction SilentlyContinue
            if ($proc) {
                $out += $proc
            }
        }
        return @($out)
    } catch {
        return @()
    }
}

function Read-TailText {
    param(
        [string]$Path,
        [int]$Lines = 40
    )
    if (-not (Test-Path $Path)) {
        return ""
    }
    try {
        return ((Get-Content -Path $Path -Tail $Lines -ErrorAction Stop) -join [Environment]::NewLine)
    } catch {
        return ""
    }
}

if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        $existingPid = [int]($state.python_pid | ForEach-Object { $_ })
        if ($existingPid -gt 0) {
            $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
            if ($existing) {
                if (-not $ForceRestart) {
                    [pscustomobject]@{
                        PythonPid = $existing.Id
                        StartedAt = $existing.StartTime
                        Path = $existing.Path
                    } | Format-List
                    exit 0
                }
                Stop-Process -Id $existing.Id -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
    }
}

Stop-StaleWrapper
Stop-ExistingMarketAgent
Start-Sleep -Seconds 1

Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue
Remove-Item $runnerFile -Force -ErrorAction SilentlyContinue

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
if ($Token) {
    $env:TELEGRAM_BOT_TOKEN = $Token
} else {
    Remove-Item Env:TELEGRAM_BOT_TOKEN -ErrorAction SilentlyContinue
}

$runnerLines = @(
    "@echo off",
    "set PYTHONUTF8=",
    "set PYTHONNOUSERSITE=1",
    "cd /d $workdir",
    """$python"" ""$agentScript"" --log-level INFO 1>>""$stdout"" 2>>""$stderr"""
)
$runnerLines | Set-Content -Path $runnerFile -Encoding ASCII

$wrapper = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", "`"$runnerFile`"") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -PassThru

if (-not $wrapper -or -not $wrapper.Id) {
    throw "Detached market agent wrapper did not start."
}

$agent = $null
for ($i = 0; $i -lt 10; $i++) {
    Start-Sleep -Seconds 2
    $agent = Get-MarketAgentPythonProcesses | Sort-Object StartTime -Descending | Select-Object -First 1
    if ($agent) {
        break
    }
    if ($wrapper.HasExited) {
        break
    }
}

if (-not $agent -and $wrapper.HasExited) {
    $stderrTail = Read-TailText -Path $stderr -Lines 80
    if ($stderrTail) {
        throw "Market agent wrapper exited before python process was confirmed. Stderr tail: $stderrTail"
    }
    throw "Market agent wrapper exited before python process was confirmed."
}

@{
    wrapper_pid = $wrapper.Id
    python_pid = if ($agent) { $agent.Id } else { 0 }
    started_at = (Get-Date).ToString("o")
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

[pscustomobject]@{
    WrapperPid = $wrapper.Id
    PythonPid = if ($agent) { $agent.Id } else { 0 }
    Stdout = $stdout
    Stderr = $stderr
} | Format-List
