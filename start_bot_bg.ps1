param(
    [string]$Token = $env:TELEGRAM_BOT_TOKEN,
    [switch]$ForceRestart = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$stdout = Join-Path $root "bot_stdout.log"
$stderr = Join-Path $root "bot_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "bot_bg.json"
$runnerFile = Join-Path $runtimeDir "bot_bg_runner.ps1"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

function Get-BotPythonProcesses {
    Get-Process python,py -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $python }
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

if (-not $Token) {
    throw "TELEGRAM_BOT_TOKEN is not set."
}

$existing = @(Get-BotPythonProcesses)
if ($existing.Count -gt 0) {
    if (-not $ForceRestart) {
        $existing |
            Select-Object Id, ProcessName, StartTime, Path |
            Format-Table -AutoSize
        exit 0
    }
    $existing | Stop-Process -Force -ErrorAction SilentlyContinue
}

Stop-StaleWrapper
Start-Sleep -Seconds 1

Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue

$env:TELEGRAM_BOT_TOKEN = $Token
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

$stdoutEsc = $stdout -replace '"', '\"'
$stderrEsc = $stderr -replace '"', '\"'
$workdirEsc = $workdir -replace '"', '\"'
$pythonEsc = $python -replace '"', '\"'
$tokenEsc = $Token -replace '"', '\"'
$cmd = 'cmd /c "set PYTHONUTF8=& set PYTHONNOUSERSITE=1& set TELEGRAM_BOT_TOKEN=' +
    $tokenEsc +
    '& cd /d ' + $workdirEsc +
    '& ' + $pythonEsc + ' bot.py 1>>' + $stdoutEsc + ' 2>>' + $stderrEsc + '"'

$wrapperPid = $null
try {
    $wrapper = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmd
    }
    if ($wrapper -and $wrapper.ReturnValue -eq 0 -and $wrapper.ProcessId) {
        $wrapperPid = $wrapper.ProcessId
    }
} catch {
}

if (-not $wrapperPid) {
    $wrapperProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden -PassThru
    if (-not $wrapperProc -or -not $wrapperProc.Id) {
        throw "Detached bot process did not start."
    }
    $wrapperPid = $wrapperProc.Id
}

$bot = $null
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 2
    $bot = Get-BotPythonProcesses | Sort-Object StartTime -Descending | Select-Object -First 1
    if ($bot) {
        break
    }
}
if (-not $bot) {
    throw "Bot python process did not start."
}

@{
    wrapper_pid = $wrapperPid
    python_pid = $bot.Id
    started_at = (Get-Date).ToString("o")
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

[pscustomobject]@{
    WrapperPid = $wrapperPid
    PythonPid = $bot.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-List
