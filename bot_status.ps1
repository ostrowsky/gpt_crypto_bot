$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$stdout = Join-Path $root "bot_stdout.log"
$stderr = Join-Path $root "bot_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "bot_bg.json"

$bot = Get-Process python,py -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $python } |
    Sort-Object StartTime -Descending |
    Select-Object -First 1

$state = $null
if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
    } catch {
    }
}

[pscustomobject]@{
    Running = [bool]$bot
    PythonPid = if ($bot) { $bot.Id } else { $null }
    Started = if ($bot) { $bot.StartTime } else { $null }
    WrapperPid = if ($state) { $state.wrapper_pid } else { $null }
    Stdout = $stdout
    Stderr = $stderr
} | Format-List

if (Test-Path $stderr) {
    Write-Host ""
    Write-Host "stderr tail:"
    Get-Content $stderr -Tail 20
}
