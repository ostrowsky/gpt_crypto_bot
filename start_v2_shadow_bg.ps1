param(
    [switch]$ForceRestart = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$script = Join-Path $workdir "v2_shadow_worker.py"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "v2_shadow_bg.json"
$runnerFile = Join-Path $runtimeDir "v2_shadow_bg_runner.cmd"
$stdout = Join-Path $root "v2_shadow_stdout.log"
$stderr = Join-Path $root "v2_shadow_stderr.log"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($pid in @($state.wrapper_pid, $state.python_pid)) {
            if ($pid) { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }
        }
    } catch {}
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

try {
    $workers = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.ExecutablePath -eq $python -and
        $_.CommandLine -like "*v2_shadow_worker.py*"
    }
    foreach ($worker in $workers) {
        Stop-Process -Id $worker.ProcessId -Force -ErrorAction SilentlyContinue
    }
} catch {}

Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue
Remove-Item $runnerFile -Force -ErrorAction SilentlyContinue

$runnerLines = @(
    "@echo off",
    "set PYTHONUTF8=",
    "set PYTHONNOUSERSITE=1",
    "cd /d $workdir",
    """$python"" ""$script"" --log-level INFO 1>>""$stdout"" 2>>""$stderr"""
)
$runnerLines | Set-Content -Path $runnerFile -Encoding ASCII

$wrapper = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", "`"$runnerFile`"") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3
$agent = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.ExecutablePath -eq $python -and
    $_.CommandLine -like "*v2_shadow_worker.py*"
} | Sort-Object ProcessId -Descending | Select-Object -First 1

if (-not $wrapper -or -not $wrapper.Id) {
    throw "Detached v2 shadow wrapper did not start."
}

@{
    wrapper_pid = $wrapper.Id
    python_pid = if ($agent) { $agent.ProcessId } else { 0 }
    started_at = (Get-Date).ToString("o")
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

[pscustomobject]@{
    WrapperPid = $wrapper.Id
    PythonPid = if ($agent) { $agent.ProcessId } else { 0 }
    Stdout = $stdout
    Stderr = $stderr
} | Format-List

