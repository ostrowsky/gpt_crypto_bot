param(
    [int]$RestartDelaySec = 15
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$script = Join-Path $root "files\rl_headless_worker.py"
$workdir = Join-Path $root "files"
$runtimeDir = Join-Path $root ".runtime"
$loopLog = Join-Path $runtimeDir "headless_loop.log"

if (-not (Test-Path $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path $script)) {
    throw "RL headless worker script not found: $script"
}
if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

Push-Location $workdir
try {
    while ($true) {
        $started = (Get-Date).ToString("o")
        Add-Content -Path $loopLog -Encoding UTF8 -Value "$started headless_loop: worker start"
        & $python $script @args
        $code = $LASTEXITCODE
        $stopped = (Get-Date).ToString("o")
        Add-Content -Path $loopLog -Encoding UTF8 -Value "$stopped headless_loop: worker exit code=$code, restart in ${RestartDelaySec}s"
        Start-Sleep -Seconds $RestartDelaySec
    }
} finally {
    Pop-Location
}
