param(
    [switch]$FullDiscover
)

$ErrorActionPreference = "Stop"

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

Set-Location "$PSScriptRoot\files"

if ($FullDiscover) {
    Write-Host "Running FULL legacy unittest discovery..."
    & "$PSScriptRoot\pyembed\python.exe" -m unittest discover -s . -p "test*.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Full unittest discovery failed with exit code $LASTEXITCODE"
    }
    Write-Host "FULL TESTS PASSED"
    exit 0
}

$releaseSuite = @(
    "test_build_info",
    "test_suspicious_reentry_shadow",
    "test_suspicious_reentry_scorecard",
    "test_replay_hold_after_weak_sell",
    "test_learning_progress_report",
    "test_replay_protected_trailing_exit_candle",
    "test_recompute_watchlist_filtered_top_metrics"
)

Write-Host "Running RELEASE smoke/regression suite: $($releaseSuite -join ', ')"
& "$PSScriptRoot\pyembed\python.exe" -m unittest @releaseSuite
if ($LASTEXITCODE -ne 0) {
    throw "Release test suite failed with exit code $LASTEXITCODE"
}

Write-Host "RELEASE TESTS PASSED"
Write-Host "Note: full legacy discovery is available with -FullDiscover and is not part of release restart gate."
