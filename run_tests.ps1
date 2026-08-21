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
    "test_replay_partial_exit_after_weak_sell",
    "test_replay_trailing_tail_after_partial_exit",
    "test_replay_early_exit_gated_tail_selector",
    "test_replay_observable_tail_selector",
    "test_report_entry_admission_shadow_reward",
    "test_report_blocked_winner_causal_reward",
    "test_report_portfolio_replacement_shadow_reward",
    "test_replay_replacement_policy_variants",
    "test_replay_chase_guard_variants",
    "test_objective_leader_admission_replay",
    "test_replay_daily_objective",
    "test_market_signal_agent_replacement_policy",
    "test_learning_progress_report",
    "test_replay_protected_trailing_exit_candle",
    "test_recompute_watchlist_filtered_top_metrics",
    "test_rl_headless_worker",
    "test_report_research_universe_impact",
    "test_research_universe_shadow_collector",
    "test_report_research_universe_shadow_scorecard",
    "test_v2_shadow_observer"
    "test_evidence_capacity_phase0"
    "test_phase0_canonical_audit"
)

Write-Host "Running RELEASE smoke/regression suite: $($releaseSuite -join ', ')"
& "$PSScriptRoot\pyembed\python.exe" -m unittest @releaseSuite
if ($LASTEXITCODE -ne 0) {
    throw "Release test suite failed with exit code $LASTEXITCODE"
}

Write-Host "RELEASE TESTS PASSED"
Write-Host "Note: full legacy discovery is available with -FullDiscover and is not part of release restart gate."
