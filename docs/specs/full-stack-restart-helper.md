# Full-Stack Restart Helper

Status: implemented  
Last updated: 2026-05-17

## Purpose

Provide one explicit local operator command that verifies the repository, restarts the
three runtime services, and fails loudly if any service does not come back healthy.

## Scope

- Run the repository test entrypoint before restart.
- Restart the trading bot, RL worker, and market agent with their existing PowerShell launchers.
- Reuse the status scripts' `-FailIfNotRunning` contract.
- Print the relevant log locations and recent stderr tails on failure.

## Non-Goals

- No deployment orchestration beyond the local Windows workspace.
- No dependency installation.
- No replacement for individual service launchers.
- No mutation of trading settings.

## Acceptance Criteria

1. The helper exits non-zero if tests fail.
2. The helper exits non-zero if any restarted service is not healthy.
3. The helper prints enough status/log information for an operator to triage a failed restart.
4. The helper can run either interactively or with `--no-pause`.

## Verification

- Parse the batch file manually against the current script names.
- Keep this helper aligned with:
  - `run_tests.ps1`
  - `start_bot_bg.ps1`
  - `start_rl_worker_bg.ps1`
  - `start_market_agent_bg.ps1`
  - the three `*_status.ps1` scripts.
