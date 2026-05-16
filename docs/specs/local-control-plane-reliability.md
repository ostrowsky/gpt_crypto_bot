# Local Control-Plane Reliability

Status: implemented  
Last updated: 2026-05-16

## Purpose

Make local bot supervision less ambiguous after crashes, stale wrappers, or repeated launch attempts.

## Scope

- Status scripts can fail explicitly when a process is not running.
- Bot status prefers a validated real bot PID over wrapper/log freshness heuristics.
- Background startup understands `files/bot.lock`, avoids duplicate launches, and supports force-restart cleanup.
- The local execution supervisor no longer passes the Telegram token on the process command line.
- The test runner discovers the full local `test*.py` suite.

## Why This Exists

- A fresh log file or wrapper process is not the same thing as a healthy bot process.
- Duplicate local bot launches can corrupt operator assumptions and local state.
- Passing secrets as command-line arguments broadens accidental exposure.

## Acceptance Criteria

1. `bot_status.ps1`, `market_agent_status.ps1`, and `rl_worker_status.ps1` support `-FailIfNotRunning`.
2. `bot_status.ps1` only reports `Running = true` for a validated bot process.
3. `start_bot_bg.ps1` reads and clears stale lock PIDs safely, and does not launch a duplicate bot unless forced.
4. `local_exec_supervisor.py` launches the bot without embedding the Telegram token in command arguments.
5. `run_tests.ps1` discovers `test*.py`, not only `test_bot.py`.

## Verification

- Run the status scripts both with and without active workers.
- Run `run_tests.ps1` after restoring the full trusted test suite.
- Validate force-restart behavior against a stale `files/bot.lock`.
