# Release Test Harness Safety

Date: 2026-06-01
Status: implemented release gate

## Problem

`run_tests.ps1` previously launched unittest discovery but did not explicitly fail on a non-zero native process exit code. As a result, `restart_full_stack.bat` could continue after a red test run.

Full legacy unittest discovery currently contains many historical tests that do not match the active production codebase. Treating that whole set as the restart gate would block safe operational restarts for unrelated legacy drift.

## Decision

Use an explicit release smoke/regression suite as the default restart gate, and keep full discovery available as an explicit diagnostic mode.

Default `run_tests.ps1` must:

- run the curated release suite;
- check `$LASTEXITCODE`;
- throw on failure;
- print `RELEASE TESTS PASSED` only after a zero exit code.

`run_tests.ps1 -FullDiscover` must:

- run full legacy unittest discovery;
- check `$LASTEXITCODE`;
- fail honestly when legacy discovery is red.

## Current Release Suite

- build/version reporting;
- shadow suspicious re-entry runtime helpers;
- suspicious re-entry daily scorecard;
- learning progress report;
- replay protected trailing/re-entry helpers;
- watchlist-filtered top-mover metric recomputation.

## Non-goals

- Do not silently mark full legacy discovery green.
- Do not fix every historical legacy test in this change.
- Do not change production BUY/SELL logic.

## Follow-up

Create a separate legacy-test cleanup roadmap: either update stale tests to current architecture or quarantine them with explicit ownership and rationale.
