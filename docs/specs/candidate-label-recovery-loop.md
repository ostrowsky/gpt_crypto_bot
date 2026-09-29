# Candidate Label Recovery Loop

Status: data-repair implementation; no trading-policy relaxation
Date: 2026-09-29
Parent: `continuous-signal-improvement.md`, stage P0 measurement/label recovery.

## Root cause and scope

The collector requests only 120 recent candles. Labels whose observation candles
have fallen outside that window cannot mature through that collector. Historical
backfill existed, but was not called by the recurrent headless training loop.
The live worker also stopped its collector after Windows permission failure on
2026-09-18. Known PermissionError (including wrapped causes) pauses collection
fail-closed for 300 seconds and retries; unknown integrity failures still stop it.
Repair must operate independently of collector enablement and training readiness.
This is not permission to weaken aged_label_coverage or promote a model.

## Behavior

- Before each recurrent training preflight, run candidate-outcome-v2 historical
  recovery in an isolated child process, concurrency 4 and timeout 900 seconds.
- Select only valid provenance observations and genuinely mature missing targets;
  fetch paginated ranges beyond the rolling window, independent of current watchlist.
- Require aligned, closed, finite positive OHLC, valid volume and consistent candle
  close timestamps. Invalid responses cannot provide labels. Exact T+N endpoints
  additionally require contiguous candles through a closed successor.
- Attach endpoint, range, retrieval timestamp and series digest to recovered labels;
  retain existing label values/provenance. Use existing cross-process dataset lock
  and atomic rewrite, so concurrent appends are preserved and retries are idempotent.
- Remaining unavailable targets stay UNKNOWN and keep training quality gates active.
  Do not delete rows or lower denominator to make coverage pass; invalid-provenance
  observations remain excluded by existing training guards, not re-certified.
- Persist recovery status, requested/remaining row and target counts, failures,
  start/finish and error in runtime reports and worker dataset status. Incomplete
  recovery is BLOCKED and logs an error, never a training success.
- Recovery never authorizes training or changes runtime BUY/SELL eligibility.

## Safety, rollback and limitations

`CANDIDATE_LABEL_RECOVERY_ENABLED=False` disables only historical repair; existing
collector and all dataset/production guards remain active. Parent cancellation
terminates its child. Failed/timeout recovery leaves original evidence intact;
subsequent preflight still decides training eligibility. Timeout may leave pending
labels, never synthesizes them. Separate worker reporting/training timestamp repair,
independent evaluator and model promotion remain later work.
Deployment may restart this project's headless worker with its original arguments;
it must not stop the Telegram bot or another project's worker or claim improvement.

## Validation

Data-only canary: first run one controlled historical recovery, inspect residual
counts and label provenance, then observe the restarted worker's first collection
and recovery cycle before considering repair healthy. Any failed write or invalid
market response must remain visible and fail closed. Model/trading shadow/canary
promotion is explicitly out of scope and remains blocked by independent evaluation.

Focused tests cover out-of-window recovery, exact contiguous horizons, missing
bars, non-finite prices, existing-label immutability, invalid exchange OHLC/times,
pagination, provenance exclusion, worker integration, timeout and cancellation.
Run dataset/provenance/worker regression suites and Truth Harness change profile.
Exercise recovery on the maximum available pending observation history; compare
requested/remaining counts and recompute readiness. No trading-policy backtest is
needed for a label-only repair, and no trading gate may be relaxed on that basis.
