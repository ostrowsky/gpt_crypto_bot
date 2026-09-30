# Historical Evaluation and Safe Release Controller

Date: 2026-09-30
Status: retrospective audit and fail-closed release-readiness controller;
production adoption adapter NOT delivered by this stage.
Parent: continuous-signal-improvement.md. Applies TH-01 through TH-12.

## Historical audit

Snapshot the entire maximum available candidate-outcome-v2 dataset under the
cooperative dataset IO lock. Freeze candidate, source evaluator and dataset SHA256
in an immutable run directory before inference. Reusing a run must not replace its
inputs, cutoff, result or model. The trainer never runs inside the evaluator.
No date-limited convenience cohort or hand-picked symbols are allowed.

Evaluate only observations after the latest train/validation/test feature, label
or label-recording exposure plus 48 hours. All earlier rows are exposed/excluded;
unknown boundaries fail closed. Record full available feature range and excluded
counts, eligible interval, incomplete groups and invalid/overdue rows. The existing
independent Top-1 evaluator supplies causal, grouped T+5 comparisons and daily CI.
Missing observations cannot establish complete universe/uptime coverage; gaps
are reported and remain a promotion blocker. Never rebrand this retrospective
post-exposure proxy audit as a preregistered sealed holdout, realized PnL or a
maximum-period unified portfolio backtest. No policy relaxation is authorized.

Historical and prospective evidence have separate artifacts. Historical results
never replace the prospective manifest or reset its frozen candidate/cohort.
All tests and prior exposures, not just training rows, set the cutoff. The report
retains every selected pair and daily delta; periods with no eligible pairs are
unknown. At least 30 observed UTC days and 100 pairs are required for the proxy
verdict; negative effects are diagnostic even when underpowered.

## Controller

Run independently of fitting on a frozen audit result and fresh forward result.
Verify candidate, manifest, snapshot and evaluator-source hashes; recompute numeric
pair summaries/CI from the result trace using a separate implementation. Reject
modified traces, stale/future reports (24h), unknown exposure, other candidates,
or mismatched contracts. A stored PASS string is never sufficient.

States are BLOCKED, REJECTED and SHADOW (evaluation only). Historical proxy
direction can reject a candidate but cannot grant CANARY or PROMOTED. Mandatory
release blockers explicitly include sealed historical protocol, maximum-period
paired unified ten-slot replay after fees/slippage, verified point-in-time
coverage, prospective shadow minimum sample and canary monitor/production adapter.
These unsupported capabilities are UNKNOWN, not mocked passing gates. No controller
write can change active model files, positions, BUY/SELL config or gate thresholds.

Persist append-only, hash-chained transitions with timestamp, candidate identity,
reason and input evidence digests. Preserve rejections; recomputing the same
evidence is idempotent. Corrupt/truncated chain or mismatched candidate blocks
state updates. Exclusive controller lock prevents concurrent updates. After
restart, rebuild state from the ledger, not from an editable latest-status file.
Every decision includes runtime_eligible=False and achievement_claimed=False.

Headless runs release readiness every hour in a separate process with a 600-second
timeout; failures stay BLOCKED in worker status. The completed immutable historical
run is selected by an evaluator-published runtime pointer. Missing pointer blocks.
Rollback switch is SIGNAL_RELEASE_CONTROLLER_ENABLED=False plus headless
restart; last-known-good production is untouched. Data-only shadow/canary checks
must demonstrate malformed evidence cannot move production. Actual rollout needs
a separately specified bounded adapter with atomic champion fallback; this stage
must not claim closure of the entire self-improvement loop.

## Verification

Tests: exposure embargo, maximum-period scope, immutable snapshots/reuse,
historical versus forward separation, causal labels, missing/duplicate rows,
trace recomputation, CI/count tampering, stale evidence, hash mismatch, controller
concurrency, idempotence, ledger tampering/restart and no production mutation.
Run the real maximum-available historical audit, record exact coverage and verdict,
run controller against it and current forward evidence, then focused tests,
Truth Harness change --staged and git diff --check before commit/push.
