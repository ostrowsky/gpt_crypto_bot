# Prediction-error optimization loop

2026-10-04. Implemented trainer search; production adoption remains gated.

## Objective and boundaries (TH-01..12)

Minimize Brier error of the existing candidate-quality probability against the
existing causal forward EV-quality label. This is not price forecasting, exit
learning, realized profitability or permission to relax BUY/SELL thresholds.
Use all eligible historical rows in the immutable evaluator-exported snapshot;
legacy/unknown and immature labels are excluded by the existing provenance and
dataset-quality preflight. Preserve purged chronological train/validation/test
groups. Never choose parameters using test errors or labels.

Search exactly three preregistered CatBoost quality-classifier configurations,
sequentially with thread_count=1, fixed seed and no filesystem side effects from
CatBoost. Include the current default parameters as the reference. Fit on train
only (no validation-driven early stopping); choose raw-probability validation
Brier minimum, stable reference-first tie break. Calibrate the chosen model using
validation only, as in the existing trainer. Regressor/ranker/teacher parameters,
EV target, trading gates, holding periods, stops and portfolio limits are fixed.

Report reference and selected raw-probability test Brier, per-day paired error
deltas, deterministic day bootstrap CI, row/day counts, class balance, false
positive/negative counts and majority constant baseline. Mark insufficient test
day support UNKNOWN; absence of positive lower CI is NOT_PROVEN, not improvement.
Even SUPPORTED is a prediction diagnostic only, never an independent portfolio
certificate. The existing frozen prospective evaluator, maximum historical
portfolio confirmation, sealed/shadow/canary, parity, application receipts and
rollback controller remain mandatory. Known live BUY/replay FAIL blocks adoption.

The scheduled trainer calls this search by default, records the result in status
and candidate metadata, and still emits runtime_eligible=false. An unchanged
immutable input snapshot with a successfully published matching candidate is a
no-op, not a repeated fit or new improvement. Digest drift fails closed before
publishing. Explicit deployment prediction_error_search=false rolls back to the
previous trainer; CLI training remains backward compatible unless opted in.

`local_learning_runtime.py --serial` runs export, train, evaluator, portfolio and
controller sequentially in one limited-CPU process using the existing deployment,
role cadences, supervisor lock and stop request. No concurrent fits or backtest
bursts. Exceptions remain BLOCKED. This connects learning to existing independent
verification and release/rollback without granting trainer release authority.
The scheduler's closed_loop flag remains false until genuine application/outcome
evidence proves production improvement; starting a scheduler is not that proof.

## Verification and rollout

Test bounded search and tie break, test isolation, paired daily uncertainty,
empty/nonfinite rejection, output metadata, scheduled wiring, and unchanged-input
deduplication. Run relevant tests, full/staged Truth Harness and diff checks.
Perform an actual single limited-load historical training attempt using the
maximum exported snapshot. Dataset failure must be reported as BLOCKED, never
waived. No strategy improvement claim until independent maximum/forward gates
pass. No trading restart or production enablement is part of this change.
