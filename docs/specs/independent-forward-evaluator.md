# Independent Forward Evaluator

Status: independent forward evaluation foundation; promotion NOT implemented
Date: 2026-09-30
Parent: `continuous-signal-improvement.md`, P0 independent evaluation.

## Contract

Historical trainer validation/test scores cannot be rebranded as independent
sealed holdout evidence. A separate evaluator process freezes one candidate and
preregisters a future cohort after registration plus a 48-hour embargo. Trainer
updates cannot replace the frozen candidate or reset the evaluation window.
Registration requires explicit train/validation/test exposure boundaries preceding
registration. Unknown exposure rejects the candidate. Immutable candidate bytes
and a manifest carry SHA256; every report binds model, manifest and input dataset.
The independent manifest/report is evaluator-owned, not written by the trainer.
This is application-level separation, not an OS permissions/security boundary.

Fixed contract `independent-forward-top1-v1`: candidate-score Top-1 baseline,
deterministic ID ties, learned final-score Top-1 challenger, exact causal T+5
percent return. Decision groups share timeframe and closed-bar timestamp, not
per-symbol processing timestamps. Shared inference/feature encoding is permitted; training,
hyperparameter selection, trainer verdicts and trainer metric calculations are not.
Only candidate-outcome-v2 with valid decision/label provenance, decision within
one minute of feature availability and supported 15m/1h/4h timeframes is eligible.
Future labels, unknown/duplicate IDs, nonfinite results, forbidden feature names,
wrong exact label time, and overdue labels fail closed. Incomplete decision groups
are excluded whole, not scored on the survivors. Missing data remain UNKNOWN.
The dataset is streamed read-only. Label/teacher fields never reach prediction.

## Evidence and acceptance

Preregister at least 30 UTC days and 100 paired multi-candidate groups. Compare
daily means of paired T+5 deltas; a deterministic 2000-resample daily bootstrap
reports 95% CI. Lower bound > 0 gives PASS_PROXY, upper bound < 0 REJECTED;
otherwise INCONCLUSIVE. Insufficient evidence gives UNKNOWN. Invalid or overdue
evidence gives BLOCKED. Ratios include counts, base rate and lift (undefined if
baseline positive count is zero). No future coverage is promised or synthesized.

Learning-quality verdict and signal-quality verdict are distinct. Signal quality
stays UNKNOWN: T+5 ranking does not include fees, actual admission, HOLD/SELL,
10-position capacity, actual fills or portfolio alpha. All reports have
runtime_eligible=False and achievement_claimed=False. No evaluator outcome can
enable BUY/SELL. Apply TH-01 through TH-12; maximum-available-period causal replay,
independent sealed historical training design, shadow/canary risk budgets and
promotion controller remain subsequent work. Freeze/reuse safeguards do not
prove cross-regime profitability. Current candidate cannot be swapped by retraining.

## Operations, rollback and validation

Headless worker runs the evaluator separately every hour, timeout 600 seconds;
failure appears as BLOCKED in status. Cancellation terminates its child. Runtime
artifacts live under `.runtime/independent_evaluation`, never source control.
Data-only canary: register once, inspect zero/insufficient future evidence as
UNKNOWN, verify hashes and confirm active production model is untouched.
Rollback: set INDEPENDENT_SIGNAL_EVALUATION_ENABLED=False and restart headless;
last-known-good production is unchanged.
Tests cover freeze immutability, exposure, tampered artifacts/criteria, causal
labels, leakage, duplicates, partial groups, deterministic baseline, missing data,
bootstrap direction and separate signal-quality UNKNOWN. Do not claim all of
P0 independent evaluation complete: this delivers the prospective foundation only.
