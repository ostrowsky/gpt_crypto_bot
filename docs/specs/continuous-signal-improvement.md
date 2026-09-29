# Continuous Signal Improvement

Status: required architecture; implementation pending, not a production achievement
Date: 2026-09-29

## Binding objective

The bot MUST continuously improve signal quality through feedback from verified
historical market data and the outcomes of its earlier decisions. This includes
BUY, rejected candidates, WATCH, HOLD, SELL, cooldown and portfolio replacement.
Collection and training without a functioning path to independently validated
production adoption is an unacceptable incomplete learning loop, not a delivered
self-learning capability. The loop must actively diagnose and recover blockers.
Continuous improvement is an obligation to run and close the evidence cycle;
it is not a guarantee that every candidate, day or market regime improves.
An unsuccessful candidate MUST be rejected, never deployed merely to show activity.

## Required closed loop

Versioned observations -> mature outcome labels -> dataset quality gate ->
candidate training -> independent evaluation -> shadow -> bounded canary ->
atomic production promotion -> forward evaluation -> retain or rollback -> feedback.

Learning must influence actual signal decisions after these gates pass. It must
not remain permanently shadow-only. Permitted automatic changes must be explicit:
initially bounded BUY ranking/admission weights; later separately validated HOLD/
SELL parameters. Unrestricted code rewriting or removal of safety gates is excluded.
All live decisions record policy/model version and promotion identity.

## Independent evaluation contract

The evaluator MUST be separate from the trainer and production decision path.
It has separate artifacts, immutable input manifests and a read-only interface
to models/decisions; it cannot train, tune, or write active policy configuration.
The trainer cannot author its own promotion verdict or consume sealed holdout
labels. A promotion controller accepts only evaluator-signed/digest-bound evidence
for the exact candidate, dataset, baseline and preregistered experiment.
Independence is architectural, not a requirement to use another language model.

Two distinct evaluations are mandatory:

- Learning quality: provenance, label maturity/coverage, leakage, chronological
  purged holdout, baseline lift, calibration, regime stability and uncertainty.
- Signal quality: BUY precision, early capture, realized movement capture,
  premature exits/continuation, giveback, turnover, execution latency and unified
  portfolio net return/drawdown after fees/slippage against a named benchmark.

Training/teacher/shadow scores cannot substitute for realized objective evidence.
Every ratio includes numerator/denominator; missing or partial evidence is UNKNOWN,
not zero or success. Metric contracts, target population and compared windows must
match. All TH-01 through TH-12 in `truth-harness.md` apply.

## Data and promotion safety

Use closed, timestamp-verified candles and point-in-time universe/candidate
snapshots. Features must precede labels; log accepted and rejected decisions,
reason chains, exact entry/exit identities and execution outcomes. Counterfactual
labels are explicitly simulated and cannot be called realized trades.
Detect missing/late/duplicate labels, selection bias and downtime. Quarantine
invalid rows; backfill recoverable labels idempotently without inventing outcomes.

Every candidate changing trading behavior is replayed on the maximum available
verified historical period and actual candidate population, with chronological
train/validation/sealed-test boundaries, embargo/purge and after-cost portfolio
constraints. Unsupported historical coverage yields UNKNOWN and blocks promotion.
Register hypotheses and freeze acceptance thresholds before evaluation; preserve
rejected trials, account for repeated testing and rotate exhausted holdouts.

Promotion requires passing data, holdout, risk, shadow and canary gates. Never
disable those gates to repair a blocked pipeline. Bound turnover, false positives,
drawdown, exposure, portfolio capacity, candidate drift and model influence.
Keep an immutable last-known-good policy and immediate rollback switch. Evaluator
failure, stale evidence, manifest mismatch or breached risk budget blocks promotion;
production continues on the last-known-good policy, not an unvalidated candidate.
Exact numerical budgets, minimum sample sizes, schedules and canary allocation
must be specified and tested in stage-specific specs before production enablement.

## Operational accountability

States: COLLECTING, LABELING, TRAINING, EVALUATING, SHADOW, CANARY, PROMOTED,
REJECTED, BLOCKED, ROLLED_BACK. Each transition has timestamp, reason and artifact IDs.
BLOCKED is a fault requiring an alert, owner, repair deadline and recovery action;
it cannot be silently reported as fresh training. Insufficient market evidence
is explicitly distinguished from a broken pipeline. Training attempt completion,
successful fitting, evaluator approval and production adoption are separate facts.

The morning report MUST show new/mature/quarantined/pending-overdue rows, latest
successful fit, independently evaluated deltas with N/CI, active vs candidate
versions, actual promotions/rollbacks, live outcome deltas and outstanding blockers.
It must answer whether learning changed production decisions and whether an
improvement is proved, inconclusive or contradicted.

## Priority implementation plan and acceptance gates

1. **P0 Measurement and label recovery.** Fix aged_label_coverage at its source;
   audit decision/position identity, recover mature outcomes from closed history,
   quarantine unrecoverable rows, add backlog/coverage alerts. Gate: reconstructible
   ratios and idempotent labeling; no invalid rows enter training. Tests: missing,
   delayed, duplicated labels, candle closure, restart and incomplete history.
2. **P0 Independent evaluator.** Freeze metric contracts, manifests and baseline;
   implement sealed chronological holdout and separate learning/signal verdicts.
   Gate: leakage, stale/partial data or trainer-written evidence cannot approve a
   candidate. Tests: feature/label timing, embargo, artifact mismatch, denominator
   mismatch and immutable holdout. Full Truth Harness must report actual status.
3. **P1 Reliable recurrent training.** Run on new mature feedback with bounded
   cadence, checkpoint/retry, dataset gate and an immutable candidate registry.
   Gate: a successful fit creates a candidate, never an active model. Tests:
   concurrent runs, interrupted writes, restart, insufficient/new-invalid data.
4. **P1 Maximum-period BUY validation.** Compare frozen champion vs learned
   challenger on identical actual candidate snapshots, full available period,
   untouched time-separated test and unified 10-position portfolio after costs.
   Gate: preregistered benefit/risk thresholds pass; repeated trials/rejections
   remain visible. Tests: causal replay, portfolio parity, costs and regime splits.
5. **P1 Forward shadow and bounded canary.** Record paired decisions and outcomes;
   implement promotion controller, bounded BUY influence and atomic version switch.
   Gate: independent historical and forward evidence pass before canary; canary
   risk/sample criteria pass before wider automatic adoption. Tests: fail-closed
   evaluator outage, version mismatch, risk breach, rollback and restart recovery.
6. **P2 Exit feedback and joint optimization.** Feed premature exits and post-exit
   continuation into candidate HOLD/SELL strategies; test together with BUY rather
   than optimizing isolated trades. Gate: maximum-period portfolio and forward
   validation show better monetization without unacceptable drawdown/late exits.
   Tests: causal-at-exit features, mature continuation labels, fees and re-entry.
7. **P2 Continuous audit and morning report.** Independently evaluate active policy,
   detect deterioration, rollback and route errors back to training. Gate: end-to-end
   test proves observation -> fit -> evaluator -> canary -> live decision -> outcome
   -> next iteration; a rejected candidate cannot become active. Scheduled report
   delivery, blocked-loop alerts and all stage transitions must survive restart.

## Completion definition

Complete only when an end-to-end automated feedback cycle operates, independently
validated candidates can affect production within specified risk bounds, and their
real outcomes feed subsequent iterations. A model file, collection counter,
training timestamp, commit or shadow metric alone does not satisfy this spec.
This document records requirements and a plan; it does not enable BUY/SELL changes.
