# Immutable Policy Epoch and Label Provenance

Date: 2026-08-13
Status: measurement safety; production trading policy unchanged

## Problem

The candidate-ranker dataset mixes observations produced by different live
policies, and legacy labels do not record when or how they became available.
The training report also described a row-index split as a chronological
holdout even though a decision timestamp could straddle a split boundary.
Consequently, a favorable model metric could not prove causal, out-of-sample
improvement or even identify the policy that generated its candidates.

## Objective fit

This work makes later entry, selection, and exit experiments auditable. It
serves safer and more explainable operation; it does not itself improve or
claim improvement in capture, precision, exits, or portfolio alpha.

## Scope

- Give every new ML/critic observation an immutable `policy_epoch`, policy
  hash, feature cutoff, decision-record time, and source identity.
- Give every newly written forward, trade-outcome, learning, and teacher label
  an immutable definition plus availability/recording time.
- Write a forward label only for the exact T+N bar and only after a subsequent
  bar proves that target candle has closed; never label from a forming candle
  or silently substitute a later candle across a data gap.
- Keep the observation epoch immutable when a collector candidate is upgraded
  to a later decision; record the later decision epoch separately and retain
  prior decision provenance when the epoch changes.
- Train the candidate ranker from provenance-verified rows by default.
- Split train/validation/test only at complete decision-bar group boundaries,
  purging rows whose labels were not yet available at the next split boundary,
  and publish the exact split periods, epoch counts, label contract, and
  evaluation scope in both the report and model payload.
- Refuse runtime loading of a candidate-ranker payload that is not explicitly
  marked decision-grade under this provenance contract.
- Audit the maximum locally available critic dataset without rewriting or
  pretending that legacy rows belong to the current policy.

## Policy-epoch semantics

`policy_epoch` identifies semantically distinct decision behavior. A new epoch
is required when a change can alter candidate eligibility, score/routing/gate
order, BUY/SELL/re-entry or portfolio actions, decision-time feature/label
semantics, active-universe eligibility, costs/capacity, or an effective runtime
override for any of those surfaces.

Documentation, tests, observability-only fields, and refactors with proven
decision-trace equivalence belong to the same semantic epoch even though their
code/build hashes differ. Market regime is an observation attached to evidence,
not a policy epoch. It may justify a separately registered regime-conditional
hypothesis but cannot relabel the policy that generated historical decisions.

Evidence from an older epoch is retained as `directly_comparable`,
`transportable_with_bridge`, or `historical_only`. Cross-epoch pooling requires
an explicit overlap/decision-trace bridge on the same candidate population. An
epoch transition never deletes or silently upgrades old rows.

The shipped implementation currently errs conservatively and can over-segment
epochs when a non-behavioral source hash changes. That is safe but statistically
inefficient. A future equivalence bridge may mark such evidence transportable;
it must not rewrite the immutable epoch stored on existing observations.

Explicitly unchanged:

- BUY/SELL, score, blocker, cooldown, replacement, sizing, and portfolio rules;
- historical dataset rows and generated model/report artifacts;
- the truth status of legacy evidence: missing provenance remains
  `legacy_unknown` and is never backfilled by inference.

## Primary metrics

- `verified_rows / labeled_rows` with both counts;
- `legacy_unknown_rows`;
- policy-epoch counts and feature-time range;
- train/validation/test row and complete-group counts;
- cross-split timestamp overlap (must be zero);
- model `evidence_status` and `runtime_eligible`.

Model proxy scores remain diagnostic. They do not count as north-star or
portfolio achievement.

## Acceptance criteria

1. A new record preserves its root observation `policy_epoch` across decision
   upserts.
2. New labels record a stable definition and a label availability/record time.
   A forming or missing exact target candle leaves the label pending.
3. Legacy rows are excluded from default ranker training and reported as
   unknown, not assigned to the current epoch.
4. No decision group occurs in more than one split.
   The latest train label precedes the first validation feature, and the latest
   validation label precedes the first test feature.
5. The model payload and training-session report expose `feature_time`,
   `label_time`, `label_definition`, `evaluation_scope`, epoch coverage, and
   runtime eligibility.
6. The runtime rejects unproven/legacy candidate-ranker payloads when the
   provenance guard is enabled.
7. Focused tests and the maximum-period local provenance audit pass.
8. If the verified cohort is too small, the worker emits a fresh, non-runtime,
   non-achievement readiness report and leaves the existing model artifact
   untouched. Freshness must not be presented as successful retraining.

## Risks and trade-offs

- Retraining can pause until enough new provenance-verified rows mature. This
  is intentional fail-closed behavior; an old model must not be presented as
  newly self-improving.
- Hashing all relevant config and decision-source files creates a new epoch for
  conservative non-behavioral source changes in the current implementation.
  False separation is preferable to silently merging policies, but the control
  plane reports this as over-segmentation and may use only a verified bridge to
  recover comparability.
- A decision upsert can occur under a later epoch than observation. Both epochs
  are retained; training requires verified decision provenance.
- Legacy rows may still be inspected in a report explicitly marked
  `diagnostic_only`; such reports are never runtime-eligible and cannot support
  promotion.

## Backtest / verification gate

This is observability and evidence governance, not a trading-policy
relaxation, so a PnL backtest cannot approve or reject it. Verification is:

1. focused provenance, split-boundary, runtime-guard, and harness tests;
2. existing dataset atomicity/ranker tests;
3. a streaming audit over the maximum available local critic history, reporting
   actual date range and verified/legacy denominators;
4. Truth Harness `change --staged` and `git diff --check`.

No capture/PnL uplift may be claimed from this change. A later model promotion
still requires maximum-period causal replay, untouched chronological holdout,
and unified ten-slot portfolio alpha after costs.

## Rollback switch

`POLICY_PROVENANCE_REQUIRED_FOR_RANKER=False` restores legacy ranker loading
and training eligibility for emergency diagnostics. It must not be used to
make an achievement or production-promotion claim. The provenance fields are
append-only and remain safe for older readers.
