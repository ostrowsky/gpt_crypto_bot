# Candidate Dataset Quality Guardrails and Online Learning

Date: 2026-08-27
Status: approved implementation; production trading policy unchanged

## Problem

The provenance-verified ranker cohort was counted as it matured, but forward
returns were produced through the open-position lifecycle.  As a result, all
training-eligible rows were accepted (`take`) decisions while thousands of
`blocked` and `shadow` observations remained unlabeled long after T+5.  A row
count threshold alone could therefore start training on a selected and
non-representative population.

## Objective fit

The candidate ranker must learn from the actual candidate population competing
for scarce alert/portfolio capacity.  This change improves evidence quality and
learning-loop availability; it does not claim improved capture, precision,
exit quality, or portfolio performance.

## Contract

1. New observations carry immutable
   `dataset_contract=candidate-outcome-v2`.  Older observations remain
   queryable but are excluded from default training.
2. The independent market collector matures exact T+3/T+5/T+10 returns for
   every candidate action (`candidate`, `take`, `blocked`, or `shadow`) from
   closed candles.  Position-only labels remain supplemental and may be absent
   for candidates that were not bought.
3. Missing future candles remain missing.  They are never converted to zero,
   success, or failure.
4. Online training begins before 500 rows, but only after a quality preflight
   passes.  The initial safe micro-cohort is 120 rows and subsequent shadow
   retraining requires at least 20 new rows.
5. The preflight is fail-closed and reports numerator/denominator evidence for:
   contract coverage, aged T+3/T+5/T+10 maturity, action diversity, target-class
   balance, teacher coverage/positives, complete decision groups, constant or
   non-finite features, and a viable purged chronological split.
6. A trained online artifact remains `shadow_online_training` and
   `runtime_eligible=false`.  Promotion requires the existing maximum-period
   top-1/top-3/top-5 pre-gate and paired portfolio replay.
7. The supervised headless launcher rejects `--disable-collector` by default,
   so a routine restart cannot silently recreate position-only selection bias.
   Emergency override requires the explicit
   `GPT_BOT_ALLOW_UNLABELED_COLLECTION=1` environment marker and is exposed in
   wrapper status evidence.
8. Candidate-wide forward labels are committed in one batch mutation per
   collector cycle.  Per-symbol full-file rewrites are forbidden because they
   cause lock contention, partial coverage, and action-dependent write loss.

## Initial guardrails

- minimum mature rows: `120`;
- maximum aged T+3/T+5/T+10 pending rate: `5%`, using T+10 availability plus a two-bar
  collection grace;
- at least two observed action classes and no single action above `95%` of the
  mature cohort;
- at least 10 positive and 10 non-positive quality targets;
- at least 30 decision groups and 10 groups with two or more candidates;
- at least 20 teacher-labeled rows and 5 teacher top-gainer positives;
- non-empty purged chronological train/validation/test partitions;
- no non-finite feature values.

These are collection-safety minima, not statistical proof of trading uplift.
The top-N promotion pre-gate still requires at least 100 eligible competitions
per required slice.

## Online lifecycle

`COLLECT -> MATURE -> PREFLIGHT -> SHADOW_TRAIN -> HOLDOUT_EVALUATE`

Any failed guard returns to `MATURE` with a named blocker.  It does not write a
new model artifact.  A successful shadow training run writes evidence but does
not alter BUY/SELL, score gates, portfolio replacement, or Telegram signals.

## Verification

- focused fixtures cover candidate-wide maturation, aged-label failure,
  batch maturation, action-selection bias, empty split failure, and a passing
  multi-action cohort;
- maximum locally available dataset audit reports old-contract rows separately
  and confirms they cannot enter the new training cohort;
- Truth Harness `full` and `change --staged` remain mandatory.

## Canary and rollout

The canary is a data-plane canary, not a subset of Telegram users: the first
complete collector cycle runs with the production alert policy unchanged and
the online artifact disconnected from runtime scoring.  It passes only when
the worker reports `collector.enabled=true`, writes current-contract
observations, completes the single batch maturation transaction without a
dataset-lock error, and the preflight still reports `runtime_eligible=false`.
Failure stops the headless worker and leaves the existing Telegram bot and
production ranker untouched.  Subsequent cycles remain shadow-only until the
separate promotion contract passes.

## Rollback

`RANKER_ONLINE_LEARNING_ENABLED=False` stops online model training while data
collection and quality reporting continue.  `RL_WORKER_ENABLE_COLLECTOR=False`
stops independent candidate maturation.  Neither switch launders or deletes
existing evidence.  The launcher's emergency CLI override is intentionally
separate and must not be used during normal operation.
