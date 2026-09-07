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
9. Current-contract observations are written to the dedicated
   `critic_dataset_v2.jsonl` evidence stream.  The historical
   `critic_dataset.jsonl` remains immutable legacy evidence and is never mixed
   into online-training denominators.  This bounds rewrite/lock duration and
   makes a clean collection restart auditable without deleting history.
10. Collector writes are strict and fail closed.  An append or maturation
    transaction that cannot acquire the dataset lock raises a dataset-integrity
    error; the row is not counted as written, the collector is marked disabled,
    a stop marker is created, and the supervised headless worker exits instead
    of retrying an apparently successful partial cycle.
11. The candidate collector does not append or mature the legacy
    `ml_dataset.jsonl` by default.  That stream has no v2 provenance contract,
    is not an input to the online candidate ranker, and its per-pair rewrites
    make a bounded collector cycle impossible.  A diagnostic rollback switch
    may re-enable it explicitly, but such rows remain ineligible for v2
    training.
12. Reprocessing an unchanged candidate decision is a no-op. It must finish
    during the unlocked preview scan and must not acquire the cross-process
    dataset lock or rewrite the JSONL stream.
13. The cross-process lock wait budget is 120 seconds by default and can be
    changed only with `GPT_BOT_CRITIC_LOCK_TIMEOUT_SEC`. This exceeds the
    measured 23-second full serialization time of the current 125 MB stream;
    timeout still fails closed and never counts a lost append as successful.
14. Atomic snapshot replacement has a separate 120-second Windows reader
    contention budget, configurable with
    `GPT_BOT_CRITIC_REPLACE_TIMEOUT_SEC`. A transient reader handle is retried
    while the writer lock remains held; exhaustion is an integrity failure and
    cannot be counted as a completed label update.
15. Recovery after collector downtime uses paginated Binance klines from the
    earliest mature missing target through the proof bar after T+N. It updates
    only exact closed-bar T+3/T+5/T+10 outcomes. Missing exchange history stays
    `unknown` and is reported; it is never imputed as zero, success, or failure.
16. Multi-process JSONL append order is not assumed to be chronological. The
    preflight sorts verified rows by immutable `feature_time` before building
    purged train/validation/test partitions, matching the training loader.
    Cross-timeframe competition groups also use `feature_time` (closed-bar
    availability), never the candle-open `ts_signal`. The training loader sorts
    by the same key so every CatBoost ranking query remains contiguous.

## Policy-epoch bridge and target availability

`policy_epoch` is a manually versioned semantic decision identity. Exact
config/source/watchlist hashes remain attached to every observation, but an
operational collector repair or notification wording change does not create a
new decision epoch. Any production eligibility, score, routing, BUY/SELL,
re-entry, sizing, replacement, capacity, or cost change must bump the semantic
epoch.

The two raw epochs already present in `critic_dataset_v2.jsonl` are registered
as decision-equivalent to `decision-policy-v1-20260827`:

- `pe1-f7fdfbdbba47b9f3`;
- `pe1-648646fcc3ebb415`.

The only source-hash difference between them is `monitor.py` commit
`548f2c98a62ca6bbbdd8153e9d427f082d265d8f`, which changed operator-facing
Telegram wording and same-day entry labeling but not admission, ranking,
portfolio, or exit decisions. Reports retain raw counts and expose the bridge;
an unregistered epoch still fails `unbridged_policy_epochs`.

Teacher target `label_time` means when the target outcome became objectively
available: 12:00 local for `midday`, and the next local midnight for `final`.
`recorded_at` remains the actual later annotation time. A delayed report may
not move objective availability forward and collapse a chronological split.

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

Recovery follows the same strict order: restore dataset integrity and mature
labels; verify the registered policy epoch and purged split; train/evaluate in
shadow; run maximum-period candidate replay and the canonical 30-day ten-slot
after-cost portfolio replay; only then consider a bounded production canary.

## Verification

- focused fixtures cover candidate-wide maturation, aged-label failure,
  batch maturation, action-selection bias, empty split failure, and a passing
  multi-action cohort;
- focused fixtures also cover idempotent candidate updates, paginated recovery,
  registered versus unknown policy epochs, and objective teacher availability;
- maximum locally available dataset audit reports old-contract rows separately
  and confirms they cannot enter the new training cohort;
- Truth Harness `full` and `change --staged` remain mandatory.

Applicable invariants: TH-01 through TH-07 and TH-09 through TH-12. In
particular, shadow training is proxy evidence (TH-02), purged availability is
chronological OOS evidence (TH-03/TH-04), and no production claim is permitted
without candidate-population maximum-period replay and unified after-cost
portfolio alpha (TH-06/TH-11).

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
