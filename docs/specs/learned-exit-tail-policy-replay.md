# Learned Exit-Tail Policy Replay

Status: research-only; production SELL and cooldown behavior unchanged.

## Objective

Test whether the causal post-exit discriminator can reduce false early exits by
leaving a protected fraction of a position open only for exits classified as
high continuation risk.

## Pre-registered design

- Population: every mature exit in the maximum available final
  `signal_quality` history that has a complete cached candle path. No
  retrospective `early_exit`, top-mover, capture-ratio, or future-return field
  may select a case.
- Label: `wrong_exit_continuation`, defined by the existing discriminator as at
  least 0.75 percentage points of favorable continuation beyond the larger of
  exit PnL and pre-exit MFE.
- Split: chronological by whole calendar day, 70% train and 30% test, with at
  least three train days. A day cannot appear in both partitions.
- Model: train the existing categorical causal-at-exit discriminator on the
  train partition only.
- Replay transport preserves entry and exit prices from the source final
  reports; missing prices fail action coverage instead of being treated as a
  zero-delta outcome.
- Admission: calculate the 80th percentile of train risk scores and apply that
  fixed threshold to the later test partition. Test labels cannot tune the
  threshold.
- Actions tested on admitted exits only:
  - sell 50% and trail 50% for 5 or 10 bars;
  - sell 70% and trail 30% for 5 or 10 bars;
  - tail exits on EMA20 loss, a 1.0 percentage-point adverse cap, or horizon.
- Unselected exits retain the recorded baseline exit.

## Decision gate

A candidate can advance only to a portfolio-aware replay when all conditions
hold on the untouched chronological test partition:

- at least 50 path-complete test exits and 20 selected exits;
- selected average delta above +0.10 percentage points;
- selected median delta is non-negative;
- selected worse-rate is at most 35%;
- selected delta p10 is at least -0.75 percentage points;
- total test average delta is above +0.02 percentage points;
- total test median PnL and win rate do not fall below baseline.

Passing this gate does not authorize production. Failure is a terminal reject
for this registered policy family; the learner remains diagnostic/shadow-only.

## Evidence and artifacts

The replay writes generated JSON/text under `.runtime/reports/`; those runtime
artifacts are not committed. A dated human-readable maximum-period result is
committed under `docs/reports/`.

## Maximum-period result (2026-09-08)

The registered family is terminally rejected. On 1,407 path-complete exits in
the later 32-day test partition, the train-only threshold selected 314 exits.
All four actions had positive mean delta but negative median delta, harmed
59.55% to 65.29% of selected exits, and reduced portfolio-level win rate.
Positive mean uplift was concentrated in a minority of large continuations and
is not a safe decision rule. See
`docs/reports/learned-exit-tail-policy-max-period-2026-09-08.md`.

The next admissible hypothesis must predict the advantage of a concrete action
(`tail PnL - immediate-exit PnL`) rather than use continuation classification as
a proxy for action value.
