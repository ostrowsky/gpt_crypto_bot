# Early Signal One-Bar Persistence Replay

Status: research-only validation complete; `inconclusive_underpowered`
Date: 2026-08-29

No production or shadow policy was promoted. The machine-readable replay is a
runtime artifact and is intentionally not committed.

## Purpose

Test whether one additional closed 15-minute observation can distinguish useful
early score-gate alerts from noise without repeating the rejected `+120m`
confirmation delay.

This package does not change Telegram, BUY, score, portfolio, or exit behavior.

## Candidate Population

The maximum available completed local-day window is used. One opportunity is the
first successfully delivered score-gate early alert per `(local_day, symbol)`.
Telegram retries do not increase the denominator.

The candidate must be causally joined within 120 seconds to its preceding
`blocked_learning_label` and satisfy:

- timeframe `15m`;
- mode in `trend`, `strong_trend`, or `retest`;
- live top-gainer score in `[32, 34)`;
- a final critic exists for the local day.

Final-top membership is read only from `exchange_top_gainers` rows with
`in_watchlist=true`. `watchlist_top_gainers` is not an alternative denominator.

## Causal Feature Alignment

`bar_ts` is the candle-open timestamp. Its features become observable only at
`bar_ts + 15m` because the collector deliberately reads the last closed candle.
The base observation is the latest row whose feature-availability time is at or
before the alert, within 20 minutes. The confirmation observation must be
exactly one 15-minute bar later; its availability time is also shifted by one
bar and is the earliest actionable confirmation time. Both rows must be
in-watchlist, `15m`, and decision-time only.

Forward labels, later BUY state, and final critic membership are forbidden from
variant predicates.

## Pre-Registered Variants

No threshold sweep is allowed in the first run.

1. `persistence_structure`
   - confirmation slope is positive;
   - slope change is at least `-0.05` percentage points;
   - normalized MACD histogram change is at least `-0.02`.
2. `persistence_rank`
   - all `persistence_structure` conditions;
   - 24-hour rank does not worsen.
3. `persistence_quality`
   - all `persistence_rank` conditions;
   - confirmation relative volume is at least `0.8`;
   - upper-wick percentage is no larger than body percentage;
   - RSI is at most `75`.

## Labels And Metrics

Keep three meanings separate:

- operational confirmation: a later main-bot BUY on the same local day;
- objective confirmation: final canonical watchlist-top membership;
- forward market confirmation: confirmation-row T+10 return after 20 bps cost.

For baseline and each variant report:

- opportunities, active local days, candidates/day;
- operational BUY conversion;
- final-top precision and retained final-top count;
- T+10 net average, median, positive rate, and mature count;
- deterministic day-cluster bootstrap intervals for differences versus baseline.

## Temporal Validation

Split ordered candidate-population local days `60%/20%/20%` into train,
validation, and untouched holdout. Days are assigned before causal feature-pair
coverage is applied, so collector gaps cannot move the temporal boundaries.
Purge one complete local day on both sides of each boundary. The three variants
are frozen by this document and are not selected or tuned on holdout.

## Decision Gate

The terminal replay decision is:

- `advance_to_independent_shadow` only when a frozen variant has at least 30
  holdout candidates, retains at least five holdout final-top cases, improves
  final-top precision by at least 2 percentage points, does not reduce later-BUY
  conversion by more than 5 points, and has non-negative T+10 net median that is
  no worse than baseline;
- `inconclusive_underpowered` when coverage or outcome counts are below the gate;
- otherwise `rejected`.

Even a pass authorizes only independent shadow telemetry. Any BUY change still
requires current-policy maximum-period portfolio replay with costs, valid policy
epoch, rollback, and forward evidence.

## Frozen Validation Result — 2026-08-29

Terminal decision: `inconclusive_underpowered`.

- 3,825 delivered score-gate messages produced 3,422 causal label joins, 544
  eligible deliveries, and 457 first-per-day-symbol opportunities.
- Only 48/457 opportunities (10.5%) had an exact causal base/next-closed-bar
  pair. The other 409 are missing evidence, not negative labels.
- The aligned cases span 26 non-contiguous local days. Days were assigned from
  the full candidate population before coverage filtering. After the registered
  split and boundary purge: train `n=15`, validation `n=10`, untouched holdout
  `n=22`.
- Holdout baseline: final-top `1/22=4.55%`, later BUY `10/22=45.45%`, T+10 net
  median `+0.55%` after 20 bps.
- `persistence_structure`: final-top `1/13=7.69%` (`+3.15pp`) but later BUY
  `5/13=38.46%` (`-6.99pp`) and lower T+10 median `+0.15%`.
- `persistence_rank`: final-top `1/12=8.33%` (`+3.79pp`) and later BUY
  `5/12=41.67%` (`-3.79pp`), but lower T+10 median `+0.11%`.
- `persistence_quality`: `5` cases, no final-top, later BUY `2/5=40.0%`.

The holdout has fewer than the required 30 candidates and five final-top
outcomes. The apparent precision lift rests on one positive case. Structure
fails operational non-inferiority; rank stays inside that margin but fails the
T+10 median constraint. Therefore none advances to independent shadow.

Before rerun, telemetry must capture an alert-time closed-bar snapshot and an
exact one-bar follow-up for every eligible candidate, with feature time,
decision time, policy epoch, watchlist hash, and immutable candidate ID. A
coverage guard must fail the study before terminal analysis when causal-pair
coverage is below its registered minimum.

## Truth-Harness Constraints

- TH-01/TH-02: denominators and data status are explicit.
- TH-03: features are aligned to closed observations only.
- TH-04: day-separated holdout and purge are mandatory.
- TH-05: final-top, operational BUY, and T+10 are not conflated.
- TH-07: missing feature pairs are coverage loss, never negative labels.
- TH-10: retries are deduplicated by day and symbol.
- TH-11: research success cannot bypass canonical portfolio evidence.
