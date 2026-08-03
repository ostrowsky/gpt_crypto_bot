# V2 Exit Advantage Model Family Comparison

Status: research-only  
Last updated: 2026-05-19

## Purpose

The first ridge exit-advantage baseline trained successfully but failed the
holdout proxy gate. This package tests whether the failure is caused by using a
linear decision surface.

The question is:

```text
Can simple nonlinear / binned causal models select SELL frames that beat naive
always-sell and never-sell advantage proxies on chronological holdout data?
```

## Candidates

Use transparent model families only:

- single-feature quantile bins;
- two-feature quantile-bin lookup tables;
- train-only bin statistics with minimum support;
- holdout decisions based on predicted bin mean `sell_advantage`.

No live behavior, Telegram signal, or production SELL change is authorized.

## Protocol

1. Load the action-level exit advantage dataset.
2. Keep the same chronological train / holdout split as the linear baseline.
3. Fit quantile bins on the train slice only.
4. Estimate train bin mean advantage for each bin/cell.
5. On holdout, SELL if predicted bin mean exceeds a threshold.
6. Compare against:
   - `never_sell` proxy;
   - `always_sell` proxy;
   - `oracle_positive_sell` upper bound.

## Acceptance Criteria

A candidate can advance to full offline replay only if it:

- beats `always_sell` captured advantage on holdout;
- beats `never_sell`;
- does not require selling nearly every frame unless that also beats always-sell;
- reports support, sell rate, bad-sell count, and strong precision.

This is still a proxy gate. Full environment replay remains mandatory before any
operator-facing signal.


## Audit Result

Latest run:

- rows: `99,935`;
- train rows: `69,954`;
- holdout rows: `29,981`;
- bins: `6`;
- minimum train support per bin/cell: `100`;
- holdout `always_sell` proxy: `22,721.463582`;
- holdout `oracle_positive_sell` upper bound: `65,048.375943`.

Best candidate:

- family: `single_bin`;
- feature: `adx`;
- selected threshold: `-2.0`;
- sell rate: `100.00%`;
- captured advantage: `22,721.463582`.

This exactly matches the `always_sell` proxy and therefore fails the gate.

## Interpretation

The binned nonlinear baselines did not solve the exit-selection problem. The best
model family degenerates into selling every holdout frame. More selective bins
found some positive pockets, but none captured enough advantage to beat the naive
holdout baseline.

This means the current causal feature set is not yet sufficient for robust
sell-vs-hold discrimination under the latest chronological regime. The bottleneck
is now feature / context design, not model-family complexity.

## Decision

Reject this package for replay promotion.

Next research direction:

1. add explicit trade-path context to the action-level dataset:
   - bars held;
   - unrealized PnL;
   - MFE;
   - giveback;
   - candidate action;
2. rerun the model-family comparison;
3. require beating `always_sell` before full environment replay.

Rationale: current features describe market/symbol structure, but the exit action
also depends on where the current trade is in its own lifecycle.

## Context-Aware Revalidation Protocol — 2026-08-03

The first model-family audit had two limitations that make its result
diagnostic rather than a valid promotion gate:

1. `bars_held`, unrealized PnL, MFE, giveback, and candidate action were present
   at the top level of every action row, but the audit read only the nested
   structural feature map;
2. the best prediction threshold was selected on the same holdout used for the
   final decision.

The revalidation must therefore:

- add the causal position-path fields and a `candidate_action_sell` indicator;
- split complete local days into train, validation, and untouched holdout;
- purge complete days around both split boundaries;
- fit quantile edges and bin means on train only;
- select both model family and threshold on validation only;
- read the holdout once and require captured advantage to beat both never-sell
  and always-sell without degenerating into a near-always-sell policy.

Passing still authorizes only a full offline environment replay. Production
SELL behavior remains unchanged.

## Context-Aware Revalidation Result — 2026-08-03

The corrected maximum-history dataset contained `99,935` action frames across
18 local days. Complete-day splitting with one-day purges produced `53,565`
train, `10,531` validation, and `14,209` untouched holdout rows; four boundary
days were excluded.

Validation selected a pair-bin model using `price_vs_ema20_pct` and
`belief_open_mass` at threshold `0.0`. Position-path features were available to
every candidate model but were not selected as the best validation family.

On untouched holdout:

- sell rate: `1.6891%`;
- captured advantage: `62.322330`;
- strong precision: `50.0%`;
- bad sells: `62`;
- always-sell captured advantage: `21,869.736704`;
- never-sell captured advantage: `0`.

Decision: `research_only_rejected_underperforms_always_sell_proxy`. Adding the
available causal trade-path context does not solve action-level exit selection.
Do not run the full environment replay and do not change production SELL.
