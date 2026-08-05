# Early RSI-WEAK Exit Causal Replay

Date: 2026-08-05
Status: research-only complete; all frozen variants rejected, production SELL unchanged

## Problem

`check_exit_conditions()` classifies RSI divergence as a soft `WEAK:` warning
whose message says that the stop is tightened. The live monitor currently treats
the same result as a full SELL after the mode-specific grace period. Early
`15m/retest` positions can therefore close after two bars even while their price
structure remains intact.

The existing tail-selector research is not sufficient for this decision because
it filters on MFE/giveback and can exclude the exact low-MFE early-exit cohort.

## Frozen hypotheses

The replay evaluates the following predeclared policies without tuning them on
the final holdout:

- current full SELL baseline;
- persistent tightened ATR trails with `k` in `{0.9, 1.2, 1.4}` after any RSI
  `WEAK:` exit;
- the same tightened trails limited to `15m/retest`;
- `15m/retest` minimum weak-exit grace of `{3, 4, 5}` bars;
- `15m/retest` exit only after two consecutive RSI-WEAK observations;
- `15m/retest` structure veto with tightened trail `k` in `{0.9, 1.2, 1.4}`.
- `15m/retest` veto confirmed by the last fully closed `1h` candle, with the
  same tightened-trail grid;
- protected partial tails of `{25%, 50%}` only for the 1h-confirmed
  `15m/retest` cohort.

The structure veto is causal and requires the last fully closed candle to have:

- close at or above EMA20;
- EMA20 at or above EMA50;
- ADX at least `24`;
- EMA20 slope at least `+0.10%`;
- positive MACD histogram.

The 1h confirmation is also causal. Its last fully closed candle must have
close at or above EMA20, EMA20 at or above EMA50, ADX at least `22`,
non-negative EMA20 slope, and positive MACD histogram. The partial-tail
profiles realize the rest of the position at the original SELL and apply the
replayed policy only to the stated tail fraction.

Hard exits remain active in every variant. A replayed position is force-closed
after ten additional bars if neither its ATR trail nor a hard exit fires.

## Dataset and causality

- Source exits: every `event=exit` RSI-divergence `WEAK:` row in
  `files/bot_events.jsonl`.
- Price paths: local `.runtime/signal_quality_cache` OHLCV only.
- The decision feature row must be a fully closed candle at the event timestamp.
- The cached decision close must match the logged exit price within `0.25%`;
  larger timestamp/price mismatches are missing labels.
- Indicators are recomputed from cached candles; future candles are used only to
  score the frozen policy outcome.
- Cache files are indexed once and only fragments overlapping the case window
  are loaded. Missing paths remain missing labels rather than being inferred.
- Chronological train/validation/holdout splitting is performed by whole UTC
  day (`60%/20%/20%`) to avoid same-day leakage.

## Metrics

For each policy and split report:

- average and median PnL delta against the actual full SELL;
- conservative net delta after a `5 bps` policy-change penalty;
- worse-case rate and p10 delta;
- win rate, post-exit adverse excursion, exit efficiency, and giveback;
- `15m/retest`, early (`bars_held <= 2`), and combined early-retest slices.

## Promotion gate

A policy is only eligible for portfolio replay when:

- validation and untouched holdout each contain at least ten applicable cases;
- validation and holdout average net delta are positive;
- validation and holdout median net delta are non-negative;
- holdout worse-rate is at most `45%`;
- holdout p10 net delta is no worse than `-1.0%`.

Passing this gate does not authorize production. The selected policy must next
pass the maximum available portfolio replay with fees/slippage, turnover,
position-slot occupancy, drawdown, early-capture, and top-mover monetization.

## Result

The maximum locally labelable period contained `831/938` valid RSI-WEAK exits
from `2026-05-03` through `2026-08-04`. Whole-day splits were:

- train: `392` cases / `46` days;
- validation: `206` cases / `15` days;
- holdout: `233` cases / `16` days.

No policy passed the promotion gate. Broad tightened trails worsened holdout
average net delta by `-0.10` to `-0.16pp` and harmed roughly `70-72%` of cases.
The 1h-confirmed retest policies had positive means but negative validation and
holdout medians and harmed `56-61%` of holdout cases. Keeping only a 25% tail
reduced holdout p10 to `-0.50pp`, but its median remained `-0.07pp` and its
worse-rate remained `61%`.

The fresh 2026-08-05 UNI case was not in the mature holdout because ten future
cached bars were not yet available. It is a useful failure example, but it does
not override the historical rejection.
