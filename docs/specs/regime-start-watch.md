# Multi-Day Regime-Start Watch

Status: shipped shadow-only; operator WATCH rejected by maximum-period validation
Last updated: 2026-07-14

## Problem

`POLUSDT` exposed a horizon mismatch. The bot observed causal intraday recovery
on 2026-07-01, but the observation stayed in `scout_shadow` because production
admission is optimized for same-day top gainers. Repeated tactical BUY/EXIT
cycles did not communicate that the observations belonged to one developing
multi-day trend.

The existing replay-only `trend_start` mode is an entry-policy experiment and
was rejected. `regime_start` is deliberately different: it is a non-trading
WATCH/shadow event for the transition into a possible multi-day bullish regime.

## Objective Fit

The feature supports earlier recognition and operator visibility of persistent
trends without weakening the profitable Top-Gainer BUY gate or changing exits.

## Scope

This change:

- evaluates only closed `4h` and `1d` candles;
- uses a causal `4h` recovery transition with the latest already-closed daily
  context;
- emits a deduplicated `regime_start_shadow` event independently of BUY
  admission, portfolio capacity, cooldown, and Top-Gainer score;
- provides a maximum-period historical audit with chronological holdout labels;
- may emit a Telegram WATCH notification only when the maximum-period gate
  passes and the separate config switch is enabled.

This change does not:

- open, close, resize, rank, or replace a position;
- bypass any existing gate;
- reinterpret the old `trend_start` BUY experiment;
- use the current incomplete `4h` or `1d` candle;
- use future values in the detector.

## Causal Profile: `base_recovery_v1`

The first profile represents a transition, not a continuously true state.

Local `4h` requirements:

- close is above EMA20 and EMA20 slope is positive;
- the prior bar did not already satisfy the complete local regime predicate;
- MACD histogram is positive and improving;
- RSI is in the non-exhausted recovery band;
- ADX and relative volume exceed minimum activity floors;
- price is not excessively extended from EMA20.

Latest closed `1d` context available at the `4h` decision time:

- daily RSI is not exhausted;
- daily MACD histogram is improving;
- daily close is not materially below EMA7;
- short daily return is not already a chase.

After a start, the same symbol is suppressed for a configurable number of `4h`
bars. A fresh event requires both cooldown expiry and a new false-to-true local
transition.

## Primary Metrics

- chronological train/holdout signal count;
- useful precision, where useful means forward 5-day MFE is at least `8%` and
  forward 5-day MAE is above `-8%`;
- median forward 3/5/10-day close return;
- median forward 5-day MFE and MAE;
- signals per calendar day in the evaluated period;
- per-symbol examples, including the first POL signal in July 2026.

## Acceptance Criteria

The profile may be enabled as an operator-visible WATCH only if the maximum
available watchlist history satisfies all of the following on the chronological
holdout:

1. at least 100 labeled signals over an evaluation period of at least 30 calendar days;
2. useful precision is at least `35%`;
3. median 5-day close return is positive;
4. median 5-day MFE is at least `6%`;
5. alert pressure is no more than `5` signals per calendar day;
6. useful precision is no more than 10 percentage points below train;
7. the July 2026 POL case is detected no later than the close of 2026-07-02 UTC.

If any criterion fails, shadow event collection may ship, but Telegram WATCH
must remain disabled. No result from this audit can enable a BUY.

## Risk / Trade-Offs

- recovery transitions are more common than successful multi-day trends and can
  create operator noise;
- direct daily confirmation can be late, while looser daily context can admit
  bear-market bounces;
- newly listed symbols have less warm-up history and must fail closed until all
  indicators are finite;
- exchange-wide market shocks correlate false starts across many symbols.

## Backtest / Verification Gate

- Fetch the maximum Binance `4h` and `1d` history available for every current
  watchlist symbol, from listing (or 2017-08-17) through the last closed bar.
- Cache fetched runtime data outside committed source files.
- Use a chronological 70/30 calendar split; do not randomly shuffle signals.
- Run focused unit tests for causality, transition semantics, daily alignment,
  deduplication, labeling, and promotion decisions.
- Run the audit, touched-module compilation, focused tests, and
  `git diff --check` before release.

## Rollback Switches

- `REGIME_START_SHADOW_ENABLED=False` disables scanning and event logging.
- `REGIME_START_TELEGRAM_ENABLED=False` disables operator-visible WATCH alerts
  while retaining shadow measurement.

## 2026-07-14 Maximum-Period Decision

`base_recovery_v1` was evaluated from 2017-08-17 through
2026-07-13 22:07 UTC on all 105 symbols returned by the current runtime
watchlist. Binance coverage was `105/105`.

Chronological results:

- train: 3,233 labeled signals, `33.44%` useful precision, median 5d return
  `-0.48%`, median 5d MFE `5.78%`, `1.42` signals/calendar day;
- holdout: 3,592 labeled signals, `29.70%` useful precision, median 5d return
  `-1.07%`, median 5d MFE `5.26%`, `3.68` signals/calendar day;
- `POLUSDT` was causally detected at 2026-07-02 00:00 UTC at `0.07124`, before
  the first production BUY at `0.07370`.

The profile failed the pre-registered precision, positive median return, and
median MFE gates. It is therefore shipped only as `regime_start_shadow` data;
`REGIME_START_TELEGRAM_ENABLED` remains `False`.

A stricter structural hypothesis (`slope>=0.15`, `ADX>=22`, `vol_x>=0.8`,
tighter 4h/daily price and RSI bands) was also checked on the same maximum
period. It retained the POL detection but worsened holdout useful precision to
`26.69%`, with median 5d return `-0.41%` and median MFE `4.67%`. That variant is
rejected and is not present in runtime configuration.
