# Trend-Start Mode

Status: replay-only research mode, first profile rejected  
Last updated: 2026-05-16

## Purpose

`trend_start` is a dedicated early-entry hypothesis for coins that may later
finish among the same-day watchlist top movers. It exists because the current
BUY path can already discover early structure through scout/shadow layers, but
production entries are optimized for stronger current top-gainer evidence and
therefore may arrive late or not at all.

## Why It Is Separate

`trend_start` must not be implemented as a bypass over `impulse_speed`.
Backtests on 2026-05-16 rejected that approach:

- 30d baseline: `855` trades, `+31.80%` total PnL.
- 30d broad bypass: `1395` trades, `-15.37%` total PnL.
- 30d strict bypass: `1036` trades, `-28.56%` total PnL.

The hypothesis therefore gets its own entry contract, metrics, and replay gate.

## Entry Contract

Initial offline profile:

- timeframe: `15m`;
- positive same-day move already visible;
- price above or very near EMA20;
- EMA20 slope positive and still accelerating;
- early-but-not-exhausted RSI band;
- moderate ADX, before mature-trend territory;
- volume above baseline;
- no large extension from EMA20;
- recent high close enough to avoid late chase;
- MACD histogram not materially negative.

The first replay implementation intentionally uses only locally reproducible
features. A later live version may additionally require wake-up scout evidence,
but only after the replay profile itself proves useful.

## Objective Metrics

Primary:

- `capture_rate` on final top15;
- `trade_precision`;
- `capture_ratio_at_entry`;
- `lead_time_to_final_top_min`;
- total and average PnL.

Secondary:

- false-positive count by mode;
- giveback and exit efficiency;
- incremental contribution over the current production baseline.

## Promotion Gate

`trend_start` may move from replay-only to live only if a 7d and 30d comparison
shows:

1. no degradation in final top15 capture;
2. materially earlier entry (`capture_ratio_at_entry` and/or lead time);
3. no material damage to 30d PnL and precision;
4. behavior remains additive rather than duplicating existing modes.

Until then it remains an offline research mode.

## 2026-05-16 First Replay Result

The initial profile improved timing slightly but failed the economic gate after
the replay `intraday_change_pct` measurement was fixed:

- 7d baseline: `302` trades, `-74.16%` total PnL, `0.2596` average capture ratio.
- 7d `trend_start`: `295` trades, `-90.82%` total PnL, `0.2496` average capture ratio.
- 30d baseline: `1618` trades, `-259.00%` total PnL, `0.2268` average capture ratio.
- 30d `trend_start`: `1592` trades, `-294.97%` total PnL, `0.2319` average capture ratio.

Decision:

- reject the first profile;
- keep `trend_start` replay-only;
- future profiles must improve timing without worsening 30d PnL.
