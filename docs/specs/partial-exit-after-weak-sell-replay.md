# Partial Exit After Weak Sell Replay

Date: 2026-06-01
Status: research-only replay; no production SELL changes

## Problem

The hold-after-weak-sell replay found a real exit monetization signal: short holds after weak/suspicious SELL often improve PnL. But full-position holds also create a high harm rate, especially for some EMA-break and false-positive cases.

## Hypothesis

For weak or suspicious exits, selling part of the position immediately and holding the remaining tail for a short horizon can preserve much of the continuation upside while reducing downside compared with holding the full position.

## Non-goals

- Do not change live SELL logic.
- Do not change stop-loss or trailing-stop execution.
- Do not infer order-book fills, fees, or slippage from this replay.
- Do not promote a partial-exit policy directly from aggregate results.

## Replay Design

Inputs:

- the same `signal_quality_*_final.json` exit rows used by hold-after-weak-sell replay;
- cached OHLCV candles from `.runtime/signal_quality_cache`;
- labeled hold outcomes from `files/replay_hold_after_weak_sell.py`.

Policies:

- baseline: current full SELL at original exit;
- `partial_50_hold_2/5/10`: sell 50% at original exit, hold 50% to horizon;
- `partial_70_hold_2/5/10`: sell 70% at original exit, hold 30% to horizon;
- reason-filtered breakdowns, especially `weak_signal` vs `ema_break` vs false-positive cases.

Approximate policy PnL:

`partial_pnl = sold_fraction * baseline_pnl + (1 - sold_fraction) * horizon_hold_pnl`

This is intentionally conservative/simple. It does not model dynamic stops for the remainder; that comes in the next candle-path replay if the edge survives.

## Acceptance Gate

A partial policy may advance to candle-path trailing-tail replay only if:

- average and median delta are positive;
- worse-rate is materially below full hold for the same horizon;
- the edge is concentrated in interpretable buckets such as `weak_signal` or `early_exits`, not in noisy aggregate artifacts;
- false-positive buckets are not made materially worse.

Production adoption requires a separate replay with fees/slippage, remainder stop logic, and no duplicate trade artifacts.

## Research Harness Reliability

Replay runs over the maximum available history must be bounded by unique `(symbol, timeframe)` candle-cache loads, not by repeated file scans per trade case. The hold-label helper is shared by hold, partial-exit, trailing-tail, and observable-tail selector replays; therefore it must memoize candle series within a single replay invocation.

This is a measurement/research reliability requirement only. It must not change live SELL behavior or any replay metric definition.
