# Trailing Tail After Partial Exit Replay

Date: 2026-06-01
Status: research-only replay; no production SELL changes

## Problem

Partial-exit replay showed that keeping a tail after weak/suspicious exits can improve exit monetization, especially for early exits. But fixed-horizon tails still hurt false-positive buys and some broken structures.

## Hypothesis

For weak or suspicious exits, sell part of the position immediately and keep the tail only while short-term structure remains acceptable. A simple candle-path trailing-tail policy should preserve continuation upside while reducing harm versus fixed full holds.

## Non-goals

- Do not change live SELL logic.
- Do not model exact exchange fills, fees, or slippage.
- Do not promote a policy directly from aggregate replay.
- Do not apply tail retention to false-positive-looking trades without a veto/gate.

## Replay Design

Inputs:

- `signal_quality_*_final.json` exit rows;
- cached OHLCV candles from `.runtime/signal_quality_cache`;
- eligibility and labeling helpers from `files/replay_hold_after_weak_sell.py`.

Policies:

- baseline: current full SELL at original exit;
- `tail50_h10_ema20_cap100`: sell 50%, hold tail up to 10 bars, close tail when close falls below EMA20 or tail adverse reaches -1.00%;
- `tail50_h10_ema20_cap150`: same with -1.50% cap;
- `tail70_h10_ema20_cap100`: sell 70%, hold 30% tail with same structural stop.

Approximate policy PnL:

`policy_pnl = sold_fraction * baseline_pnl + tail_fraction * tail_exit_pnl`

Tail exit uses future candle close at the first structural stop, adverse cap, or max horizon. This is still a candle-path approximation, not production execution.

## Acceptance Gate

A policy may advance to shadow/live dry-run only if:

- aggregate average and median delta are positive;
- early-exit / weak-signal slices improve;
- false-positive slice is not materially worsened;
- harm rate is lower than fixed full hold;
- duplicate artifacts and pending labels are audited before promotion.
