# Hold After Weak Sell Replay

Date: 2026-06-01
Status: research-only replay; no production SELL changes

## Problem

The daily learning report shows exit monetization as the main bottleneck: negative/weak exit efficiency and meaningful giveback. Some SELL decisions may be directionally right but too early or too binary.

## Hypothesis

For weak or suspicious exits, holding the position for a short fixed horizon after the original SELL (`+2`, `+5`, `+10` bars) improves realized PnL enough to justify a later conditional exit/partial-exit model.

## Non-goals

- Do not change live SELL logic.
- Do not bypass cooldown.
- Do not open re-entry positions.
- Do not promote a hold policy directly from aggregate replay.

## Replay Design

Inputs:

- `signal_quality_*_final.json` trade/exit rows;
- cached OHLCV candles from `.runtime/signal_quality_cache`.

Eligible cases:

- weak/ATR/EMA/stop-like exit reason, or early exit / high giveback / post-exit continuation tag;
- sufficient MFE before exit;
- closed or evaluable exit timestamp;
- available future candles.

Policies:

- baseline: original `pnl_pct`;
- `hold_2`: close after 2 bars;
- `hold_5`: close after 5 bars;
- `hold_10`: close after 10 bars.

Metrics:

- average/median PnL delta;
- win-rate delta;
- worse-case rate;
- adverse excursion after original exit;
- reason/mode breakdown.

## Acceptance Gate

A hold policy may advance only to a stricter candle-path / partial-exit replay if it improves average and median PnL without creating excessive adverse excursion. It must not be adopted in production from this replay alone.
