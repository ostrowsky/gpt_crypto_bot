# Research Universe Shadow Collector

Status: research/measurement-only  
Date: 2026-06-04

## Problem

The bot's live trading watchlist is intentionally narrow. That is correct for operator control and live BUY/SELL accountability, but it limits the learning loop to a small fraction of Binance top-mover examples.

The `research-universe-impact-audit` showed that the positive-label pool could expand materially if the bot learned from a wider Binance USDT universe.

## Goal

Add a shadow-only collector that observes a wider Binance USDT universe and writes research observations into a separate dataset.

The collector must support learning/replay research without changing live behavior.

## Guardrails

- Do not modify `watchlist.json`.
- Do not alter BUY/SELL gates.
- Do not open positions.
- Do not emit Telegram alerts.
- Do not write into `ml_dataset.jsonl` or `critic_dataset.jsonl` by default.
- Store research observations in a separate JSONL file.
- Keep configurable caps for symbol count, timeframes, batch size, cycle interval, and per-symbol timeout.
- Start with a bounded liquid-symbol ramp-up before attempting uncapped full-universe collection.
- Promotion from research universe to trade watchlist requires a separate replay/liquidity/operator gate.

## Dataset

Default file:

```text
files/research_universe_shadow.jsonl
```

Each row should include:

- symbol / timeframe / closed bar timestamp;
- whether the symbol is currently in the trade watchlist;
- 24h rank/change/quote-volume context;
- rule-signal classification from the existing V1 structural detector;
- selected scalar market-structure features;
- forward labels for T+3/T+5/T+10 when mature.

## Runtime Integration

The collector may run from the headless worker as a separate background task.

Default rollout should be capped to avoid starving the existing live/research loops. Increase the cap only after runtime status shows healthy cycle duration and no API degradation. Each symbol fetch must be bounded by timeout so a long-tail API hang cannot block the worker indefinitely.

It should report status in `rl_worker_status.json`, including:

- enabled/running;
- last started/finished;
- last error;
- symbols scanned;
- rows written;
- labels updated;
- in-progress cycle status while a long collector cycle has not yet returned to the worker loop.

## Expected Effect

The bot can learn from a broader market without expanding live trading. The first measurable target is not PnL; it is increased positive-label coverage and better early-trend pattern discovery.
