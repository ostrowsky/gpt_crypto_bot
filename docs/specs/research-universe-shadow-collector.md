# Research Universe Shadow Collector

Status: research/measurement-only  
Date: 2026-08-03

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

Label maintenance must be incremental at the collection-cycle level: all
symbols fetched in one cycle are applied in one streaming dataset pass, rather
than rereading and rewriting the complete dataset once per symbol. A malformed
JSONL row must not block later mature labels. Such rows are removed from the
active dataset during the atomic rewrite and copied to
`research_universe_shadow_quarantine.jsonl` with their line number and parse
error. Runtime state and quarantine output remain uncommitted artifacts.

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
- in-progress cycle status while a long collector cycle has not yet returned to the worker loop;
- batch-level progress for pairs scanned, rows written, and labels updated.
- malformed rows quarantined during the label pass.

The active JSONL file must be replaced atomically after a successful label
pass. If no label or quarantine change is needed, the original file must stay
byte-for-byte unchanged.

The maintenance command
`python backfill_research_universe_shadow_labels.py` must derive the earliest
and latest incomplete observation for every symbol/timeframe pair, fetch that
entire range plus the longest label horizon, and fill every label that has
matured. It must never truncate a concurrent append: the atomic replacement is
aborted if the source file size or modification time changes during the pass.

## Expected Effect

The bot can learn from a broader market without expanding live trading. The first measurable target is not PnL; it is increased positive-label coverage and better early-trend pattern discovery.

## Maximum-Period Backfill — 2026-08-03

The repaired job covered all 328 accumulated symbol/timeframe pairs from
2026-06-05 through 2026-08-03, fetched 811,338 candles, wrote 391,660 mature
T+3/T+5/T+10 labels, and quarantined the single malformed row with zero failed
pairs. Across 130,691 mature T+5 observations, average return was `-0.0346%`,
positive rate `45.63%`; outside-watchlist average was `-0.0611%`. Existing
`alignment` and `trend` rule names did not show enough unconditional edge to
justify live-watchlist or BUY expansion. Candidate patterns remain replay-only
and require chronological held-out validation.
