# Research Universe Shadow Scorecard

Status: research/measurement-only  
Date: 2026-06-04

## Problem

The research-universe shadow collector expands observation coverage beyond the live trade watchlist, but raw JSONL rows do not tell whether this broader universe is improving the learning loop.

Without a scorecard, the bot can collect more data while still failing to answer the important question: which wider-universe patterns or symbols deserve replay-gated promotion research?

## Goal

Add a daily research-only scorecard for `files/research_universe_shadow.jsonl`.

The scorecard must summarize mature forward labels, pattern quality, and promotion candidates without changing live trading behavior.

## Guardrails

- Do not modify `watchlist.json`.
- Do not change BUY/SELL gates.
- Do not write to production ML/critic datasets.
- Do not treat outside-watchlist opportunities as live bot misses.
- Promotion candidates are research candidates only and require separate replay/liquidity/operator approval.
- Telegram output is disabled by default to avoid operator noise.

## Metrics

Default horizon: `ret_5`.

The scorecard should report:

- total rows and mature rows;
- inside vs outside trade-watchlist rows;
- positive rate and average/median return;
- high-return rate for stronger labels;
- performance by `rule_signal`;
- top outside-watchlist symbols by mature label quality;
- top feature-pattern buckets for hypothesis generation.

## Promotion Triage

A symbol can be listed as a promotion candidate only if it has enough mature rows and passes minimum positive-rate and average-return thresholds.

This is not production approval. It means only: create a separate replay experiment for adding or shadow-prioritizing that symbol.

## Runtime Integration

The headless worker may run the scorecard once per daily reporting slot and save outputs into `.runtime/reports`.

Default files:

```text
.runtime/reports/research_universe_shadow_scorecard_latest.json
.runtime/reports/research_universe_shadow_scorecard_latest.txt
```
