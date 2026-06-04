# Research Universe Impact Audit

Status: research/measurement-only  
Date: 2026-06-04

## Problem

The bot's live objective is measured against Binance/exchange top movers filtered to the approved trading watchlist. That is correct for live BUY/SELL accountability, but it also means the learning loop sees only a small subset of Binance top-mover examples.

Expanding the live watchlist directly to the full Binance universe would change the trading game, increase noise, increase API/runtime load, and likely degrade operator UX.

## Goal

Measure whether a wider Binance research universe would improve the bot's self-improvement loop without changing live trading.

The audit must answer:

- how many exchange top-mover examples are inside vs outside the current watchlist;
- how much larger the positive-label pool would be if research used the wider universe;
- which outside-watchlist symbols repeatedly appear in exchange top movers;
- whether the result supports research-only collection, trade-watchlist promotion candidates, or no action.

## Guardrails

- Research/reporting only.
- Do not change `watchlist.json`.
- Do not change BUY/SELL gates.
- Do not send Telegram alerts for outside-watchlist symbols.
- Do not call outside-watchlist misses a live bot failure.
- Promotion candidates require separate liquidity, spread, replay, and operator approval gates.
- Promotion candidates must be normalized exchange-style symbols only: ASCII alphanumeric symbols ending in `USDT`.

## Expected Output

Create a report from the longest available local `top_gainer_critic_*_final.json` window:

- current trade-watchlist size;
- days loaded and date range;
- exchange top events;
- exchange top events inside watchlist;
- watchlist capture and early-capture rates;
- exchange-wide diagnostic capture and early-capture rates;
- positive-label expansion factor;
- repeated outside-watchlist top movers;
- recommendation.

## Decision Rule

If outside-watchlist exchange top events are materially larger than inside-watchlist events, recommend a `research_universe` / `trade_watchlist` split:

- learn broadly from liquid Binance USDT symbols;
- trade only approved watchlist symbols;
- propose promotion candidates only after replay gates.
