# Blocked-Winner Causal Reward Table

Date: 2026-06-01
Status: research-only measurement; no BUY/SELL changes

## Problem

The bot counts blocked winners, but that does not tell which blockers are harmful and which are protective. Without a causal reward table, we can over-relax a useful guard or keep a harmful one.

## Goal

Create a daily/rolling blocker reward table:

- group blocked events by normalized reason code;
- estimate positive opportunity lost on missed/late watchlist top movers;
- estimate protection value from false candidates blocked by that reason;
- rank blockers by net harm/protection;
- surface top harmful examples for follow-up replay.

## Inputs

- `top_gainer_critic_*_final.json` for watchlist top labels;
- `files/bot_events.jsonl` and `files/agent_events.jsonl` for causal blocked events and entries;
- normalized blocker taxonomy from `blocking.py` via existing event replay helper.

## Reward Proxy

For each `(day, symbol, first_reason_code)`:

- harmful reward: missed top mover opportunity from first block, or late-entry opportunity when capture ratio <= threshold;
- protection credit: false candidate penalty avoided when symbol was not a watchlist top mover;
- neutral: already bought before block/rescue or no opportunity.

This report is not a PnL replay. It only decides which blocker families deserve candle-level behavior replay.

## Acceptance Gate

A blocker may advance to targeted behavior replay if:

- harmful opportunity is material;
- harmful top cases are repeated across multiple day-symbols;
- protection ratio is not overwhelming;
- examples are available for candle-level replay.

No blocker relaxation is allowed from this table alone.
