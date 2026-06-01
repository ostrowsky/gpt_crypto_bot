# Entry Admission Shadow Reward Loop

Date: 2026-06-01
Status: research-only measurement loop; no production BUY changes

## Problem

Entry admission still relies mostly on fixed gates and retrospective audits. The bot can see blocked winners, but it does not yet convert blocked/admitted candidate outcomes into a daily reward signal for admission policy improvement.

## Goal

Create a daily shadow reward report for entry admission/rescue candidates:

- identify causal blocked-candidate opportunities from existing event logs;
- label them using watchlist-filtered top-gainer critic outcomes;
- estimate reward for missed/late top movers;
- subtract penalty for false candidate pressure;
- rank candidate rescue policies by net proxy reward.

## Inputs

- `.runtime/reports/top_gainer_critic_*_final.json` for watchlist top labels;
- `files/bot_events.jsonl` and `files/agent_events.jsonl` for blocked/entry events;
- existing normalized blocker taxonomy from `audit_early_block_rescue_event_replay.py`.

## Candidate Policies

Research-only grids over:

- reason set: agent gates, score gate, cooldown, chase guard combinations;
- maximum first-block hour;
- minimum repeated blocks.

## Reward Proxy

For each selected `(day, symbol)` candidate:

- positive reward: missed/late watchlist top mover opportunity from first block;
- zero reward: already bought before the rescue timestamp or already early enough;
- negative reward: fixed false-candidate penalty for non-top candidates.

This is not a trading PnL replay. It is a ranking signal for which admission hypotheses deserve deeper candle-level behavior replay.

## Acceptance Gate

A policy may advance only to candle-level behavior replay if:

- net proxy reward is positive and materially above zero;
- top precision is acceptable;
- false candidate count is bounded;
- it rescues multiple missed/late watchlist winners;
- all proposed behavior changes remain disabled until replay/backtest evidence exists.

## Non-goals

- No BUY gate relaxation.
- No Telegram BUY/WATCH change.
- No use of non-watchlist Binance top movers in the denominator.
- No production policy promotion from this proxy alone.
