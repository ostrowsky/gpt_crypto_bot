# Partial Profit-Take Exit Replay

Date: 2026-05-31
Status: research-only replay variant; no live SELL changes

## Problem

Recent daily reports show strong early capture on watchlist top movers, but poor monetization: high MFE/giveback and low exit efficiency. Holding exits longer and suspicious-exit re-entry have not passed replay gates.

## Hypothesis

A causal partial profit-take may preserve part of MFE without delaying the final exit. This tests whether the bot should lock some profit on strong favorable movement while leaving the remaining position under existing exit logic.

## Replay Variant

Variant: `partial_profit_take`.

Behavior:

- if current candle close reaches a configured profit threshold, mark a one-time partial exit;
- final replay PnL becomes weighted PnL of partial leg plus remaining leg;
- final SELL timing and remaining position exit logic are unchanged;
- portfolio limits, entries, and live trading logic are unchanged.

Default research parameters:

- partial fraction: 50%;
- trigger: +3.0% current PnL;
- optional minimum MFE: +3.0%.

## Acceptance Gate

Do not promote unless replay improves or at least does not worsen:

- total PnL;
- average PnL;
- win-rate;
- giveback / realized capture;
- false positive or churn proxies.

This is measurement-only until a separate live implementation spec is approved.
