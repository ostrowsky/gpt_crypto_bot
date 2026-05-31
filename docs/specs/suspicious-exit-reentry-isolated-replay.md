# Suspicious Exit Re-entry Isolated Replay

Date: 2026-05-31
Status: research-only replay variant; no live entry/exit changes

## Problem

Broad replay of `suspicious_exit_reentry` changed two things at once:

1. it enabled suspicious-exit cooldown bypass;
2. it also participated in the top-gainer score-gated variant family, skipping thousands of candidates.

That makes the result useful as a policy-bundle test, but not as an isolated test of the re-entry idea.

## Goal

Add an isolated replay variant that behaves like baseline/normal candidate flow, but allows cooldown bypass only after a high-risk suspicious exit and only for confirmed normal candidates.

## Variant

`baseline_suspicious_reentry`

Expected behavior:

- no general top-gainer score gate;
- normal candidate universe is preserved;
- normal portfolio/replacement behavior is preserved;
- only delta is suspicious-exit re-entry window/cooldown bypass;
- no production trading changes.

## Acceptance Gate

Compare against baseline on 14d and 30d full watchlist:

- PnL total and average;
- win-rate;
- ret_3/ret_5/ret_10;
- exit efficiency and giveback;
- cooldown harm;
- number of re-entry windows and admitted re-entries.

If this passes while the policy-bundle version fails, the next step is Telegram shadow re-entry alerts, not live trading.
