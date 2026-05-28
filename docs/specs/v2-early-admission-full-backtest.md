# V2 Early Admission Full-Candidate Backtest

Last updated: 2026-05-28

## Problem

Quick replay on known watchlist top movers showed that V2 first-upside often appears earlier than V1 BUY. That is not sufficient for adoption because it ignores false-favorable V2 signals. We need a full-candidate backtest over all V2 upside symbols inside the watchlist universe and all locally available days with corrected top-gainer outcomes.

## Objective Fit

The backtest evaluates whether V2 can improve early capture and exit opportunity inside the fixed watchlist universe without inflating false positives.

## Scope

Add a research-only backtest that:

- uses all local days where both V2 shadow events and corrected `top_gainer_critic_*_final.json` exist;
- considers one first candidate per `symbol/day` for V2 policies;
- limits candidates to symbols present in the bot watchlist;
- labels positives as `Binance top-N ∩ watchlist` from corrected top-gainer reports;
- computes entry price from V2 event features;
- computes hold-to-close return and MFE-to-day-high from local V2 15m history;
- compares V2 policies against V1 actual first buys on the same corrected objective universe.

## Out Of Scope

- No production BUY/SELL changes.
- No portfolio-cap simulation in this first pass.
- No order-book/slippage model beyond optional fee bps parameter.
- No automatic approval.

## Primary Metrics

- candidate count;
- top precision and top recall;
- false-favorable count/rate;
- median/average hold-to-close return after fees;
- median/average MFE-to-day-high;
- median capture remaining for true top movers;
- decision: reject / research-only / advance to portfolio replay.

## Acceptance Criteria

- CLI writes JSON and text reports.
- Unit tests cover top label joins, false-favorable accounting, and policy ranking.
- Missing candle history is counted as partial coverage, not silently ignored.

## Risk / Trade-offs

This is still a bar/day proxy replay. It can validate whether early V2 admission has statistical promise, but production adoption still requires portfolio-aware replay with fees, slippage, and real exit rules.

## Verification Gate

- `python -m unittest test_v2_early_admission_full_backtest`
- `python files/backtest_v2_early_admission.py --json`

## Status

Research-only. No trading behavior changes.
