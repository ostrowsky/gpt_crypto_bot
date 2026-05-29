# Daily learning next actions — 2026-05-29

Status: active research / measurement-only  
Owner: Codex  
Date: 2026-05-29 Europe/Budapest

## Objective

Turn the 2026-05-29 learning report into concrete, falsifiable next actions
without changing live BUY or SELL behavior.

The current report says:

- 7d early-capture improved to target level: about 25.7%;
- latest day captured and early-captured 66.7% of watchlist top movers;
- coverage is still partial: 186/206 candles;
- trend miss-rate remains high;
- exit efficiency remains weak.

## Decision frame

Do **not** broaden entry gates while early-capture is at/above target and
coverage is partial. The immediate work is measurement hardening and exit
monetization research.

## Required checks

1. Coverage triage
   - Identify whether `coverage=partial` is caused by missing files, missing
     symbols, incomplete candle windows, duplicate/manual artifacts, or evaluator
     denominator issues.
   - Mark whether the partial coverage is metric-affecting or safe.
   - If missing candle series correspond only to inactive exchange symbols
     (`BREAK`, `HALT`, `END_OF_LIFE`, or otherwise non-`TRADING`), classify the
     gap as safe/inactive rather than a data-pipeline failure.

2. Blocked-winner audit
   - Inspect the latest final top-gainer critic report.
   - List the exact blocked winner(s), dominant reason, first block, latest
     block, opportunity loss, and whether the reason recurs across recent days.
   - Do not recommend production relaxation from a single case.

3. Exit-quality replay/audit
   - Use the existing exit-quality auditor across the longest available
     signal-quality window.
   - Identify the worst SELL buckets and symbols by opportunity loss, giveback,
     early exits, and post-exit continuation.
   - Produce hypotheses only; SELL changes require separate replay.

## Acceptance

- Produce JSON/TXT artifacts under `.runtime/reports`.
- If tooling changes are needed, they must be measurement-only and covered by
  unit tests.
- Do not commit runtime artifacts.
- Do not change production trading decisions in this package.
