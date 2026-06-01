# Learning Progress Report Denominator and Freshness Semantics

Date: 2026-06-01
Status: implementation

## Problem

The daily learning report can produce misleading alerts when the watchlist-filtered top-mover denominator is zero. If no Binance top movers are in the bot watchlist, same-day watchlist top-mover capture is not applicable for that day and must not be treated as a 0% failure.

The report can also show measurement as stale after a process restart because worker status fields are reset even though dated report files exist on disk.

## Goal

Make the learning report robust enough for operator decisions:

1. `watchlist_top_count == 0` means target metric is not applicable for the latest day, not failed.
2. Serious early-capture alerts require a positive denominator.
3. Render text should say when there were no watchlist top movers yesterday.
4. Measurement freshness should fall back to actual dated report files when worker status is empty.
5. No trading logic changes.

## Acceptance Criteria

- A day with `watchlist_top_count == 0` does not emit `early capture only 0.0%` as serious.
- Rendered report says `watchlist top movers: 0 — метрика дня не применима` or equivalent.
- Measurement component is `ok` when latest critic and signal-quality files exist for the latest day, even if worker status lacks last-target fields.
- Unit tests cover both cases.
