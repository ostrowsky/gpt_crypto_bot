# Watchlist-Filtered Top-Mover Denominator

Last updated: 2026-05-28

## Problem

The top-gainer critic previously used `top N within the watchlist universe` as the denominator for `watchlist_top_*` metrics. This made weak watchlist-relative coins look like missed same-day top movers when the actual Binance top movers were mostly outside the watchlist. The operator-facing metric must instead answer: among Binance top movers that are actually tradable by this bot because they are in the watchlist, how many did the bot buy and buy early?

## Objective Fit

This keeps the primary metric aligned with the bot's actual operating universe. The bot should not be penalized in `watchlist_top_capture_rate_pct` for Binance top movers that are outside the watchlist, and it should not inflate missed-winner counts with merely top-ranked watchlist symbols that are not in Binance top-N.

## Scope

Change `top_gainer_critic` so that:

1. `exchange_top_gainers` remains Binance top-N for context.
2. `watchlist_top_gainers` becomes `exchange_top_gainers` filtered to `in_watchlist=True`.
3. `watchlist_top_count`, `watchlist_top_bought`, `watchlist_top_missed`, early capture, blocked winners, and missed reasons are computed from that filtered set.
4. The previous `top N within watchlist universe` is retained separately as `watchlist_universe_top_gainers` / `watchlist_universe_top_count` for diagnostics only.

Example: if Binance top-N has 10 coins, 4 are in watchlist, and the bot bought 2 of those 4, `watchlist_top_capture_rate_pct` must be `2/4 = 50%`, not `2/10` and not `2/topN_inside_watchlist_universe`.

## Out Of Scope

- No BUY / SELL logic changes.
- No watchlist composition changes.
- No V2 policy changes.

## Primary Metrics

- `watchlist_top_count` = `len(exchange_top_N ∩ watchlist)`.
- `watchlist_top_capture_rate_pct` = bought among that filtered set.
- `watchlist_top_early_capture_rate_pct` = early captures among that filtered set.
- `blocked_winner_count` = blocked/missed among that filtered set only.

## Acceptance Criteria

- Unit test covers the 2/4 denominator case.
- Existing top-gainer tests still pass.
- Reports expose `watchlist_universe_top_gainers` only as diagnostic, not as primary capture denominator.

## Risk / Trade-offs

The primary denominator can be small on days when the watchlist misses most exchange movers. This is correct for measuring bot execution inside its universe, but reports must still show `exchange_top_in_watchlist` so watchlist coverage problems remain visible.

## Verification Gate

- `python -m unittest test_top_gainer_critic`
- Smoke report on 2026-05-27 if local data is available.

## Rollback Switch

Revert `top_gainer_critic.py` to the prior denominator logic. No runtime trading behavior is affected.

## Status

Measurement correction only. No trading behavior changes.
