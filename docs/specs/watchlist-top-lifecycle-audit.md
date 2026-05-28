# Watchlist Top Lifecycle Audit

Last updated: 2026-05-28

## Problem

After correcting the objective denominator to `Binance top-N ∩ watchlist`, the remaining improvement work is not watchlist coverage. The bot must improve two things inside the fixed watchlist universe:

1. **Early capture**: buy watchlist top movers before most of the daily move is gone.
2. **Exit monetization**: preserve more MFE and avoid turning correct captures into weak or negative realized PnL.

Current reports show aggregate metrics, but they do not provide a compact per-symbol lifecycle from V2 radar / blocks / BUY to exit outcome.

## Objective Fit

This is measurement-only tooling for earlier same-day top-mover capture and better trend exits inside the existing watchlist universe.

## Scope

Add a daily lifecycle audit for `watchlist_top_gainers` from the corrected top-gainer critic report. For each symbol it should show:

- day change and status;
- first V2 upside transition and first V2 confirmation;
- first block and dominant/latest block reason;
- first BUY time, mode, source, entry price;
- capture ratio at entry and lead time to final top cutoff;
- latest exit time, reason, PnL, exit efficiency, giveback;
- diagnosis for early-capture failure and exit failure.

## Out Of Scope

- No BUY / SELL behavior changes.
- No gate relaxation.
- No V2 promotion.
- No watchlist-universe expansion.

## Primary Metrics

- `early_failures`: bought top movers with `capture_ratio_at_entry < 0.35`, plus missed filtered top movers.
- `v2_to_buy_delay_min`: delay from first V2 upside to first BUY.
- `block_to_buy_delay_min`: delay from first block to first BUY when both exist.
- `exit_failures`: bought top movers with negative / low `exit_efficiency`, high giveback, or negative PnL.
- `post_exit_runup_pct`: when available in source rows; otherwise explicitly null.

## Acceptance Criteria

- CLI writes JSON and text reports for a target day.
- The report is based only on `watchlist_top_gainers` after denominator correction.
- Unit tests cover early-failure and exit-failure diagnosis.
- Missing V2 or exit data is marked as unknown, not interpreted as good or bad.

## Risk / Trade-offs

This report can expose plausible failure modes but does not prove a rule change. Any proposed entry/exit change must still go through replay/backtest.

## Verification Gate

- `python -m unittest test_watchlist_top_lifecycle_audit`
- Smoke run:
  `python files/report_watchlist_top_lifecycle.py --date 2026-05-27`

## Rollback Switch

No runtime switch is required; the audit is read-only. Delete/ignore generated reports if needed.

## Status

Research-only diagnostic. No trading behavior changes.
