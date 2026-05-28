# Exit-quality auditor

Status: shipped measurement-only  
Owner: Codex  
Last updated: 2026-05-28 Europe/Budapest

## Objective

Add a daily/rolling exit-quality auditor for the current production bot without
changing BUY or SELL behavior. The report must make exit monetization failures
visible enough to decide which exit hypotheses deserve replay.

The auditor answers:

1. Are captured watchlist/top-mover trends being monetized or given back?
2. Which symbols/modes/reasons create the worst MFE loss?
3. Are failures mostly early exits, late exits, open positions marked down, or
   negative exits after favorable movement?
4. Which concrete cases should be used for the next replay gate?

## Non-goals

- Do not change live SELL logic.
- Do not lower exit thresholds.
- Do not recommend production adoption from summary statistics alone.
- Do not treat lower giveback as sufficient if total PnL/capture worsens.

## Inputs

Primary input is existing signal-quality reports:

- `.runtime/reports/signal_quality_YYYY-MM-DD_final.json`

The auditor consumes the report summary and the visible case rows from:

- `late_entries`
- `early_exits`
- `false_positive_buys`
- `trades` when reports were generated with `--include-trades`

If reports do not include full trade rows, the auditor must mark case-level
coverage as partial while still using summary-level medians.

## Metrics

Rolling summary:

- `closed_trades_total`
- `late_exits_total`
- `early_exits_total`
- `exit_efficiency_median`
- `giveback_pct_median`
- `pnl_pct_median`
- `negative_exit_case_count`
- `high_giveback_case_count`
- `top_mover_exit_failure_count`
- `case_rows_loaded`
- `case_coverage_status`

Per-case fields:

- day, symbol, timeframe, source, mode;
- entry/exit timestamps and prices;
- pnl, MFE, future favorable move;
- exit efficiency, giveback;
- top-mover rank/change when available;
- normalized exit reason;
- diagnostic tags.

## Diagnostic tags

A case can have multiple tags:

- `top_mover_exit_failure`: top mover rank is inside the configured top-N and
  exit efficiency is weak, PnL is negative, or giveback is high.
- `negative_after_mfe`: PnL <= 0 while MFE was positive.
- `high_giveback`: giveback is above threshold.
- `late_exit`: evaluator classified exit as late.
- `early_exit`: evaluator classified exit as early.
- `open_marked_down`: position was still open in the evaluator window but marked
  with negative PnL.
- `post_exit_continuation`: future favorable move exceeded MFE seen before exit.

## Acceptance / verification

- Unit test with synthetic reports must verify summary aggregation, de-duplication,
  tags, and worst-case ordering.
- `py_compile` must pass.
- Smoke run on local reports must produce JSON and TXT under `.runtime/reports`.

## Promotion rule

The auditor itself is measurement-only and can ship without replay. Any SELL
change proposed from it must later pass replay on:

- realized PnL;
- exit efficiency;
- giveback;
- false early exits;
- watchlist top-mover capture monetization.
