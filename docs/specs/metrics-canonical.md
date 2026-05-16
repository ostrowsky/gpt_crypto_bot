# Canonical Metrics Map

Last updated: 2026-05-16

Use one primary metric per business question so roadmap decisions do not drift
between incompatible success definitions.

| Business question | Canonical metric | Supporting metrics | Notes |
|---|---|---|---|
| Do we see same-day watchlist winners? | `capture_rate` | `watchlist_top_bought`, `blocked_winners` | Split BUY, WATCH, and diagnostics. |
| Do we buy winners early enough? | `early_captures` | `capture_ratio_at_entry`, `lead_time_to_final_top_min`, `entry_day_range_percentile` | Count-only capture is insufficient. |
| Are BUY decisions selective enough? | `trade_precision` | `false_positive_buys` | More BUYs are not automatically better. |
| Which blockers cost us winners? | `blocked_reason_harm` | `missed_reason_counts`, `blocked_winners` | Use normalized blocker codes. |
| Do we preserve MFE after entry? | `exit_efficiency` | `giveback_pct`, `reversal_delay_bars` | Exit quality must be measured separately from entry quality. |
| Are cooldowns harming continuing trends? | `cooldown_harm` | post-exit run-up | Do not relax cooldowns without evidence. |
| Is WATCH useful without becoming noise? | `WATCH precision` | alerts/day, WATCH recall | WATCH quality must not be mixed with BUY precision. |
| Is the portfolio using scarce slots well? | `replacement_uplift` | `blocked_winners` under `10/10`, concentration measures | Requires replay before live changes. |
| Is the evaluator trustworthy today? | `coverage.status` | event-file presence, trades paired, candle coverage | A zero report is not automatically informative. |

## Decision Rules

1. Prefer the canonical metric for the question being answered.
2. Use supporting metrics to explain the canonical metric, not replace it opportunistically.
3. Do not compare features with different action layers as if they shared one denominator:
   - BUY affects capital and PnL;
   - WATCH affects operator noise;
   - diagnostics affect learning quality.

