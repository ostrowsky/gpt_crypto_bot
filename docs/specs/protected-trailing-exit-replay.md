# Protected trailing exit replay

Status: research-only  
Owner: Codex  
Date: 2026-05-29 Europe/Budapest

## Objective

Evaluate whether weak/EMA/stop exits should sometimes switch a position into a
protected trailing state instead of closing immediately.

This is motivated by exit-quality reports showing:

- negative median exit efficiency;
- high giveback;
- many `post_exit_continuation` cases;
- top-mover exit failures after already captured favorable movement.

## Non-goals

- Do not change live SELL behavior.
- Do not remove hard stops.
- Do not let an oracle-like continuation estimate become a production rule.
- Do not recommend adoption without a full candle-path replay.

## Inputs

Primary input:

- `.runtime/reports/signal_quality_YYYY-MM-DD_final.json`

The replay uses visible case rows from `trades`, `early_exits`,
`false_positive_buys`, and `late_entries`. If reports lack full trade rows,
coverage must be marked partial.

## Research policies

For eligible exit cases, estimate the value of a protected trailing state:

- baseline: current realized `pnl_pct`;
- protected_25: captures 25% of post-exit continuation opportunity;
- protected_50: captures 50%;
- protected_75: captures 75%.

Eligibility:

- exit reason bucket is EMA/weak/ATR/stop/unknown open;
- case has `future_favorable_pct` above current PnL by a configurable threshold;
- case is not a no-data row.

These are opportunity estimates, not production backtests.

## Acceptance

- Produce JSON/TXT artifacts under `.runtime/reports`.
- Include coverage status, eligible counts, estimated uplift, top cases, and a
  decision string.
- Unit tests must verify eligibility, uplift math, and research-only decision.

## Promotion rule

Any production SELL change must later be tested by candle-path replay with:

- realized PnL;
- exit efficiency;
- giveback;
- false hold damage;
- top-mover monetization;
- portfolio replacement impact.

