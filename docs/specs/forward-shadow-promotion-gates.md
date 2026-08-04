# Forward Shadow Promotion Gates

Date: 2026-08-04
Status: measurement-only

## Purpose

Turn the two replay-approved shadow profiles into reproducible independent
forward cohorts without reusing train, validation, or holdout evidence:

- `research_early_trend_catboost_v1` for additive early-trend detection;
- `exclude_ema_and_false_cleanup` for retaining a protected 50% exit tail.

The report is an evidence gate, not a trading action. It does not change BUY,
SELL, Telegram, positions, portfolio capacity, or watchlist membership.

## Early-Trend Cohort

- Read the frozen model metadata and accept only dataset rows strictly after
  `created_at_utc` with the exact exported profile.
- Use only 15-minute rows inside the trade watchlist.
- Count at most the first candidate per symbol and Europe/Budapest local day.
- A row is mature only when both T+5 and T+10 labels are present.
- Join final critics only after cohort construction. The canonical denominator
  is `exchange_top_gainers` filtered by `in_watchlist=true`.
- Compare the additive candidate union with the first existing V1 structural
  signal on the same forward days. Report recall, precision, newly captured
  top movers, and earlier captures.

The cohort remains in collection below 30 mature first candidates or five
local days. It then requires five critic days and five canonical top movers.
The frozen proxy guardrails are primary precision at least 9%, strict precision
at least 4.5%, non-negative average T+5, average T+10 at least 0.12%, no lower
canonical recall, no more than 2 percentage points precision loss, and at least
one newly or earlier captured canonical top mover.

Passing advances only to fee/slippage and ten-slot portfolio replay.

## Observable-Tail Cohort

- Read structured `observable_tail_shadow_candidate` and label events for the
  exact selector.
- Associate T+2/T+5/T+10 labels with the most recent matching earlier
  candidate by symbol, timeframe, selector, and exit price.
- Treat T+10 as the maturity boundary.
- Report the counterfactual 50% partial-tail delta before costs.

The cohort remains in collection below 30 mature T+10 cases or five local
days. It then requires positive average and median T+10 partial-tail delta and
no more than 35% worse cases. Passing advances only to a full portfolio replay
that includes fees, slippage, realized PnL, drawdown, and early-capture
guardrails.

## Failure Safety

Malformed JSONL rows are counted and skipped. Missing or invalid model
metadata fails the early-trend gate closed. Every decision is
`production_eligible=false`; production promotion remains a separate reviewed
change with a rollback switch.
