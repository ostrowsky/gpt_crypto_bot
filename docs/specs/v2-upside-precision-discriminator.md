# V2 Upside Precision Discriminator

Last updated: 2026-05-27

## Problem

The V2 scorecard shows that the shadow Markov / belief observer can see many same-day top movers, but it also emits many favorable-looking upside symbols that do not become same-day watchlist top movers. Before adding RL complexity or promoting V2 into entry admission, we need to know whether the first V2 upside signal contains enough causal information to separate useful early upside from false-favorable noise.

## Objective Fit

This serves earlier same-day top-mover capture by improving selection quality, not by increasing signal volume. The goal is to reduce V2 false-favorable pressure while preserving top-mover recall.

## Scope

Add a research-only audit that builds one row per `symbol/day` at the first V2 upside transition:

- first `emerging_move`, `confirmed_trend`, or `mature_trend` event;
- event state, previous state, action, confidence, reason;
- first-event features such as ADX, RSI, slope, volume multiple, daily range, MACD histogram, price-vs-EMA;
- later same-day V2 confirmation/deescalation counts;
- outcome from `top_gainer_critic_*_final.json`: same-day watchlist top mover, bought by V1, capture ratio, exit efficiency when present.

The audit ranks simple feature slices by precision/recall trade-off and reports whether any slice is strong enough to advance to replay.

## Out Of Scope

- No BUY / SELL behavior changes.
- No new realtime Telegram alerts.
- No RL policy update.
- No production promotion from this audit alone.

## Primary Metrics

- `baseline_precision_pct`: useful V2 first-upside rows divided by all first-upside rows.
- `slice_precision_pct`: useful rows inside a feature slice.
- `slice_recall_pct`: useful rows captured by a slice.
- `false_favorable_reduction_pct`: reduction in false-favorable rows versus baseline if the slice were used as a gate.
- `advance_to_replay`: true only if a slice improves precision materially while retaining enough useful movers.

## Acceptance Criteria

- CLI writes JSON and text reports for a configurable date window.
- Missing outcome reports are marked as partial and excluded from decision claims.
- The report includes baseline, ranked slices, best slice, and an explicit decision.
- Unit tests cover first-upside extraction, top-mover outcome join, and slice ranking.

## Risk / Trade-offs

Simple feature slices can overfit small windows and can accidentally remove rare early winners. Therefore this audit can only advance a candidate to replay; it cannot approve production behavior.

## Verification Gate

- `python -m unittest test_v2_upside_precision_discriminator`
- Smoke run:
  `python files/audit_v2_upside_precision_discriminator.py --days 7`

## Rollback Switch

No runtime switch is required because this is an offline report. Delete or ignore the generated reports if the audit is not useful.

## Status

Research-only diagnostic. No trading behavior changes.
