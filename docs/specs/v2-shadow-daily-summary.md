# V2 Shadow Daily Summary

Status: expedited shadow-only  
Last updated: 2026-05-17

## Purpose

Replace noisy per-transition Telegram streams with one concise daily operator report for
the provisional v2 observer.

## Operator Question

At the end of a day, the operator should quickly understand:

1. how many symbols entered early positive lifecycle states;
2. how many advanced from `emerging_move` to `confirmed_trend`;
3. whether the observer mostly produced discovery candidates or churn;
4. which symbols deserve manual review tomorrow.

## Scope

### In scope

- build a daily report from `v2_shadow_events.jsonl`;
- exclude bootstrap observations;
- report counts for:
  - `emerging_move`
  - `confirmed_trend`
  - de-escalations back to `noise`
- list concise latest symbol examples;
- send one Telegram message inside the existing daily post-factum reporting flow;
- save JSON + TXT artifacts under `.runtime/reports`.

### Out of scope

- judging profitability before outcome labels exist;
- using the report to change live BUY/SELL behavior;
- separate scheduler or additional long-running worker.

## Metrics

- `upside_discovery_events`
- `confirmed_trend_events`
- `unique_upside_symbols`
- `deescalation_to_noise_events`
- `confirmation_ratio = confirmed_trend_events / upside_discovery_events`

## Alert Contract

- one daily message only;
- no raw downside stream;
- empty-day report is allowed and should say `0`, not disappear silently.

## Acceptance Criteria

1. Daily summary excludes bootstrap rows.
2. Summary artifact is reproducible from event log alone.
3. Telegram rendering is compact enough for operator use.
4. Summary is triggered by the existing daily signal-quality loop, not a new scheduler.
5. Unit tests cover counting and empty-day behavior.

