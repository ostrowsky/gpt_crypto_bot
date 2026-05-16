# Peak Risk Shadow

Last updated: 2026-05-16 13:29 Europe/Budapest

## Purpose

Observe overextended profitable positions before any exit rule is changed.
This feature logs lifecycle telemetry only; it never emits a SELL and never
tightens a trail in production.

## Trigger

For an open profitable position, compute a 0-100 score from:

- RSI overextension
- price edge above EMA20
- positive MACD deceleration
- already-realized open profit

Rows are logged as `peak_risk_shadow` when score crosses a new bucket at or
above the configured threshold.

## Metrics

- shadow events per day
- score-bucket distribution
- share that precede a local peak within N bars
- false-positive continuation rate after alert
- later relation to `exit_efficiency` and `giveback_pct`

## Promotion Gate

Any future conversion from telemetry to tighter trail / Telegram alert must be
validated separately on replay and must improve exit quality without reducing
trend capture.
