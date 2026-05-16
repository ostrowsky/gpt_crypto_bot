# Agent Entry Quality Gates

Status: implemented, replay/backtest evidence required before relaxation  
Last updated: 2026-05-16

## Purpose

Keep the market-signal agent from turning broad contextual awareness into low-quality BUY pressure.
The agent may surface interesting candidates, but the production entry path should remain stricter
than the WATCH/diagnostic layer.

## Scope

- Use a stricter ADX floor for `trend` entries than for generic agent entries.
- Require explicit same-day strength confirmation for `4h_leader_watch`.
- Add optional soft-block WATCH alerts for interesting blocked candidates without opening positions.

## Why This Exists

- The bot optimizes for early capture of real same-day top movers, not simply more activity.
- 4h context is useful ranking evidence, but not sufficient as a standalone entry reason.
- Operator visibility should improve without weakening BUY precision.

## Acceptance Criteria

1. `trend` uses `AGENT_TREND_MIN_ADX` instead of the generic `AGENT_MIN_ADX`.
2. `4h_leader_watch` requires the configured minimum same-day move and volume strength.
3. Soft-block alerts are rate-limited to at most one alert per symbol/timeframe/rule/local-day.
4. No soft-block alert opens a position or bypasses the normal entry gate.
5. Unit tests cover the stricter trend ADX floor and 4h leader strength gate.

## Promotion / Relaxation Gate

Any future relaxation of these gates must be replay-tested against:

- trade precision;
- false-positive buys;
- blocked winners;
- realized PnL after fees;
- capture rate under the 10-slot portfolio constraint.
