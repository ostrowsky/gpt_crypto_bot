# V2 Shadow Alert Policy

Status: expedited shadow-only  
Last updated: 2026-05-17

## Purpose

Keep the v2 shadow observer useful without turning Telegram into a raw state-transition
stream.

## Problem

The first observer emitted all non-bootstrap material transitions. That is too broad:

- `noise -> reversal`
- `emerging_move -> noise`
- other downside / de-escalation transitions

are useful as research telemetry, but noisy as operator alerts when there is no open
position attached to the symbol.

## Alert Contract

### Telegram-eligible now

Only **upside discovery** transitions:

- `emerging_move`
- `confirmed_trend`

### Trace-only now

All other states remain fully logged but do **not** notify Telegram:

- `noise`
- `mature_trend`
- `exhaustion`
- `reversal`

Exit-oriented shadow alerts may be reconsidered later only when they are linked to
actual held positions or an explicit exit-observation workflow.

## Runtime Contract

- only the clean release worker may be running for operator observation;
- dev workers must not be left running alongside release workers;
- Telegram alerts must never be the only copy of a v2 decision: the decision trace is
  canonical.

## Acceptance Criteria

1. No Telegram is sent for `noise -> reversal`.
2. No Telegram is sent for transitions back to `noise`.
3. Telegram remains available for `emerging_move` / `confirmed_trend`.
4. All transitions are still present in append-only research telemetry.
5. Unit tests cover alert eligibility.

