# Main top-gainer intraday feature parity

Status: bugfix  
Owner: Codex  
Last updated: 2026-05-28 Europe/Budapest

## Problem

Main bot blocked `BATUSDT` on 2026-05-28 with:

- `top-gainer score 29.32 < 34.00`;
- logged `today_change_pct=0.0`;
- logged `forecast_return_pct=0.0`.

The agent entered BAT because its feature path saw positive local-day movement
and a forecast proxy. Investigation showed two feature-parity gaps in the main
path:

1. `_intraday_change_pct_from_data()` expected a dict-like object with `.get()`,
   but live monitor passes a numpy structured candle array. The function catches
   the exception and returns `0.0`, removing the intraday component from
   `_top_gainer_live_score()`.
2. `strategy.CoinReport` does not expose `today_change_pct` or
   `forecast_return_pct`, so candidate logs store zeros even when the candle
   structure implies a valid top-mover move.

## Objective

Make main top-gainer scoring use the same causal local-day movement semantics as
agent scoring and make diagnostics show the non-zero values used by scoring.

## Scope

- Fix main intraday change helper to support numpy structured candle arrays and
  dict-like candle containers.
- Use Europe/Budapest local-day open, matching agent semantics.
- Add runtime-safe forecast proxy helper for main candidate logging/scoring.
- Keep production thresholds unchanged.
- Do not relax `TOP_GAINER_SCORE_GATE_MIN_SCORE`.

## Non-goals

- No BUY/SELL threshold changes.
- No V2 admission change.
- No retroactive production decision.

## Acceptance

- Unit tests prove structured-array intraday change no longer returns zero.
- Unit tests prove BAT-like score increases when local-day movement is present.
- `py_compile` passes for touched files.
- Change is committed and pushed to `main`.

## Risk

This is decision-path code: restoring the intraday component can increase the
number of candidates passing `top_gainer_score_gate`. Because the threshold is
unchanged and the component was already intended by the formula, this is treated
as a bugfix, but it should be monitored in the next daily learning report for
false-positive pressure.
