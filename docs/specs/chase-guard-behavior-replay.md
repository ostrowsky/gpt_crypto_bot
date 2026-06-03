# Chase Guard Behavior Replay

Date: 2026-06-03
Status: research-only replay variants; no live BUY gate changes

## Problem

The daily blocker reward table flagged `chase_guard` as harmful enough to advance
to behavior replay. The live chase guard protects against late overheated entries,
so it must not be relaxed directly from blocker counts.

## Hypotheses

Replay-only variants:

- `chase_guard_off`: broad control; disables chase guard entirely.
- `chase_guard_rsi_off`: keeps daily-range hard chase guard, disables only the
  RSI-overheat subguard.
- `chase_guard_rsi_82`: keeps daily-range hard chase guard and raises RSI max
  from 76 to 82.

## Guardrails

- Do not change production `monitor._top_gainer_chase_guard_reason`.
- Do not promote a variant only because it increases activity.
- Require full-watchlist replay on the longest feasible windows.
- Promotion requires early-capture improvement without unacceptable PnL,
  trade-precision, false-positive, or late-entry damage.
