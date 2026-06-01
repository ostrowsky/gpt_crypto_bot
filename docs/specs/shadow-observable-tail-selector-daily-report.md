# Shadow Observable Tail Selector Daily Report

Date: 2026-06-01
Status: reporting/measurement-only; no production SELL changes

## Problem

Observable tail selector replay found a promising research selector, but it must be watched in the daily self-improvement loop before any live SELL behavior changes.

## Goal

Add a compact shadow tail-selector section to the daily learning progress report so the operator can see whether exit monetization research is progressing, stable, or failing.

## Scope

- Run `files/replay_observable_tail_selector.py` from the daily learning report path.
- Summarize the top selector, decision, train/test coverage, test delta, median delta, worse-rate, allowed-rate, and false-positive allowed-rate.
- Render a concise Telegram-friendly line after `shadow re-entry`.
- Add next-action logic:
  - if selector passes shadow gate, keep collecting shadow evidence and do not change production SELL;
  - if no selector passes, continue feature/gate research;
  - if report is missing or errored, mark it as measurement gap.

## Non-goals

- No live SELL changes.
- No Telegram trade signal changes.
- No auto-approval of tail selector.
- No use of hindsight labels as production features.

## Acceptance

- Unit tests cover missing, passed, and failed selector summaries.
- `run_tests.ps1` release suite passes.
- Daily report JSON contains `shadow_tail_selector`.
- Rendered text contains a `shadow tail selector` line.
