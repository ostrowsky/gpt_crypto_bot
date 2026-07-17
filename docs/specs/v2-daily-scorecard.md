# V2 Daily Scorecard

Last updated: 2026-07-17

## Problem

The V2 Markov / belief / RL research path produces shadow events and offline reports, but the operator cannot quickly see whether V2 is becoming more useful day to day or week to week. The existing V2 daily summary counts states only; it does not connect V2 upside discovery to the bot objective or show trend progress.

## Objective Fit

This is measurement-only tooling for the north star: earlier capture of same-day watchlist top movers with safe exits. It helps decide whether V2 is improving as a radar/policy candidate before any production BUY/SELL promotion.

## Scope

Add a daily V2 scorecard that joins:

- V2 shadow state transitions from `files/v2_shadow_events.jsonl`;
- same-day top-mover outcomes from `top_gainer_critic_*_final.json`;
- daily history from prior saved scorecards / source reports for day-over-day and week-over-week comparisons.

The scorecard must report:

- V2 upside discovery volume;
- confirmation ratio;
- V2 top-mover recall and precision against watchlist top movers;
- V2-to-V1 handoff among top movers, meaning V2 saw upside and V1 actually bought;
- false-favorable pressure, meaning V2 upside symbols not present in same-day top movers;
- day-over-day and 7d-vs-previous-7d deltas.
- explicit primary-denominator metadata plus a separately labeled watchlist-relative diagnostic;
- causal V2-to-V1 handoff, where the first same-day V2 upside event must precede the first V1 BUY;
- event-stream coverage gaps and incomplete calendar windows.

## Out Of Scope

- No BUY / SELL / portfolio behavior changes.
- No new RL policy promotion.
- No Telegram realtime V2 alerts.
- No automatic gate relaxation based on this report.

## Primary Metrics

- `v2_top_recall_pct`: share of same-day watchlist top movers seen by V2 as `emerging_move` or `confirmed_trend`.
- `v2_top_precision_pct`: share of V2 upside symbols that became same-day watchlist top movers.
- `v2_confirmation_ratio`: `confirmed_trend` events divided by `emerging_move + confirmed_trend` events, matching the V2 daily summary.
- `v2_handoff_bought_pct`: share of top movers with V2 upside whose first V1 BUY happened at or after the first same-day V2 upside event.
- `v2_false_favorable_symbols`: V2 upside symbols absent from same-day watchlist top movers.

The primary top-mover denominator remains `exchange Top-N ∩ watchlist`, as defined by the watchlist-filtered denominator spec. `watchlist_universe_top_gainers` is reported only as a diagnostic and must not replace the primary metric.

Rates with a zero denominator are `null` / `n/a`, not zero. Day-over-day means the immediately preceding calendar day. Week-over-week means the exact target-day-minus-6 through target-day window versus the preceding seven calendar days; a window is not compared unless all seven days have complete coverage.

For the V2 transition stream, any start, internal, or end-of-day gap above 90 minutes marks the day `partial`. The threshold was selected from the maximum available joined history: complete-looking days had maximum gaps at or below 52.2 minutes, while the next observed gap was 123.4 minutes.

## Acceptance Criteria

- A CLI can generate JSON and text scorecards for a target day.
- The scorecard marks missing outcome reports as `partial` instead of silently implying zero performance.
- Text output includes day-over-day and week-over-week progress.
- RL worker sends the scorecard text instead of the old count-only V2 summary when daily V2 Telegram is enabled.
- Unit tests cover objective joins and progress deltas.
- Unit tests cover causal ordering, zero denominators, coverage gaps, and fixed calendar windows.

## Risk / Trade-offs

The precision denominator is limited to the day’s top-mover report, not a full tradable-universe outcome table. Therefore false-favorable pressure is a diagnostic pressure metric, not a final PnL claim. This is acceptable because the scorecard is explicitly research-only.

Tiny primary denominators must be flagged in operator text. A `1/1` result is valid arithmetic but is not sufficient promotion evidence.

## Maximum-History Validation — 2026-07-17

The corrected measurement was replayed over all 39 days with both V2 events and final top-gainer outcomes. The 90-minute continuity gate retained 27 complete days and marked 12 partial. Across complete days:

- primary micro recall: 100.0%;
- primary micro precision: 3.16%;
- causal V2-to-V1 handoff: 84.0%;
- watchlist-relative diagnostic recall: 97.14%;
- watchlist-relative diagnostic causal handoff: 54.55%.

The replay supports the measurement correction but does not support promoting V2 into production BUY. Trading gates remain unchanged.

## Verification Gate

- `python -m unittest files.test_v2_daily_scorecard`
- Smoke run on current runtime reports:
  `python files/report_v2_daily_scorecard.py --date 2026-05-26 --json`

## Rollback Switch

Set `V2_SHADOW_DAILY_SUMMARY_TELEGRAM_ENABLED = False` to stop daily V2 Telegram reporting. The report is read-only and can also be removed from `rl_headless_worker.py` without affecting trading behavior.

## Status

Shipped as measurement-only / research-only. No production trading behavior changes.
