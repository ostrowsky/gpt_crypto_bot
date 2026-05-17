# P0 Measurement Hardening

Status: implemented measurement-only  
Last updated: 2026-05-17

## Purpose

Close the current self-improvement loop gap by making long-window reporting robust and by
adding two daily diagnostic views that explain:

- where final watchlist winners are lost before BUY;
- how much trend value is lost after BUY through weak exits.

## Scope

1. Harden `bot-progress-report`:
   - tolerate missing / `None` metric fields;
   - deduplicate duplicate/manual artifacts by target day;
   - label each requested day as `complete`, `partial`, `zero_activity`, or `missing`.
2. Add `files/report_trend_start_funnel.py`:
   - for final watchlist top movers, show first structural scout, first wake-up scout,
     first block, first BUY, and final outcome.
3. Add `files/report_exit_quality.py`:
   - aggregate signal-quality final reports into a focused exit lifecycle summary.

## Non-Goals

- No BUY/SELL behavior change.
- No automatic promotion of hypotheses.
- No use of a diagnostic metric as production evidence by itself.

## Acceptance Criteria

1. Long-window progress reports complete even when some days have no lead-time metric.
2. Duplicated reports for the same target day do not double count rolling analytics.
3. Missing and low-information days are visible rather than silently mixed into valid windows.
4. Winner funnel rows expose the first loss point without manual log archaeology.
5. Exit audit exposes giveback and exit-efficiency trends across the available window.
