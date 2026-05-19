# Failure Casebook

Status: diagnostic-only  
Last updated: 2026-05-19

## Purpose

Stop low-yield model iteration and return to concrete failure analysis.

The casebook answers:

```text
Which specific historical cases cost the bot the most, and what single targeted
hypothesis should be replayed next?
```

It is intentionally compact. It should produce inspectable examples, not another
large ML dataset.

## Inputs

Use existing reports only:

- `.runtime/reports/signal_quality_*_final.json`;
- `.runtime/reports/top_gainer_critic_*_final.json`;
- `.runtime/reports/watchlist_top_gainer_goal_*.json` when available.

## Output

A small report with:

- worst exit / MFE giveback cases;
- late-entry top mover cases;
- missed / blocked top-winner cases;
- false-positive buys;
- repeated blocker patterns;
- a short hypothesis shortlist.

## Rules

- No new model training.
- No Telegram / live behavior changes.
- No production BUY or SELL changes.
- A hypothesis is only useful if a positive result would imply one concrete
  replayable behavior change.

## Acceptance Criteria

The package passes if it produces:

- a ranked case list with source files and dates;
- enough fields to inspect each case without re-reading all raw reports;
- no more than three proposed next hypotheses;
- an explicit replay gate for each hypothesis.


## First Run Result

Latest run over all available final reports loaded:

- `13` signal-quality final reports;
- `44` top-gainer critic final reports.

The compact casebook identified three concrete failure tracks:

1. **Missed / blocked winners dominate opportunity cost**.
   - Worst examples include `DOGSUSDT` on `2026-05-05` with `73.514%`
     opportunity from first block, and `DOGSUSDT` on `2026-05-07` with
     `40.886%`.
   - Dominant blocker families include `agent_mode_disabled`,
     `agent_leader_filter`, and `top_gainer_score_gate`.
2. **Late entries into already-failed moves are visible and replayable**.
   - Examples include `TIAUSDT`, `DOTUSDT`, `AEVOUSDT`, `AAVEUSDT`, and
     `APEUSDT` with `capture_ratio_at_entry = 0.0` and negative post-entry
     opportunity.
3. **Exit/MFE pain exists, but should be targeted by case slice, not broad ML**.
   - Worst examples include `WIFUSDT` early exit with `27.27%` MFE and
     `19.09%` giveback, and `UMAUSDT` with `14.15%` MFE and `8.68%` giveback.

## Decision

Stop generic exit-model iteration for now.

The next productive replay package should be one of:

1. `early-block-to-entry rescue` for repeated early blocks before late entries;
2. `MFE-protection case replay` for high-MFE giveback slices;
3. `false-positive entry slice veto` for the worst losing no-trend modes.

The recommended next package is **early-block-to-entry rescue**, because it
attacks the largest visible opportunity-cost cases and leads directly to a
specific replayable behavior change.
