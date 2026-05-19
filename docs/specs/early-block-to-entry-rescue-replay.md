# Early Block To Entry Rescue Replay

Status: diagnostic-only  
Last updated: 2026-05-19

## Purpose

Turn the failure-casebook finding into one bounded replay question:

```text
When a final top mover was repeatedly blocked early, how much opportunity was
lost between the first block and the eventual entry or day close?
```

This is not a production rule. It is a proxy replay over existing critic reports
to decide whether a deeper candle-level replay is worth writing.

## Hypothesis

A narrow rescue path may improve early top-mover capture if it only considers
symbols that already produced repeated early structural blocks, instead of
relaxing BUY gates globally.

## Inputs

- `.runtime/reports/top_gainer_critic_*_final.json`

## Candidate Families

Evaluate simple report-level variants:

- allowed blocker families;
- maximum first-block hour;
- minimum blocked count;
- minimum positive opportunity from first block.

## Metrics

For each variant report:

- rescued missed winners;
- rescued late-bought winners;
- proxy opportunity gain;
- cases with non-positive first-block opportunity;
- top concrete examples.

## Guardrails

This report uses final top-gainer critic files, so it cannot measure false
positives outside the final top list. Therefore it cannot authorize production.

A variant only advances if:

- it captures meaningful missed/late-entry opportunity;
- it keeps non-positive opportunity cases low within the top list;
- it defines a concrete candle-level replay gate before any live behavior.

## First Run Result

Latest proxy audit over existing final top-gainer critic reports:

- source rows: `645` unique day/symbol top-gainer rows;
- best admissible variant:
  - reason set: `agent_plus_score`;
  - blocker reasons: `agent_mode_disabled`, `agent_leader_filter`, `top_gainer_score_gate`;
  - max first-block hour: `12`;
  - min blocked count: `5`;
  - min opportunity from first block: `1.0%`;
- selected cases: `43`;
- rescued missed winners: `43`;
- rescued late-bought winners: `0`;
- non-positive opportunity cases: `0`;
- proxy opportunity gain: `340.041%`;
- average proxy gain per rescued case: `7.907930%`.

Top examples include:

- `DOGSUSDT` on `2026-05-05`: `73.514%` opportunity from first block;
- `DOGSUSDT` on `2026-05-07`: `40.886%`;
- `TONUSDT` on `2026-05-05`: `28.109%`;
- `STRKUSDT` on `2026-05-08`: `19.620%`;
- `NEARUSDT` on `2026-05-06`: `16.654%`.

## Decision

Advance to candle-level replay.

Important limitation: this proxy audit only looks at final top-gainer critic
rows, so it cannot measure how many non-winners would have been incorrectly
rescued. The next gate must replay causal candle/event data and explicitly report
false-positive impact.

Required next replay gate:

```text
Improve capture_ratio_at_entry and net PnL without increasing false_positive_buys
by more than 10%.
```
