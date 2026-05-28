# V2 Wake-up to V1 Bridge Replay

Last updated: 2026-05-28

## Problem

Naive V2 early admission has high recall but too many false-favorable candidates. The more realistic architecture is V2 as an early wake-up sensor and V1 as the admission/risk controller. This bridge must be replayed before any live behavior change.

## Objective Fit

Improve early capture inside the fixed watchlist universe without replacing V1 gates or expanding the universe.

## Scope

Replay candidate policies where:

- V2 emits first upside for a watchlist symbol/day;
- V1 critic_dataset rows inside a causal time window after that wake-up provide structural confirmation;
- the selected row is evaluated with existing critic labels and corrected top-gainer outcomes.

Profiles:

- `v2_wakeup_v1_structural`: V1 `entry_ok`, `alignment_ok`, or `surge_ok` with sane trend structure.
- `v2_wakeup_v1_momentum`: stricter slope/volume/RSI/EMA conditions.
- `v2_wakeup_v1_observation_window`: any V1 structural candidate in the window.

## Out Of Scope

- No production BUY/SELL changes.
- No portfolio-cap simulation in this pass.
- No automatic threshold changes.

## Primary Metrics

- top precision and recall for corrected `Binance top-N ∩ watchlist` positives;
- ret5 precision / average ret5 from critic labels;
- negative ret5 rate;
- median delay from V2 wake-up to V1 confirmation;
- candidate pressure vs V1 actual.

## Acceptance Criteria

- CLI writes JSON/text report.
- Unit tests cover V2 wake-up + V1 row matching and false-positive accounting.
- Decision is explicit: reject, research-only, or advance to portfolio-aware replay.

## Status

Research-only diagnostic. No trading behavior changes.
