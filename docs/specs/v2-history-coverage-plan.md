# V2 History Coverage Plan

Status: research-only planning  
Last updated: 2026-05-17

## Purpose

Define what "enough continuous history" means for the first v2 state-model experiments
and describe the next data-population work for the canonical history store.

## Problem

The current legacy sequence data is not sufficient for learning hidden lifecycle states:

- 15m legacy coverage currently has no usable contiguous transitions;
- the new canonical store exists, but is still empty until explicitly populated;
- without a coverage target, it is easy to start HMM / RL work on whatever data happens
  to be available rather than on a defensible dataset.

## Initial Coverage Target

For the first serious state-reconstruction experiments:

| Dimension | Minimum target |
|---|---:|
| Watchlist symbols | full active watchlist snapshot |
| Timeframes | `15m`, `1h` |
| Continuous history per symbol/timeframe | 60 calendar days |
| Max missing bars inside a training slice | 0 |
| Required valid symbols before first aggregate experiment | 80% of active watchlist |
| Holdout policy | later spec, but no training on the final evaluation window |

Why 60 days:

- enough to include multiple local market regimes and repeated same-day movers;
- still small enough for first deterministic validation and data QA;
- not claimed to be sufficient for production promotion.

## Population Strategy Candidates

### Candidate A — exchange historical candles into canonical store

Pros:

- cleanest sequence source;
- direct continuity guarantees;
- independent from fragmented legacy logs.

Risks:

- exchange/API availability;
- rate limits;
- need for reproducible fetch manifests.

### Candidate B — reconstruct from legacy rows

Pros:

- immediately local;
- preserves existing row lineage.

Risks:

- already proven inadequate for 15m continuity;
- not suitable as the primary path.

Decision:

- use Candidate A as the primary path;
- use legacy rows only for diagnostics / joins, not as the canonical history source.

## Next Implementation Package

Create a separate feature package for:

1. historical candle fetch adapter;
2. deterministic manifest of requested slices;
3. write-through into `LocalHistoryStore`;
4. per-symbol continuity report;
5. retry-safe resumability;
6. no production import.

## Acceptance Criteria For The Later Population Package

1. Every stored slice names its source and request window.
2. Missing bars are explicit.
3. Re-running the same fetch is idempotent.
4. Aggregate report states how many symbols met the 60-day target.
5. No state-model work starts until a coverage report is attached.

## Rollback / Safety

- planning document only;
- no production behavior change.

