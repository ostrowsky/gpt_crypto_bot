# V2 Canonical Market History

Status: research-only  
Last updated: 2026-05-17

## Purpose

Define the canonical continuous OHLCV history contract that future v2 state models will
consume instead of relying on fragmented legacy row logs as their primary sequence source.

## Problem

The first v2 coverage audit showed that the current local `ml_dataset` is not suitable as
the main history source for early-trend state modeling:

- `15m`: 1,292 observations, 1,292 sequences, 0 transitions;
- `1h`: materially better, but still partial;
- current rows are useful diagnostics, not a canonical continuous market history.

The v2 architecture therefore needs its own explicit market-history abstraction with
continuity guarantees before labels, HMMs, MCMC, or RL are introduced.

## Scope

### In scope

- canonical OHLCV bar type;
- history slice type with source metadata;
- continuity validation by timeframe;
- explicit missing-bar reporting;
- conversion from raw rows into canonical slices;
- unit tests.

### Out of scope

- fetching candles from exchanges;
- persistent storage format selection;
- live websocket ingestion;
- indicator computation;
- model training;
- production imports.

## Canonical Bar Contract

Each bar includes:

- `symbol`
- `timeframe`
- `open_ts_ms`
- `open`
- `high`
- `low`
- `close`
- `volume`

## Canonical Slice Contract

Each slice includes:

- `symbol`
- `timeframe`
- ordered bars;
- `source`
- optional `start_ts_ms` / `end_ts_ms`;
- continuity report:
  - `expected_step_ms`
  - `missing_intervals`
  - `is_contiguous`

## Acceptance Criteria

1. v2 can represent a continuous OHLCV slice independently of legacy datasets.
2. Duplicate / unsorted raw rows are normalized deterministically.
3. Missing intervals are explicit, not silently skipped.
4. Unit tests cover contiguous and gapped slices.
5. No live code imports the v2 module.

## Verification

- `python -m unittest test_v2_history.py`

## Next Gate

After this contract:

1. choose the canonical source/build process for historical slices;
2. populate v2 history for coverage-valid windows;
3. only then build hindsight lifecycle labels.

## Rollback / Safety

- pure research-only types and helpers;
- no I/O side effects;
- removable with no production effect.

