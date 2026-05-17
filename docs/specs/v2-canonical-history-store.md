# V2 Canonical History Store

Status: research-only  
Last updated: 2026-05-17

## Purpose

Create the first concrete storage layer for canonical v2 OHLCV history slices so future
backfill, labeling, and state-model work can use repeatable continuous history instead of
fragmented legacy row logs.

## Problem

The v2 coverage audit proved that current legacy datasets are not a safe primary source
for state modeling, especially on `15m`. We already have a pure canonical history
contract, but no concrete place to persist and reload those slices with provenance.

## Scope

### In scope

- a local JSONL-backed canonical history store under an explicit root directory;
- deterministic upsert / dedupe by bar open timestamp;
- source provenance metadata per stored slice;
- loading a validated `HistorySlice`;
- listing stored symbol/timeframe keys;
- unit tests.

### Out of scope

- fetching candles from Binance or any other exchange;
- deciding final long-term storage format;
- live ingestion;
- backfill orchestration;
- model training.

## Storage Contract

Default research root:

```text
files/.runtime/v2_history/
  SYMBOL/
    TIMEFRAME.jsonl
    TIMEFRAME.meta.json
```

Bar rows use the canonical schema from `files/v2/history.py`.

Metadata records:

- `symbol`
- `timeframe`
- `source`
- `rows`
- `start_ts_ms`
- `end_ts_ms`
- `updated_at_utc`

## Acceptance Criteria

1. Store can write and reload a canonical history slice.
2. Duplicate timestamps are deduplicated deterministically.
3. Loaded slices still pass continuity validation.
4. Provenance metadata is written and readable.
5. Unit tests cover roundtrip, dedupe, and key listing.
6. No production code imports the v2 package.

## Verification

- `python -m unittest test_v2_history_store.py`

## Next Gate

After this package:

1. define the backfill/source policy for populating the canonical store;
2. fetch or reconstruct enough contiguous 15m/1h history for training-grade windows;
3. only then proceed to hindsight lifecycle labels.

## Rollback / Safety

- research-only;
- writes only to the explicit v2 history root;
- removable with no effect on current production trading.

