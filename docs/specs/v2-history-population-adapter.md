# V2 History Population Adapter

Status: research-only  
Last updated: 2026-05-17

## Purpose

Populate the canonical v2 history store with repeatable OHLCV slices and report whether
the first coverage target is actually met.

## Problem

The v2 roadmap now has:

- a canonical bar contract;
- a local canonical store;
- a coverage target:
  - 60 days;
  - `15m` and `1h`;
  - contiguous slices;
  - at least 80% of the active watchlist.

What is still missing is the adapter that turns an external historical-candle source into
stored canonical slices with provenance and coverage diagnostics.

## Scope

### In scope

- reusable asynchronous historical candle client;
- deterministic population manifest;
- write-through into `LocalHistoryStore`;
- aggregate report by symbol/timeframe;
- coverage classification against the current 60-day target;
- dry-run-friendly pure helpers and unit tests.

### Out of scope

- production imports;
- live websocket ingestion;
- feature calculation;
- legacy `ml_dataset` repair;
- HMM / RL work.

## Source Decision

Primary source for this first adapter:

- Binance historical klines REST endpoint.

Why:

- it already matches the current exchange universe;
- fresh replay successfully uses the same source in this environment;
- the v2 store remains independent even if the adapter initially talks to the same
  exchange.

## Population Contract

For each `(symbol, timeframe)` request:

1. fetch closed bars for the requested UTC window;
2. convert to canonical bars;
3. upsert into `LocalHistoryStore`;
4. validate continuity;
5. record manifest/report row:
   - requested window;
   - received rows;
   - stored rows;
   - source;
   - contiguous yes/no;
   - missing interval count.

## First Coverage Target

Copied from `docs/specs/v2-history-coverage-plan.md`:

- `60` calendar days;
- `15m`, `1h`;
- no missing bars inside a valid slice;
- at least `80%` of active watchlist with valid slices.

## Acceptance Criteria

1. Adapter code is under `files/v2/` or a v2-specific one-shot entrypoint.
2. Unit tests cover candle conversion and coverage classification.
3. Manifest/report output is deterministic and machine-readable.
4. Local smoke can populate at least one symbol/timeframe slice.
5. No production code imports the v2 package.
6. Roadmap status updates from "now" only after a real coverage report exists.

## Verification

- `python -m unittest test_v2_history_population.py`
- smoke:
  - `python files/populate_v2_history.py --symbols BTCUSDT --timeframes 15m 1h --days 3 --json`

## Next Gate

- If 60-day population succeeds broadly:
  - proceed to lifecycle labeling spec.
- If source or continuity problems block the target:
  - solve data acquisition first;
  - do **not** start state-model work on partial history.

## Rollback / Safety

- research-only;
- writes only to the v2 history store root;
- generated history can be deleted without touching production bot state.

## First Local Smoke Result

Executed on 2026-05-17:

```text
python files/populate_v2_history.py --symbols BTCUSDT --timeframes 15m 1h --days 3 --json
```

| Slice | Fetched rows | Expected rows | Contiguous | Full window covered |
|---|---:|---:|---:|---:|
| `BTCUSDT 15m` | 288 | 288 | yes | yes |
| `BTCUSDT 1h` | 72 | 72 | yes | yes |

Decision:

- the adapter is valid on a real external source;
- the next gate is no longer code feasibility, but scaling the population run to the full
  watchlist / 60-day target and attaching an aggregate coverage report.

## First Full 60-Day Population Result

Executed on 2026-05-17 against the active 105-symbol watchlist:

| Metric | Value |
|---|---:|
| requests | 210 |
| valid symbols | 95 |
| valid-symbol ratio | 90.48% |
| coverage target passed | yes |
| fetch errors | 0 |
| incomplete symbol/timeframe slices | 20 |

Symbols without full requested coverage:

- `ACAUSDT`
- `BAKEUSDT`
- `EOSUSDT`
- `LRCUSDT`
- `MDTUSDT`
- `MKRUSDT`
- `OXTUSDT`
- `RNDRUSDT`
- `SNTUSDT`
- `TRUUSDT`

Decision:

- the 80% minimum data gate is passed;
- v2 can proceed to lifecycle-label design on the `95` valid symbols;
- the 10 invalid symbols must remain excluded from first state-model experiments unless a
  later symbol-resolution / alternate-source step recovers them.
