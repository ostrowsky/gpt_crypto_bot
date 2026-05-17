# V2 Shadow Signal Observer

Status: expedited shadow-only  
Last updated: 2026-05-17

## Purpose

Provide observable v2-style signals immediately, before the learned state model exists,
so the operator can inspect tomorrow whether lifecycle-oriented signals are useful.

## Important Boundary

This is **not** the final v2 model:

- not HMM;
- not Bayesian filtering;
- not RL;
- not eligible to trade.

It is a provisional deterministic observer that emits lifecycle-state estimates and
shadow action recommendations using live data, while the real v2 learning path continues.

## Why This Exists

The owner wants to observe v2 signal quality tomorrow rather than wait until the full
labeling / modeling stack is complete.

The correct bridge is:

- expose the new architecture's *shape* now;
- keep behavior shadow-only;
- log enough evidence to compare tomorrow;
- avoid relabeling old BUY modes as "v2".

## Scope

### In scope

- provisional symbol-state estimation:
  - `noise`
  - `emerging_move`
  - `confirmed_trend`
  - `mature_trend`
  - `exhaustion`
  - `reversal`
- provisional shadow actions:
  - `watch`
  - `elevate_priority`
  - `buy_candidate`
  - `hold`
  - `tighten_exit`
  - `sell_candidate`
- append-only shadow event log;
- background worker with heartbeat/status;
- Telegram shadow alerts for material transitions only;
- inclusion in the unified `restart_full_stack.bat`.

### Out of scope

- live orders;
- replacing current BUY/SELL logic;
- claiming learned-policy status;
- HMM / RL training.

## Provisional Estimation Rule

The first observer may use only live-available features from the last closed bar:

- slope;
- ADX;
- RSI;
- volume multiple;
- daily range;
- MACD histogram;
- relation to EMA20.

The rules are explicit and intentionally temporary. They must later be replaced or
benchmarked against the learned state model.

## Event Contract

Each emitted event records:

- symbol / timeframe / bar timestamp;
- previous and current lifecycle state;
- shadow action;
- confidence proxy;
- live feature snapshot;
- reason string;
- whether the event is a bootstrap observation;
- source=`v2_shadow_observer`.

Stale closed bars are skipped rather than converted into current-state decisions.

## Operator Contract

Tomorrow the operator should be able to inspect:

- which symbols first entered `emerging_move`;
- which were promoted to `confirmed_trend`;
- where `exhaustion` / `sell_candidate` appeared;
- how those transitions aligned with later market outcomes.

## Runtime Contract

Because this is the first v2 background worker, it must satisfy
`docs/specs/v2-unified-runtime-integration.md`:

- own launcher;
- own status checker;
- included in `restart_full_stack.bat`;
- clean-release compatible.

## Acceptance Criteria

1. Worker runs without affecting production orders.
2. Worker logs shadow events and status heartbeat.
3. Telegram sends only on material state/action transitions.
4. The unified BAT starts and validates the worker.
5. There is a one-shot report for recent shadow signals.
6. Tests cover state estimation, action mapping, and event dedupe.

## Verification

- unit tests;
- one-shot worker run;
- unified restart helper smoke;
- recent-shadow report generation.

## Rollback / Safety

- stop the worker;
- remove it from the BAT;
- no production positions are touched.
