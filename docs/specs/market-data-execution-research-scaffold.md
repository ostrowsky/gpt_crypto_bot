# Market Data And Execution Research Scaffold

Status: research-only  
Last updated: 2026-05-17

## Purpose

Preserve a small, reusable foundation for future experiments around:

- websocket-driven market data;
- short-window impulse detection;
- order-book-derived features;
- simple execution realism and slippage estimation;
- append-only feature/event capture;
- registry-based research strategies.

This package exists so those ideas can be evaluated deliberately later instead of being
recreated ad hoc or accidentally mixed into the production path.

## Included Components

- `files/ws/binance_stream.py`
  - lightweight websocket client for closed klines and partial order-book updates.
- `files/runtime/price_tracker.py`
  - short-window in-memory tick buffer with impulse detection.
- `files/features/microstructure.py`
  - pure order-book feature helpers.
- `files/paper/execution.py`
  - simple L2 book-walk fill simulator for research estimates.
- `files/storage/feature_logger.py`
  - append-only JSONL event logger.
- `files/strategies/registry.py`
  - minimal registry for isolated research strategy experiments.

## Explicit Non-Goals

- No production BUY/SELL decisions.
- No replacement of the current REST/replay path.
- No live order routing.
- No assumption that websocket data is already gap-free, idempotent, or production-hardened.
- No direct promotion without a later architecture review, replay parity work, and resilience tests.

## Why Research-Only

Professional live trading requires more than having websocket code:

- reconnect and gap-fill semantics;
- event ordering and idempotency;
- restart recovery;
- latency and clock-drift handling;
- consistent historical/live feature parity;
- slippage model validation against realized fills.

Those guarantees are not claimed here yet.

## Acceptance Criteria

1. Pure research helpers have focused unit tests.
2. The websocket dispatcher is importable and can route closed-klines / order-book payloads.
3. Nothing in this package is imported by the production trading path by default.
4. The package remains small and composable enough for later replay/live-parity work.

## Verification

- `python -m py_compile` for all scaffold modules.
- Unit tests for:
  - imbalance / microprice / slope helpers;
  - market-buy and market-sell book walking;
  - short-window impulse detection;
  - websocket dispatch routing.

## Promotion Gate

Before any production adoption:

1. add explicit architecture/spec for live ingestion;
2. prove reconnect/gap-fill/idempotency behavior;
3. add replay/live parity tests;
4. compare execution estimates against realized fills;
5. run replay/backtest or shadow evidence showing metric benefit without harming current objectives.
