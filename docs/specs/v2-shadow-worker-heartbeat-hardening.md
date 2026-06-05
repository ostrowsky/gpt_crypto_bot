# V2 Shadow Worker Heartbeat Hardening

Status: runtime-safety fix  
Date: 2026-06-05

## Problem

`v2_shadow_worker.py` updates `files/.runtime/v2_shadow_status.json` only after a full watchlist/timeframe scan completes. A long scan, slow Binance request, or stuck symbol can make the V2 shadow observer look stale even while the process is alive and writing traces.

This weakens operator trust and makes it hard to distinguish a real dead V2 observer from a long in-progress cycle.

## Goal

Make V2 shadow status observable during a scan.

## Requirements

- Write status at cycle start.
- Refresh heartbeat during the scan.
- Include in-progress counters: scanned, emitted, stale, errors.
- Include current symbol/timeframe while scanning.
- Bound each symbol/timeframe scan with a configurable timeout.
- Do not change V2 signal logic, Telegram eligibility, BUY/SELL behavior, or datasets.

## Guardrails

- Research/shadow-only runtime fix.
- No production BUY/SELL changes.
- No new Telegram spam.
- Release gate must pass before restart.
