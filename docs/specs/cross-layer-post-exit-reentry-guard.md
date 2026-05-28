# Cross-layer Post-exit Re-entry Guard

Status: shipped safety bugfix  
Date: 2026-05-28

## Objective

Prevent the bot from sending rapid contradictory SELL -> BUY signals for the
same symbol across V1 main bot and market agent layers.

## Problem

On 2026-05-28 HBARUSDT produced:

1. V1/main SELL at 16:45 local:
   - reason: weak RSI divergence;
   - PnL: +1.33%;
   - bars held: 2.
2. Market agent BUY at 17:00 local on the same symbol.

This is bad operator behavior even when each local component is internally
consistent. The main bot had just declared momentum weakness, while the agent
did not inherit that post-exit cooldown.

## Scope

- No BUY threshold relaxation.
- No SELL rule change.
- No change to scoring.
- Add a safety gate so the market agent respects recent main-bot exits.

## Requirements

1. The market agent must read recent `bot_events.jsonl` exits.
2. If the main bot exited a symbol recently, the agent must block new entries
   for at least `COOLDOWN_BARS * 15m`.
3. Weak exits must be treated at least as strictly as normal exits.
4. Runtime files remain append-only telemetry; the guard is derived from
   existing event logs.
5. Unit tests cover fresh main exit blocking and expired cooldown pass-through.

## Acceptance Criteria

- A main SELL cannot be followed by an agent BUY on the same symbol one bar
  later.
- The block is logged as `main_exit_cooldown`.
- Tests pass.
