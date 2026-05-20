# Telegram Positions Freshness

Status: shipped operational fix  
Last updated: 2026-05-20

## Problem

The Telegram `positions` button could send a stale unified portfolio snapshot on
the first press. The handler read `_cached_positions_text()` first and only then
started `_refresh_position_cache_async()` in the background.

That made closed agent positions appear as still open even though
`positions.json` and `agent_positions.json` were already updated.

## Fix

For the `positions` callback, rebuild the unified portfolio text before sending
the Telegram message. The cache remains fallback-only if rebuilding fails.

## Guardrails

- No trading logic changes.
- No BUY/SELL changes.
- No portfolio ranking formula changes.
- Only Telegram/operator display freshness changes.

## Acceptance Criteria

- First `positions` button press uses current `positions.json` and
  `agent_positions.json`.
- Stale cache is not used unless fresh rebuilding fails.
- Background startup cache refresh still works.
