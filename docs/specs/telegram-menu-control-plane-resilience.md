# Telegram Menu Control-Plane Resilience

Date: 2026-06-13
Updated: 2026-08-25
Status: implementation

## Problem

Repeated `📋 Открыть меню` presses can create concurrent menu send attempts while
Telegram is slow. The bot receives updates, but several simultaneous
`sendMessage` calls time out and the operator sees no response.

## Goal

Make the menu/control plane observable and resilient:

- recognize menu text robustly by emoji/semantic text, not only exact string;
- coalesce duplicate menu sends per chat while one menu response is already in
  progress;
- use a longer timeout for menu/control replies than for ultra-fast callbacks;
- audit control delivery attempts/results in `bot_events.jsonl`.
- keep Telegram update polling and menu handling responsive while monitoring
  persists large evidence datasets.
- restore persisted position metadata with at most one sequential scan of each
  historical evidence file during cold start, independent of position count.

## Guardrails

- No trading logic changes.
- No BUY/SELL gate changes.
- Do not log raw chat ids.
- Evidence writes remain ordered and complete; moving them off the event loop
  must not silently drop records.
- The change must not alter signal selection, entry, exit, scoring, or portfolio
  policy.

## Acceptance Criteria

- A burst of repeated `📋 Открыть меню` messages triggers at most one concurrent
  menu response per chat.
- `telegram_delivery` events exist for control/menu sends: attempt/ok/failed.
- Text detection works for `📋 Открыть меню`, `/menu`-like text and hide-menu
  emoji text.
- A blocking evidence write runs outside the asyncio event-loop thread: a
  concurrent short timer/menu coroutine must make progress before that write
  completes.
- Heavy critic/ML evidence mutation uses a dedicated serialized executor so it
  cannot starve Telegram polling or race concurrent JSONL rewrites.
- Restoring multiple open positions performs no per-position full-file scan:
  critic records are resolved in one batch and ranker shadow events are tailed
  once, while missing evidence remains missing rather than being invented.

## Maximum-load verification

This is an operational scheduling change, not a trading-policy hypothesis, so
a market replay is not an applicable promotion gate. Verification uses the
maximum currently available live evidence files (critic and ML JSONL) and a
synthetic blocking-write regression test. Cold-start verification uses the
maximum current positions and evidence files and asserts one scan per evidence
source rather than one scan per position. Trading-policy backtests remain
unchanged because selection, entry, exit, scoring, and portfolio policy are
unchanged.

## Canary

After restart, keep the existing production policy and run an operator canary:
observe consecutive Telegram `getUpdates` polling while the monitor persists
critic/ML evidence, then send `/menu` or press `📋 Открыть меню`. The canary
passes only when the menu response is delivered without a monitoring-sized
polling gap and no `409`/`401` Telegram error appears.

Result 2026-08-25: PASS. One new Python instance restored 10 positions, started
polling, delivered the menu with `sendMessage 200`, and continued consecutive
`getUpdates 200` calls without `401` or `409` errors.

## Rollback

Rollback is a code revert followed by a single-instance bot restart. Trigger it
if evidence writes are lost/reordered, JSONL lock errors increase, the bot
cannot shut down cleanly, or Telegram polling still stalls under evidence load.
No BUY/SELL configuration rollback is needed because this change does not alter
trading policy.
