# Telegram Menu Control-Plane Resilience

Date: 2026-06-13
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

## Guardrails

- No trading logic changes.
- No BUY/SELL gate changes.
- Do not log raw chat ids.

## Acceptance Criteria

- A burst of repeated `📋 Открыть меню` messages triggers at most one concurrent
  menu response per chat.
- `telegram_delivery` events exist for control/menu sends: attempt/ok/failed.
- Text detection works for `📋 Открыть меню`, `/menu`-like text and hide-menu
  emoji text.
