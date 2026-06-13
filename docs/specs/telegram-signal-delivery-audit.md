# Telegram Signal Delivery Audit

Date: 2026-06-13
Status: implementation

## Problem

The bot can create an entry/exit event in `bot_events.jsonl`, but the operator
may not see the Telegram message. Before this spec there is no structured event
that links a trading signal to Telegram delivery attempts/results. This makes
questions like "why was there no Telegram signal for AXL?" expensive and
ambiguous.

## Goal

Add structured, low-noise Telegram delivery audit events for signal broadcasts.
The audit must answer:

- was a signal message rendered;
- which symbol/timeframe/kind it referred to;
- which chat id was targeted, without exposing the chat id in clear text;
- whether Telegram send succeeded, failed, or was skipped;
- whether a raw/PTB fallback was used.

## Scope

- Main bot broadcast path used by `monitoring_loop` BUY/SELL/WATCH messages.
- Market agent direct Telegram send path.
- No trading logic changes.
- No message content body is stored, only a short preview and extracted metadata.

## Event Schema

`bot_events.jsonl` / `agent_events.jsonl` records:

```json
{
  "event": "telegram_delivery",
  "source": "bot|market_agent",
  "delivery_stage": "attempt|ok|failed|skipped",
  "delivery_path": "broadcast|raw|ptb_fallback|agent",
  "message_kind": "buy|sell|watch|v2_shadow|portfolio|other",
  "sym": "AXLUSDT",
  "tf": "15m",
  "chat_id_hash": "...",
  "error_class": "TimeoutError",
  "text_preview": "..."
}
```

## Acceptance Criteria

- Every main-bot broadcast message logs an `attempt` and then `ok` or `failed`
  per target chat id.
- Empty chat-id set logs `skipped`.
- Market agent sends are audited similarly.
- Chat ids are hashed, never logged raw.
- Unit tests cover signal metadata extraction and hashed chat ids.
