# Telegram Startup Bootstrap Resilience

Date: 2026-06-13
Status: implementation

## Problem

The Telegram control process can exit during startup when Telegram API bootstrap
calls (`deleteWebhook`, `getMe`, polling bootstrap) hit transient
`TimedOut`/`ConnectTimeout`. This leaves trading workers alive but the operator
cannot use menu commands.

## Goal

Make the Telegram bot process survive transient API/network slowness during
startup.

## Changes

- Increase HTTPX connect/read/write timeouts for Telegram control requests.
- Use `bootstrap_retries=-1` in `run_polling` so startup bootstrap retries
  indefinitely instead of exiting on one timeout.

## Guardrails

- No trading logic changes.
- No strategy/gate changes.
- Keep polling mode unchanged.
