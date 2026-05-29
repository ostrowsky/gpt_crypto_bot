# Protected trailing exit candle-path replay

Status: research-only implementation  
Owner: Codex  
Date: 2026-05-29 Europe/Budapest

## Objective

Add a replay-only strategy variant that tests a causal version of protected
trailing exits on candle paths.

Unlike the opportunity audit, this variant must not use future favorable move.
It can only react to information available at each replay bar.

## Policy under test

When a position receives a weak/EMA/trail-like exit reason:

1. If the trade already had enough MFE and is not deeply negative, do not close
   immediately.
2. Activate `protected_exit_active`.
3. Tighten trailing stop with a small ATR multiplier and optional profit floor.
4. Close later on the protected trailing stop or after a bounded confirmation
   window if weakness persists.

## Non-goals

- Do not change live SELL logic.
- Do not promote the policy from one replay window.
- Do not protect catastrophic loss exits indefinitely.

## Acceptance

- Add `protected_trailing_exit` as a replay variant only.
- Keep entry/portfolio behavior equivalent to `score_replace`.
- Unit tests must verify that the first eligible weak exit is held and that the
  protected state later exits via the tightened stop.

## Promotion rule

Production SELL adoption requires multi-window replay showing improvement in:

- total PnL after fees;
- exit efficiency;
- giveback;
- top-mover monetization;
- no unacceptable false-hold damage.

