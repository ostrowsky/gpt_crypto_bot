# Suspicious Exit Re-entry Replay

Date: 2026-05-29
Status: research-only replay variant; no live entry/exit changes

## Problem

Candle replay rejected multiple attempts to fix exit monetization by delaying exits. The discriminator can identify suspicious exits, but holding the original position still worsened baseline.

## Hypothesis

For suspected wrong exits, a safer action may be to allow a short, confirmation-based re-entry instead of delaying the original SELL.

This preserves hard exits while testing whether the bot can recapture continuation after an early/incorrect exit.

## Replay Variant

Variant: `suspicious_exit_reentry`.

When a closed trade receives a high exit-discriminator risk score, replay opens a short re-entry watch window for that symbol. During that window, cooldown may be bypassed only if a normal candidate appears and passes a stricter top-gainer score floor.

## Guardrails

- Research-only.
- Does not alter original SELL timing.
- Does not create synthetic entries without a normal candidate.
- Does not bypass portfolio limits.
- Requires stricter candidate quality than normal cooldown bypass.

## Acceptance Gate

Promote only if replay improves PnL/average PnL or exit/capture metrics without a material increase in false positives, churn, or drawdown proxy.
