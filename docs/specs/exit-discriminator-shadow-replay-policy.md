# Exit Discriminator Shadow Replay Policy

Date: 2026-05-29
Status: research-only replay variant; no live SELL changes

## Problem

The exit failure discriminator found an out-of-sample concentration of likely wrong exits, but report-level evidence is not enough to change live SELL behavior.

## Hypothesis

A replay-only policy can delay only high-risk exits identified by causal-at-exit features, instead of delaying all WEAK or all trailing exits.

## Replay Variant

Variant: `exit_discriminator_shadow_policy`.

The first implementation uses a conservative hand-translated shadow score from discriminator segments that were visible at exit time:

- early/late capture context;
- early exit timing proxy;
- top-mover context;
- large MFE / giveback context;
- reason buckets such as WEAK or stop/trail;
- trend/strong_trend/impulse_speed modes.

If the score is high enough, replay delays the exit for a small capped window and tightens an ATR/profit-floor trail. The policy is intentionally replay-only.

## Acceptance Gate

Do not promote unless candle replay improves or at least does not worsen:

- total PnL;
- average PnL;
- win-rate;
- exit efficiency;
- giveback;
- late-loss cases.

Failure means the discriminator remains useful for diagnosis, not for live exit control.
