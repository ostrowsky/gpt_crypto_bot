# Shadow Suspicious Re-entry Daily Scorecard

Date: 2026-05-31
Status: daily measurement layer for shadow-only alerts

## Problem

Shadow re-entry alerts are useful only if they become measurable. Otherwise they are just another Telegram stream and can create the same false confidence as unvalidated BUY/SELL changes.

## Goal

Produce a daily scorecard for `suspicious_reentry_shadow` events:

- how many shadow re-entry candidates appeared;
- how many have mature forward labels;
- T+2/T+5/T+10 returns after the shadow alert;
- positive/negative pressure;
- whether the policy looks promising enough for a later replay/promotion spec.

## Non-goals

- Do not change production cooldown.
- Do not open positions.
- Do not alter SELL rules.
- Do not auto-promote the policy.

## Acceptance Criteria

1. The report reads only event logs and market data.
2. Missing or immature labels are marked explicitly as pending/partial.
3. The report writes dated and latest JSON/TXT artifacts under `.runtime/reports`.
4. Telegram delivery can be enabled from the daily worker, but the report remains decision-support only.
5. Unit tests cover positive, negative, and pending-label cases without network access.
6. The live path logs one-shot upstream watch decisions (`registered`, `rejected_exit_score`, `rejected_mfe`) for closed positions considered by the shadow re-entry watch. A day with zero `suspicious_reentry_shadow` alerts is not diagnosable unless the upstream watch-decision funnel is available.

## Promotion Gate

A production re-entry policy can be proposed only after the shadow scorecard shows persistent positive forward returns and acceptable downside across multiple market regimes, followed by replay validation.
