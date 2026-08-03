# Shadow Suspicious Re-entry Daily Scorecard

Date: 2026-08-03
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
7. The daily scorecard reports upstream decision counts, registration rate, registered examples, and same-day alert/registration pressure. The latter is diagnostic pressure, not cohort conversion, because a watch may cross midnight.
8. A day with neither alerts nor upstream watch decisions is `partial`, not `complete`.
9. Unit tests patch runtime telemetry writers so synthetic symbols cannot leak into `bot_events.jsonl`.
10. A registered watch whose configured window has not expired at report time is pending and makes the scorecard `partial`.
11. Runtime metrics include only symbols in the runtime watchlist. Raw and excluded non-watchlist counts remain visible under data quality so historical test contamination is not silently erased.
12. Every `registered` upstream watch receives a counterfactual T+2/T+5/T+10 label even when final confirmation never emits an alert. The actionable reference is the next candle open after registration; returns and ten-bar excursion are measured from that price with no look-ahead.
13. The report keeps alert and registered-watch cohorts separate. A zero-alert day can therefore reject or advance final-confirmation research instead of waiting indefinitely for alerts that the current gate never creates.

## Promotion Gate

A production re-entry policy can be proposed only after the shadow scorecard shows persistent positive forward returns and acceptable downside across multiple market regimes, followed by replay validation.

## Maximum-History Validation — 2026-07-17

The complete available event log contained 1,382 raw upstream watch decisions but zero final `suspicious_reentry_shadow` alerts. For 2026-07-16 specifically, the raw funnel contained 41 decisions: 38 rejected by exit score, one rejected by MFE, and two registered. One registered row was synthetic `AAAUSDT` telemetry leaked by a unit test. After watchlist validation, the operational funnel is 40 decisions with one real registration. Test isolation and explicit non-watchlist filtering are now required by this specification. The real LDO 1h watch was registered at 17:04 local and its eight-bar window expired at 01:04, seven minutes after the original scorecard was generated, so that original report should have been `partial`.

Because the live funnel has produced no forward-label candidates, the evidence does not support a production re-entry change or a threshold relaxation. The scorecard correction is measurement-only; cooldown and entry behavior remain unchanged.

## Accumulation Repair — 2026-08-03

Continued collection produced registered watches but still no final alerts, so
the old alert-only outcome layer remained structurally unable to accumulate
labels. The registered-watch counterfactual cohort is now the upstream evidence
source. Promotion still requires at least ten mature registered examples,
positive average T+5 above 0.25%, positive rate of at least 55%, and a separate
maximum-history replay of the final confirmation. These measurement changes do
not enable production re-entry.

The maximum-history threshold replay must label every upstream decision at the
next candle open, subtract the configured simulated round-trip fee, split the
cohort chronologically 70/30, select thresholds on the first segment only, and
judge them on the held-out segment. A relaxed threshold is not eligible even
for shadow recalibration unless held-out mature count is at least ten, average
net T+5 improves by at least 0.10 percentage point, positive-after-cost rate is
not lower, and median ten-bar drawdown is not worse by more than 0.25 point.

The 2026-08-03 maximum-history replay labeled 2,167 of 2,175 valid upstream
decisions. Current thresholds (`exit_score >= 0.68`, `MFE >= 1.0%`) had train
net T+5 `-0.1900%` (`n=123`) and holdout `+1.9405%` (`n=37`). The train-selected
relaxation (`exit_score >= 0`, `MFE >= 1.5%`) improved train only to `-0.1511%`
but fell to `+0.8843%` on holdout, an out-of-sample deterioration of
`-1.0562pp`. The promotion gate failed; keep thresholds and production re-entry
unchanged.
