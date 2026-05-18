# V2 Market-Environment Separability Audit

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test whether current market-level observations can distinguish days where the
leading temporal exit policy is favorable from days where it is unfavorable.

## Weak Teacher Label

- `candidate_favorable` if day-level reward delta vs control is positive;
- `candidate_unfavorable` otherwise.

This is not a production label. It is a diagnostic proxy for policy-favorability.

## Metrics

- favorable / unfavorable day counts;
- feature means by label;
- ranked mean differences;
- day-level reward deltas;
- verdict: `separable_candidate`, `weak_signal`, or `inconclusive`.

## Guardrails

- This audit may justify a classifier baseline, not a live switch.
- Low sample count must be stated explicitly.
- Any future policy switch must beat both fixed policies in a separate replay.

## Next Gate

- If separability is visible, build a transparent classifier baseline and replay a switched policy.
- If not, collect richer market features before attempting policy selection.

## Backtest Result

Day-level weak teacher distribution:

- `candidate_favorable`: `14` days
- `candidate_unfavorable`: `4` days

Verdict:

- `separable_candidate`

Top observed differences:

| Feature | Favorable mean | Unfavorable mean |
|---|---:|---:|
| `adx` | `25.498566` | `26.721527` |
| `projected_leader_score_trend` | `6.454344` | `7.661399` |
| `daily_range_pct` | `2.794286` | `3.265766` |
| `rsi` | `49.947797` | `49.506028` |
| `projected_forecast_proxy_pct` | `1.145449` | `1.365957` |

## Interpretation

The environment appears separable enough to justify a first classifier baseline,
but not in the simplistic form "strong market = favorable policy".

Some unfavorable days have *more* visible trend structure than favorable days.
That means the classifier target must remain **policy-favorability**, not a human
label such as bullish / bearish or strong / weak.

The sample is still small (`4` unfavorable days), so the result is directional,
not promotion-grade.

## Decision

- Advance to a transparent market-environment classifier baseline.
- Train / define it against policy-favorability, not descriptive market labels.
- The next replay must compare:
  1. fixed base control;
  2. fixed temporal candidate;
  3. classifier-switched policy.
