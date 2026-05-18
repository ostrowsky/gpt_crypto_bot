# V2 Exhaustion Discrimination Audit

Status: research-only  
Last updated: 2026-05-18

## Purpose

Measure whether the current feature set can distinguish a still-productive mature
trend from the start of genuine exhaustion before attempting a smarter exit policy.

The exit-baseline audit showed that simply selling earlier is harmful:

- lower giveback;
- but worse realized PnL and worse total reward.

Therefore the next question is:

> can we tell apart `mature_trend` and `exhaustion` better than by lowering a sell
> threshold?

## Compared Classes

- negative class: `mature_trend`;
- positive class: `exhaustion`.

## Candidate Feature Families

### Belief features

- `P(mature_trend)`;
- `P(exhaustion)`;
- `P(reversal)`;
- late-state mass;
- belief entropy.

### Projected structural features

- projected forecast proxy;
- projected v1 leader score;
- slope;
- ADX;
- RSI;
- `vol_x`;
- daily range;
- price versus EMA20.

### Lifecycle-shape features from canonical bars

- giveback from day peak;
- close-to-peak gap;
- recent return;
- rolling peak persistence.

## Required Report

- rows by class;
- per-feature class means;
- standardized mean difference;
- top-ranked separating features by absolute effect size.

## Acceptance Criteria

1. Uses only information available at the current frame.
2. Uses the same OOS v2 slice as prior policy audits.
3. Does not fit a classifier yet.
4. Names whether there is enough visible separation to justify a first rule baseline
   or whether richer modeling is required.

## Next Gate

- If a few interpretable features separate the classes materially:
  - build an exhaustion-aware transparent exit baseline.
- If separation is weak:
  - add richer temporal features before policy work.

## First OOS Result

The current dense feature set already shows interpretable separation:

| Feature | Mature mean | Exhaustion mean | Abs effect |
|---|---:|---:|---:|
| `rsi` | `62.890` | `52.490` | `1.094` |
| `belief_late_mass` | `0.243` | `0.584` | `0.991` |
| `belief_mature` | `0.464` | `0.172` | `0.801` |
| `belief_exhaustion` | `0.182` | `0.402` | `0.728` |
| `price_vs_ema20_pct` | `1.264` | `0.138` | `0.625` |

Verdict: `interpretable_rule_candidate`.

The next package should test a rule-based exhaustion discriminator rather than
immediately jumping to a learned exit model.
