# V2 Entry Admission Reward Replay

Status: research-only  
Last updated: 2026-05-18

## Purpose

Validate whether improved admission geometry becomes improved sequential reward in
the offline decision environment.

The previous audit showed that projected v1 structure improves recall-preserving
admission quality versus belief-only thresholds. This package asks the next, stricter
question:

> does the policy actually trade less badly when that admission layer is enforced?

## Compared Policies

| Policy | Description |
|---|---|
| `threshold_policy_base` | current best threshold action policy (`open=0.70`, `sell=0.70`) |
| `belief_admission_policy` | threshold action policy + belief-only admission |
| `belief_plus_projected_v1_admission_policy` | threshold action policy + recall-preserving combined admission |

## Admission Rules

| Rule | Contract |
|---|---|
| belief-only | `P(emerging_move)+P(confirmed_trend) >= 0.70` |
| combined recall-preserving | `P(emerging_move)+P(confirmed_trend) >= 0.50` and projected v1 leader score `>= 3.0` |

## Required Metrics

- trade count;
- total reward;
- named reward components;
- delta versus current threshold base;
- entry-state mix.

## Acceptance Criteria

1. All policies run on the same OOS episodes.
2. Admission affects only whether a new position may open, not hindsight labels.
3. The combined rule is judged by reward and false-buy reduction, not by classifier
   precision alone.
4. If admission improves precision but not reward, state that before adding model
   complexity.

## Next Gate

If recall-preserving admission improves reward:

1. keep it as the next transparent policy baseline;
2. analyze remaining gap to oracle;
3. then decide whether supervised admission is worth building.

## First OOS Result

| Policy | Trades | Total reward | Noise entries | Emerging entries |
|---|---:|---:|---:|---:|
| threshold base | `2213` | `-554.283` | `1529` | `373` |
| belief + projected-v1 admission | `2152` | `-384.481` | `1471` | `371` |

Delta versus current threshold base:

- total reward: `+169.802`;
- false-buy penalty: `-1529.0 -> -1471.0`;
- giveback penalty: `-4526.589 -> -4342.686`;
- emerging entries retained: `371 / 373`.

Interpretation:

- v1-projected admission improves real sequential behavior, not only classifier
  metrics;
- it removes meaningful noise while preserving almost all early entries;
- reward is still negative, so this is a validated next baseline, not a finished
  policy.
