# V2 Market-Environment Target Design

Status: research-only  
Last updated: 2026-05-18

## Purpose

Define what counts as "ground truth" for the market-environment belief model.

The bot does not need a descriptive label such as bullish / bearish. It needs a
policy-relevant target:

> which policy family receives higher objective-aligned reward from this point
> forward?

## Target Families

### 1. `day_policy_advantage`

Current coarse teacher:

- favorable if candidate reward beats base over the whole day;
- unfavorable otherwise.

Use:

- broad diagnostic baseline only.

Weakness:

- assumes one environment state for the entire day;
- hides intraday transitions.

### 2. `future_horizon_policy_advantage`

Preferred next teacher:

- at decision time `t`, compare candidate vs base reward over a future horizon;
- tested horizons:
  - `1h`
  - `2h`
  - `rest_of_day`

Use:

- main target for belief diagnostics and later classifier learning.

### 3. `structural_future_state`

Explanatory teacher:

- future occupancy / rates of:
  - noise;
  - emerging;
  - confirmed;
  - mature;
  - exhaustion;
  - reversal.

Use:

- explain why one policy is favorable;
- do not substitute directly for reward-based policy truth.

## Research Questions

1. How often does the day-level target disagree with future-horizon targets?
2. Which horizon is stable enough to train against?
3. Does the current day-level target likely explain part of the belief-v1 failure?

## Acceptance Criteria

1. Compare all horizons on the same OOS episodes.
2. Report class balance and disagreement with day-level labels.
3. Report structural future-state summaries for each horizon label.
4. Do not change the belief model until the target choice is clearer.

## Next Gate

- If day-level labels disagree materially with future-horizon labels:
  - stop using day-level favorability as the main belief target.
- If one horizon is substantially cleaner:
  - use it for the next environment-belief diagnostic / model.

## Audit Result

The current day-level target disagrees materially with future-horizon policy
advantage:

| Horizon | Samples | Favorable | Unfavorable | Disagreement with day label |
|---|---:|---:|---:|---:|
| `1h` | `179` | `109` | `70` | `0.357542` |
| `2h` | `179` | `114` | `65` | `0.363128` |
| `rest_of_day` | `179` | `55` | `124` | `0.469274` |

## Interpretation

The day-level policy-favorability label is too coarse for intraday environment
belief. It hides state transitions inside the day and gives the belief model a
noisy target.

The short future-horizon targets are also much better balanced:

- `1h`: `109 / 70`
- `2h`: `114 / 65`

versus the earlier day-level set:

- day-level: `14 / 4`

This likely explains a meaningful part of the first belief-v1 failure.

## Decision

- Stop treating `day_policy_advantage` as the main target for environment belief.
- Use it only as a coarse reporting baseline.
- Promote `future_horizon_policy_advantage` as the next training / diagnostic
  target family.
- Start with `1h` and `2h` horizons; compare both in the next belief diagnostic.
