# V2 Belief Action Calibration

Status: research-only  
Last updated: 2026-05-18

## Purpose

Measure whether belief-state confidence thresholds can convert useful lifecycle
beliefs into less destructive actions before introducing learned policies.

The first naive action mapping failed OOS:

- oracle reward: `+5867.071`;
- naive belief-policy reward: `-1864.085`;
- belief-policy trades: `2994` versus `461` oracle trades.

The failure is therefore in the action layer, not only in the state layer.

## Grid

| Parameter | Values |
|---|---|
| open threshold on `P(emerging_move) + P(confirmed_trend)` | `0.30`, `0.40`, `0.50`, `0.60`, `0.70` |
| sell threshold on `P(exhaustion) + P(reversal)` | `0.30`, `0.40`, `0.50`, `0.60`, `0.70` |

## Required Report

For each variant:

- trade count;
- total reward;
- named reward components;
- delta versus naive belief policy;
- delta versus oracle.

## Acceptance Criteria

1. All variants run on the same OOS episodes as the baseline report.
2. Calibration never uses hindsight labels for actions.
3. Best variant is chosen by reward, then sanity-checked against trade explosion and
   reward decomposition.
4. If all thresholded variants remain worse than zero, say so before attempting RL.

## Next Gate

If a calibrated threshold policy meaningfully improves over naive belief mapping:

1. inspect where the remaining oracle gap comes from;
2. add multi-symbol / portfolio episodes;
3. only then consider contextual or learned policies.

## First OOS Result

The best tested variant was:

- `open_threshold=0.70`;
- `sell_threshold=0.70`.

It improved the naive belief policy, but did not make it good enough:

| Policy | Trades | Total reward |
|---|---:|---:|
| `belief_policy_v1` | `2994` | `-1864.085` |
| best threshold policy | `2213` | `-554.283` |
| `lifecycle_oracle` | `461` | `+5867.071` |

The best threshold policy reduced:

- `false_buy_penalty`: `-2058.0 -> -1529.0`;
- `giveback_penalty`: `-5675.634 -> -4526.589`.

Conclusion:

- thresholding is directionally useful;
- thresholding alone is insufficient;
- the next required step is policy-gap audit by phase / loss source before RL.
