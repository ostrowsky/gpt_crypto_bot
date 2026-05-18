# V2 Market-Environment Horizon-Belief Diagnostic

Status: research-only  
Last updated: 2026-05-18

## Purpose

Evaluate whether `1h` / `2h` future-horizon policy-advantage targets are more
usable for environment belief than the rejected whole-day target.

This package is diagnostic only. It does not change live or shadow policy.

## Targets

- `candidate_favorable_1h`
- `candidate_unfavorable_1h`
- `candidate_favorable_2h`
- `candidate_unfavorable_2h`

Ground truth at anchor `t`:

```text
candidate_favorable_H if candidate reward over [t, t + H] > base reward
candidate_unfavorable_H otherwise
```

## Features

At each anchor, use only rows available before or at `t`:

- `adx`
- `projected_leader_score_trend`
- `daily_range_pct`
- `projected_forecast_proxy_pct`
- `price_vs_ema20_pct`
- `noise_share`
- `emerging_share`
- `mature_share`
- `belief_late_mass`

## Diagnostic Protocol

For each horizon:

1. sort samples chronologically;
2. use expanding prior samples only;
3. require both classes in prior history;
4. predict with nearest centroid;
5. report:
   - coverage;
   - accuracy;
   - favorable precision / recall;
   - wrong-confident proxy;
   - baseline majority accuracy.

## Acceptance Criteria

1. Horizon target must have materially better sample count / class balance than
   day-level labels.
2. Expanding causal classifier should beat majority baseline before it is worth
   trying another switched replay.
3. If both horizons fail, improve features / target design before policy
   switching.

## Next Gate

- If `1h` or `2h` diagnostic beats majority baseline:
  - run a horizon-belief switched replay.
- If neither beats baseline:
  - collect better environment observations or change target granularity.

## Audit Result

| Horizon | Samples | Class balance | Accuracy | Majority baseline | Favorable precision | Favorable recall | Wrong-confident share | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `1h` | `179` | `109 / 70` | `0.536723` | `0.608939` | `0.627451` | `0.592593` | `0.129944` | `below_majority` |
| `2h` | `179` | `114 / 65` | `0.531429` | `0.636872` | `0.638095` | `0.603604` | `0.034286` | `below_majority` |

## Interpretation

The horizon targets are better balanced than day-level labels, but the current
feature set plus nearest-centroid causal classifier still fails to beat a simple
majority baseline.

`2h` has fewer wrong-confident errors than `1h`, but both horizons are below the
minimum bar for switched-policy replay.

This means the bottleneck is not only the ground-truth horizon. The current
market observations are still too weak or too poorly transformed for causal
environment inference.

## Decision

- Do not run a horizon-belief switched replay from this classifier.
- Keep `1h` / `2h` horizon targets as better ground truth candidates.
- Next work should improve the observation layer:
  - add feature deltas / slopes;
  - add dispersion / breadth features;
  - test simpler threshold separability before another policy replay.
