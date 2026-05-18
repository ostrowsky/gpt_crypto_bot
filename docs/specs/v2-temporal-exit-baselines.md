# V2 Temporal Exit Baselines

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test whether short-window **trajectories** are more useful than one-bar static
thresholds for exit timing.

The previous exhaustion-aware manual rules all failed OOS despite reducing
giveback, which means snapshot features alone are not yet enough for a better
sell rule.

## Hypotheses

| Profile | Hypothesis |
|---|---|
| `base_sell_0_70` | current control profile |
| `late_mass_acceleration` | rising late-state mass is more informative than static late mass |
| `mature_decay_late_rise` | exhaustion is actionable when mature belief decays while late belief rises |
| `rsi_ema_decay` | momentum/extension decay over several bars is more useful than one-bar weakness |
| `consensus_temporal` | require agreement across belief rise and structural decay |

## Temporal Features

All temporal deltas use the current row minus the row `3` bars earlier for the
same symbol:

- `late_mass_delta_3`
- `mature_delta_3`
- `rsi_delta_3`
- `price_vs_ema20_delta_3`

Rows without a valid 3-bar predecessor must not trigger a temporal sell rule.

## Rule Definitions

| Profile | Sell rule |
|---|---|
| `base_sell_0_70` | `late_mass >= 0.70` |
| `late_mass_acceleration` | `late_mass >= 0.55 and late_mass_delta_3 >= 0.15` |
| `mature_decay_late_rise` | `late_mass >= 0.50 and mature_delta_3 <= -0.15 and late_mass_delta_3 >= 0.10` |
| `rsi_ema_decay` | `late_mass >= 0.45 and rsi_delta_3 <= -4 and price_vs_ema20_delta_3 <= -0.40` |
| `consensus_temporal` | `late_mass >= 0.50 and late_mass_delta_3 >= 0.10 and rsi_delta_3 <= -3 and price_vs_ema20_delta_3 <= -0.25` |

## Required Backtest Metrics

- total reward;
- trade count;
- named reward components;
- exit-state mix;
- average realized PnL;
- average giveback;
- average bars held;
- delta versus `base_sell_0_70`.

## Acceptance Criteria

1. All profiles reuse the same fixed admission layer and same OOS replay sample.
2. Every hypothesis must be replayed, not just the best-looking one.
3. A temporal profile may advance only if total reward improves versus base.
4. Lower giveback without higher total reward is still insufficient.

## Next Gate

## Backtest Result

All five profiles were replayed on the same OOS sample:

- episodes: `1710`
- bars/actions: `163305`
- fixed admission layer:
  - `belief open_mass >= 0.70`
  - `projected admission_mass >= 0.50`
  - `projected leader_score_trend >= 3.0`

| Profile | Total reward | Delta vs base | Avg giveback / trade | Verdict |
|---|---:|---:|---:|---|
| `base_sell_0_70` | `-384.480955` | `0.000000` | `-2.026452` | control |
| `late_mass_acceleration` | `-609.802326` | `-225.321371` | `-1.917141` | reject |
| `mature_decay_late_rise` | `-261.579298` | `+122.901657` | `-2.096515` | **advance** |
| `rsi_ema_decay` | `-542.792264` | `-158.311309` | `-1.914179` | reject |
| `consensus_temporal` | `-552.732209` | `-168.251254` | `-1.927646` | reject |

## Interpretation

The winning profile is **not** an earlier-exit shortcut. It actually has slightly
worse average giveback than the control, but wins on total reward because it:

- reduces noisy re-entries (`1471 -> 1332` noise entries);
- trades less often (`2152 -> 1982`);
- keeps useful positions longer (`44.23 -> 51.22` average bars held);
- improves average reward per trade (`-0.178898 -> -0.131377`).

The specific transferable lesson is therefore narrower:

> a useful exit cue is not “late belief is high”, but “mature belief is decaying
> while late belief is rising”.

## Decision

- Promote `mature_decay_late_rise` as the next **research control candidate**.
- Reject `late_mass_acceleration`, `rsi_ema_decay`, and `consensus_temporal`.
- Do not generalize this result into “temporal features solved exits”; it is one
  validated hypothesis that still needs robustness testing.

## Next Gate

Before any stronger claim:

1. run sensitivity tests around the winning thresholds;
2. test stability across windows / market regimes;
3. compare it against a learned supervised exit score, not just more manual rules.
