# V2 Exhaustion-Aware Exit Baselines

Status: research-only  
Last updated: 2026-05-18

## Purpose

Turn the exhaustion-discrimination findings into explicit, replay-tested exit
hypotheses.

Every hypothesis in this package must be evaluated on the same OOS offline replay
before it may influence the next roadmap step.

## Evidence Used

The prior discrimination audit found interpretable mature-vs-exhaustion separation:

- lower RSI in exhaustion;
- higher late-state belief mass;
- lower mature-state belief;
- higher exhaustion belief;
- price much closer to EMA20.

## Hypotheses

| Profile | Hypothesis |
|---|---|
| `base_sell_0_70` | current control profile |
| `late_mass_rsi_weak` | late belief is actionable only when momentum has weakened |
| `late_mass_ema_loss` | late belief is actionable only when price has lost EMA20 extension |
| `exhaustion_belief_combo` | explicit exhaustion belief plus loss of mature belief is stronger than raw late mass |
| `consensus_exhaustion` | require agreement across late belief, RSI weakening, and EMA loss |

## Rule Definitions

| Profile | Sell rule |
|---|---|
| `base_sell_0_70` | `late_mass >= 0.70` |
| `late_mass_rsi_weak` | `late_mass >= 0.55 and rsi <= 56` |
| `late_mass_ema_loss` | `late_mass >= 0.55 and price_vs_ema20_pct <= 0.35` |
| `exhaustion_belief_combo` | `P(exhaustion) >= 0.35 and P(mature) <= 0.25` |
| `consensus_exhaustion` | `late_mass >= 0.50 and rsi <= 56 and price_vs_ema20_pct <= 0.35` |

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

1. All hypotheses use the same fixed admission layer and same OOS episodes.
2. Every hypothesis appears in the replay report, whether it wins or loses.
3. A profile may advance only if it improves total reward versus base.
4. Lower giveback without higher total reward is not enough.

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
| `base_sell_0_70` | `-384.480955` | `0.000000` | `-2.026452` | control wins |
| `late_mass_rsi_weak` | `-587.578684` | `-203.097729` | `-1.931656` | reject |
| `late_mass_ema_loss` | `-562.876290` | `-178.395335` | `-1.919357` | reject |
| `exhaustion_belief_combo` | `-531.279968` | `-146.799013` | `-1.934487` | reject |
| `consensus_exhaustion` | `-656.398278` | `-271.917323` | `-1.908420` | reject |

## Interpretation

Every new rule reduced average giveback per trade, but every new rule also made
total reward worse than the control profile.

That means the exhaustion-discrimination signal is **not yet directly actionable
as a hand-written sell rule**. The current evidence does not justify another
manual threshold search over the same static features.

The most likely failure mode is that the rules are firing earlier in a way that:

- trims some giveback;
- but also cuts trend-hold reward and realized PnL;
- while reopening the policy more often into additional noisy entries.

## Decision

- Keep `base_sell_0_70` as the current transparent exit control.
- Reject all four new hand-written exit rules.
- Do **not** promote any exhaustion-aware manual rule into the next baseline.

## Next Gate

The next exit package should move from one-bar threshold rules to richer temporal
exit modeling, for example:

- short-window belief trajectory features;
- slope/decay features over late-state mass;
- supervised exit targets tied to future MFE retention / giveback;
- replay comparison against the same `base_sell_0_70` control.

Any new hypothesis must again be evaluated on the same OOS replay before it is
allowed to influence roadmap direction.
