# V2 Offline Policy Baselines

Status: research-only  
Last updated: 2026-05-18

## Purpose

Add fixed control policies before any offline RL work so the environment has
interpretable anchors:

1. `always_flat`;
2. `lifecycle_oracle`;
3. `belief_policy_v1`.

Without these anchors, a later RL score would be uninterpretable.

## Policy Definitions

| Policy | Rule |
|---|---|
| `always_flat` | never opens a position |
| `lifecycle_oracle` | opens on hindsight `emerging_move` / `confirmed_trend`, sells on `exhaustion` / `reversal` |
| `belief_policy_v1` | opens on predicted `emerging_move` / `confirmed_trend`, sells on predicted `exhaustion` / `reversal` |

Open positions are force-closed on the terminal frame so every episode has explicit
realized reward.

## Required Report

For every policy:

- episode count;
- action count;
- trade count;
- total reward;
- named reward-component totals.

## Acceptance Criteria

1. All policies run over the same OOS episodes.
2. Episode results are deterministic.
3. `always_flat` remains a valid zero-action baseline.
4. `lifecycle_oracle` is reported as an optimistic upper-bound reference, not a
   deployable policy.
5. `belief_policy_v1` is judged against the oracle gap, not in isolation.

## Next Gate

If the belief policy produces coherent but weaker reward than the oracle:

1. inspect the gap by lifecycle phase;
2. add multi-symbol / portfolio allocation episodes;
3. only then compare learned policy baselines.

## First OOS Result

| Policy | Trades | Total reward | Interpretation |
|---|---:|---:|---|
| `always_flat` | `0` | `0.000` | valid null baseline |
| `lifecycle_oracle` | `461` | `+5867.071` | optimistic upper-bound reference |
| `belief_policy_v1` | `2994` | `-1864.085` | state estimates are not yet safe to map directly into BUY / SELL actions |

The failure mode is explicit rather than mysterious:

- `false_buy_penalty`: `-2058.0`;
- `giveback_penalty`: `-5675.634`;
- `trade_count`: `2994` versus `461` for the oracle.

This means the next research task is **policy-gap analysis / action calibration**,
not immediate offline RL. The belief layer is already informative, but the first
naive action policy converts too much uncertain state mass into trades.
