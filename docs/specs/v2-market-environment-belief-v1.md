# V2 Market-Environment Belief V1

Status: research-only  
Last updated: 2026-05-18

## Purpose

Replace one hard environment guess with a rolling belief over whether the current
day is favorable or unfavorable for the leading temporal exit policy.

The prior contour showed:

- switching is valuable with perfect environment knowledge;
- the first one-shot causal classifier is harmful.

The next question is whether the bot can improve by **accumulating evidence over
time and abstaining while uncertain**.

## Belief State

First research belief vocabulary:

- `candidate_favorable`
- `candidate_unfavorable`

This is intentionally narrower than the final taxonomy. It is a first policy-
favorability belief, not yet a full latent market ontology.

## Observation Schedule

- update after every rolling `4h` market prefix;
- first update after `16` bars;
- then every `8` bars;
- train centroids only on prior completed days;
- use only rows available up to the current update time.

## Belief Update

1. compute nearest-centroid distances to prior favorable / unfavorable classes;
2. convert inverse distances into observation probabilities;
3. blend with prior belief:

```text
posterior = 0.70 * prior + 0.30 * observation
```

4. policy selection:
   - choose `candidate` only if `P(favorable) >= 0.65`;
   - choose `base` only if `P(unfavorable) >= 0.65`;
   - otherwise abstain and keep current policy.

## Replay Policies

- `fixed_base`
- `fixed_candidate`
- `oracle_switched`
- `causal_prefix_switched`
- `belief_switched`

## Required Metrics

- total reward;
- delta vs fixed policies;
- number of belief updates;
- policy switches per day;
- day-level chosen policy sequence;
- confidence / abstention counts.

## Acceptance Criteria

1. No same-day future leakage.
2. The belief path must be inspectable per day.
3. `belief_switched` must be compared against both fixed policies and the prior
   hard-prefix switch.
4. If belief switching still loses, improve environment observations / labels
   before adding more policies.

## Next Gate

- If belief switching wins:
  - expand environment taxonomy and robustness checks.
- If belief switching still loses:
  - audit whether the bottleneck is feature quality, target quality, or too few
    completed unfavorable days.

## Backtest Result

| Policy | Total reward | Delta vs fixed base | Delta vs fixed candidate |
|---|---:|---:|---:|
| `fixed_base` | `-384.480955` | `0.000000` | `-152.487919` |
| `fixed_candidate` | `-231.993036` | `+152.487919` | `0.000000` |
| `oracle_switched` | `+114.008885` | `+498.489840` | `+346.001921` |
| `causal_prefix_switched` | `-449.023609` | `-64.542654` | `-217.030573` |
| `belief_switched` | `-460.660379` | `-76.179424` | `-228.667343` |

## Interpretation

The first rolling belief model does **not** improve the contour.

Observed failure pattern:

- it spends many updates in `abstain`;
- confidence often arrives late;
- prior inertia keeps earlier choices alive after evidence weakens;
- the historical class base is very small and imbalanced.

So the architecture remains valid, but `belief_v1` is rejected as an operational
selector.

## Decision

- Keep market-environment belief as the target architecture.
- Reject this first inverse-distance + fixed-blend implementation.
- The next package should be diagnostic:
  - measure belief lag;
  - measure wrong-confidence episodes;
  - separate feature quality from label quality and class-imbalance effects.
