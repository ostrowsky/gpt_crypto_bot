# V2 Market-Environment Edge Target Audit

Status: research-only  
Last updated: 2026-05-19

## Purpose

Test whether the environment target should distinguish real policy edge from
near-zero noise.

The prior binary target:

```text
candidate_favorable if reward_delta > 0
candidate_unfavorable otherwise
```

may be too noisy because tiny positive/negative deltas are not actionable for
policy switching.

## Hypothesis

If we introduce a no-edge zone:

```text
candidate_edge if reward_delta >= threshold
base_edge      if reward_delta <= -threshold
no_edge        otherwise
```

then causal prediction over actionable samples should improve enough to justify
a future abstaining policy selector.

## Thresholds

Evaluate:

- `0.5`
- `1.0`
- `2.0`
- `5.0`
- `10.0`

for both `1h` and `2h` horizon samples.

## Diagnostic Protocol

Use the already replay-generated market-observation samples:

```text
.runtime/reports/v2_market_observation_features_15m.json
```

For each threshold and horizon:

1. sort samples chronologically;
2. train only on prior actionable samples;
3. classify only actionable samples;
4. report:
   - actionable coverage;
   - class balance;
   - accuracy;
   - majority baseline;
   - candidate-edge precision / recall;
   - verdict.

## Acceptance Criteria

This audit does not authorize policy switching. It only selects whether an
edge-aware target is promising.

A threshold is promising only if:

- actionable sample count remains meaningful;
- causal accuracy beats majority by at least `3pp`;
- candidate-edge precision is not worse than the previous binary diagnostic.

## Next Gate

- If promising:
  - run an abstaining switched replay where `no_edge` keeps the safer current
    control policy.
- If not:
  - prioritize stronger external market observations.


## Audit Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_market_environment_edge_targets.py --json
```

Source sample file:

```text
.runtime/reports/v2_market_observation_features_15m.json
```

### Summary

The no-edge target did **not** solve market-environment separability.

| Horizon | Best threshold by accuracy edge | Accuracy | Majority baseline | Actionable share | Verdict |
|---|---:|---:|---:|---:|---|
| `1h` | `0.5` | `0.583333` | `0.723881` | `0.748603` | reject |
| `2h` | `0.5` | `0.603896` | `0.683544` | `0.882682` | reject |

Higher thresholds raise candidate-edge precision, but mostly because they remove
hard samples and make the actionable set more imbalanced. They do not create a
causal classifier that beats the majority baseline.

Selected detail:

| Horizon | Threshold | Actionable | Candidate precision | Candidate recall | Accuracy vs majority |
|---|---:|---:|---:|---:|---:|
| `1h` | `0.5` | `134 / 179` | `0.773333` | `0.604167` | `-0.140548` |
| `1h` | `5.0` | `54 / 179` | `0.968750` | `0.673913` | `-0.178062` |
| `2h` | `0.5` | `158 / 179` | `0.724490` | `0.676190` | `-0.079648` |
| `2h` | `2.0` | `132 / 179` | `0.777778` | `0.736842` | `-0.083685` |

## Interpretation

The target was not the primary bottleneck. Removing small reward deltas creates a
cleaner semantic label, but the current observation layer still cannot identify
which policy family will win next.

This matters because the previous oracle switch showed large theoretical value,
but both causal switchers and belief switchers lost. The failure is now clearly
an inference problem, not proof that adaptive policy switching has no value.

## Decision

Rejected for promotion.

Do **not** run an abstaining switched replay from this classifier. The gate stays
closed until the market observation layer beats majority by the required margin
on causal horizon targets.

## Next Step

Prioritize a stronger market-breadth observation store instead of more simple
label reshaping. Candidate features:

- percentage of tracked symbols above EMA20 / EMA50;
- percentage with positive intraday return;
- return dispersion and top-minus-median breadth;
- volume expansion breadth;
- new intraday high / low breadth;
- BTC / ETH trend and risk proxy context.
