# V2 Selected-Feature Market Switch Replay

Status: research-only  
Last updated: 2026-05-19

## Purpose

Validate whether the selected market-environment features from the v1
market-structure audit translate into policy-switching reward, not only
classification accuracy.

## Hypothesis

If the selected causal features really identify when the candidate temporal exit
policy is better than the base policy, then a chronological switched policy should
improve horizon reward versus fixed base and fixed candidate baselines.

## Selected Feature Sets

From `docs/specs/v2-v1-market-structure-feature-audit.md`:

- `1h`:
  - `market_ret4_positive_share`;
  - `prefix_projected_leader_score_trend`.
- `2h`:
  - `market_btc_ret4_pct`;
  - `market_volume_gt_mean20_share`.

## Replay Protocol

Use samples from:

```text
.runtime/reports/v2_market_breadth_observation_store_15m.json
```

For each horizon:

1. sort samples chronologically;
2. train nearest-centroid classifier only on prior samples;
3. at each anchor, choose:
   - candidate policy if predicted `candidate_favorable`;
   - base policy otherwise;
4. compute reward delta versus fixed base from `reward_delta`;
5. compare against:
   - fixed base;
   - fixed candidate;
   - oracle switch.

## Acceptance Criteria

This is still research-only. A selected-feature switch is promising if:

- switched reward delta vs fixed base is positive;
- switched reward is better than fixed candidate;
- accuracy remains above majority by at least `3pp`;
- wrong-confident share is documented.

Even if passed, the next gate is a full offline environment replay, not live
production usage.


## Replay Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_selected_feature_market_switch_replay.py --json
```

| Horizon | Accuracy edge | Fixed candidate vs base | Switched vs base | Switched vs candidate | Verdict |
|---|---:|---:|---:|---:|---|
| `1h` | `+0.040779` | `+881.234181` | `+820.605376` | `-60.628805` | reject |
| `2h` | `+0.043128` | `+86.582719` | `-27.288883` | `-113.871602` | reject |

Oracle opportunity remains large:

- `1h` oracle delta vs base: `+986.150368`;
- `2h` oracle delta vs base: `+1905.486827`.

But the selected-feature switch loses to the simpler fixed candidate on `1h` and
loses even to fixed base on `2h`.

## Interpretation

Classification accuracy was not enough. The errors are asymmetric: a small number
of wrong candidate-favorable predictions can carry very large negative reward.
The `2h` run is the clearest warning:

- classifier accuracy edge: `+4.31pp`;
- wrong-confident share: `0.20`;
- wrong-confident candidate loss: `-1668.774893`.

So the next model must optimize reward-weighted mistakes, not plain accuracy.

## Decision

Rejected for switched-policy promotion.

The selected v1/market features are useful observations, but the policy selector
needs reward-aware calibration / asymmetric loss before another switched replay.

## Next Gate

Build a reward-weighted market-environment selector that penalizes false
candidate-favorable predictions according to their realized downside. Evaluate it
against:

- fixed base;
- fixed candidate;
- oracle switch;
- selected-feature unweighted switch.
