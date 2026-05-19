# V2 Reward-Weighted Market Selector

Status: research-only  
Last updated: 2026-05-19

## Purpose

The selected-feature market switch passed plain classification gates but failed
reward gates. The failure mode is asymmetric: false `candidate_favorable`
predictions can create very large losses.

This spec adds a reward-weighted selector that treats policy choice as a costed
decision, not as a balanced classification task.

## Hypothesis

A causal selector that estimates expected reward delta and applies downside-aware
admission should outperform the unweighted selected-feature switch.

The decision should be:

```text
choose candidate policy only when expected_reward_delta_after_downside_penalty > threshold
otherwise choose base policy
```

## Candidate Selector Families

Evaluate simple, transparent selectors before any RL:

1. weighted k-nearest neighbors:
   - estimate reward delta from nearest prior samples in selected feature space;
   - optionally penalize negative neighbors by a downside multiplier.
2. confidence-gated centroid:
   - use selected-feature centroid prediction;
   - only accept candidate when confidence exceeds a threshold.
3. hybrid:
   - require both predicted favorable class and positive penalized expected delta.

## Replay Protocol

Use the same causal horizon samples as selected-feature switch replay:

```text
.runtime/reports/v2_market_breadth_observation_store_15m.json
```

For each horizon:

1. sort samples chronologically;
2. use only prior samples as history;
3. compute candidate/base action;
4. sum realized `reward_delta` only when candidate is selected;
5. compare with:
   - fixed base;
   - fixed candidate;
   - unweighted selected-feature switch;
   - oracle switch.

## Acceptance Criteria

The selector is promising only if:

- switched reward vs base is positive;
- switched reward is better than fixed candidate;
- switched reward is better than the unweighted selected-feature switch;
- candidate selection count is not trivially tiny;
- wrong-candidate downside is lower than unweighted switch.

Passing this spec does not authorize live trading. It only authorizes a full
offline environment replay gate.


## Replay Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_reward_weighted_market_selector.py
```

### Best selectors

| Horizon | Best selector | Switched vs base | Fixed candidate vs base | Switched vs candidate | Verdict |
|---|---|---:|---:|---:|---|
| `1h` | `knn_k8_down1.0_thr0.0` | `+821.600433` | `+881.234181` | `-59.633748` | reject |
| `2h` | `knn_k5_down3.0_thr0.0` | `+800.875290` | `+86.582719` | `+714.292571` | pass |

### Key comparison

The previous unweighted selected-feature switch failed:

- `1h`: `+820.605376` vs base, but `-60.628805` vs fixed candidate;
- `2h`: `-27.288883` vs base, `-113.871602` vs fixed candidate.

The reward-weighted selector materially improves the `2h` horizon:

- candidate count: `93 / 175`;
- wrong-candidate loss: `-525.083989` versus unweighted wrong-confident
  candidate loss `-1668.774893`;
- switched delta vs base: `+800.875290`;
- switched delta vs fixed candidate: `+714.292571`.

## Interpretation

This confirms that the selector must optimize reward-weighted action errors, not
plain class accuracy. It also suggests that the market-environment belief is more
useful on a `2h` policy horizon than on a `1h` horizon in this dataset.

The `1h` selector still loses to simply using the candidate policy. That path is
not ready.

## Decision

- `1h`: rejected.
- `2h`: advance to the next research gate.

The next gate is a full offline environment replay using the `2h`
reward-weighted selector as an environment admission/switch layer. No live BUY /
SELL behavior changes are authorized.
