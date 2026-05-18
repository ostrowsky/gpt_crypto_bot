# V2 Market-Environment Switch Replay

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test the full environment-aware policy-selection loop:

1. fixed base policy;
2. fixed temporal candidate;
3. oracle-switched upper bound;
4. causal classifier-switched policy.

## Why Two Switched Policies

`oracle_switched` answers:

> is policy switching worth doing at all if environment inference were perfect?

`causal_prefix_switched` answers:

> can the bot make a useful non-leaking policy choice from observations available
> during the live day?

## Candidate Policies

- `base_sell_0_70`
- `mature_decay_0_20_late_rise_0_10`

## Causal Classifier Baseline

- observation window: first `16` bars of the market day (`4h` on `15m`);
- training: expanding prior days only;
- features:
  - `adx`
  - `projected_leader_score_trend`
  - `daily_range_pct`
  - `projected_forecast_proxy_pct`
  - `price_vs_ema20_pct`
- classifier: nearest centroid over favorable / unfavorable prior-day classes;
- before a valid two-class classifier exists, default to `base`.

The selected policy may affect only decisions after the prefix cutoff.

## Required Metrics

- total reward for all four policies;
- delta versus fixed base;
- delta versus fixed candidate;
- oracle selection count;
- causal selection count;
- day-by-day chosen policy and reward delta.

## Acceptance Criteria

1. `oracle_switched` should beat both fixed policies, otherwise switching itself
   is not worth pursuing.
2. `causal_prefix_switched` must be evaluated without same-day future leakage.
3. A classifier that does not beat both fixed policies is not promotion-ready,
   even if the oracle upper bound is large.

## Next Gate

- If oracle wins but causal loses:
  - improve environment inference, not the policy family.
- If both win:
  - proceed to richer environment belief modeling and robustness checks.
- If oracle fails:
  - stop the switching track and revisit policy families.

## Backtest Result

| Policy | Total reward | Delta vs fixed base | Delta vs fixed candidate |
|---|---:|---:|---:|
| `fixed_base` | `-384.480955` | `0.000000` | `-152.487919` |
| `fixed_candidate` | `-231.993036` | `+152.487919` | `0.000000` |
| `oracle_switched` | `+114.008885` | `+498.489840` | `+346.001921` |
| `causal_prefix_switched` | `-449.023609` | `-64.542654` | `-217.030573` |

## Interpretation

The full contour splits cleanly:

1. **Policy switching is worth doing.**
   - The oracle switched policy beats both fixed policies by a large margin.
2. **The first causal classifier is not good enough.**
   - The nearest-centroid prefix classifier, trained only on prior days and using
     the first `4h` of same-day observations, loses to both fixed policies.

This means the current bottleneck is **environment inference**, not the current
exit-policy family.

The result is exactly what an opponent-modeling analogy would predict:

- perfect knowledge of the opponent/environment helps a lot;
- wrong environment identification makes adaptive behavior worse than a fixed
  strategy.

## Decision

- Keep the environment-aware architecture track.
- Reject the first causal nearest-centroid classifier baseline.
- Do not weaken the switching requirement just to obtain a positive result.
- The next package should improve **belief over environment**, not add another
  policy branch.

## Next Gate

Build richer, probabilistic environment inference:

- rolling intraday observations instead of one hard `4h` prefix snapshot;
- uncertainty-aware belief rather than one hard class;
- explicit evaluation of classifier calibration / lag;
- switched-policy replay that can abstain when environment belief is uncertain.
