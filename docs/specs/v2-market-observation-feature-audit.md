# V2 Market Observation Feature Audit

Status: research-only  
Last updated: 2026-05-19

## Purpose

Test whether richer market-observation features make `1h` / `2h`
future-horizon policy advantage more predictable.

The prior horizon-belief diagnostic showed that better targets alone are not
enough: current prefix means plus nearest-centroid inference failed to beat
majority baseline.

## Added Observation Families

At each anchor, use only data available up to that anchor.

### Prefix means

Keep prior features:

- ADX
- projected leader score
- daily range
- forecast proxy
- price-vs-EMA20
- state shares
- belief late mass

### Recent-window means

Compute the same features over the most recent `8` bars only.

### Deltas

Compute:

```text
recent_mean - prior_prefix_mean
```

for each feature, when enough prior bars exist.

### Cross-sectional breadth / dispersion proxies

Using latest available row per symbol at the anchor:

- share with `price_vs_ema20_pct > 0`
- share with `rsi >= 50`
- share with `projected_forecast_proxy_pct > 1`
- mean / standard deviation of forecast proxy
- mean / standard deviation of leader score

## Diagnostic Protocol

For each horizon:

1. build samples chronologically;
2. train only on prior samples;
3. compare:
   - nearest centroid;
   - simple diagonal Gaussian Naive Bayes;
4. report majority baseline, accuracy, favorable precision/recall, and
   wrong-confident share.

## Acceptance Criteria

Do not run switched-policy replay unless at least one causal diagnostic beats
majority baseline by `3pp` on either `1h` or `2h`.

## Next Gate

- If richer observations beat baseline:
  - replay horizon-belief switching.
- If they still fail:
  - add genuinely new market data sources / labels rather than more classifiers.

## Audit Result

| Horizon | Model | Accuracy | Majority baseline | Favorable precision | Favorable recall | Wrong-confident share | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `1h` | nearest centroid | `0.610169` | `0.608939` | `0.729412` | `0.574074` | `0.016949` | `near_majority` |
| `1h` | Gaussian NB | `0.570621` | `0.608939` | `0.677778` | `0.564815` | `0.406780` | `below_majority` |
| `2h` | nearest centroid | `0.582857` | `0.636872` | `0.682692` | `0.639640` | `0.011429` | `below_majority` |
| `2h` | Gaussian NB | `0.571429` | `0.636872` | `0.663636` | `0.657658` | `0.405714` | `below_majority` |

## Interpretation

Richer observations help, but not enough.

Compared with the previous horizon diagnostic, `1h` nearest-centroid accuracy
improved from `0.536723` to `0.610169`, and wrong-confident share dropped from
`0.129944` to `0.016949`.

However, the result only matches the majority baseline; it does not beat it by
the required `3pp`. `2h` remains below majority. Gaussian NB is not usable here
because it produces too many wrong-confident calls.

## Decision

- Do not run switched-policy replay from this observation set.
- Keep richer prefix/recent/delta/breadth features as useful diagnostics.
- The next improvement must add stronger market observations or change label
  granularity, not merely swap in another simple classifier.
