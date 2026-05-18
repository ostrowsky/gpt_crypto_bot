# V2 Temporal Exit Failure-Slice Audit

Status: research-only  
Last updated: 2026-05-18

## Purpose

Explain why the leading temporal exit candidate wins three chronological OOS
windows but loses the latest one.

This package is diagnostic. It must not invent a new rule until the failure mode
is visible in data.

## Candidate Under Review

```text
late_mass >= 0.50
and mature_delta_3 <= -0.20
and late_mass_delta_3 >= 0.10
```

## Questions

1. Does the losing window have a different lifecycle-state mix?
2. Does the candidate lose because it changes entries, exits, hold time, or reward
   composition?
3. Which market-state proxies differ most between winning and losing windows?
4. Is there enough evidence to justify a later conditional policy, or is the
   evidence still inconclusive?

## Required Diagnostics

For each chronological window:

- true-state row mix;
- projected structural feature means;
- belief feature means;
- control reward components;
- candidate reward components;
- reward-component deltas;
- trade summaries for control and candidate.

Then compare:

- average winning-window profile;
- losing-window profile;
- ranked feature deltas.

## Guardrails

- This audit may explain a failure mode, not promote a new rule.
- A visible difference is not automatically a causal conditioning feature.
- Any later conditional rule must be tested in a separate replay package.

## Next Gate

## Backtest Result

The losing window is materially different from the average winning window.

### Winning-window average vs losing latest window

| Signal | Winning avg | Losing window | Delta |
|---|---:|---:|---:|
| `noise` share | `0.701496` | `0.815457` | `+0.113961` |
| `mature_trend` share | `0.085289` | `0.047014` | `-0.038275` |
| `exhaustion` share | `0.021858` | `0.014675` | `-0.007183` |
| `rsi` | `50.940924` | `46.538370` | `-4.402554` |
| `adx` | `24.868415` | `28.549651` | `+3.681236` |
| `daily_range_pct` | `3.121401` | `2.234432` | `-0.886969` |
| `price_vs_ema20_pct` | `0.078627` | `-0.187338` | `-0.265965` |

### Reward decomposition

In the winning windows the candidate improves the average delta mainly through:

- `giveback_penalty`: `+71.961469`
- `false_buy_penalty`: `+34.333333`
- `realized_pnl_reward`: `+16.049873`

In the losing window:

- `false_buy_penalty` still improves: `+49.000000`
- but `giveback_penalty` no longer helps: `-6.503492`
- and `realized_pnl_reward` becomes strongly worse: `-82.209928`

## Interpretation

The candidate appears useful when the market is actually producing enough mature
trend structure to monetize. The losing latest window is more noise-dominant,
contains fewer mature/exhaustion rows, has lower RSI, narrower daily range, and
already sits below EMA20 on average.

That is consistent with a **weak-structure / noise-dominant failure slice**, not
with a generic late-exit problem.

## Decision

- The failure mode is coherent enough to justify a separate conditional-policy
  experiment.
- Do not change the current candidate yet.
- The next package should test whether a simple regime proxy can disable or
  soften the temporal exit family in weak-structure conditions.

## Next Gate

Specify conditional hypotheses from this failure slice and replay them
separately. Any conditioning rule must beat both:

1. the original `base_sell_0_70` control;
2. the unconditional temporal candidate.
