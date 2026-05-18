# V2 Temporal Exit Robustness

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test whether the first winning temporal exit profile is a real signal or a
single lucky threshold combination.

The prior package found one OOS winner:

```text
late_mass >= 0.50
and mature_delta_3 <= -0.15
and late_mass_delta_3 >= 0.10
```

That result is not enough for promotion without a local sensitivity audit.

## Hypothesis

If the underlying signal is real, nearby threshold combinations should preserve
positive uplift versus the same control profile.

## First Robustness Grid

Keep `late_mass >= 0.50` fixed and vary the two newly discovered temporal
conditions:

| Parameter | Values |
|---|---|
| `mature_delta_3 <=` | `-0.10`, `-0.15`, `-0.20` |
| `late_mass_delta_3 >=` | `0.05`, `0.10`, `0.15` |

This yields a `3 x 3` grid around the winning center.

## Required Backtest Metrics

- total reward;
- reward delta versus `base_sell_0_70`;
- win count across the local grid;
- center-cell reward;
- best-cell reward;
- worst-cell reward.

## Acceptance Criteria

1. Every grid cell must run on the same OOS replay sample.
2. The original center must remain positive versus control.
3. A majority of the local grid should remain positive before we call the signal
   locally robust.
4. If uplift exists only in one isolated cell, treat it as threshold overfit.

## Next Gate

## Backtest Result

The local `3 x 3` OOS grid was fully positive versus the same control profile:

- positive cells: `9 / 9`
- positive share: `1.000000`
- control reward: `-384.480955`
- center reward delta: `+122.901657`
- best reward delta: `+152.487919`
- worst reward delta: `+91.861713`

| Mature decay threshold | Late-rise threshold | Reward delta vs base |
|---:|---:|---:|
| `-0.10` | `0.05` | `+92.469696` |
| `-0.10` | `0.10` | `+92.469696` |
| `-0.10` | `0.15` | `+91.861713` |
| `-0.15` | `0.05` | `+122.901657` |
| `-0.15` | `0.10` | `+122.901657` |
| `-0.15` | `0.15` | `+122.692174` |
| `-0.20` | `0.05` | `+152.487919` |
| `-0.20` | `0.10` | `+152.487919` |
| `-0.20` | `0.15` | `+152.487919` |

## Interpretation

The signal is locally robust:

- the original center remains positive;
- every nearby cell is positive;
- the late-rise threshold is nearly flat inside the tested range;
- the effect strengthens as the mature-belief decay requirement gets stricter.

This does **not** yet prove promotion readiness. It proves only that the prior
winner was not an isolated one-cell accident.

## Decision

- Keep `mature_decay_late_rise` as the leading research exit family.
- Treat mature-belief decay as the dominant candidate feature.
- Advance to window / regime stability testing before any stronger promotion.

## Next Gate

Run the same control-versus-candidate comparison across time segments and, where
available, market-regime slices. Promotion is blocked until the uplift is shown
to survive beyond the aggregate OOS average.
