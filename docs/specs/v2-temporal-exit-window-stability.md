# V2 Temporal Exit Window Stability

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test whether the locally robust `mature_decay_late_rise` exit family keeps its
uplift across separate time windows instead of depending on one aggregate OOS
period.

## Candidate

Use the locally strongest transparent profile from the prior robustness grid:

```text
late_mass >= 0.50
and mature_delta_3 <= -0.20
and late_mass_delta_3 >= 0.10
```

## Windowing Protocol

1. Build the same OOS episodes as the existing replay audits.
2. Sort episodes by `local_day`.
3. Split them into four chronological windows with near-equal episode counts.
4. Replay the same control and candidate on every window separately.

## Required Metrics

- date span per window;
- episode count;
- control reward;
- candidate reward;
- reward delta;
- win/loss count across windows;
- aggregate reward delta.

## Acceptance Criteria

1. Candidate must remain positive on the aggregate OOS sample.
2. Candidate should win in a majority of chronological windows.
3. Any losing windows must be reported explicitly; no averaging them away.
4. If uplift is concentrated in one window only, do not advance the family.

## Next Gate

## Backtest Result

Candidate:

```text
late_mass >= 0.50
and mature_delta_3 <= -0.20
and late_mass_delta_3 >= 0.10
```

Aggregate OOS:

- control reward: `-384.480955`
- candidate reward: `-231.993036`
- aggregate delta: `+152.487919`

Chronological windows:

| Window | Date span | Delta vs base |
|---|---|---:|
| `window_1` | `2026-04-30 -> 2026-05-04` | `+54.380468` |
| `window_2` | `2026-05-04 -> 2026-05-09` | `+123.891073` |
| `window_3` | `2026-05-09 -> 2026-05-13` | `+74.502550` |
| `window_4` | `2026-05-13 -> 2026-05-17` | `-100.286171` |

## Interpretation

- The candidate passes the predefined majority gate:
  - `3 / 4` windows positive;
  - aggregate OOS uplift remains positive.
- But the latest chronological window is materially negative.

That means the signal is promising but **not yet regime-agnostic**. The final
window must be explained before any promotion claim.

## Decision

- Keep the candidate as the leading research exit family.
- Do not promote it yet.
- Advance to regime / failure-slice analysis focused on why
  `2026-05-13 -> 2026-05-17` reverses the benefit.

## Next Gate

Build a regime/failure audit that compares winning and losing windows using
available aggregate market-state proxies and per-trade composition. The purpose
is to determine whether the candidate can be conditioned safely, or whether it
should remain research-only.
