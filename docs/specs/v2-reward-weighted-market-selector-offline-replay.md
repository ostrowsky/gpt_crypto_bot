# V2 Reward-Weighted Market Selector Offline Replay

Status: research-only  
Last updated: 2026-05-19

## Purpose

Move the `2h` reward-weighted market selector from horizon-sample reward replay
into the full offline decision environment.

The previous audit proved that the selector can improve horizon reward. This spec
checks whether it still helps when used as a causal switch between the base exit
policy and the temporal candidate exit policy inside full symbol-day rollouts.

## Selector Under Test

Use the best `2h` selector from `v2-reward-weighted-market-selector.md`:

```text
features: market_btc_ret4_pct, market_volume_gt_mean20_share
model: reward-weighted kNN
k: 5
downside_multiplier: 3.0
edge_threshold: 0.0
```

## Replay Protocol

1. Build normal v2 offline symbol-day episodes.
2. Build causal market selector choices from the `2h` market-breadth samples.
3. For each live-like decision frame:
   - before the first available market selector anchor for the day: use base;
   - after an anchor: use the latest selector choice for that day;
   - if selector chooses candidate: use temporal candidate policy;
   - otherwise use base policy.
4. Compare against:
   - fixed base;
   - fixed temporal candidate;
   - selected market switch.

## Acceptance Criteria

The selected switch advances only if full offline reward beats both fixed base
and fixed candidate, and does not create a large trade-count explosion.

This remains research-only and does not authorize live BUY / SELL changes.
Telegram output, if enabled later, must be shadow-only and anti-spam controlled.


## Replay Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_reward_weighted_market_selector_offline_replay.py
```

The full offline environment replay rejected the selector:

| Policy | Total reward | Delta vs fixed base | Delta vs fixed candidate | Trades |
|---|---:|---:|---:|---:|
| fixed base | `-384.480955` | `0.000000` | `-152.487919` | `2152` |
| fixed candidate | `-231.993036` | `+152.487919` | `0.000000` | `1957` |
| reward-weighted market switch | `-451.303399` | `-66.822444` | `-219.310363` | `2056` |

Selector telemetry:

- `2h` anchors: `175`;
- candidate anchors: `93` (`53.14%`);
- wrong candidate anchors: `26`;
- wrong candidate horizon loss: `-525.083989`.

## Interpretation

The selector helped in horizon-sample reward replay, but failed when embedded in
the full environment. This means the horizon selector is not yet aligned with the
actual path-dependent entry/hold/sell lifecycle.

Likely causes:

- anchor-level 2h reward does not map cleanly to all subsequent frame-level exit
  decisions;
- carrying the latest anchor choice forward can apply stale market beliefs;
- entry timing and exit timing interact, while the selector was trained only on
  exit-policy advantage windows;
- candidate/base policies differ less cleanly in full rollout than in isolated
  horizon windows.

## Decision

Rejected for policy promotion.

Do not send this selector to Telegram as a trade-like V2 signal. It may be logged
as research telemetry, but not as an operator-facing BUY/SELL recommendation.

## Next Gate

Before Telegram output, build a dedicated **live shadow telemetry** layer that
reports only:

- market-environment observation snapshot;
- selector expected-delta estimate;
- whether it is research-eligible;
- explicit gate status: `offline_full_replay_failed`.

Real-time Telegram should remain disabled for this selector until a full offline
replay beats fixed candidate.
