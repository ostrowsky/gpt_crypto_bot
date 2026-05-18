# V2 Exit Quality Baselines

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test transparent exit policies after residual-gap decomposition identified exit
monetization as the next largest reward lever.

## Compared Exit Profiles

All profiles use the same improved admission layer. Only the sell decision changes.

| Profile | Rule |
|---|---|
| `base_sell_0_70` | sell when `P(exhaustion)+P(reversal) >= 0.70` |
| `early_sell_0_60` | sell when `P(exhaustion)+P(reversal) >= 0.60` |
| `exhaustion_sensitive` | sell when `P(exhaustion) >= 0.35` or combined late mass `>= 0.60` |
| `reversal_sensitive` | sell when `P(reversal) >= 0.30` or combined late mass `>= 0.60` |
| `hybrid_peak_guard` | sell when late mass `>= 0.60`, or when late mass `>= 0.45` after a meaningful favorable move |

## Required Metrics

- total reward;
- trade count;
- named reward components;
- exit-state mix;
- average realized PnL;
- average giveback;
- average bars held.

## Acceptance Criteria

1. Admission layer is fixed across all variants.
2. No hindsight labels are used for action selection.
3. A better profile must improve total reward, not only lower giveback.
4. If earlier exits reduce giveback but damage trend monetization, state that explicitly.

## Next Gate

If one transparent exit profile improves reward:

1. keep it as the next policy baseline;
2. inspect remaining gap to oracle;
3. then decide whether to build supervised exit modeling.

## First OOS Result

No tested early-exit profile beat the current base:

| Profile | Total reward | Avg giveback | Avg realized PnL |
|---|---:|---:|---:|
| `base_sell_0_70` | `-384.481` | `-2.026` | `0.210` |
| `early_sell_0_60` | `-576.457` | `-1.951` | `0.176` |
| `exhaustion_sensitive` | `-741.815` | `-1.820` | `0.161` |
| `reversal_sensitive` | `-683.343` | `-1.903` | `0.168` |
| `hybrid_peak_guard` | `-613.790` | `-1.914` | `0.182` |

Interpretation:

- simply exiting earlier lowers giveback a little;
- but it destroys more trend monetization than it saves;
- the next exit improvement must improve **exhaustion discrimination**, not merely
  lower the sell threshold.
