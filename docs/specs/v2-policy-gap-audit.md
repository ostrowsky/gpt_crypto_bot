# V2 Policy Gap Audit

Status: research-only  
Last updated: 2026-05-18

## Purpose

Explain why the best calibrated threshold policy remains far below the lifecycle
oracle before any learned policy is attempted.

## Questions

1. In which true lifecycle states does the policy open positions?
2. In which true lifecycle states does it exit?
3. Which trade-level reward components create the largest oracle gap?
4. Is the next bottleneck primarily:
   - admission;
   - exit;
   - or both?

## Compared Policies

- lifecycle oracle;
- best current threshold policy:
  - `open_threshold=0.70`;
  - `sell_threshold=0.70`.

## Required Report

For each policy:

- entries by true lifecycle state;
- exits by true lifecycle state;
- trade count;
- average realized PnL;
- average giveback penalty;
- average total reward per trade;
- average bars held.

## Acceptance Criteria

1. Uses the same OOS episodes as prior baseline reports.
2. Keeps admission and exit failure modes separate.
3. Does not recommend RL if a simpler structural defect is still obvious.

## First OOS Result

| Policy | Trades | Avg reward / trade | Avg realized PnL | Avg giveback |
|---|---:|---:|---:|---:|
| `lifecycle_oracle` | `461` | `+12.727` | `+5.062` | `-3.095` |
| `threshold_o0.70_s0.70` | `2205` | `-0.251` | `+0.189` | `-2.053` |

The calibrated threshold policy still opens mostly in the wrong lifecycle state:

- `noise` entries: `1529`;
- `emerging_move` entries: `373`;
- `confirmed_trend` entries: `49`.

Conclusion:

- the dominant remaining defect is **admission**, not only exit;
- the next policy layer should decide whether capital may be allocated at all before
  downstream action selection;
- RL remains premature until this admission gap is modeled explicitly.
