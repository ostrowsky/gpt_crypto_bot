# V2 Residual Gap Decomposition

Status: research-only  
Last updated: 2026-05-18

## Purpose

Explain what still separates the improved admission policy from the lifecycle oracle
after the first validated admission gain.

The combined admission replay improved total reward:

- threshold base: `-554.283`;
- improved admission: `-305.751`.

The next question is no longer “does admission help?” It is:

> what is now the dominant remaining source of loss: residual admission error or exit
> monetization?

## Compared Policies

- lifecycle oracle;
- improved admission policy:
  - threshold action policy;
  - recall-preserving projected-v1 admission.

## Required Decomposition

### Admission side

- entries by true state;
- share of:
  - productive entries (`emerging_move`, `confirmed_trend`);
  - noise entries;
  - late entries (`mature_trend`, `exhaustion`, `reversal`).

### Exit side

- exits by true state;
- average realized PnL;
- average giveback;
- average total reward per trade;
- bars held.

### Reward gap

- component-level difference versus oracle:
  - early capture;
  - false buy;
  - realized PnL;
  - MFE retention;
  - giveback;
  - trend hold.

## Acceptance Criteria

1. Uses the same OOS episode set as prior policy reports.
2. Separates admission defects from exit defects.
3. Names the dominant next bottleneck explicitly.
4. Does not recommend RL before the dominant residual defect is characterized.

## Next Gate

If residual admission remains dominant:

- build a supervised / richer admission model.

If exit monetization dominates:

- build a dedicated exit-quality layer before broader policy learning.

## First OOS Result

The improved admission policy still has material admission error:

- productive entry share: `19.563%`;
- noise entry share: `68.355%`;
- late entry share: `12.082%`.

However, the largest remaining reward gaps versus oracle are:

| Reward component | Gap versus oracle |
|---|---:|
| `giveback_penalty` | `-2915.976` |
| `realized_pnl_reward` | `-1883.641` |
| `false_buy_penalty` | `-1471.000` |

Conclusion:

- residual admission is still not solved;
- the next **largest monetization lever** is exit quality;
- the Roadmap should branch:
  - keep admission improvement as an active line;
  - prioritize a dedicated exit-quality layer next.
