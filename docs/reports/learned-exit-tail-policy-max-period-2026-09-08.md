# Learned Exit-Tail Policy: Maximum-Period Result

Date: 2026-09-08
Status: terminal reject for the registered policy family; production unchanged.

## Evidence contract

- Source: all locally available mature final signal-quality reports through
  2026-09-07.
- Split: 72 whole train days followed by 32 untouched test days.
- Cases: 2,730 train; 1,543 test; 1,407 test cases with complete candle paths.
- Model inputs: causal-at-exit fields only.
- Threshold: train-only risk q80 = 0.53.
- Selected test exits: 314 / 1,407 (22.32%).
- Action coverage: 1,407 / 1,407; no selected action was missing.
- Production effect: none.

## Results

| Policy | Selected mean delta | Selected median delta | Selected p10 | Worse rate | Overall mean delta | Policy win rate | Baseline win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| sell 50%, tail 50%, 5 bars | +0.1468pp | -0.1583pp | -0.6658pp | 59.55% | +0.0328pp | 29.28% | 32.20% |
| sell 50%, tail 50%, 10 bars | +0.1684pp | -0.3076pp | -0.7031pp | 65.29% | +0.0376pp | 28.78% | 32.20% |
| sell 70%, tail 30%, 5 bars | +0.0881pp | -0.0950pp | -0.3995pp | 59.55% | +0.0197pp | 31.20% | 32.20% |
| sell 70%, tail 30%, 10 bars | +0.1010pp | -0.1845pp | -0.4219pp | 65.29% | +0.0225pp | 30.92% | 32.20% |

Baseline test average PnL was -0.4230% and median PnL was -0.4520%.

## Terminal decision

No policy passed the pre-registered safety gate. Every policy failed the
non-negative selected median, maximum 35% worse-rate, and non-decreasing win
rate requirements. The positive average delta comes from a minority of large
continuations and does not establish a generally beneficial action.

The continuation-risk learner remains useful as a shadow diagnostic, but its
score must not alter SELL or cooldown. The next hypothesis must learn the
counterfactual advantage of each concrete tail action and validate it with the
same chronological, maximum-period discipline before portfolio replay.
