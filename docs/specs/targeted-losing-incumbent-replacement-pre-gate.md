# Targeted Losing-Incumbent Replacement Pre-Gate

Date: 2026-08-03
Status: research-only causal pre-gate

## Problem

Unfiltered portfolio replacement is neutral on average and negative at the
median. Replacements of losing incumbents look better, but a production rule
must also control the profitable replacements it would block.

Historical replacement events do not persist the candidate-ranker probability.
The first causal quality proxy is therefore the decision-time leader-score
difference already present in every replacement reason.

## Hypothesis

Allow replacement only when the incumbent is losing at the rotation decision
and the incoming leader score exceeds it by a train-selected minimum delta.
This should create positive replacement delta without excessive regret from
blocked profitable rotations.

## Protocol

- Load every closed replacement outcome from the incremental event cohort.
- Use only decision-time incumbent PnL and leader-score delta as policy inputs.
- Split complete replacement days chronologically 70/30 and purge one boundary
  day on both sides.
- Select leader delta from `0/5/10/15/20` on train only.
- Evaluate the frozen threshold on untouched holdout and the latest 14-day
  stability window.

Each split must have average delta above `+0.10pp`, non-negative median delta,
positive rate at least `50%`, and blocked-positive regret no higher than `25%`.
Minimum allowed counts are 30 train cases and 10 holdout/recent cases.

## Promotion Boundary

Passing authorizes a paired maximum-period ten-slot replay with fees, slippage,
and current exits. Failing keeps the current guarded replacement behavior and
does not justify adding ranker fields or changing live rotation.

## Maximum-History Result — 2026-08-03

The incremental cohort contained `507` closed replacements. The chronological
split produced `413` train and `74` untouched holdout cases; the latest 14-day
window contained `30` cases. Train selected leader delta `0`, so higher leader
thresholds did not improve the frozen objective.

| Split | Allowed | Avg delta | Median delta | Positive | Blocked regret | Gate |
|---|---:|---:|---:|---:|---:|---|
| train | `164` | `+0.3901pp` | `-0.0265pp` | `48.78%` | `33.73%` | fail |
| holdout | `55` | `+0.4507pp` | `+0.4270pp` | `63.64%` | `57.89%` | fail |
| recent 14d | `30` | `+0.4174pp` | `+0.4080pp` | `63.33%` | `0.00%` | pass |

The recent regime is favorable, but the train median/positive rate fail and the
holdout rule blocks too many profitable alternatives. Higher train thresholds
made median, positive rate, and regret worse. Decision:
`reject_targeted_losing_incumbent_replacement`. Keep the current replacement
guard and wait for another independent stability window before designing a
regime-conditional version.
