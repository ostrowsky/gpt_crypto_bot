# V2 Belief Calibration Audit

Status: research-only  
Last updated: 2026-05-18

## Purpose

Tune the first belief filter before any policy learning by measuring the trade-off
between:

- early-state recall;
- overall lifecycle reconstruction;
- late-state degradation.

## Grid

| Parameter | Values |
|---|---|
| self-transition bias | `0.55`, `0.65`, `0.70`, `0.75`, `0.85` |
| emission temperature | `0.50`, `0.75`, `1.00`, `1.25`, `1.50` |

## Required Report

For every variant:

- weighted accuracy;
- macro F1;
- recall by state;
- delta versus isolated nearest-centroid baseline.

The report must also identify:

1. best macro-F1 variant;
2. best emerging-move recall variant;
3. balanced candidate that improves `emerging_move` while not collapsing `reversal`.

## Acceptance Criteria

1. Every variant is evaluated on the same OOS split.
2. No test-window fitting.
3. Best trade-off is stated explicitly, not selected by cherry-picking one metric.
4. If no balanced variant exists, say so before moving to policy learning.

## Result

The first audit found three different optima:

| Selection rule | Variant | Why it matters |
|---|---:|---|
| Best macro F1 | `self_bias=0.85`, `temperature=1.50` | strongest overall lifecycle reconstruction, but reversal recall falls to `0.361` |
| Best emerging-move recall | `self_bias=0.55`, `temperature=1.50` | catches the most early moves (`0.620` recall), but collapses reversal recall to `0.078` |
| Balanced research default | `self_bias=0.85`, `temperature=0.75` | lifts macro F1 to `0.319`, emerging recall to `0.388`, and keeps reversal recall at `0.409` |

The balanced variant is the default for the next research stage because the bot's
objective needs both:

- earlier recognition of persistent moves;
- enough late-state awareness to avoid learning policies that hold through reversals.

The high-emerging-recall variant is useful as an ablation, not as the mainline
filter for offline policy learning.
