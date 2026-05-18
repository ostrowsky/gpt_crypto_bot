# V2 Entry Admission Baselines

Status: research-only  
Last updated: 2026-05-18

## Purpose

Measure whether the inherited v1 feature language improves v2 admission beyond
belief-only rules.

The question is not whether projected v1 scores look reasonable. The question is:

> do they help reject `noise` while preserving useful `emerging_move` admissions?

## Compared Baselines

| Baseline | Rule |
|---|---|
| `belief_only` | admit when `P(emerging_move) + P(confirmed_trend) >= threshold` |
| `projected_v1_only` | admit when projected v1 leader score >= threshold |
| `belief_plus_projected_v1` | require both belief mass and projected v1 score |
| `belief_plus_projected_v1_plus_temporal` | require belief + projected v1 + prior structural scout |

## Grid

| Parameter | Values |
|---|---|
| belief threshold | `0.50`, `0.60`, `0.70` |
| projected-v1 leader threshold | `3.0`, `5.0`, `8.0` |

## Required Metrics

- admitted rows;
- admitted true-state mix;
- `noise` admission rate;
- `emerging_move` recall;
- admission precision where positives are:
  - `emerging_move`;
  - `confirmed_trend`.

## Acceptance Criteria

1. Every baseline uses the same OOS rows.
2. Temporal evidence uses only prior events.
3. The best baseline is chosen by the trade-off between:
   - lower `noise` admission;
   - preserved `emerging_move` recall.
4. If v1 features reduce noise only by collapsing recall, say so explicitly.

## Next Gate

If one family beats belief-only meaningfully:

1. run it through the offline decision environment;
2. compare reward versus threshold policy;
3. then decide whether to build supervised admission or keep a transparent gate.

## First OOS Result

The audit exposed two different truths:

1. precision-only selection is misleading;
2. projected v1 structure is still useful when recall is protected.

| Variant | Precision | Emerging recall | Noise admission |
|---|---:|---:|---:|
| best belief-only (`belief>=0.70`) | `0.180` | `0.384` | `0.739` |
| recall-preserving combined (`belief>=0.60`, `leader>=3.0`) | `0.190` | `0.355` | `0.701` |
| precision-max combined + temporal (`belief>=0.70`, `leader>=8.0`) | `0.246` | `0.035` | `0.572` |

Interpretation:

- projected v1 structure adds real admission value:
  - precision improves;
  - noise admission falls;
  - early recall is only modestly reduced in the recall-preserving regime.
- hard temporal requirements are currently too restrictive for the main path:
  - they improve precision,
  - but collapse the early-capture objective.
- the next gate should replay **recall-preserving** admission in the offline decision
  environment, not the precision-max variant.
