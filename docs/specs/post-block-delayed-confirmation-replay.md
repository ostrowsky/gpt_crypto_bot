# Post-Block Delayed Confirmation Replay

Status: research-only  
Last updated: 2026-05-19

## Purpose

Continue the winning direction from the post-block experiment suite:
`top15_and_tradable_120m` with delayed confirmation.

The suite found a high-precision confirmation pattern after an early block, but
that alone does not prove a tradable entry. This replay tests whether entering
**after** the 120-minute confirmation window still leaves enough forward edge.

## Candidate Rule

A post-block row is selected when:

```text
rel_ret_120m_pct >= 2.0
and volume_x_120m >= 1.5
```

Interpretation: the symbol outperformed BTC by at least `2%` over the 120 minutes
after the block and had at least `1.5x` recent-volume expansion.

## Replay Semantics

- Candidate time: first qualifying repeated block from the post-block dataset.
- Entry time: close of the `+120m` bar after candidate time.
- Forward evaluation: `60m`, `120m`, `240m`, and end-of-day after entry.
- Fees: reported as gross returns for research; production adoption would require
  full fee/slippage replay.

## Guardrails

- No live behavior change.
- No Telegram signal.
- No BUY relaxation.
- If forward returns are weak after the delayed entry, reject the direction as
  classification-only / too late.

## Acceptance Criteria For Further Research

Continue only if the selected holdout slice has:

- non-trivial support (`>= 10` candidates);
- better post-entry return than the unfiltered post-block holdout baseline;
- acceptable adverse excursion versus forward upside;
- enough remaining upside to justify a later full replay with fees and exits.

## First Run Result - 2026-05-19

Input dataset: `.runtime/reports/post_block_causal_discriminator_dataset_15m.jsonl`.

Selected holdout slice:

- candidates: `12`;
- top15 count: `7`;
- top15 precision: `58.333%`.

Post-entry performance after entering at the `+120m` confirmation close:

| Metric | Selected delayed confirmation | All post-block holdout baseline |
|---|---:|---:|
| mean `60m` return | `-0.889721%` | `-0.177884%` |
| mean `120m` return | `-1.135294%` | `-0.418279%` |
| mean `240m` return | `-0.878785%` | `-0.362176%` |
| mean EOD return | `-9.298740%` | `-7.494584%` |
| mean MFE to EOD | `8.687730%` | `3.719923%` |
| mean MAE to EOD | `-11.985605%` | `-9.438091%` |
| positive `120m` rate | `33.333%` | `33.080%` |
| positive `240m` rate | `41.667%` | `40.494%` |

Decision: `research_only_rejected_no_forward_edge_after_confirmation`.

Interpretation: the delayed confirmation rule is a good hindsight/top15 selector,
but it does not create a post-entry trading edge. By the time the confirmation is
observable, the entry is too late or too exposed to reversal/chase risk.

Consequence: do not continue this branch as an entry policy. The next useful
research direction is to move the decision point earlier and add causal trajectory
features that are known before the 120-minute confirmation completes, or to use
this pattern only as a post-factum diagnostic label.

