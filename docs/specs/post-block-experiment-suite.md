# Post-Block Experiment Suite

Status: research-only  
Last updated: 2026-05-19

## Purpose

Run a bounded series of post-block discriminator experiments and continue only the
most successful direction.

This prevents low-yield iteration over one weak target. The experiment compares
several target definitions on the same chronological split.

## Experiment Targets

- `useful_missed_winner`: final top15, not bought, positive opportunity from the
  first block;
- `final_top15`: became final watchlist top15 by cutoff;
- `tradable_60m`: after candidate, reached meaningful upside within 60m without
  excessive adverse move;
- `tradable_120m`: after candidate, reached meaningful upside within 120m without
  excessive adverse move;
- `top15_and_tradable_120m`: final top15 and tradable within 120m.

## Protocol

1. Use the existing post-block causal discriminator dataset.
2. Split chronologically by `local_day`.
3. Evaluate the same transparent rule families for every target.
4. Rank experiments by holdout precision lift, positive count, and bounded
   candidate pressure.
5. Continue only the winning target direction.

## Guardrails

- No live behavior changes.
- No Telegram signal.
- No BUY relaxation.
- Do not continue a target if it only wins by selecting a trivial tiny sample.

## Acceptance Criteria

The suite passes if it produces:

- per-target base rates;
- best holdout rule for each target;
- a selected next direction or explicit rejection of all directions;
- a clear next research action.

## First Run Result - 2026-05-19

Input dataset: `.runtime/reports/post_block_causal_discriminator_dataset_15m.jsonl`.

Split:

- rows: `1,937`;
- train: `1,411`;
- holdout: `526`.

Selected direction: `top15_and_tradable_120m`.

Best holdout rule:

```text
rel_ret_120m_pct >= 2.0
and volume_x_120m >= 1.5
```

Holdout result:

- base positive rate: `1.711%`;
- candidates: `12`;
- positives: `7`;
- precision: `58.333%`;
- precision lift: `34.09x`;
- recall: `77.778%`;
- bad ratio: `41.667%`.

Decision: continue this direction as a **delayed post-block confirmation replay**,
not as an immediate early-entry relaxation.

Important interpretation: the winning features are observable only after the
post-block `120m` confirmation window. This is not a universal early-trend mode
and not a causal BUY signal at the first block. The next valid experiment is a
focused behavior replay that asks whether entering after this delayed confirmation
improves objective metrics without unacceptable chase risk.

Rejected direction: keep avoiding more threshold sweeps on
`useful_missed_winner` as the primary label. The previous audit showed it was too
sparse and unstable; the suite found a better target by requiring both final
`top15` outcome and actual tradability within `120m`.


