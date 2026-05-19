# Post-Block Causal Discriminator Audit

Status: research-only  
Last updated: 2026-05-19

## Purpose

Evaluate whether the post-block causal discriminator dataset contains a simple,
usable separation signal.

The question is:

```text
Can causal post-block confirmation features select useful missed winners with
materially better precision than the raw early-block rescue baseline?
```

## Inputs

- `.runtime/reports/post_block_causal_discriminator_dataset_15m.jsonl`

## Protocol

1. Split chronologically by `local_day`.
2. Evaluate transparent rule families first:
   - market-relative return thresholds;
   - volume expansion thresholds;
   - range expansion thresholds;
   - conjunctions of return + expansion.
3. Evaluate train-only binned lookup models over one/two features.
4. Rank candidates by holdout useful-winner precision and bounded candidate
   pressure.

## Target

Primary positive class:

```text
label_useful_missed_winner == true
```

Secondary diagnostic class:

```text
label_top15 == true
```

## Gate

A discriminator may advance only if holdout metrics satisfy:

- useful precision materially above raw base rate;
- candidate count bounded;
- useful recall not zero/trivial;
- top15 precision better than the rejected event-level rescue baseline.

No live behavior, Telegram signal, or BUY change is authorized by this audit.


## First Audit Result

Latest chronological split:

- rows: `1,937`;
- train rows: `1,411`, days `2026-04-23` through `2026-05-09`;
- holdout rows: `526`, days `2026-05-10` through `2026-05-17`.

Holdout base rates:

- useful missed winner rate: `2.66%`;
- final top15 rate: `19.58%`;
- bad candidate rate: `80.42%`.

Best holdout variant:

- family: `rule_return_and_expansion`;
- features: `rel_ret_30m_pct` + `range_x_120m`;
- thresholds: `rel_ret_30m_pct >= 0.5`, `range_x_120m >= 2.0`;
- candidate count: `7`;
- useful missed winners: `1`;
- top15 candidates: `3`;
- bad candidates: `4`;
- useful precision: `14.29%`;
- top15 precision: `42.86%`;
- useful recall: `7.14%`.

Result: rejected.

## Interpretation

The best simple discriminator improves useful precision above the raw holdout base
rate, but only by collapsing to a tiny number of candidates. It captures one
useful missed winner on holdout, which is not enough to justify behavior replay.

This means the current causal confirmation features are directionally useful but
not yet sufficient as a robust rescue selector.

## Decision

Do not advance to behavior replay.

The next productive move is not another broad threshold sweep. Options are:

1. improve labels from `final top15` / `missed winner` to a more direct
   tradability label, e.g. forward return after candidate minus adverse excursion;
2. add rank/score trajectory features from v1 candidate scoring if available;
3. expand training coverage before trying a learned classifier.

Until one of those is done, keep post-block discriminator research-only.
