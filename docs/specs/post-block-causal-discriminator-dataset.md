# Post-Block Causal Discriminator Dataset

Status: research-only  
Last updated: 2026-05-19

## Purpose

Build the first dataset for a causal-time discriminator after early blocks.

The target problem is not:

```text
early block => BUY
```

It is:

```text
after an early block, did the symbol show causal confirmation that distinguishes a
future useful top mover from ordinary noise?
```

## Unit Of Observation

One row is a `(local_day, symbol, first_eligible_block)` candidate.

The first eligible block is selected from event logs using only information
available at event time:

- blocked event before/equal max local hour;
- normalized reason in an allowed set;
- minimum repeated blocked events up to the candidate point.

## Features

Use only causal features available at or shortly after the first eligible block:

- block context:
  - hour;
  - reason code;
  - source;
  - timeframe;
  - block count before candidate;
- OHLCV context from local canonical history:
  - return from block close to `15m`, `30m`, `60m`, `120m` forward closes;
  - max high / min low move inside those horizons;
  - volume expansion versus recent bars;
  - range expansion versus recent bars;
  - market-relative return if BTC history is available.

## Labels

Post-factum labels are used only as targets:

- `label_top15`: final watchlist top15 by cutoff;
- `label_useful_missed_winner`: final top15, not bought, positive opportunity
  from first block;
- `label_bad_candidate`: not final top15.

## Acceptance Criteria

The package passes if it produces:

- rows for event-level early block candidates;
- feature coverage by horizon;
- class balance;
- enough positive and negative samples for a chronological discriminator audit.

No live behavior, Telegram signal, or BUY gate change is authorized by this
dataset alone.


## First Build Result

Latest build:

- output: `.runtime/reports/post_block_causal_discriminator_dataset_15m.jsonl`;
- audit: `.runtime/reports/post_block_causal_discriminator_dataset_audit_15m.json`;
- rows: `1,937` post-block candidates;
- settings:
  - max local hour: `12`;
  - minimum repeated blocks: `3`;
  - allowed reasons: `agent_mode_disabled`, `agent_leader_filter`,
    `top_gainer_score_gate`.

Labels:

- `top15`: `341` (`17.60%`);
- `useful_missed_winner`: `67` (`3.46%`);
- `bad_candidate`: `1,596` (`82.40%`).

Reason distribution:

- `agent_mode_disabled`: `1,383`;
- `top_gainer_score_gate`: `281`;
- `agent_leader_filter`: `273`.

Feature coverage is `100%` for all selected causal OHLCV / BTC-relative features
across `15m`, `30m`, `60m`, and `120m` horizons.

## Interpretation

This dataset confirms why naive rescue failed: the positive class is small and
highly imbalanced. Only `3.46%` of post-block candidates are useful missed
winners under the current label definition.

However, the top positive examples show strong causal confirmation patterns after
the block, especially large market-relative returns and volume/range expansion in
`60m` / `120m` horizons. This is exactly the signal family a discriminator should
try to separate from ordinary repeated-block noise.

## Decision

Advance to a simple chronological discriminator audit.

The next package should compare transparent models/rules on this dataset and must
optimize for precision and candidate pressure, not raw recall:

- train/holdout split by local day;
- baseline rules over `rel_ret_*`, `volume_x_*`, `range_x_*`;
- small logistic/ridge or binned lookup only after rule baselines;
- gate: materially improve over the event replay precision while keeping
  candidate count bounded.

No live behavior is authorized by this dataset alone.
