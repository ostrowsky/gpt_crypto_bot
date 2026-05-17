# V2 Sequence Dataset Builder

Status: research-only  
Last updated: 2026-05-17

## Purpose

Create the first sequence-dataset layer for `files/v2/` from already logged local data,
without pretending that the current data coverage is already sufficient for HMM or RL
training.

## Problem

The v2 architecture needs ordered observations and transitions, not isolated BUY modes.
The repository already contains two useful sources:

- `files/ml_dataset.jsonl`
  - intended to log bar snapshots for every monitored coin;
- `files/critic_dataset.jsonl`
  - candidate/event-centered records with richer decision metadata.

However, the current local `ml_dataset.jsonl` is sparse:

- 2,304 rows;
- 105 symbols;
- only 11 calendar days represented;
- visible gaps in history.

Therefore the first v2 dataset package must solve **contract and measurement** before
model training.

## Scope

### In scope

- read `ml_dataset.jsonl`;
- normalize rows into ordered observation records;
- group them by `(symbol, timeframe, local_day)`;
- deduplicate repeated bars;
- split sequences when timestamps are not contiguous for the timeframe;
- emit transitions between adjacent observations;
- expose coverage diagnostics:
  - rows read / accepted / rejected;
  - days covered;
  - symbols covered;
  - sequences built;
  - transitions built;
  - gap breaks;
  - duplicates removed;
  - coverage status.

### Out of scope

- hindsight lifecycle-state labels;
- HMM fitting;
- MCMC estimation;
- RL training;
- merging `critic_dataset` and `ml_dataset`;
- any production import.

## Dataset Contract

Each observation carries:

- `symbol`
- `timeframe`
- `ts_ms`
- `features`
- `local_day`
- optional top-gainer teacher metadata if already present

Each transition carries:

- current observation
- action placeholder (`ignore` for passive bar snapshots)
- reward placeholder (`0.0` until reward labeling is introduced)
- next observation
- `done=False`
- optional label placeholder

## Coverage Status

- `usable_partial`
  - at least one sequence and one transition exist, but coverage is not broad enough to
    call the dataset training-grade;
- `insufficient`
  - no usable transitions exist.

No current local output may be called `training_ready` until a later spec defines and
passes stronger coverage criteria.

## Acceptance Criteria

1. Builder is pure and research-only under `files/v2/`.
2. Duplicate bars do not create duplicate transitions.
3. Time gaps split sequences instead of silently connecting unrelated bars.
4. Output summary clearly states current coverage limitations.
5. Unit tests cover dedupe, contiguous grouping, and gap splitting.
6. Smoke run on current local `ml_dataset.jsonl` succeeds.
7. No live code imports the v2 package.

## Verification

- `python -m unittest test_v2_core.py test_v2_sequence_dataset.py`
- smoke:
  - `python files/build_v2_sequence_dataset.py --json`

## Next Gate

After this package:

1. add a separate spec for hindsight lifecycle labeling;
2. decide whether `ml_dataset` coverage must be repaired/backfilled before model work;
3. only then start state-reconstruction experiments.

## Rollback / Safety

- delete the builder and spec with no production impact;
- builder reads existing logs only and writes reports only when explicitly requested.

## First Local Smoke Result

Executed on 2026-05-17 against the current local `files/ml_dataset.jsonl`:

| Metric | Value |
|---|---:|
| rows read | 2,304 |
| rows accepted | 2,304 |
| days covered | 11 |
| symbols covered | 105 |
| sequences built | 1,540 |
| transitions built | 764 |
| gap breaks | 1,148 |
| coverage status | `usable_partial` |

Decision:

- the builder is valid;
- the current dataset is **not** learning-grade yet;
- before HMM / RL work, either coverage must be improved or future experiments must be
  explicitly limited to coverage-valid slices only.
