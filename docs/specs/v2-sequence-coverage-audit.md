# V2 Sequence Coverage Audit

Status: research-only  
Last updated: 2026-05-17

## Purpose

Explain whether the current local history is sufficient for v2 state-model work and
identify where contiguous observation coverage is being lost.

## Problem

The first v2 sequence smoke run found:

- 2,304 accepted bar rows;
- 1,540 sequences;
- only 764 transitions;
- 1,148 gap breaks.

That is far too fragmented to treat the local sequence history as training-grade.
Before lifecycle labeling or HMM experiments, the bot needs a clear answer to:

> Which symbol/timeframe/day slices are contiguous, which are broken, and how much usable
> history actually exists?

## Scope

### In scope

- audit output built from `files/ml_dataset.jsonl`;
- per-timeframe metrics;
- per-day metrics;
- longest contiguous sequence metrics;
- transition density;
- top gap-heavy symbol/timeframe/day slices;
- machine-readable JSON plus concise text output.

### Out of scope

- repairing or backfilling the dataset;
- reading temporary dataset artifacts as canonical history;
- HMM / RL training;
- production changes.

## Metrics

- `rows_accepted`
- `days_covered`
- `sequences_built`
- `transitions_built`
- `gap_breaks`
- `transition_density`
- `longest_sequence_bars`
- `longest_sequence_minutes`
- per-day sequence / transition counts
- per-timeframe sequence / transition counts

## Acceptance Criteria

1. The audit reuses the v2 sequence builder instead of duplicating logic.
2. It reports at least daily, timeframe, and longest-sequence coverage.
3. It explicitly says whether the current history is `insufficient`, `usable_partial`,
   or later `training_ready`.
4. It stays read-only.
5. Unit tests cover the coverage summary.

## Verification

- `python -m unittest test_v2_coverage_audit.py`
- smoke:
  - `python files/report_v2_sequence_coverage.py --json`

## Next Gate

Use the audit to decide between:

1. repairing/backfilling `ml_dataset` history;
2. building a separate canonical OHLCV sequence store;
3. limiting first state-model experiments to coverage-valid windows only.

## Rollback / Safety

- report-only;
- no production import;
- removable with no behavioral effect.

## First Local Result

Executed on 2026-05-17 against the current local `files/ml_dataset.jsonl`:

| Metric | Value |
|---|---:|
| rows accepted | 2,321 |
| days covered | 11 |
| sequences built | 1,549 |
| transitions built | 772 |
| gap breaks | 1,157 |
| transition density | 0.3326 |
| longest sequence | 11 bars / 660 min |
| coverage status | `usable_partial` |

By timeframe:

| Timeframe | Observations | Sequences | Transitions |
|---|---:|---:|---:|
| `15m` | 1,292 | 1,292 | 0 |
| `1h` | 1,029 | 257 | 772 |

Decision:

- the local `ml_dataset` is not suitable as the primary v2 sequence source;
- especially on `15m`, it currently contains isolated observations rather than usable
  contiguous sequences;
- the next architecture step should define a canonical v2 market-history source instead
  of silently training on fragmented legacy logs.
