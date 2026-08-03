# ML Dataset Atomic Rewrite

Date: 2026-08-03
Status: recovery safety fix

## Problem

The bot and headless RL collector can append labels to the same
`ml_dataset.jsonl`. Legacy label functions read the complete file and then used
`Path.write_text`, which truncates the destination before writing. A concurrent
reader could observe that empty/truncated interval and replace the dataset with
only the rows it saw. On 2026-08-03 this reduced a 257.6 MB dataset to one row.

## Behavior

- `ML_FILE` is anchored to the module directory and no longer depends on the
  process working directory.
- Append and label-rewrite operations use the same Windows cross-process file
  lock.
- Rewrites stream source rows into a PID/thread-specific temporary file and
  atomically replace the destination only after the complete output is closed.
- A failed rewrite removes its own temporary file and leaves the prior dataset
  intact.
- Malformed rows remain observable through warning counts and are removed only
  during a successful atomic rewrite.
- Recovery backfill scans `bot_events.jsonl` line by line as UTF-8 with invalid
  bytes ignored; a host cp1251 default or one damaged byte cannot abort gap
  reconstruction before writes begin.

## Guardrails

This changes dataset persistence only. It does not change signal rules, feature
values, label definitions, model thresholds, positions, or Telegram behavior.

## Verification

Focused tests reject whole-file reads, verify target-label mutation, preserve
unrelated rows, remove temporary files, and exercise append plus rewrite through
the shared lock.
