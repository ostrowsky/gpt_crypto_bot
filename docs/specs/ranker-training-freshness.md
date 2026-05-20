# Ranker Training Freshness

Status: shipped operational fix  
Owner: Codex  
Date: 2026-05-20

## Problem

The RL/headless worker restores training progress from runtime artifacts before
deciding whether to retrain the candidate ranker. If those artifacts come from a
different local dataset window, `last_trained_rows` can be larger than the
currently available labeled ranker rows.

That creates a hidden self-improvement failure: the worker sees current data,
but suppresses retraining because the current row count never reaches the old
larger baseline.

## Goal

Keep the learning loop fresh for the dataset the running bot is actually using,
without auto-promoting any unvalidated trading rule.

## Requirements

- If the current ranker dataset has enough labeled rows and its file mtime is
  newer than the restored training baseline, retrain even when row count moved
  backwards.
- Restore `dataset_mtime` from the latest training report so restart decisions
  compare both row count and data freshness.
- Keep the change operational-only: it may retrain the shadow ranker and write
  reports/models, but it must not relax BUY/SELL gates.
- Cover the rollback case with a unit test.

## Acceptance

- `should_train(...)` returns true for a newer dataset whose row count is below
  the restored historical baseline.
- unchanged or older datasets do not retrain solely because rows are lower.
- RL worker status continues to expose `last_rows_total` and
  `last_dataset_mtime`.

