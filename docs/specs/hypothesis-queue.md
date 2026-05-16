# Hypothesis Queue

Last updated: 2026-05-16 13:29 Europe/Budapest

## Purpose

Convert repeated evaluator/scout findings into a machine-readable research
queue without letting diagnosis mutate live behavior directly.

## Inputs

- Latest `signal_quality_*_final.json`
- Latest `top_gainer_critic_*_final.json`
- Existing replay-confirmed feedback in `.runtime/signal_quality_feedback.json`

## Output

`files/build_quality_hypotheses.py` writes:

- `.runtime/reports/quality_hypotheses_latest.json`
- `.runtime/reports/quality_hypotheses_latest.txt`

Each hypothesis includes:

- `id`
- `priority`
- `problem`
- `evidence`
- `proposal`
- `required_backtest`
- `auto_apply_allowed`

## Rules

1. The queue is diagnostic-only.
2. `auto_apply_allowed` is always `false` unless a separate replay gate already exists.
3. Prefer one narrow hypothesis per error class:
   - missed top movers
   - late entries
   - early exits
   - cooldown harm
4. Rejected or mixed replay results stay visible so the bot does not keep
   rediscovering the same bad idea.
