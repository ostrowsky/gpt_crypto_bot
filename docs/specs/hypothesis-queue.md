# Hypothesis Queue

Last updated: 2026-08-05 20:25 Europe/Budapest

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

## Rejected or mixed early-exit hypotheses — 2026-08-05

Maximum local causal replay of `831` RSI-WEAK exits rejected all static
production relaxations:

| Hypothesis | Result | State |
|---|---|---|
| WEAK always tightens ATR instead of SELL | negative validation/holdout median; about 70% harmed | rejected |
| retest grace 3–5 bars | no setting positive on both validation and holdout average | rejected |
| two consecutive WEAK observations | holdout net avg `-0.26pp`, median `-0.58pp` | rejected |
| 15m causal structure veto | holdout median negative; `77%+` harmed at `k=1.4` | rejected |
| last-closed 1h confirmation | positive mean, negative median, `56-61%` harmed | mixed/rejected for production |
| static 25%/50% protected tail | downside compressed, but median negative and `61%` harmed | mixed/rejected for production |

Next allowed exit hypothesis: a causal discriminator trained on universal
post-exit labels for **all** WEAK exits, including low-MFE exits currently
excluded by suspicious-reentry registration. Static threshold variants above
must not be reopened without a new observable feature or new forward evidence.
