# Feature Spec Index

Last updated: 2026-05-16 13:29 Europe/Budapest

## Objective

Keep every material bot capability tied to the same north star:
earlier capture of same-day watchlist top movers with a single unified
10-position portfolio and profitable exits near trend reversal.

## Spec Catalog

| Capability | Status | Canonical spec / source | Primary metrics | Next gate |
|---|---|---|---|---|
| Scout objective and metric contract | shipped | `SCOUT_OPTIMIZATION_SPEC.md` | `watchlist_top_bought`, `early_captures`, `false_positive_buys` | keep current |
| Signal-quality evaluator | shipped | `skills/signal-quality-evaluator/SKILL.md` | `miss_rate`, `capture_ratio_at_entry`, `exit_efficiency`, `giveback_pct` | feed hypothesis queue |
| 1m wake-up scout | shipped shadow-assisted | `SCOUT_OPTIMIZATION_SPEC.md` | `wakeups`, `admitted`, `buy_conversion`, `top_mover_conversion` | live funnel review |
| Signal-quality feedback policy | shipped narrow auto-apply | `files/signal_quality_feedback.py` | `cooldown_harm`, replay-confirmed quality deltas | keep auto-apply limited to replay-confirmed cooldown |
| Peak-risk lifecycle telemetry | shipped shadow-only | `docs/specs/peak-risk-shadow.md` | event count, `peak_within_n_bars`, false-positive continuation rate | collect shadow rows before any exit change |
| Hypothesis queue | shipped diagnostic-only | `docs/specs/hypothesis-queue.md` | ranked hypotheses, linked evidence, replay status | no auto-apply |
| Unified portfolio replacement | planned | `SCOUT_OPTIMIZATION_SPEC.md` | opportunity cost, replacement uplift, capture under `10/10` | replay grid required |
| Why-no-signal traces | planned | `SCOUT_OPTIMIZATION_SPEC.md` | blocker chain per missed top mover | design after funnel baseline |

## Governance

1. Any new live decision path needs a feature spec, a rollback switch, and a replay acceptance rule.
2. Shadow-only instrumentation may ship before replay when it does not alter BUY/SELL behavior.
3. Evaluator findings may create hypotheses automatically, but production changes still require replay evidence.
4. ML/RL changes must report effect on the bot objective, not only on surrogate PnL metrics.
