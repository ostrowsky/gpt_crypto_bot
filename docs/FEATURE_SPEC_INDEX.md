# Feature Spec Index

Last updated: 2026-05-16 13:29 Europe/Budapest

## Objective

Keep every material bot capability tied to the same north star:
earlier capture of same-day watchlist top movers with a single unified
10-position portfolio and profitable exits near trend reversal.

## Spec Catalog

| Capability | Status | Canonical spec / source | Primary metrics | Next gate |
|---|---|---|---|---|
| Spec-first workflow | shipped | `docs/specs/spec-first-workflow.md` | regression safety, reproducibility | use for every non-trivial change |
| Scout objective and metric contract | shipped | `SCOUT_OPTIMIZATION_SPEC.md` | `watchlist_top_bought`, `early_captures`, `false_positive_buys` | keep current |
| Signal-quality evaluator | shipped | `skills/signal-quality-evaluator/SKILL.md` | `miss_rate`, `capture_ratio_at_entry`, `exit_efficiency`, `giveback_pct` | feed hypothesis queue |
| 1m wake-up scout | shipped shadow-assisted | `SCOUT_OPTIMIZATION_SPEC.md` | `wakeups`, `admitted`, `buy_conversion`, `top_mover_conversion` | live funnel review |
| Early trend-start mode | replay-only research, first profile rejected | `docs/specs/trend-start-mode.md` | `capture_rate`, `capture_ratio_at_entry`, `lead_time_to_final_top_min`, `trade_precision`, `PnL` | next narrower replay profile |
| Agent entry quality gates | shipped | `docs/specs/agent-entry-quality.md` | `trade_precision`, `false_positive_buys`, `blocked_winners` | replay before any relaxation |
| Local control-plane reliability | shipped | `docs/specs/local-control-plane-reliability.md` | process health, duplicate-launch avoidance | keep operational-only |
| Local context MCP | local tooling | `docs/specs/local-context-mcp.md` | context-read latency, report compactness | keep outside trading path |
| Full-stack restart helper | shipped local tooling | `docs/specs/full-stack-restart-helper.md` | restart success, operator recovery time | keep operational-only |
| Repository skills tooling | shipped local tooling | `docs/specs/repo-skills-tooling.md` | workflow reuse, review consistency | keep outside trading path |
| Market-data / execution scaffold | research-only | `docs/specs/market-data-execution-research-scaffold.md` | replay/live parity groundwork, slippage realism | no production import without later gate |
| Local artifact hygiene | shipped local tooling | `docs/specs/local-artifact-hygiene.md` | status readability, secret hygiene | keep operational-only |
| P0 measurement hardening | shipped measurement-only | `docs/specs/p0-measurement-hardening.md` | report coverage, funnel loss point, exit lifecycle | use before next algorithm change |
| Agent mode rescue replay | research-only | `docs/specs/agent-mode-rescue-replay.md` | `capture_rate`, `trade_precision`, `pnl_total`, entry timing | compare `agent_allowed` vs `agent_mode_rescue` before any live change |
| Wake-up confirmed agent rescue | research-only | `docs/specs/wakeup-confirmed-agent-rescue.md` | `capture_rate`, `trade_precision`, `pnl_total`, temporal rescue admissions | evaluate only on windows with wake-up coverage |
| Belief-state v2 architecture | research-only | `docs/specs/belief-state-v2-architecture.md` | state reconstruction quality, objective-aligned reward, later policy uplift | build sequence dataset before any RL |
| V2 sequence dataset builder | research-only | `docs/specs/v2-sequence-dataset-builder.md` | sequence coverage, transition count, gap diagnostics | label lifecycle states before model fitting |
| V2 sequence coverage audit | research-only | `docs/specs/v2-sequence-coverage-audit.md` | transition density, contiguous coverage, longest usable history | choose data-repair strategy before state modeling |
| Signal-quality feedback policy | shipped narrow auto-apply | `files/signal_quality_feedback.py` | `cooldown_harm`, replay-confirmed quality deltas | keep auto-apply limited to replay-confirmed cooldown |
| Peak-risk lifecycle telemetry | shipped shadow-only | `docs/specs/peak-risk-shadow.md` | event count, `peak_within_n_bars`, false-positive continuation rate | collect shadow rows before any exit change |
| Hypothesis queue | shipped diagnostic-only | `docs/specs/hypothesis-queue.md` | ranked hypotheses, linked evidence, replay status | no auto-apply |
| Unified portfolio replacement | planned | `SCOUT_OPTIMIZATION_SPEC.md` | opportunity cost, replacement uplift, capture under `10/10` | replay grid required |
| Why-no-signal traces | shipped first read-only version | `docs/specs/p0-observability-foundation.md` | blocker chain per missed top mover | connect to daily reports |
| P0 observability foundation | shipped first additive slice | `docs/specs/p0-observability-foundation.md` | `blocked_reason_harm`, evaluator coverage | expand blocker context fields |
| Canonical metrics map | shipped | `docs/specs/metrics-canonical.md` | objective decision consistency | use in roadmap reviews |

## Governance

1. Follow `docs/specs/spec-first-workflow.md` for every non-trivial change.
2. Any new live decision path needs a feature spec, a rollback switch, and a replay acceptance rule.
3. Shadow-only instrumentation may ship before replay when it does not alter BUY/SELL behavior.
4. Evaluator findings may create hypotheses automatically, but production changes still require replay evidence.
5. ML/RL changes must report effect on the bot objective, not only on surrogate PnL metrics.
