---
name: trading-app-review
description: Review crypto trading applications and trading-bot codebases against this bot's objective, architecture, algorithms, and current professional SOTA practices. Use when Codex must compare another trading application with this bot, assess strengths and weaknesses, perform architecture and algorithm gap analysis, and produce a prioritized roadmap for improving this bot toward earlier top-mover capture and better trend exits.
---

# Trading App Review

## Purpose

Review a trading application as a source of transferable lessons for this bot.

Always answer four questions:
1. What is objectively strong or weak in the reviewed application?
2. What is stronger or weaker than this bot today?
3. What separates both systems from current professional SOTA practice?
4. Which concrete changes should this bot adopt first to improve its target metrics?

## Required Inputs

- The reviewed application's repo, docs, paper, or product material.
- This bot's objective and metrics from `SCOUT_OPTIMIZATION_SPEC.md`.
- This bot's current progress from `skills/bot-progress-report`.
- Fresh external SOTA references when comparing with professional market practice.

## Workflow

1. Read `SCOUT_OPTIMIZATION_SPEC.md`.
2. Run `bot-progress-report` for the latest relevant window so the comparison uses the bot's current state, not memory.
3. Inspect the reviewed application's:
   - architecture;
   - data ingestion;
   - signal generation;
   - ranking/selection;
   - execution and risk;
   - exit logic;
   - feedback/learning;
   - observability and operator controls;
   - test/backtest discipline.
4. Read `references/review-rubric.md`.
5. For SOTA comparison, use current external sources:
   - prefer official product docs, engineering blogs, exchange docs, and primary research papers;
   - verify claims that may have changed recently;
   - distinguish proven industry practice from marketing claims.
6. Produce the report in this order:
   - executive verdict;
   - reviewed app strengths;
   - reviewed app weaknesses;
   - comparison with this bot;
   - SOTA benchmark comparison;
   - gap analysis;
   - prioritized roadmap for this bot;
   - risks, trade-offs, and backtest requirements.

## Required Evaluation Axes

- `Goal fit`: does the system optimize for the same outcome as this bot or for a different game?
- `Signal timeliness`: how early it detects persistent trend starts.
- `Selection quality`: how it ranks scarce opportunities under a portfolio cap.
- `Exit quality`: how it detects trend exhaustion and preserves MFE.
- `Learning loop`: labels, delayed feedback, online/offline split, drift handling.
- `Architecture`: separation of live path, research path, evaluator path, and control plane.
- `Resilience`: retries, degradation, idempotency, recovery after restarts.
- `Observability`: traceability from candle to signal to block to outcome.
- `Experiment discipline`: replay, walk-forward, holdout, leakage control, promotion gates.
- `Operator UX`: latency of controls, explainability, concise reports.

## Mandatory Output

Use this structure:

1. `Executive Verdict`
2. `What The Reviewed App Does Better`
3. `Where The Reviewed App Is Weaker`
4. `Compared With My Bot`
5. `Compared With Professional SOTA`
6. `Gap Analysis`
7. `Roadmap For My Bot`
8. `What Must Be Backtested Before Adoption`

For each meaningful finding include:
- why it matters;
- which target metric it affects;
- whether it is transferable as-is, only as inspiration, or not transferable.

## Guardrails

- Do not reward complexity for its own sake.
- Do not confuse higher activity with better capture.
- Do not call a design SOTA merely because it uses ML, RL, or many services.
- Do not compare systems with different objectives as if they were solving the same task.
- Do not recommend production adoption without a replay/backtest gate.
- When evidence is weak, say `inconclusive`, not `good`.

## Companion References

- Read `references/review-rubric.md` for the scoring frame.
- Read `references/report-template.md` when writing a full user-facing report.
