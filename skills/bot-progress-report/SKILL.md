---
name: bot-progress-report
description: Build recurring progress reports for this crypto trend bot, focused on learning progress, scout performance, ML/RL component health, and distance to the bot objective metrics. Use when Codex needs daily, weekly, or rolling-window reporting about whether the bot is improving toward early capture of same-day watchlist top movers and better trend exits.
---

# Bot Progress Report

## Purpose

Report whether the bot is moving toward its stated objective:
- capture same-day watchlist top movers earlier;
- keep a high-quality unified portfolio;
- improve ML/RL candidate selection without confusing lower noise with actual goal progress.

## Workflow

1. Read `SCOUT_OPTIMIZATION_SPEC.md` for the objective and target metrics.
2. Run `scripts/build_progress_report.py` for the requested window.
3. Read the generated JSON first, then summarize:
   - objective progress;
   - scout progress;
   - ML progress;
   - RL/data-pipeline health;
   - current bottlenecks and next actions.
4. Keep retrospective evaluator metrics separate from live scout metrics.
5. State clearly when a metric improved locally but still moved away from the business objective.

## Core Inputs

- `.runtime/reports/top_gainer_critic_YYYY-MM-DD_final.json`
- `.runtime/reports/watchlist_top_gainer_goal_YYYY-MM-DD_22h.json`
- `.runtime/reports/signal_quality_YYYY-MM-DD_final.json`
- `.runtime/reports/rl_train_latest.json`
- optional `bot_events.jsonl` / `agent_events.jsonl` only when a requested report needs event-level detail

## Required Sections

- `Goal`: top-mover recall, early capture, precision, trend capture, and distance to target.
- `Scout`: discovered/bought top movers, missed reasons, first-alert timeliness, false positives.
- `ML`: ranker deltas, calibration, top-N ranking quality, whether ranker helps the objective or only average return.
- `RL/Data`: rows collected, critic rows, dataset freshness, training cadence, data health.
- `Quality`: evaluator metrics for late entries, early exits, giveback, missed trends.
- `Verdict`: better / flat / worse relative to previous window, with the main blocker.

## Guardrails

- Do not claim learning progress from row count alone.
- Do not call ML progress positive if teacher top-gainer rate or capture ratio worsened materially.
- Do not blend WATCH quality with BUY quality.
- Use the latest complete local day for daily reports; use complete days only for multi-day comparisons.
- If a requested report period lacks enough complete days, say so and fall back to the largest complete window available.

## Commands

```powershell
.\pyembed\python.exe .\skills\bot-progress-report\scripts\build_progress_report.py --days 14
.\pyembed\python.exe .\skills\bot-progress-report\scripts\build_progress_report.py --days 7 --output .\.runtime\reports\bot_progress_7d.json
```
