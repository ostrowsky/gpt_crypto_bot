---
name: signal-quality-evaluator
description: Post-factum quality evaluator for this crypto trend bot's BUY/SELL signals. Use when Codex must assess whether existing crypto bot signals entered early or late, exited early or late, captured enough of a sustained trend, missed top-mover trends, or produced false-positive BUYs; supports daily, rolling multi-day, symbol-specific, timeframe-specific, bot/agent-specific analysis and produces metrics for critic/RL optimization without generating new signals or training models directly.
---

# Signal Quality Evaluator

## Purpose

Evaluate signals after they happened. Do not generate BUY/SELL signals, do not change production gates, and do not train models directly.

Use this skill to answer:
- Did the bot enter near the start of a sustained uptrend or late?
- Did it exit near the end of the trend, too early, or after giving back profit?
- How much of the trend did it catch?
- Which sustained trends were missed?
- Which BUYs were false positives?

This complements `files/replay_backtest.py` and `files/offline_backtest.py`; it does not replace them.

## Quick Start

Automatic run:
- `files/rl_headless_worker.py` runs this evaluator once per local day for the previous day.
- Default schedule is `00:15 Europe/Budapest`, configurable in `files/config.py` via `SIGNAL_QUALITY_EVALUATOR_*`.
- Reports are written to `.runtime/reports/signal_quality_YYYY-MM-DD_final.json` and `.runtime/reports/signal_quality_YYYY-MM-DD_final.txt`.
- Telegram summary is sent when `SIGNAL_QUALITY_EVALUATOR_TELEGRAM_REPORTS_ENABLED=True`.

Run the bundled evaluator from the repo root:

```powershell
python .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --days 7
```

With embedded Python:

```powershell
.\pyembed\python.exe .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --days 7
```

Common scopes:

```powershell
.\pyembed\python.exe .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --date 2026-05-05
.\pyembed\python.exe .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --days 7 --symbol ICPUSDT
.\pyembed\python.exe .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --days 7 --tf 15m
.\pyembed\python.exe .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --days 7 --source agent
.\pyembed\python.exe .\skills\signal-quality-evaluator\scripts\evaluate_signals.py --days 7 --json --output .\.runtime\reports\signal_quality_7d.json
```

## Workflow

1. Read the objective in `SCOUT_OPTIMIZATION_SPEC.md`: early capture of same-day top movers with a unified 10-position portfolio.
2. Run `scripts/evaluate_signals.py` for the requested window/symbol/timeframe/source.
3. Interpret metrics by error class, not as one blended score:
   - `late_entries`: entry was too close to the trend peak.
   - `early_exits`: trend continued after exit.
   - `late_exits`: exit gave back too much of maximum favorable excursion.
   - `false_positive_buys`: BUY was not followed by a sustained trend.
   - `missed_trends`: sustained trend existed but no BUY matched it.
4. Use the output to form hypotheses for backtests. Do not apply production changes directly from evaluator output.
5. Validate any proposed rule change with `files/replay_backtest.py` before enabling it.

## Core Metrics

- `capture_ratio_at_entry`: remaining trend move after entry divided by full detected trend move. Higher means earlier entry; below `0.35` is late by default.
- `realized_capture_ratio`: realized BUY-to-SELL move divided by full detected trend move.
- `mfe_capture_ratio`: max favorable move during the trade divided by full detected trend move.
- `exit_efficiency`: realized PnL divided by maximum favorable excursion before exit.
- `giveback_pct`: max favorable excursion minus realized PnL.
- `miss_rate`: missed detected trends divided by detected trends.
- `false_positive_rate`: false-positive BUYs divided by BUYs.
- `top_mover_caught_trends` / `top_mover_missed_trends`: detected trends on final top movers in the loaded universe.

## Parameters

Useful defaults:
- `--trend-min-pct 3.0`: post-factum sustained uptrend requires at least +3%.
- `--trend-min-bars 4`: trend must last at least 4 bars from start to peak.
- `--reversal-pct 1.2`: trend episode closes after a 1.2% reversal from peak.
- `--late-entry-capture-max 0.35`: entries with less than 35% remaining trend are late.
- `--early-exit-after-pct 1.0`: exit is early if price runs at least 1% further after exit inside the same trend.
- `--late-exit-giveback-pct 1.0`: exit is late if giveback is at least 1%.
- `--false-positive-max-fav-pct 1.0`: BUY is false positive if no detected trend matched it and future MFE stays below 1%.

## Output Discipline

When reporting results to the user:
- Lead with concrete counts and rates.
- Separate late entry, early exit, late exit, false-positive BUY, and missed trend.
- Mention whether `--symbol` or `--max-symbols` made top-mover ranking partial.
- Treat evaluator output as diagnosis. Require replay/backtest confirmation for production changes.

## Architecture Notes

For component replacement and simplification guidance, read `references/architecture-fit.md`.
