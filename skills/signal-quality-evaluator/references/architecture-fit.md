# Architecture Fit

## Role

`signal-quality-evaluator` is a measurement skill. It should sit after signal generation:

`signals -> executed/open/closed events -> evaluator -> metrics -> hypothesis -> replay_backtest -> production change`

It must not sit inside the live decision path.

## What It Can Replace

- It can partially replace ad hoc manual "why was there no signal?" investigations when the question is post-factum quality.
- It can absorb some daily top-gainer critic diagnostics once its output covers the exact Telegram report contract.
- It can replace manual spreadsheet-style calculations for entry lateness, exit efficiency, trend capture, false positives, and misses.

## What It Should Not Replace

- Do not replace the scout/monitor. The scout is a live candidate generator; this skill is retrospective.
- Do not replace `replay_backtest.py`. Replay is the gate for production rule changes; this skill creates hypotheses and reward/penalty metrics.
- Do not replace the market agent. The agent owns live portfolio actions and cooldown behavior.
- Do not directly write model labels or change model weights. Export metrics first, then let the RL/ML pipeline consume them explicitly.

## Simplification Opportunities

1. Merge overlapping critic metrics into this skill once the daily report format is preserved.
2. Keep `top_gainer_critic.py` as the scheduled Telegram reporter until this skill emits the same daily final/midday sections.
3. Use this skill as the shared metric source for:
   - daily critic report,
   - RL reward shaping,
   - "why no signal" postmortems,
   - threshold optimization reports.
4. Keep live signal generation and post-factum judgment separate. This lowers feedback-loop risk.

## Additional Useful Skills

- `why-no-signal-explainer`: reads blocked events and current candles, explains the exact blocker chain for a symbol.
- `trend-start-hypothesis-auditor`: uses evaluator output to propose one narrow threshold/cooldown/exit hypothesis at a time.
- `exit-quality-auditor`: focuses only on `exit_efficiency`, `giveback_pct`, post-exit runup, and weak-exit reason quality.
- `portfolio-exposure-auditor`: verifies unified 10-symbol exposure, duplicate main/agent positions, cluster concentration, and replacement quality.

Recommended order:
1. Keep this evaluator as the canonical retrospective metric source.
2. Add `why-no-signal-explainer` if user-facing explanations still require log archaeology.
3. Add `exit-quality-auditor` only after enough evaluator rows show systematic early/late exit harm.
