# Daily Learning Progress Report

Status: shipped reporting/observability  
Last updated: 2026-05-20

## Purpose

Give the operator one daily 09:00 local-time answer to the question:

```text
Is the bot improving, standing still, or degrading against its target metrics?
```

The report must not claim learning merely because scripts ran. It separates:

- measurement coverage;
- top-mover capture and early capture;
- entry timing and exit monetization;
- blocked / missed winner pressure;
- whether feedback or model-training components actually changed anything;
- decisions awaiting operator approval.

## Schedule

Default schedule: `09:00 Europe/Budapest`, summarizing the latest completed day
from final reports.

## Inputs

- `top_gainer_critic_*_final.json`;
- `signal_quality_*_final.json`;
- `watchlist_top_gainer_goal_*_22h.json`;
- `.runtime/signal_quality_feedback.json`;
- `.runtime/rl_worker_status.json`.

## Guardrails

- Reporting only.
- No BUY/SELL changes.
- No automatic approval of hypotheses.
- If data is partial/stale, the report must say so clearly.
- Telegram delivery must be idempotent per daily slot, even if the worker restarts inside the delivery window.
- The 09:00 report is a fast aggregation layer. It must not run heavy research
  replays inline. If an optional research component cache is stale, use the
  cached artifact and mark that component as stale instead of blocking the daily
  report.
- A `watchlist_top=0` day is valid only when the final top-gainer critic for the
  target day exists. If the critic is missing, the denominator is unknown and the
  report must be `СТАТУС НЕПОЛНЫЙ`, not `ДЕНЬ НЕИНФОРМАТИВЕН`.
- Shadow re-entry reporting must not say that outcome data is absent merely
  because final alert confirmation produced zero alerts. When registered-watch
  counterfactual T+5 labels exist, the report must show that upstream cohort,
  keep it distinct from final alerts, and keep production re-entry disabled.

## Required Output

A concise Telegram-friendly report with:

- verdict: developing / flat / degrading;
- one-line main metric summary;
- where winners are lost;
- previous decisions and whether they helped;
- alerts;
- next operator actions.

## Delivery Idempotency

The worker must persist a sent marker for each `target_day::learning_progress` slot before Telegram delivery.

Expected behavior:

- If the same slot was already marked sent, skip Telegram delivery.
- Use an atomic lock/marker so two workers cannot send the same daily report concurrently.
- Keep report generation allowed for diagnostics, but prevent duplicate Telegram messages for the same slot.
