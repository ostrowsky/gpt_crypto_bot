# Daily Critic Scheduler Recovery

Date: 2026-08-03
Status: shipped measurement reliability fix

## Problem

The 22:00 watchlist-goal report and the midday/final top-gainer critic shared
one asynchronous loop. A goal job that ran past midnight blocked the 00:00
critic window. Once the first 15 minutes passed, the missing final artifact was
never due again, and a restart erased the in-memory slot state.

## Behavior

- Watchlist goal and top-gainer critic run in independent worker tasks.
- A missing final critic within the configurable seven-day recovery window
  remains due at every later check until `top_gainer_critic_<day>_final.json`
  exists. Yesterday is tried first, then older gaps.
- A due final takes precedence over the current midday critic.
- Artifact existence makes catch-up persistent across restarts.
- Last successful critic and goal slot/counters are restored from worker status
  so operational health does not reset on restart.
- Failures remain visible in worker status and retry on the bounded scheduler
  interval; no missing artifact is silently converted to a zero denominator.
- Event JSONL inputs are consumed line by line. Daily critic memory must depend
  on the selected day's events, not on the total multi-month log size; the
  worker must not materialize complete `bot_events.jsonl` or
  `agent_events.jsonl` files.
- Before JSON decoding, event lines are filtered by every UTC calendar date
  touched by the requested local-day window. The scan makes no ordering
  assumption, so late-appended records remain eligible while unrelated months
  avoid JSON parsing cost.
- Worker heartbeat summaries and ranker row counters likewise scan critic and
  ML JSONL inputs incrementally. Routine status writes must remain bounded in
  memory as those datasets grow.

## Guardrails

This changes scheduling and measurement only. It does not change signals,
entry/exit gates, positions, watchlist membership, or Telegram BUY semantics.

## Verification

Focused tests cover missed-final catch-up outside the nominal window, final
precedence, independent goal scheduling, and state restoration from a restart
snapshot. A regression test also rejects whole-file event-log reads while
preserving malformed-line and non-object filtering. Status aggregation and row
count tests reject whole-file reads and preserve the existing counters.
