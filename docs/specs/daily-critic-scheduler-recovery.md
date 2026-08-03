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

## Guardrails

This changes scheduling and measurement only. It does not change signals,
entry/exit gates, positions, watchlist membership, or Telegram BUY semantics.

## Verification

Focused tests cover missed-final catch-up outside the nominal window, final
precedence, independent goal scheduling, and state restoration from a restart
snapshot.
