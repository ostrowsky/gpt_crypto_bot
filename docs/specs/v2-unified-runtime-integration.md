# V2 Unified Runtime Integration

Status: planned  
Last updated: 2026-05-17

## Purpose

Guarantee that any future long-running v2 service starts, stops, and reports health
through the same operator workflow as the current production bot stack.

The operator must not need one command for the legacy stack and a second hidden command
for v2.

## Problem

The current production stack already has one explicit operator entrypoint:

```text
restart_full_stack.bat
```

That helper starts and validates:

- bot;
- RL worker;
- market agent.

The v2 architecture is currently research-only and has no background process yet. Once
it gains a live-shadow or data-collection worker, starting it separately would create:

- split operational truth;
- hidden failure modes;
- accidental release drift;
- uncertainty over which stack is actually running.

## Runtime Contract

As soon as v2 introduces its first long-running background worker, that worker must ship
with:

1. a dedicated starter, for example:
   - `start_v2_history_collector_bg.ps1`
   - or `start_v2_shadow_worker_bg.ps1`;
2. a dedicated status checker:
   - `v2_history_collector_status.ps1`
   - or `v2_shadow_worker_status.ps1`;
3. restart/health integration in:
   - `restart_full_stack.bat`;
4. release-worktree compatibility:
   - it must start from the same clean release root as the rest of the bot stack.

## Unified Startup Requirement

The same one-command operator flow must launch all mandatory services:

```text
restart_full_stack.bat
  -> bot
  -> RL worker
  -> market agent
  -> mandatory v2 background services
```

The helper must fail non-zero when a mandatory v2 service is expected but unhealthy.

## What Does Not Need Runtime Integration Yet

The following do **not** require BAT integration while they remain offline-only:

- pure `files/v2/` libraries;
- one-shot dataset builders;
- one-shot audit reports;
- offline training scripts;
- replay-only experiments.

## Promotion Gate

No v2 capability may be called "live-shadow operational" until:

1. it has a background launcher;
2. it has a status checker;
3. it is included in `restart_full_stack.bat`;
4. it has been verified from a clean release worktree;
5. the operator can identify from one command which version is running.

## Acceptance Criteria

1. Future v2 worker specs explicitly state whether the worker is offline-only or runtime.
2. Every runtime v2 worker includes launcher + status + unified restart integration.
3. No separate manual BAT becomes the canonical launch path for v2.
4. `restart_full_stack.bat` remains the single operator entrypoint.

## Rollback / Safety

- planned governance only today;
- no current runtime behavior changes;
- when implemented later, v2 services must remain individually disableable by config or
  explicit inclusion policy.

