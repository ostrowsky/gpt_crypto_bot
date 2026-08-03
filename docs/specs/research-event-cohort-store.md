# Incremental Research Event Cohort Store

Date: 2026-08-03
Status: research pipeline

## Problem

Entry-admission, blocker-reward, and portfolio-replacement reports repeatedly
scanned `bot_events.jsonl` and `agent_events.jsonl`. The files now exceed 746 MB
and contain more than two million blocked events. Four concurrent refreshes did
not finish within five minutes, leaving decision artifacts stale.

## Required behavior

The research pipeline maintains a runtime SQLite cohort store with per-source
byte offsets. Each sync reads only appended bytes after the initial maximum-
period backfill.

The store persists:

- compact blocked cohorts keyed by source/day/symbol/reason, including count
  and the first causal event;
- normalized entry and exit events required by admission and replacement
  counterfactuals;
- source size, offset, modification time, and sync timestamp.

Offset advancement and cohort updates occur in the same transaction. Re-running
a completed sync is idempotent. If a source log is truncated, rows from that
source are removed and rebuilt without duplicating the other source.

## Consumers

- entry-admission shadow reward uses compact blocker counts and indexed entries;
- blocked-winner causal reward uses the same compact cohorts;
- portfolio-replacement shadow reward uses indexed entry/exit events.

Generated research artifacts include the cohort database in their input
watermarks. The database is runtime state and must not be committed.

## Guardrails

- The initial population still covers the maximum available event history.
- No live trading behavior changes.
- The bounded morning report does not trigger a maximum-period rebuild; stale
  artifacts remain diagnostic until an explicit research refresh completes.

## 15-minute blocker intervals

Schema v2 also stores compact blocker intervals keyed by source, symbol,
normalized reason, and 15-minute UTC bucket. Each row preserves exact first and
last event timestamps, first observed price/context, and count. This supports
causal trend-episode attribution without rescanning the raw event archive or
using an entire day of blockers as if they happened before the episode peak.

The schema migration rebuilds the runtime-only index once. A normal append sync
remains incremental; source truncation triggers the same safe full rebuild.
