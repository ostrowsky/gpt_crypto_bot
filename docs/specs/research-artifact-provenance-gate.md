# Research Artifact Freshness and Provenance Gate

Date: 2026-08-03
Status: measurement safety

## Problem

The morning learning report previously used file modification time as the only
freshness signal for cached research replays. A recently copied or rewritten
artifact could therefore look current even when it was produced under an old
live policy or different research configuration.

## Required behavior

Decision-bearing research artifacts must record:

- builder identity and provenance schema;
- generation time and a bounded freshness budget;
- a hash of the non-secret current live-policy configuration;
- a hash of the research configuration;
- watermarks for the latest material input files.

The morning report validates the artifact against the current live policy and
the expected research configuration. Missing provenance, builder/config/policy
hash mismatch, an invalid timestamp, or an exceeded age budget marks the
component `stale`.

## Decision guardrail

A stale cached result may remain visible for diagnosis, with its prior status
preserved as `cached_status`, but it cannot emit a current recommendation to
relax BUY, SELL, blocker, or portfolio policy. It must instead request an
explicit rebuild. Heavy replay remains outside the bounded morning report.

The first covered artifacts are:

- observable tail selector;
- entry-admission shadow reward;
- blocked-winner causal reward;
- portfolio-replacement shadow reward.

This is reporting/research safety only. It does not change live trading logic.
