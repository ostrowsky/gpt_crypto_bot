# V2 Shadow Explainability

Status: expedited shadow-only  
Last updated: 2026-05-17

## Purpose

Make tomorrow's v2 shadow signals explainable in seconds:

- why a symbol produced a shadow signal;
- why it did not;
- what exact live features and rule outcome were seen on the relevant closed bar.

## Problem

Logging only material transitions is insufficient for operator questions such as:

> "Why was there no v2 signal for CHZUSDT today?"

If a symbol stays in `noise/watch`, absence itself leaves no durable event trail unless we
record the decision snapshot.

## Scope

### In scope

- append-only per-bar v2 decision trace;
- one record per `(symbol, timeframe, closed_bar_ts)`;
- both positive and negative decisions;
- fast explainer script by symbol/date/timeframe;
- repo-local skill describing the fast lookup workflow.

### Out of scope

- retrospective model retraining;
- expensive full-history analysis for routine operator questions;
- production trading decisions.

## Decision Trace Contract

Each row must contain:

- `sym`
- `tf`
- `bar_ts`
- `observed_at`
- `state`
- `action`
- `confidence`
- `reason`
- `material_transition`
- `bootstrap`
- `previous_state`
- live feature snapshot

The log is deduplicated by `(symbol, timeframe, bar_ts)` so repeated worker scans do not
inflate the trace.

Bootstrap rows are retained for observability but excluded from operator-facing
material-signal counts because the first observation is not a real state transition.

## Fast Answer Contract

Given `symbol + optional timeframe + optional date`, the explainer must return:

1. latest decision rows;
2. whether any material shadow signal occurred;
3. latest no-signal / watch reason when there was no material signal;
4. feature snapshot used for that answer.

## Acceptance Criteria

1. Absence of a signal is explainable from the trace.
2. Positive and negative decisions use the same source of truth.
3. Lookup does not require scanning legacy logs or rerunning market analysis.
4. A repo-local skill documents the fast workflow.
5. Unit tests cover dedupe and no-signal explanation.

## Verification

- unit tests;
- one-shot worker run;
- `python files/explain_v2_shadow_signal.py --symbol BTCUSDT --json`

## Rollback / Safety

- append-only shadow telemetry only;
- no production action impact.
