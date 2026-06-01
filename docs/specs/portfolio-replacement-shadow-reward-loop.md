# Portfolio Replacement Shadow Reward Loop

Date: 2026-06-01
Status: research-only measurement; no portfolio behavior changes

## Problem

The bot has a unified 10-position portfolio cap and replacement/rotation logic, but the daily learning loop does not yet measure whether actual replacements improve scarce-slot usage.

## Goal

Create a daily/rolling shadow report for executed portfolio replacements:

- parse replacement exits from event logs;
- identify the incoming replacement candidate;
- match the candidate's later entry/exit outcome;
- compare replacement candidate PnL with the replaced position's exit PnL;
- surface whether rotation is helping, neutral, or hurting.

## Inputs

- `files/agent_events.jsonl` and `files/bot_events.jsonl`;
- optional top-gainer critic reports for watchlist top labels.

## Metrics

- replacement count;
- matched incoming candidate count;
- closed incoming candidate count;
- average replaced exit PnL;
- average incoming candidate PnL;
- average delta: incoming PnL - replaced exit PnL;
- win-rate of replacement delta;
- top/watchlist label coverage when available.
- segment table for causal debugging:
  - incoming candidate was watchlist top vs not;
  - replaced position was losing vs non-losing;
  - leader-score delta buckets.

## Non-goals

- No live portfolio behavior changes.
- No claim of true counterfactual hold-vs-replace PnL.
- No replacement threshold changes from this report alone.

## Acceptance Gate

A replacement policy may advance to deeper counterfactual replay only if the executed replacement shadow report shows repeated positive replacement delta with enough closed cases and no obvious false-candidate concentration.

## Interpretation Guardrail

This report is not a full hold-vs-replace counterfactual unless post-replacement candle paths are available. When candle paths are not available, the report must mark itself as an event-log shadow measurement and focus on:

- whether executed replacements were followed by better or worse incoming outcomes;
- which replacement segments look harmful;
- whether any segment is promising enough to justify candle-level counterfactual replay.
