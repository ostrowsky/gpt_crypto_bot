# Blocked-Winner Focus Audit

Status: shipped diagnostic tooling  
Owner: Codex  
Date: 2026-05-20

## Problem

When the operator asks why a specific top mover was or was not bought, a raw
daily count is not enough. The answer must be available without re-reading the
entire event history.

## Goal

Produce a compact per-symbol audit from the latest top-gainer critic report for
symbols the operator is watching, such as STRK, TIA, AR, and RENDER.

## Requirements

- Read the latest `top_gainer_critic_*` report.
- For each requested symbol, show:
  - bought/blocked/missing status;
  - first block time/reason/price;
  - latest block reason;
  - first BUY time/mode/price if present;
  - exit PnL/efficiency/giveback if present;
  - dominant blocker and reason-count distribution.
- Do not change live trading behavior.
- Write JSON/TXT artifacts when requested so the result can be reused in daily
  reports or Telegram summaries.

## Acceptance

- The audit answers a focused “why STRK/TIA/AR/RENDER?” question from one
  precomputed report.
- Missing symbols are marked explicitly instead of silently ignored.
- Unit coverage validates bought, blocked, and absent cases.

