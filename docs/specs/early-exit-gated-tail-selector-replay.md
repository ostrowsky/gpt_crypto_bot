# Early-Exit Gated Tail Selector Replay

Date: 2026-06-01
Status: research-only selector replay; no production SELL changes

## Problem

Trailing-tail replay showed that partial tail retention has positive edge on early exits, but aggregate performance is unsafe because false-positive buys are harmed. The bot needs a selector that decides when tail retention is allowed.

## Hypothesis

A gated selector that allows tail retention only for early-exit-like trades can preserve most tail upside while avoiding false-positive tail drag.

## Non-goals

- Do not change live SELL logic.
- Do not claim hindsight `early_exits` labels are production-observable.
- Do not promote any selector without a later observable-feature shadow selector.
- Do not use this replay to relax false-positive cleanup exits.

## Replay Design

Inputs:

- `signal_quality_*_final.json` exit rows;
- cached OHLCV candles from `.runtime/signal_quality_cache`;
- trailing-tail policy labels from `files/replay_trailing_tail_after_partial_exit.py`.

Policies:

- baseline: current full SELL;
- trailing-tail candidate: `tail50_h10_ema20_cap150`;
- gated policies:
  - `gate_oracle_early_exit`: allow tail only for evaluator `early_exits` bucket;
  - `gate_early_weak_signal`: allow tail only for `early_exits` and weak/divergence exit reason;
  - `gate_early_non_ema_break`: allow tail for `early_exits` except EMA-break exits;
  - `gate_weak_signal_only`: observable-ish weak/divergence reason without hindsight bucket, included as a cautionary baseline.

For a rejected case, the gated policy equals baseline SELL. For an allowed case, it uses the trailing-tail policy result.

## Acceptance Gate

A gated selector may advance only to observable-feature shadow modeling if:

- aggregate average and median delta are positive;
- false-positive allowed-rate is near zero or false-positive delta is not materially negative;
- early-exit slice improves;
- allowed-rate is not trivially zero;
- coverage remains sufficient.

The `oracle_early_exit` selector is an upper-bound diagnostic, not a live production selector.
