# Observable Tail Selector Replay

Date: 2026-06-01
Status: research-only observable selector replay; no production SELL changes

## Problem

Oracle early-exit gating reduces harm but relies on evaluator hindsight labels. Production needs a selector that uses only features available at or before SELL.

## Hypothesis

A small observable rule grid over exit reason, mode, current PnL, MFE, and giveback can identify cases where partial trailing-tail retention has positive edge without using hindsight `early_exits` labels.

## Observable Features

Allowed features:

- exit reason bucket;
- entry mode;
- source;
- current realized PnL at SELL;
- MFE at SELL;
- giveback at SELL;
- entry/exit timestamps for bars-held approximation.

Disallowed production features:

- evaluator bucket such as `early_exits`;
- future favorable move;
- trend object / final top mover outcome;
- post-exit candle labels.

## Replay Design

Inputs:

- same labeled tail rows as `files/replay_trailing_tail_after_partial_exit.py`;
- candidate tail policy: `tail50_h10_ema20_cap150`;
- chronological train/test split by day.

Candidate selectors are simple rules such as:

- weak-signal only;
- weak-signal + minimum MFE/giveback;
- non-EMA exits + positive PnL + giveback;
- momentum modes + high MFE.

Each candidate is scored on train and test for:

- average / median delta;
- worse-rate;
- allowed-rate;
- false-positive allowed-rate, diagnostic only;
- early-exit delta, diagnostic only.

## Acceptance Gate

A selector may advance to shadow-only live scoring only if the test slice has:

- positive average and median delta;
- worse-rate <= 30%;
- allowed-rate >= 5%;
- false-positive allowed-rate <= 10%;
- at least 10 test rows or explicitly marked low confidence.

No production SELL change is allowed from this replay alone.
