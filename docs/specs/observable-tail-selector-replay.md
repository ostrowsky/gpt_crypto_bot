# Observable Tail Selector Replay

Date: 2026-06-01
Last revalidated: 2026-07-13
Status: `non_ema_mfe150` retired after failed live-forward gate; no production SELL changes

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

- average delta across the full policy slice;
- average / median delta among selected cases;
- worse-rate among selected cases;
- allowed-rate;
- false-positive allowed-rate, diagnostic only;
- early-exit delta, diagnostic only.

## Acceptance Gate

A selector may advance to shadow-only live scoring only if the test slice has:

- full-slice average delta `> 0.10%`;
- selected-case average delta `> 0.10%` and median delta `> 0`;
- selected-case worse-rate <= 30%;
- allowed-rate >= 5%;
- false-positive allowed-rate <= 10%;
- at least 10 test rows or explicitly marked low confidence.

No production SELL change is allowed from this replay alone.

## Maximum-Period Revalidation (2026-07-13)

The replay was rerun over every available final signal-quality report and its
cached candle coverage:

- exit-quality reports loaded: `58`;
- selector rows: `668` across `55` labeled days;
- chronological split: `468` train rows and `200` test rows.

The initial full-slice ranking leader was `exclude_ema_and_false_cleanup`:

- observable rule: non-EMA exit, MFE at SELL `>= 1.0%`, and current PnL
  `> -0.5%`;
- initial full-slice test average delta: `+0.2888%`;
- test allowed-rate: `27.5%`;
- test false-positive allowed-rate: `0.0%`.

The corrected replay gate uses selected-case median and worse-rate rather than
diluting them with zero deltas from unselected rows. On the rerun:

- `exclude_ema_and_false_cleanup` was rejected because its selected-case
  worse-rate was `30.91%`, above the `30%` limit;
- `non_ema_mfe150` passed with full-slice average delta `+0.2792%`, selected-case
  median delta `+0.6275%`, selected-case worse-rate `24.39%`, allowed-rate
  `20.5%`, and false-positive allowed-rate `0.0%`.

Decision: advance `non_ema_mfe150` to shadow telemetry only. Its observable rule
is a non-EMA exit with MFE at SELL `>= 1.5%`. It may register a
hypothetical 50% retained tail and collect forward labels at the configured
horizons, but it must not delay, suppress, or resize a production SELL. Unknown
selector names fail closed. Production adoption still requires independent live
labels followed by a fee/slippage candle-path replay.

## Live-Forward Decision (2026-08-03)

Independent telemetry accumulated 147 candidates, including 100 mature T+10
labels. Retaining the hypothetical half-position produced average incremental
value of `-0.1239%`, median `-0.2881%`, and a `59%` worse rate before costs.
This fails every promotion gate and reverses the replay result. The
`non_ema_mfe150` collector is therefore disabled by default and must not be
replaced by the previous selector without a new maximum-period, held-out
replay. Production SELL behavior remains unchanged.

## Rollback

Set `OBSERVABLE_TAIL_SHADOW_ENABLED = False` to stop new shadow candidates
without changing production exits. Restoring
`OBSERVABLE_TAIL_SHADOW_SELECTOR = "non_ema_positive_giveback"` returns to the
previous shadow cohort definition.
