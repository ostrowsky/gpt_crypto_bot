# Protected Weak-Only Exit Candle Replay

Date: 2026-05-29
Status: research-only, replay gate required before any live SELL change

## Problem

The previous opportunity audit suggested that delaying some exits after favorable movement could improve realized PnL. The first causal candle-path replay variant, `protected_trailing_exit`, protected a broad set of exit reasons including WEAK, EMA, ATR trail, and stop-like exits.

Focused replay rejected that broad version: it reduced trades and worsened PnL/win-rate versus baseline. The likely cause is over-protection: it held positions after structural failures that should not be delayed.

## Hypothesis

A narrower variant may help only when the first exit is a soft `WEAK:` momentum warning, while preserving hard/structural exits:

- protect only `WEAK:` exits;
- do not protect EMA breaks;
- do not protect ATR trail stops;
- do not protect stop-like exits;
- require positive current PnL;
- require meaningful MFE before protection;
- cap the extra hold window.

## Variant

Replay variant: `protected_weak_only`.

This variant is research-only and must not affect live trading unless it beats current baseline on candle-path replay.

## Acceptance Gate

Eligible for further review only if, versus baseline on focused and broad replay windows:

- total PnL is not worse;
- average PnL is not worse;
- win-rate is not materially worse;
- exit efficiency improves;
- giveback does not increase materially;
- no increase in late-loss / round-trip failure cases.

If it fails the gate, keep it only as a documented negative experiment.
