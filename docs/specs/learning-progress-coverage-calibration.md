# Learning Progress Coverage Calibration

Date: 2026-05-31
Status: reporting/measurement-only

## Problem

Daily learning reports mark any `signal_quality` partial candle coverage as a serious incomplete-data status. Recent reports showed `186/206` coverage, but coverage triage proved all 20 missing series belonged to Binance `BREAK` symbols.

That is materially different from missing active market data. The old wording made the report look less reliable than it actually was.

## Goal

Use coverage triage to distinguish:

- metric-affecting missing active candles;
- safe partial coverage caused only by inactive/BREAK symbols.

When coverage is safe partial, the learning report should keep the caveat but should not override the main verdict or emit a serious alert.

## Guardrails

- Measurement/reporting only.
- Do not change trading logic.
- If triage is unavailable or missing symbols are active/unknown, keep conservative partial status.
- Continue separating target metric quality from diagnostic trend-episode miss-rate.

## Expected Behavior

If latest signal quality report is partial but triage assessment is `partial_safe_inactive_symbols_only`, daily report may say the bot is developing on the target metric while preserving a warning-level coverage caveat.
