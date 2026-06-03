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

## Verdict Confidence Calibration

Daily verdicts must not overstate directionality when the evidence denominator is weak.

The report should attach `confidence` and `confidence_reason` to the verdict and use them in the human-readable summary.

Low/medium-confidence conditions include:

- latest daily `watchlist_top_count` below 5;
- short or sparse rolling windows;
- metric-affecting incomplete coverage.

When early-capture drops versus the previous rolling window, the report may say `ДЕГРАДИРУЕТ` only when confidence is high. Otherwise it should say `УХУДШИЛСЯ ПО EARLY-CAPTURE` and explicitly ask for replay checks rather than implying automatic production rollback.

## Bounded Daily Report Composition

The daily learning report must be fast and reliable. It may summarize heavy shadow/replay components, but it should not block the morning report by recomputing long sweeps inline.

Expected behavior:

- Use fresh cached shadow/replay summaries when available.
- Recompute a shadow block only when the cache is missing or stale.
- Keep full replay/backtest sweeps as explicit verification jobs, not hidden work inside the Telegram report.
