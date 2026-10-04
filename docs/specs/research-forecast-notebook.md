# BTC / ETH / SOL research forecast notebook

2026-10-04. Research only; no production imports or BUY/SELL changes.

## Contract

The supplied v2 notebook is replaced by a standalone, output-cleared v3 notebook
generated from `files/research_forecast.py` with a checked-in builder. Preserve
the original Downloads file. Forecast cumulative log returns for horizons 1..15
minutes from the close of a completed one-minute candle, available at the next
minute boundary plus an explicit assumed publication delay. Historical klines
do not establish actual historical arrival times; this limitation is mandatory.

Use a single fixed UTC exclusive end for all symbols and hash cached input.
Require exactly the requested minute grid: reject missing bars, conflicting
duplicates, nonfinite/invalid OHLCV, incomplete responses, schema changes and
unclosed candles. Identical duplicates may be deduplicated. Never interpolate
future observations or compress gaps into adjacent model steps.

Split by wall-clock time into train / tuning / calibration / untouched test
(60/15/10/15 percent). Purge each origin unless its last label is available
strictly before the next split starts. Scaling and fitting use train only;
epoch/model selection uses tuning only; residual widths use calibration only.
Sequence context may cross a split boundary using already observed features.
ARIMA(1,1,0) uses causal origin-history centering/scaling and direct Yule-Walker
estimation on differences; this avoids tiny-variance iterative-MLE failures.
Do not describe a direct estimator as an iterative convergence pass. Seasonal
models still require genuine MLE convergence; failed paths are excluded wholesale.
All assets share UTC cutoffs and evaluation grids. Every model uses identical
tuning/test origins. Fixed ML models and rolling classical refits are explicitly
different preregistered forecasting policies, not an architecture-only contest.

Include persistence in selection, lock the chosen model before examining test,
and keep the original weights for the illustrative latest-snapshot forecast.
Never automatically refit or select using test. Native statistical intervals
are distinguished from empirical residual intervals. Calibrate each horizon in
log-return space using a finite-sample quantile; temporal dependence means no
distribution-free time-series coverage guarantee. Report coverage counts and
interval score, per-horizon errors and paired block-bootstrap uncertainty.
Use normalized return errors for cross-asset comparisons, not mean USDT MAE.
Insufficient independent time blocks yields UNKNOWN, and observed positive
point estimates alone do not prove stable improvement. Nonconverged or missing
optional models get explicit failure/unavailable status, no shortened test set.

## Scope, evidence and rollback (TH-01..TH-12)

This is a price forecast benchmark, not a portfolio backtest or profitability
claim. The requested 30-day history is a bounded experiment, not maximum exchange
history. No trading-policy hypothesis is promoted: maximum-period actual-bot
population replay, after-cost portfolio evidence, forward shadow/canary,
guardrails and immediate rollback remain prerequisites for any later adoption.
Rollback is deletion/reversion of the isolated research files. Preserve rejected
models, periods, counts, dependency versions, warnings and input hashes in runtime
output; do not commit market snapshots, fitted weights or generated results.

TH-01/05/10: counts, coverage, identical UTC windows, and UNKNOWN for missing
support. TH-02/11: forecast errors cannot become realized PnL or alpha.
TH-03/04: availability timestamps, purged UTC splits and unseen test labels.
TH-06/07: no production relaxation; maximum-period replay and canary required.
TH-08/09: failed hypotheses and research status remain explicit.
TH-12: spec, registered index entry, focused tests, staged harness, commit/push.

## Forecast-serving demo

Ship an opt-in FastAPI factory/CLI and Docker recipe in the isolated research
package. Require a separate API token, one worker, atomic snapshot publication,
90-second input freshness, first-horizon inference deadline and seven-day release
expiry. Publish actual inference issuance separately from assumed candle arrival.
Serve the preregistered persistence fallback when the selected model lacks
positive paired uncertainty evidence. Do not refit in inference. Health reports
train-reference feature drift and residual/coverage counts only after target
labels mature; missing labels are UNKNOWN. External TLS, logging retention,
load tests and prospective production qualification remain deployment work.
The Docker recipe is delivered unbuilt when Docker is unavailable.

## Verification

Test data/schema rejection, causal prefix invariance, label availability at
boundaries, train-only transformations, sequence continuity, tuning-only model
selection (including persistence), calibration/test isolation, finite-sample
quantiles, zero denominators, partial-model exclusion, cross-asset normalized
aggregation and deterministic notebook generation. Run an actual frozen public
Binance snapshot benchmark for available dependencies; optional heavy models are
reported as unexecuted unless they really complete. Record full Harness FAIL
independently from focused tests and the staged-change result.

### Completed verification, 2026-10-04

The frozen public snapshot covers [2026-09-04T00:00Z, 2026-10-04T00:00Z),
43,200 candles per asset. All 13 executable notebook cells completed; all four
core policies produced full paths, giving 12 model/asset results with 96 identical
test origins per asset over four complete UTC days. Tuning selected XGBoost for
BTC and Ridge for ETH/SOL; test never changed these choices. Boosting and Ridge
did not outperform persistence on this test; ARIMA's small positive point
differences do not establish stability. Every uncertainty verdict is UNKNOWN
because the preregistered ten-test-day support requirement is unmet.

Focused temporal/serving/serialization tests passed. An actual FastAPI smoke
using fresh public Binance input passed for all three symbols at the shared
2026-10-04T18:53Z input cutoff; the preregistered persistence fallback was served.
No future quality metric was fabricated: mature-label count was zero and coverage
was null. Runtime evidence stays under `.runtime/forecast_review/` and is not
staged. Docker was unavailable and the image is not claimed verified.

The repository full Harness independently returned FAIL TH-11 (stale replay
source hash in the portfolio artifact). The isolated staged-change profile is
PASS; it does not waive, repair or reinterpret the full-profile failure.
