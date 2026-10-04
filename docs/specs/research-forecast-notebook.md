# BTC / ETH / SOL research forecast notebook

2026-10-04. Research only; no production imports or BUY/SELL changes.

## Contract

### Extended confirmation protocol, 2026-10-04

The 30-day exploratory run remains a rejected/underpowered historical result.
Per user constraint, freeze a compact 120-day window [2026-05-04, 2026-09-01) UTC,
excluding ALL September/October dates already inspected in the original run.
Use published monthly Binance spot archives with SHA256 CHECKSUM verification,
explicit microsecond-to-millisecond normalization, and the existing exact-grid
validation. Archive revisions/actual historical receipt remain limitations.
SARIMA and SARIMAX become default, mandatory benchmark participants. Seasonal
parameters and exogenous scaling fit on the last 2880 train minutes only; frozen
parameters filter observed history causally and forecast the next 15 minutes.
SARIMAX uses ONLY deterministic future calendar covariates (no future volume or
price-derived exogenous regressors). No refit on tune, calibration or test.
The seasonal period 60 remains a preregistered hypothesis, not assumed fact.
Use conditional least squares for multiplicative AR-only SARIMA(1,1,0)(1,0,0,60)
and the same model with calendar regression for SARIMAX, implemented with SciPy.
Require successful finite optimizer termination; do not call it statsmodels MLE.
Fixed-parameter AR-only residual recurrences permit causal bounded-memory
forecasts without repeated 61-state filtering over the full minute history.

Report h15 return MAE and direction as distinct tasks. For direction include
always-up, train-majority and origin momentum baselines, correct/N, abstentions,
base rate, balanced accuracy and paired day-block confidence intervals against
the TRAIN-majority baseline. Persistence abstains on direction. Predeclare a
familywise 95% Bonferroni interval across all nonbaseline models/assets for
MAE/direction and report three equal chronological test subperiods plus three
expanding train-only folds. A CI crossing zero is INCONCLUSIVE despite adequate
data, not UNKNOWN for short history. Negative CI is NO_IMPROVEMENT; positive
corrected CI is SUPPORTED_DIAGNOSTIC. Do not collect data until significance is
obtained, change the cutoff after results, or claim a deployment approval.
Use 60 train / 15 tune / 15 calibration / 30 test days, at least 30 complete test
days. Older downloaded archives are excluded from the experiment. Training
may be subsampled on a fixed 15-minute UTC grid for bounded compute, with the
same causal labels and explicitly recorded training count for all supervised
models. No bot trading behavior is relaxed.

### Visual evidence and interactive comparison

Every completed method/asset has real-price versus forecast plots from the last
expanding fold INSIDE train (out-of-fold, fitted before each validation period),
final test, and the latest frozen snapshot transitioning into the full h1..h15
path and empirical interval. Display comparative MAE, RMSE, direction/base rates,
coverage counts, interval score and familywise uncertainty plus per-asset error
bars and three chronological subperiods. Lowest observed error is a descriptive
winner; statistical superiority requires a positive corrected interval.

An ipywidgets panel creates a separately timestamped research forecast for every
completed method, with immutable paths and a unique snapshot ID. Manual refresh
fetches new closed candles and overlays actuals at exact target-close timestamps
without retraining or rewriting the original prediction. Missing/future targets
remain pending, with mature-point denominators. Refresh is user-triggered, no
background infinite polling. UI state may be reinitialized by Run All; export
immutable forecast records to runtime JSON for reproducible subsequent comparison.
Do not describe old calibrated weights as a current approved production release.
Production API expiry/fallback gates remain unchanged.

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
(50/12.5/12.5/25 percent). Purge each origin unless its last label is available
strictly before the next split starts. Scaling and fitting use train only;
epoch/model selection uses tuning only; residual widths use calibration only.
Sequence context may cross a split boundary using already observed features.
ARIMA(1,1,0) uses direct scale-invariant Yule-Walker estimation on causal
origin-history differences; this avoids tiny-variance iterative-MLE failures.
Do not describe a direct estimator as an iterative convergence pass. Seasonal
CSS seasonal models require successful finite optimizer termination; failed
paths are excluded wholesale. No estimator failure is relabeled as success.
All assets share UTC cutoffs and evaluation grids. Every model uses identical
tuning/test origins. Fixed ML/seasonal models and rolling ARIMA fits are explicitly
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
claim. The requested compact 120-day history is bounded, not maximum exchange
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

### Original 30-day exploratory verification, superseded on 2026-10-04

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

### Extended verification, 2026-10-04

All six default methods completed on all three assets: 172,800 minute bars per
asset in the requested 120-day experiment, 720 identical test origins across 30
complete UTC days (2026-08-02..2026-08-31). Tune locked Ridge for BTC and ARIMA
for ETH/SOL. Persistence has lowest BTC MAE; ARIMA has lowest ETH MAE (0.048%
point reduction); SARIMA has lowest SOL MAE (0.052%). Neither tiny advantage
is statistically supported. Corrected intervals establish negative improvement
for XGBoost on BTC/ETH and SARIMAX on BTC/SOL; other nonbaseline differences
are INCONCLUSIVE. There are no short-history UNKNOWN verdicts in final test.
Always-up direction baseline is 373/720, 373/720, 374/720 for BTC/ETH/SOL;
every tested forecasting model is below that baseline in point accuracy.
All 15 executable notebook cells completed, including 54 expanding-fold traces,
four saved figures (per-method train/test/inference and comparative panels),
and widget creation. Final source/output notebook parity passed; the extra
ipywidgets dependency guard was separately reexecuted after the main run.
Final serving review also required live history >= classical_window+context
(three days for the default 2880-bar ARIMA window). This serving-only correction
does not change forecast algorithms or historical benchmark results. Declaration
and config cells were reexecuted; notebook metadata retains the original benchmark
source SHA separately from the delivered engine SHA and records this validation.

29 focused tests passed, including causal/frozen seasonal parameters, archive
checksums/microsecond conversion and immutable prospective paths. An actual BTC
prospective smoke fixed all six paths at 2026-10-04T19:35:27Z and fetched new
closed candles at 19:37:35Z: 2/15 target points were mature for each method,
with unchanged original forecasts and honest PARTIAL status. After all targets
matured, a real refresh at 19:55:42Z verified 15/15 points and COMPLETE for all
six unchanged paths. The delivered runtime notebook includes an independently
executed offline replay appendix containing that immutable public forecast and
the subsequently collected actual candles. It includes no fitted weights and
does not turn a single prospective path into evidence of stable superiority.
Input/output evidence remains runtime-only. Ignore forecast_demo_artifacts
to keep notebook downloads, snapshots and reports out of commits.
