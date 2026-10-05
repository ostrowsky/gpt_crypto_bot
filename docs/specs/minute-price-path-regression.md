# Minute price paths: CatBoost and compact DeepLOB regression

Registered 2026-10-05, research-only. TH-01..TH-12 apply. User needs an actual
forecast price path alongside past history and later realized prices; the prior
categorical report cannot infer movement magnitude from class probabilities.

## Population, information timing and targets

Reuse ALL pinned March/April BTC/ETH/SOL perpetual books, the established causal
features, 100-state sequences, source hashes and global chronological clock cuts
from `minute-direction-catboost-deeplob.md`. Verify source/book hashes first.
Predict five exact log-mid-price displacements from ONE issuance origin to
+1,+2,+3,+4,+5 minutes. Labels cross no book gap/reset. Keep existing five-minute
purge plus execution delay, train five-minute sampling, validation/calibration/
TEST minute sampling. No future prices enter features or sequence normalization.
All five targets, methods and assets use the same eligible scored cohort;
inference includes unknown futures without selecting on their outcomes.

The historical TEST has already been inspected for classification and chart
illustrations. It is chronological out of training, but this new price-path
experiment is a retrospective diagnostic on an already disclosed period, NOT
new independent holdout or prospective evidence. Register all parameters before
regression results; do not tune after price TEST or select attractive examples.

## Fixed models and calibration

CatBoostRegressor multi-output MultiRMSE: five direct targets, 600 maximum trees,
depth 6, learning rate .05, L2=10, seed42, two CPU threads, validation early stop60.
Normalize targets by their TRAIN-only mean and standard deviation per horizon.
Ridge alpha10 with TRAIN-only StandardScaler is the visible simple price control.
An internal zero-return baseline diagnoses absolute-price tracking; it is not
displayed as a forecast curve. Price levels alone can create illusory accuracy.

Compact DeepLOB regression reuses the spatial/temporal CNN -> three Inception
branches -> LSTM32 encoder with five scalar heads. This is a new untrained
regression network, not a converted classification probability or pretrained
classification output. Channels16, sequence100, Adam lr.001/weight_decay.0001,
MSE on TRAIN-standardized displacements, batch128, max8 epochs, validation
patience2, deterministic seed42, two CPU threads; no checkpoint selection on TEST.

Fit horizon-wise absolute log-return residual quantiles at nominal90% on the
separate CALIBRATION cohort only. Render exp-transformed price bands, and publish
empirical marginal coverage/width on TEST. This temporal split residual interval
is a calibration diagnostic, not an IID coverage guarantee or joint path band.
All five point forecasts are issued at once: P_hat(t+h)=P(t)*exp(r_hat_h).
No stochastic noise, future anchor, smoothing toward realized prices or
probability-to-price heuristic is allowed. A legitimately near-flat expected
path must remain near-flat, rather than be distorted for appearance.

## Evidence, graphics and production boundary

Report pooled/per-asset MAE and RMSE of displacement (basis points), MAE/RMSE
price USDT per asset, exact nonzero direction correct/N, neutral band +/-2bp
class correct/N, internal zero-return loss and nominal90% interval covered/N.
Publish all five horizons and matched daily losses. Short common TEST and reused
period do not support robust model-superiority or trading profitability claims.
No live trading gates or service are changed; no portfolio adoption is approved.

Freeze native CatBoost, Ridge and safetensors states, registered source/data
hashes, calibration quantiles, immutable predictions and exact target clocks.
Reload all native models on fixed evenly spaced inference rows and compare to
saved predictions. Verify zero changes to the old classifier result/predictions.
Runtime reports/weights/data are never committed.

Graphs: use the already registered common asset/origin examples from the frozen
direction report, selected using past availability and fixed clock anchors.
For each of Ridge, CatBoost regression and DeepLOB regression show SAME past20
minutes, issuance cutoff, forecast prices at +1..+5 minutes, optional marginal
90% intervals and realized next five minutes. Point markers identify model
outputs; connecting segments are visualization, not intermediate forecasts.
Missing facts stay gaps/UNKNOWN, never forward-filled; all models receive the
same example clock and scale. HTML supports a synchronized future-only zoom and
optional intervals (hidden initially for legibility). Offline asset/origin selectors plus PNG per
method/asset and auditable graph JSON. Link from the existing comparison report.

## Verification and rollback

Focused tests cover five target clocks and gap invalidation; split/purge and
TRAIN-only scaling; future mutation/prefix equivalence; scalar-head architecture
and native state round trip; independent calibration; fair metric denominators;
single-origin price construction and genuine HTML/PNG generation. Run relevant
tests and diff check, review intended staged sources/spec/tests, staged Truth
Harness, commit and push. Full Truth Harness FAIL TH-11 (canonical portfolio
replay source-hash mismatch) stays visible, no waiver or live adoption.
Rollback: remove offline price-path scripts/view; no production behavior changes.

## Completed diagnostic, 2026-10-05

Used the entire existing pinned archive and the fixed common chronological cuts:
TRAIN8657 / validation13352 / calibration11612 / scored TEST26224; inference26424.
CatBoost validation selected nine trees; DeepLOB selected epoch1 and stopped
after epoch3. Native state reload matched 24 predetermined inference origins for
all three models; the original classifier result and predictions stayed unchanged.
Retain immutable runtime predictions, weights, hashes and all losing outcomes.

All three regressors had higher displacement MAE than internal zero return on
each of the five horizons. CatBoost's slightly lower RMSE at +4/+5 minutes does
not establish robust superiority; this disclosed short TEST is diagnostic only.
The near-current expected price paths are actual outputs and are not distorted
into visually attractive fluctuations. No trading hypothesis is promoted.

```powershell
python files/minute_price_paths.py --books .runtime/minute_direction_partial_books_v4 --classifier .runtime/minute_direction_benchmark_20261005_converged --output .runtime/minute_price_paths_20261005
python files/render_minute_price_paths.py .runtime/minute_price_paths_20261005 --classifier .runtime/minute_direction_benchmark_20261005_converged --books .runtime/minute_direction_partial_books_v4
python files/render_minute_direction.py .runtime/minute_direction_benchmark_20261005_converged --books .runtime/minute_direction_partial_books_v4 --price-paths .runtime/minute_price_paths_20261005/price_paths.html
```
