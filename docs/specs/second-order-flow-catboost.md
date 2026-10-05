# CatBoost order flow: 5/10/30 seconds

Registered 2026-10-05, research-only. TH-01..TH-12 apply. Objective: determine
whether granular book order-flow information improves short future price
displacement and direction after the minute-price regressors failed MAE control.
No live trading policy is changed, no portfolio adoption/alpha claim is approved.

## Maximum source and availability

Use ALL 90 existing pinned BTC/ETH/SOL Binance USD-M perpetual top-20 standing
book files, predict-quant/binance-future-orderbook revision
b8590b83452d7a32fbb274ff7741b6db000b3984, verify every original LFS SHA256.
Read original ~100ms states, not the earlier ten-second samples or one-minute
trade totals. Compute Cont-style bid/ask supply-demand imbalance at each of
the first five ranks between consecutive observed full states and accumulate
these increments into one-second causal bins. This is book-state OFI, not
individual order messages, executed aggressor flow, or identified cancellations.
An improvement would motivate adding a separately verified trade tape later.

Recorded ordering is preserved. Quarantine older event clocks, never sort them
back into the past. Equal clocks keep recorded updates. Reject missing/nonfinite,
nonpositive, unordered or crossed first-five quotes and malformed 20-level
source structure. Source event-clock gaps >500ms reset context. At each one-second
origin take only a standing state at/before that clock and <=250ms old; gaps
remain NaN, never backfilled from later events. Discard OFI across resets.
Store quote age, source event counts, quote/depth and net OFI per rank.
Preparation may run independent asset samplers in parallel, with unchanged
per-asset carry/session/order. A completed asset prefix from a stopped sequential
preparation can be reused only with all raw-file/log counts, continuous clock,
part hashes and source algorithm hash verified; record the stopped run explicitly.
Publish the assembled receipt atomically once, after verifying all parts.
Exchange E is the availability assumption: independent receive timestamps are
absent, so this is retrospective event-time research, not latency parity proof.

## Targets and cohorts

Use exactly the existing global chronological calendar cuts, disclosed historical
TEST. Targets are log mid-price displacement from one origin to +5/+10/+30sec.
Additional +15/+20/+25sec direct outputs support plotting a six-point price path;
all models and metrics use the same six-target eligibility cohort. Labels require
continuous valid source context across the entire longest target. Features use
only the past60sec, valid same-session history. Train every30sec; validation,
calibration, TEST and inference every5sec. Purge31sec at every split boundary.
Inference keeps unknown future outcomes; score only mature complete labels.
Shared population/denominators for OFI, book-only ablation and simple controls.
This calendar has been viewed in prior experiments; it is not fresh independent
holdout. Parameters are fixed before new subminute results and never test-tuned.

## Fixed inputs, models and uncertainty

Book features: normalized five-rank depth, spread/microprice, top1/3/5 queue
imbalance, past1/5/15/60sec returns and realized variation, current quote age,
past event rates and asset one-hot. Flow features: signed and absolute per-rank
OFI summed over past1/5/15/60sec and normalized by observed past depth; no future
normalization. Book-only CatBoost excludes all OFI columns on the SAME rows.

CatBoost multi-output regressors: MultiRMSE, depth6, lr.05, L2=10, max600 trees,
validation early stopping60, seed42, two CPU threads. TRAIN-only mean/std target
scaling. Ridge alpha10 with TRAIN-only StandardScaler is the simple OFI control.
Microprice displacement held constant across horizons is a causal control.
Zero return appears in error tables only, not as a user-facing forecast curve.
Model selection/calibration never reads TEST. CALIBRATION supplies independent
nominal90% per-horizon absolute residual quantiles, no IID/joint coverage promise.
No synthetic price noise, smoothing toward facts, future anchoring or slope edits.

## Evidence, graphics and verification

Publish exact scored/inference counts and source gaps/quarantines per asset;
pooled/per-asset displacement MAE/RMSE bp, price errors per asset, nonzero sign
correct/N plus observed majority counts, predicted-sign distributions, class
prevalence and marginal interval covered/N. Report zero-return and microprice
controls; also report three-sign correct/N including unchanged outcomes.
Numerical sign zero is abs(log return)<=1e-12, below any instrument price tick.
Report OFI-versus-book ablation. Daily matched loss differences and
three-day block bootstrap with multiplicity across three registered horizons
and three controls; fewer than30 complete test days cannot support robust claims.
Short-horizon predictability is not trading profitability after costs/latency.

Freeze source/data hashes, cuts, parameters, native states and issued predictions.
Record Python/package versions and resolved native CatBoost parameters.
Reload native CatBoost/Ridge on predetermined inference samples. Verify earlier
classifier and minute-regression evidence hashes remain unchanged. Runtime
reports/data/weights are never committed. Plots use fixed clock examples selected
only by past availability, with common origins/scales, history, exact forecast
points at5-second endpoints, later facts, explicit gaps and optional intervals.
Offline HTML asset/origin selectors and PNG snapshots; no origin selection by
future completeness/correctness or confidence.
Provide pooled MAE/RMSE control-relative and direction comparison plots; exact
errors, correct/N and base rates remain in the adjacent table and frozen JSON.
Temporal alias/prefix invariance, future mutation, OFI quote-change signs,
gap/quarantine, purge, scaling/calibration, native state and rendering have
focused tests. Undefined metric ratios remain unknown in tables and figures.

Run tests/diff check, review intended staged spec/source/tests, staged Truth
Harness, commit/push. Full Harness FAIL TH-11 (canonical replay source-hash gap)
is retained. Rollback: remove offline experiment files; production is untouched.

## Reproduction

Use a fresh output directory for each run. The embedded Python ignores
PYTHONPATH, so bootstrap the dependency and files directories explicitly:

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
pyembed\python.exe -c "import sys,runpy;sys.path[:0]=['.runtime/forecast_dependencies','files'];runpy.run_path('files/prepare_second_order_flow_parallel.py',run_name='__main__')" --data .runtime/minute_direction_partial_data --output .runtime/second_ofi_reproduction_books
pyembed\python.exe -c "import sys,runpy;sys.path[:0]=['.runtime/forecast_dependencies','files'];runpy.run_path('files/evaluate_second_order_flow.py',run_name='__main__')" --data .runtime/minute_direction_partial_data --books .runtime/second_ofi_reproduction_books --anchor .runtime/minute_direction_benchmark_20261005_converged --output .runtime/second_ofi_reproduction_result
pyembed\python.exe -c "import sys,runpy;sys.path[:0]=['.runtime/forecast_dependencies','files'];runpy.run_path('files/render_second_order_flow.py',run_name='__main__')" --books .runtime/second_ofi_reproduction_books --report .runtime/second_ofi_reproduction_result
```

Focused tests: `test_second_order_flow_data`, `test_second_order_flow_models`,
`test_second_order_flow_provenance`, `test_second_order_flow_parallel`,
`test_render_second_order_flow`. The renderer test executes all selector,
interval and focus configurations with a mocked Plotly/DOM, without network.

## Completed diagnostic, 2026-10-05

All 90 original files were processed: 54,862,541 standing states (BTC 21,951,615;
ETH 20,365,937; SOL 12,544,989). Completed BTC was reused from the explicitly
stopped sequential preparation; ETH/SOL independent workers completed, then
all hashes/counts/parts were assembled. No algorithm or model parameters were
changed after registration/training. Shared calendar TEST is 2026-03-30
11:04 UTC through 2026-04-07 19:08:50 UTC; gaps remain excluded, not fabricated.

TRAIN 70,417; validation 130,861; calibration 112,259; scored TEST 239,619;
issued inference 253,951 (14,332 lack a complete future source path).
CatBoost OFI selected56 trees, book-only65, using validation only.
All error means below use the same239,619 rows. Direction excludes unchanged
outcomes; the full three-sign counts are in the report.

| Horizon | OFI correct / nonzero N | OFI MAE bp | Zero MAE bp | OFI RMSE bp | Zero RMSE bp |
|---|---|---|---|---|---|
| 5sec | 99,363 /156,871 (63.34%) | 1.284387 | 1.239822 | 2.188052 | 2.217346 |
| 10sec | 111,828 /188,720 (59.26%) | 1.942231 | 1.922573 | 3.121374 | 3.147287 |
| 30sec | 119,764 /218,204 (54.89%) | 3.563061 | 3.562948 | 5.389156 | 5.407691 |

Microprice direction counts are 99,361/156,871,112,007/188,720,
119,920/218,204: approximately equal at5sec and slightly better at10/30sec.
Among learned regressors OFI has the smallest pooled MAE at all registered
horizons; Zero still has smaller pooled MAE. Book-only has the smallest RMSE
at5/10sec, OFI at30sec. Additional OFI information gives negligible incremental
gain over the standing-book ablation; accurate future path shape is not established.
BTC/ETH show small MAE improvements over Zero at10/30sec, SOL does not; see
matched per-asset metrics. No pooled/portfolio or robust adoption claim follows.

Nominal90% OFI intervals cover 207,563/239,619 at5sec,206,722/239,619 at10sec,
205,458/239,619 at30sec: undercoverage, not production calibration approval.
Only7 full calendar TEST days enter paired block diagnostics; these days also
have source holes. This disclosed TEST is not fresh independent forward evidence.

Runtime evidence: `.runtime/second_order_flow_catboost_20261005/` contains
registration, resolved versions/parameters, native states, frozen predictions,
result/verification/metrics verification, CSV, comparison PNG and interactive
forecast_paths.html/JSON (10 common origins,9 price PNGs). Result SHA256:
0798edd6125d95364d4f992b0c2c753ddfc6b4cdbfdeef41fd2c501f335cb86b;
predictions SHA256:123e1ecdd1c7ed9a9f2e047721c950a98ba6d8751ca19d9285ff4dc4119f3e64.

Verification:23 focused tests passed (including denominator-free plots);
all121 actual HTML selector/focus/interval configurations passed mocked JS;
native states reload equivalently on32 predetermined origins per learned model;
120 metric rows were independently recomputed from issued predictions. Five
predetermined BTC clocks were independently JSON-decoded to verify E/as-of
quotes, age, OFI and event counts (sampled audit, not full-source certification).
Prior classifier and minute-price artifacts remain byte-identical.
Full Harness remains FAIL TH-11; staged change profile PASS. Retain this
hypothesis as a diagnostic; production trading behavior remains unchanged.
