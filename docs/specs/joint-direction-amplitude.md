# Joint direction and conditional amplitude: impulse-entry hypothesis

Registered 2026-10-06 before fitting/outcomes; offline only. Parent:
impulse-entry-catboost.md, SCOUT_OPTIMIZATION_SPEC.md. TH-01..12 apply.
Full Truth Harness at start PASS. Prior RMSE entry filter remains REJECTED.

## Fixed hypothesis and maximum population

A calibrated probability of raw-price growth plus conditional growth/fall
magnitudes may select economically viable impulse_speed entries better than a
single unconditional return regressor. Evaluate on ALL original impulse rule
candidates, not just selected trades. Keep other modes/BUY score/capacity,
replacement, cooldown, exits and sizing unchanged. Use same longest complete
recovered 186.333-day archive Apr1-Oct4 06UTC, 93/105 requested symbols, with
10-day market warmup. Previously exposed history remains retrospective.
Source-hash verified prior dataset/control/trades may be reused, not refitted or
renamed as fresh outcomes. Recheck manifests/raw/checkpoints, native candidate
kernel/source parity and prior independent audit before reuse. Copy receipts,
bind every inherited file to its actual producer and SHA. Native pickle only
from trusted workspace .runtime. Preserve unknown population/live-agent/PIT,
receive-time and actual-execution limitations; never manufacture missing rows.

## Fixed models and causal evaluation

Same ten features as previous impulse experiment: exact17 CLOSED15m candles,
returns1/4/16, vol16, volume ratio16, range/range-position16, tf1h and UTCsin/cos.
Same75m fixed horizon. Rebuild gross return from exact raw closes and check
its exact multiplicative cost conversion against prior net labels; classifier
target gross>0 (flat goes into down; no floating-point inversion sign artifact).
Two separate conditional RMSE regressors predict positive gross return and
absolute nonpositive return, respectively. At inference clamp both magnitudes
to >=0. Joint gross mean=p(up)*up_magnitude-(1-p(up))*down_magnitude; convert
to NET with exact7.5bp-or-higher config fee and5bp slip per side. Admit only
expected NET>0, without threshold/parameter/horizon search.

First30days untrained passthrough; then expanding refit every30days. Inner
whole-local-day train60%/early-stop-validation20%/calibration20% of available
history; purge75m crossing each boundary and require label maturity strictly
before fit. Classifier Logloss and magnitude RMSE CatBoost400/depth4/lr.03/
l2=10/seed42/threads2, validation-only early stopping40. Require train>=500,
validation>=100, calibration>=100 and each class train>=200/validation>=50;
both classes in calibration. Otherwise explicitly untrained passthrough.
Platt LogisticRegression(C=1, solver=lbfgs, max_iter=1000) on raw-probability
logits, fit ONLY disjoint calibration rows. Freeze three native models and
calibration coefficient/intercept before issuing each future block. Unknown
features fail closed when model exists; missing future labels never suppress
inference. Earlier matured TEST blocks may train later fits: prequential OOS,
NOT sealed holdout. No changing calibration method after results.

## Comparisons, metrics and frozen acceptance

One new portfolio policy: joint filter vs original rule-only control. Previously
rejected regressor is a descriptive same-window reference, not a second new
test or production candidate. Resimulate full joint stream on maximum period;
receipt-verified unchanged control/reference can reuse trades, but recalculate
cash accounts/marks and objective counts. Compare identical candidate population,
clock, costs and sizing, not old30-day profile. Full and last20% reporting cohort:
net/alpha/DD/exposure, trade/cost attribution, unique early/captured leader pairs,
precision denominators. Continuous TEST carries earlier holdings/capital.
Single-comparison TEST daily log-return paired3-calendar-day block bootstrap,
5000draws seed42, 95%CI. >=30completeTESTdays, full+TEST return gain>=1pp,
lowerCI>0, no fewer early/captured pairs full+TEST, precision drop<=.5pp,
DD worse<=1pp. Numerical pass only allows separately designed forward shadow;
runtime_eligible alwaysfalse. Retain failures, no retuning exposed history.

Forecast diagnostic metrics on same issued known targets: raw/calibrated/class-
climatology Brier/logloss, ten fixed reliability bins and ECE with counts,
raw-price direction correctness and train-climatology majority baseline.
Joint net MAE/RMSE vs zero NET, flat RAW-price after-cost and train-mean baseline;
all-positive expected-net origins actual net means/counts, not executed PnL.
Probability plus conditional amplitude is not a reconstructed price trajectory
or an independently estimated volatility process. Do not invent curved paths.

## Verification, rollback and delivery

Tests cover disjoint purged cohorts, gross/net inversion, cost-aware mixture,
zero/NaN decisions, calibrated probabilities, no future-label influence in
actual native fits, future tail issuance and fixed reliability denominators.
Independent verifier rebuilds all features/gross labels from rawJSON, checks
all fitting/calibration timing, reloads all native models and reconstructs all
probabilities/magnitudes/net forecasts and admissions; canonical cash/mission/
gate and probability metrics recomputed. SHA receipts bind outputs and producer
sources. Generate full/TEST equity and probability reliability plots plus report
CSV/Markdown. Spec-first, tests, full+staged Harness, diffreview, source/spec/test
commit+push; no prices/weights/runtime reports in Git. No live orders/config/
positions changed. Rollback stops own research worker. Subsequent production
requires complete parity, new forward/paper/canary evidence and immediate rollback.

## Reproduction

Embedded Python: prepend `.runtime/forecast_dependencies` and `files` to
sys.path; run scripts via runpy.run_path(...,run_name='__main__'). Set
OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1. Run:

```
files/run_joint_direction_amplitude.py
  --parent .runtime/impulse_entry_catboost_20261006_v1
  --output .runtime/joint_direction_amplitude_20261006_v1
```

Use a NEW output directory for reproduction. Then run
`files/verify_joint_direction_amplitude.py --run <output>` and
`files/render_joint_direction_amplitude.py --run <output>`. Tests:
test_joint_direction_amplitude, test_run_joint_direction_amplitude,
test_verify_joint_direction_amplitude, test_render_joint_direction_amplitude,
plus previous causal-entry and turnover/portfolio regressions. Outputs/weights
remain in runtime; inherited controls preserve original producer/hash receipts.

## Completion result (2026-10-06)

Run `.runtime/joint_direction_amplitude_20261006_v1`, result SHA256
`21125bfcb5056263458e52c6732bc3d813f8a87c1ff6d2537f0aec808bea6e32`.
Six causal blocks,18native models and6disjoint Platt fits. 57,095 original impulse
candidates,48,327 issued after first30days,1,055 positive expected-net candidates
retained. 48,300 known75m outcomes (27tail origins explicitly unknown);1,048
positive forecasts with known outcomes,7positive forecasts still lack frozen
future evidence. These are candidate counts, not executed-trade counts.

| Arm | Full net % | TEST net % | Trades | Early /2790 | Captured /2790 | TEST early /570 | TEST captured /570 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | -96.256673 | -44.809011 | 11010 | 1039 | 2219 | 253 | 480 |
| prior regressor | -94.988859 | -37.635722 | 10256 | 941 | 2078 | 224 | 437 |
| joint | -94.120106 | -35.263940 | 10306 | 937 | 2093 | 224 | 453 |

Joint full alpha -118.171788pp vs control -120.308355pp, DD94.186105 vs
96.347967%. Joint full precision4847/9725 vs control5444/10376. Correctly
paired single-comparison95%TEST CI[-14.389230,+101.791510]bp mean daily
log-return difference,38completeTESTdays; lowerbound notpositive. Full/TEST
early andtotal captures decrease,precision lower: registered gate REJECTED.
Joint has better observed account returns than both references, but all three
lose money and the improvement does not satisfy bot mission/robustness gates.

All-known OOS raw/calibrated/climatology Brier .248711/.246439/.246329;
binary growth/non-growth correctness25876/27087/27222 outof48300,actual
growth21078/48300. TEST raw/calibrated/climatology Brier .247776/.246626/
.247147; correctness9279/9267/9283 outof16699,actual growth7416/16699.
Calibration improves raw probability losses, but full-period climatology is
still better, and TEST classification correctness does not beat climatology.
ECE10 raw/calibrated/climatology: allOOS .033513/.011765/.006005;
TEST .008014/.008985/.014587. Probability accuracy and calibration are
distinct; do not claim calibration improves every metric/cohort.

Known positive expected-net candidates: meanactual75m net -0.184414%
on1048allOOS, -0.055075% on389TEST. Joint net MAE .979827 vs flatRAW-price
after-cost .974096 andtrain-mean .978847% on48300; TEST1.091165 vs1.084045/
1.083221% on16699. No consistent improvement in net-return forecast error.
Control/joint allocated-quantity raw-price PnL -1309.079945/-1125.423118 USDT,
simulated costs8316.587397/8286.587499. This additive cash attribution differs
from an independently compounded zero-cost counterfactual and cannot prove
the isolated causal value of forecasting; other-mode admissions also interact.

Retain the exact rejected architecture/threshold; no calibration/horizon/threshold
search on exposed TEST. Production unchanged. 47focused/relevant tests PASS;
10new focused tests rerun after verifier/render refinements PASS. Full Harness
PASS is a mechanical check, not permission to deploy or profitability evidence.

Independent audit PASS: all57095 raw feature/label rows rebuilt,18native models
reloaded and48327 forecasts reproduced,6Platt calibrations refit using only
disjoint past rows; all303532 screen decisions and17889marks in each of3accounts
checked. Verifier SHA256
`c746ff0f94743da2d416f43e1eb9226cef37a07694cfebf2cc3cfdb8180f9b2f`.
Python3.11.9, NumPy2.2.6, CatBoost1.2.10, scikit-learn1.7.2. Comparison
CSV/PNG/report receipts bind publication to this audited result. Own runner and
verifier completed normally; no production processes changed.
