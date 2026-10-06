# Impulse-speed expected-net-edge CatBoost

Registered 2026-10-06, before fitting/outcomes. Offline research only.
Parent: turnover-economics-replay.md; SCOUT_OPTIMIZATION_SPEC.md. TH-01..12 apply.
Start full Truth Harness PASS. Earlier capacity/amplitude/replacement hypotheses
remain rejected; this is a separately registered supervised entry hypothesis.

## Fixed question and population

Can a causal CatBoost expected-net-return filter improve the existing rule-only
ten-slot portfolio without sacrificing unique early daily Top-15 capture?
Use ALL impulse_speed rule candidates, including candidates never admitted by
control. Other modes, scores, replacement, cooldown, SELL and sizing unchanged.
Use longest complete recovered 93/105-symbol archive (186.333 evaluation days,
Apr 1 through Oct 4 06 UTC, ten-day market warmup), receipt-bound native features
and candidates from the previous experiment. Check market/source/checkpoint
hashes and regenerate fixed BTC15m/ETH1h candidate fields before reuse. Missing
population and absent PIT/live-agent/receive-time/exchange-fill parity disclosed.
Trusted local pickle restricted to workspace .runtime, never external input.

## Features, target, prequential fitting

Exact 17 closed 15m candles at each candidate origin: returns 1/4/16, log-return
volatility16, volume ratio16, range16, range position16. Append timeframe 1h
indicator and UTC time-of-day sine/cosine. No symbol ID, day-final labels, MFE,
exit outcomes, future model scores or unrestricted native feature arrays.
Target = exact 75-minute future close return AFTER multiplicative roundtrip
7.5bp-or-config-higher fee and 5bp slippage per side. Target is a fixed-horizon
proxy, not a forecast of existing SELL-path PnL. Incomplete targets excluded
from fitting/forecast metrics, never treated as loss; valid features still issued.

First 30 evaluation days: both arms unchanged, no untrained rejection. Then
expanding-origin refit every 30 days. At each fit use only labels matured strictly
BEFORE fit time. Chronological inner train80%/validation20% split by whole local
days; purge labels reaching validation boundary. CatBoostRegressor RMSE,
400 iterations, depth4, learning_rate .03, l2=10, seed42, thread_count2,
early_stopping_rounds40, use_best_model=True. No grid search or threshold tuning.
Minimum 500 train /100 validation rows; below minimum explicit untrained passthrough.
Freeze each native model before the following block; predict all available
impulse candidates. Admit only prediction >0, unknown feature fail closed when
model exists. Do not use future label availability as an inference filter.
Periodic fits may use earlier matured TEST labels: explicitly prequential OOS,
NOT a sealed test. Last20% of history is the fixed reporting cohort; already
exposed historical market is retrospective regardless of causal fitting.

## Portfolio evaluation and frozen gate

Re-simulate control and filtered full streams across the entire maximum period.
Report both full and last20% net return, BTC alpha, complete-grid drawdown,
exposure, turnover/cost cash attribution, unique leader/early pair counts and
precision counts. Continuous TEST account carries prior capital/positions;
TEST buys only are used for TEST mission. Cross-check every account mark and
fee/slip total against canonical evaluator. Compare control to previous identical
profile on identical input, not the different 30-day baseline profile.
Single registered comparison: paired 3-calendar-day moving-block bootstrap
5000 draws seed42, 95% CI of TEST daily log-return difference. At least30 complete
TEST days; full and TEST return gain >=1pp; paired lowerCI>0; full and TEST
unique early/captured pairs not lower; full precision drop <=.5pp; DD no worse
by >1pp. No minimum trade-count reduction requirement: test entry edge itself.
Numerical pass permits only separate new forward shadow design. runtime_eligible
always false; mechanical PASS does not imply profitability/production approval.
Retain rejected outcomes and counts. Never retune threshold on exposed history.

## Verification, delivery and rollback

Focused tests: future mutation/prefix equivalence, exact gaps, target costs and
maturity, purged split, inference tail survival, strict zero threshold, other
mode preservation, gate risks. Independent verifier reloads all native models,
reconstructs all feature/target rows from SHA-verified raw JSON market,
checks training cohort timing, every prediction/filter decision, screened trade
admissions and canonical account/mission metrics. Save snapshots/receipts,
comparison CSV/PNG/Markdown in runtime only. Spec/tests/source review, full and
staged Harness, diff check, commit and push. Stop research worker only on rollback;
no live BUY/SELL/config/positions/model weights modified. Later deployment needs
complete parity, fresh shadow/paper/canary, risk limits and an immediate rollback.

## Reproduction

Embedded Python bootstrap: prepend `.runtime/forecast_dependencies` and `files`
to sys.path, then run scripts with runpy.run_path(..., run_name='__main__'). Set
OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1. Runner arguments:

```
files/run_impulse_entry_catboost.py
  --market .runtime/price_volatility_bot_comparison_release
  --features .runtime/price_volatility_bot_20261005_parallel/features
  --candidates .runtime/price_volatility_bot_comparison_final
  --output .runtime/impulse_entry_catboost_20261006_v1
```

Use a NEW output directory for reproduction; existing results are immutable.
After completion run `files/verify_impulse_entry_catboost.py --run <output>`,
then `files/render_impulse_entry_catboost.py --run <output>`. Focused tests:
test_impulse_entry_catboost, test_run_impulse_entry_catboost,
test_verify_impulse_entry_catboost, test_render_impulse_entry_catboost;
include turnover/accounting regressions. Reports/weights/raw checkpoints stay
in runtime, never Git. Model training and portfolio replay run serially.

## Completion outcome (2026-10-06)

Run `.runtime/impulse_entry_catboost_20261006_v1`, result SHA256
`9aab170c454aa3d4525ff20b23b3171ecb2e512d3539038f93d7a711a2980ad8`.
Six causal fits, 57,095 impulse candidates, 48,327 issued after first30 days,
417 candidates retained by the positive-forecast filter (not 417 executed trades);
no unknown past features. 27 issued origins
lack a complete 75m future target and remain issued, excluded only from losses.
Control trades are byte-identical to prior same-profile maximum-history control,
SHA256 `0568934962f6d32c2a7576b01100e059c0190a8ed20b7161e7db6c17137d0b14`.

| Arm | Full net % | TEST net % | Trades | Early /2790 | Captured /2790 | TEST early /570 | TEST captured /570 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | -96.256673 | -44.809011 | 11010 | 1039 | 2219 | 253 | 480 |
| CatBoost | -94.988859 | -37.635722 | 10256 | 941 | 2078 | 224 | 437 |

Full alpha: -120.308355 vs -119.040541pp; DD96.347967 vs95.043310%.
Full precision5444/10376 vs4799/9675. Single-comparison 95%3-calendar-day
paired TEST CI[-20.818379,+85.874610]bp mean daily log-return difference,
38complete days. Lower bound notpositive, full/TEST early andtotal capture
decrease, precision worse: registered numerical gate REJECTED.

Control raw-price allocated-quantity PnL -1309.079945 USDT vs model
-1402.578295; simulated cost8316.587397 vs8096.307595. Savings in costs
outweigh worse raw-price PnL, but do not establish profitable entry selection.
Among all417 positive OOS forecasts mean realized75m net proxy is -0.237148%;
TEST45 positive forecasts mean+0.165747%. These are all-candidate proxy means,
not executed portfolio trades or proof of reliable positive edge. Model MAE
0.980007 vs zero-net-return1.024011% on48300 known OOS targets; TEST MAE
1.083148 vs1.129516% on16699. Error improvement does not pass portfolio gate.
Field direction_correct_n compares positive vs nonpositive NET-return targets,
not raw-price direction; do not present it as up/down price accuracy.

Retain this exact rejected configuration; no threshold/horizon retuning on the
exposed TEST. Production unchanged. Subsequent entry hypothesis must address
target/actual-exit mismatch with its own registration and fresh forward evidence.

Independent audit PASS: all57095 impulse feature/target rows reconstructed from
raw JSON, all48327 native predictions reproduced, all303532 candidate decisions
checked, all17889 marks of each account matched canonical accounting, mission
counts/forecast losses/gate recomputed. Verifier SHA256
`1485c2754beb039eb397d7a1a50a9d7d3886a34f97848836869ffdcaef5aaf8f`.
Python3.11.9, NumPy2.2.6, CatBoost1.2.10. 37 focused/relevant tests PASS,
including actual native fitting with mutated unavailable future labels.
Full Truth Harness PASS. Runtime eligibility remains false.
