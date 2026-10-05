# Minute direction: CatBoost then DeepLOB

Registered 2026-10-05 Europe/Budapest; research-only. TH-01..TH-12 apply.
Problem: hourly OHLC Ridge did not establish tradable direction. Evaluate genuine
minute targets with book/trade inputs before suggesting production adoption.
Objective fit: improve prospective entry selection; this experiment does not
certify same-day watchlist leaders or the actual ten-slot bot policy.

## Population, source and maximum period

Use ALL published June/July files for BTCUSDT, ETHUSDT, SOLUSDT in
MaximumLeverage/crypto-lob-stream, revision
873f31e729ae23b1c309cd5dcb33feed27c407de. Pin metadata and SHA256/LFS hashes.
The maximum available reconstructable public archive is used, not a selected
winning week. Publication lags current time; results are historical diagnostics.
The archive has exchange/event times without independent receive timestamps;
perfect historical availability/execution parity is not claimed.
No invented books, no OHLC masquerading as L2, no equity benchmark substitution.

Pre-training data-quality amendment: the first archive failed sequence integrity
on the entire maximum period. Only BTC=68, ETH=68, SOL=37 of 500773 ten-second
samples per asset could be certified, with no eligible 100-step sequence. A
standing anchor at 1780640988543/id94988831130 is followed only at 1780642218637,
first id94993125806: this is a real missing interval, not REST buffering. Keep
that rejected archive, hashes and coverage as evidence; do not weaken checks.
Before viewing any model results, replace the training source with ALL 90
BTC/ETH/SOL files in predict-quant/binance-future-orderbook, revision
b8590b83452d7a32fbb274ff7741b6db000b3984 (March-April 2026, ~2.21GB).
These are Binance USD-M perpetual top-20 partial-book states, independently
validated as standing ordered positive books; they are not incremental L2 diffs.
Every selected state must carry exactly 20 levels per side, no crossed quotes,
and a timestamp no later than its sample origin. No state is propagated more
than 10 seconds; clock gaps reset sequence context. Report the common maximum
clock span, each asset's missing dates and all exclusions.
Use matching official Binance UM 1m klines for closed-bar trade counts, volume
and taker buy volume, with a conservative extra ten-second availability delay.
These are aggregated trade features, not a full per-trade tape. Fetch funding
history on the exact archive window and charge settlements on held positions.
Economic results are unlevered perpetual quote diagnostics, with funding,
not spot bot performance. Capture/execution/PIT parity remains unproven.

Second pre-training data-quality amendment: the standing-book archive includes
older event timestamps recorded after newer ones (BTC March 6 has seven such
rows). Do not sort late records back into the past. Preserve recorded order and
quarantine states with E below the running maximum E across batches/files; count
all excluded rows. Equal clocks keep the last recorded state. This is safe only
for independent full states; it does not weaken incremental sequence-ID checks.
Retain the failed strict-clock preparation log and source snapshot. No model
losses were viewed before this data-handling amendment. Three single-threaded
preparation workers are allowed; models retain their registered two CPU threads.

Rejected incremental source audit: reconstruct each asset independently from standing snapshots plus grouped diffs.
Skip updates already covered by a snapshot. First applied update must span the
snapshot id+1. Subsequent updates must span the previous id+1; a gap invalidates
the book until a new snapshot. Quantity is an absolute level size; zero removes.
Prune each side to 1000 levels after complete updates. Verify best prices are
ordered, quantities positive, and bid < ask. Reset across collector outages.
Sample the past book at a 10-second UTC grid; never backfill from the next event.
Require fresh events (at most 10 seconds old) and valid complete top ten levels.
Keep invalid samples as missing and publish coverage/reset counts.
Aggregate trades on the same clock with buyer_maker determining aggressor side.
Trade history requires ID continuity and resets; do not silently bridge gaps.

## Targets and splits, frozen before losses are inspected

Forecast log mid-price change at exactly 1, 3 and 5 minutes from the origin.
Fixed three classes: down below -2bp, neutral [-2bp,+2bp], up above +2bp.
Labels are future information and never features. Report unrounded returns and
class prevalence, not only accuracy inflated by the neutral class.
Inference origins are minute boundaries with a complete past 100 x 10s book
sequence. The CatBoost and DeepLOB comparison uses exactly the same eligible
asset/origin/target rows at each horizon. Maturity and missing coverage are
explicit denominators. Exclude targets crossing a sequence-reset/gap.

Boundaries are elapsed UTC clock fractions of the common archive: 50% TRAIN,
15% validation, 10% calibration, 25% TEST, rounded to minute boundaries.
TRAIN labels close strictly before validation; validation labels before
calibration; calibration labels before TEST. Purge by the maximum five-minute
target plus execution delay. Splits are global by time across all assets.
No shuffled split, no train fit on validation/calibration/test, no test tuning.
Historical TEST stays untouched by hyperparameter/epoch/calibration selection;
the pre-registered DeepLOB configuration cannot change after CatBoost TEST.
This remains retrospective evidence, not independently collected forward proof.

## Registered models and comparison

CatBoostClassifier: separate three-class models per horizon; depth 6,
learning_rate .05, L2=10, maximum 600 iterations, validation early stopping 60,
seed 42, two CPU threads, no automatic class balancing, no file writes outside
the output root. Features: current top-ten normalized book, spread/imbalance/
microprice, causal return/depth/flow lags and rolling past trade summaries.
Training samples every five minutes; validation/calibration/TEST every minute.
Publish exact versions, selected iterations, source hashes and train counts.

DeepLOB: actual spatial CNN blocks reducing 40 book columns to 1, temporal
convolutions, three-branch Inception and LSTM, followed by three 3-class heads.
Use a registered compact CPU implementation (channels 16, Inception branches 16,
LSTM hidden 32), not a renamed plain LSTM. Sequence length 100; historical
prices are normalized to origin mid and quantities to causal sequence depth.
Adam lr=.001, weight_decay=.0001, batch 128, at most 8 epochs, patience 2 on
validation cross entropy, seed 42, two CPU threads, no random train/test mixing.
Shuffling already-separated TRAIN batches is permitted. Train every five minutes.
Select epoch on validation only. All horizons share the encoder, separate heads.

Both models receive training-only feature scaling as needed, independent
calibration-only scalar temperature (bounded .25..4), and the same fixed labels.
Controls: training class prior, momentum, and train-only logistic regression on
the CatBoost feature matrix. No Persistence line in user-facing plots.
Training-only numerical amendment: the initial logistic control hit its 300-step
LBFGS limit. Keep C=1, solver, default tolerance, TRAIN inputs and class objective;
increase the numerical iteration cap to 5000 and treat ConvergenceWarning as a
failure. This is a solver convergence repair, not model/test tuning. No TEST
scores were consulted for this decision. Preserve the stopped first training
run and source snapshot; CatBoost/DeepLOB configurations remain unchanged.
Record converged iteration counts per horizon in the repaired registration.
Publish per asset/horizon and pooled common-cohort log loss, Brier, macro F1,
balanced accuracy, confusion counts, class/base rates, raw correct/N, non-neutral
direction correct/N, calibration ECE and paired daily-block loss confidence
intervals. Blocks are days, not hundreds of correlated minute observations.
Do not call a numerical best model robust superiority when its interval crosses
zero, there are too few independent days, or multiplicity is uncontrolled.

## Economic diagnostic and production limits

Long-only independent 3-asset perpetual diagnostic account for each method/horizon; capital 1,
at most one position per asset and three positions total, budget <= equity/3,
no leverage. Fixed signal P(up) >= .5 and greater than P(down); no test-selected
threshold. Enter using the next grid's ask, exit using bid at origin+horizon plus
one grid delay; apply 7.5bp fee and 5bp adverse slippage on EACH side. A quote gap
delays exit to the first observed valid quote, never a fabricated fill. Missing
marks must be counted and cannot be advertised as fully observed drawdown.
Signals during gaps do not execute. Last admissions stop at the known experiment
end minus max horizon/delay; final holdings require actual liquidation quotes.
Report return, BTC buy-and-hold after identical costs, drawdown/mark coverage,
exposure, trades, costs and double-cost stress.
Funding is charged using the actual published settlement rate and mark price on
positions held at settlement; it is never an input before settlement. This is a
fully collateralized quoted-notional diagnostic, not a futures margin simulator.
Capture metrics are not inferred from market-wide predictions: this is not the
live bot candidate population.
No result grants BUY/SELL permission. Mandatory next gate before adoption:
maximum actual-population causal replay, full PIT/execution parity, fresh full
Truth Harness PASS and independent forward shadow. Full Harness registration
is FAIL TH-11 (stale canonical replay hash), retained separately.

## Acceptance, verification and rollback

Complete both model families on the collected maximum archive; preserve rejected
results and non-winning periods. Produce comparison tables, common-period plots,
immutable test predictions and a reproducible CLI. Focused tests cover book
sequence resets, crossing/zero levels, timestamps, future mutation and prefix
equivalence, label purging, shared cohorts, calibration separation, actual CNN/
Inception/LSTM topology, fees/slots and gap-aware execution. Review staged sources
only, run tests, diff check and staged Truth Harness, commit and push.
Runtime files, datasets, fitted weights and reports are untracked evidence.
Rollback: stop the offline process/remove the new research scripts. Running bot,
live config, credentials, positions and alerts are not modified.

## Reproduction

### Frozen single-origin price views

Add an offline `price_forecasts.html` view for Prior, Momentum, Logistic,
CatBoost and DeepLOB, with synchronized asset/origin selectors. Every model
shows the same 20-minute observed mid-price history, issuance cutoff and five
minutes of subsequently observed mid-price. These are categorical classifiers:
render their frozen DOWN/NEUTRAL/UP probabilities at exact +1/+3/+5 minutes,
with directional color bands and separate probability bars, never an invented
predicted price or interpolated forecast trajectory. The neutral class is the
registered +/-2bp log-return band relative to the issuance mid-price; it is
neither an uncertainty interval nor a transaction-cost break-even threshold.

Choose example origins from the intersection of all assets' inference clocks,
using the test start and fixed six-hour clock anchors, accepting only a valid
past 20-minute prefix. Do not consult future prices, labels, scored flags,
correctness or model confidence to select examples. Keep unknown future states
as gaps and unavailable endpoint facts as unknown; do not forward-fill, connect
gaps, or remove unfavorable cases. Display all five methods on every selected
origin and permit browsing every registered six-hour example. Export a
default-origin PNG per method/asset and an auditable JSON of selected views,
clocks, probabilities, source hashes and facts. Verify prepared book hashes and
the frozen result/prediction hashes against the native-verification receipt.
Generation changes no model, metrics, split, weights or trading behavior.

Acceptance tests: shared-clock and future-independent example selection,
future-mutation invariance of the forecast, exact target alignment, unknown
future/gap preservation, receipt/source drift rejection and actual offline
HTML/PNG generation. Rollback: remove the optional visual view/link; frozen
benchmark and production behavior remain unchanged.

```powershell
python files/plot_minute_direction_prices.py .runtime/minute_direction_benchmark_20261005_converged --books .runtime/minute_direction_partial_books_v4
```

Use Python 3.11, CatBoost 1.2.10, Torch 2.8.0 CPU, numpy 2.2.6, pandas 2.3.3,
scipy 1.15.3, scikit-learn 1.7.2, pyarrow 21.0.0 and sortedcontainers 2.4.0;
safetensors for the frozen neural checkpoint; matplotlib, plotly and tabulate for
the standalone report. The registration records actually loaded core versions.
Normal Python environment (set OPENBLAS_NUM_THREADS/OMP_NUM_THREADS to 1):

```powershell
python files/minute_direction_partial.py download --data .runtime/minute_direction_partial_data
python files/minute_direction_partial.py prepare --data .runtime/minute_direction_partial_data --output .runtime/minute_direction_partial_books_v4
python files/evaluate_minute_direction.py --books .runtime/minute_direction_partial_books_v4 --output .runtime/minute_direction_benchmark_20261005_converged
python files/render_minute_direction.py .runtime/minute_direction_benchmark_20261005_converged --books .runtime/minute_direction_partial_books_v4
python files/verify_minute_direction.py .runtime/minute_direction_benchmark_20261005_converged --books .runtime/minute_direction_partial_books_v4
```

Outputs must be new directories; an existing run is never silently overwritten.
For the embedded interpreter, insert `.runtime/forecast_dependencies` and `files`
at the beginning of `sys.path` before importing the module; it ignores PYTHONPATH.
Retain pinned source manifest, original official receipts, coverage, registration,
source snapshots, CatBoost phase outputs, final predictions and account details.
The comparison HTML embeds Plotly for offline exploration; PNG/Markdown/CSV are
portable views of the same saved metrics. No training outcomes are used to alter
the DeepLOB architecture, class band, splits, thresholds or fee assumptions.
Reload native CatBoost and safetensors checkpoints and compare 24 frozen origins
spanning TEST against saved calibrated probabilities. Recompute all pooled losses
from immutable predictions and verify source/data hashes and cohort identities.
