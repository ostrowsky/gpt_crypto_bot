# ML Researcher interview demo v3.2

Default inference displays ARIMA, ETS, SARIMA, SARIMAX, Prophet, Ridge, XGBoost,
LSTM and actual pytorch-forecasting TFT. Persistence is removed from displayed
forecasts; the last-price error remains an internal statistical control and
production fallback. This does not relax trading/serving gates.

The delivered executed notebook embeds a checksummed native model release. Run
All loads frozen weights, then publishes one common current 15-minute UTC window
for ALL methods/assets. USDT and percentage plots share identical bounds. Manual
refresh adds matured actuals without changing predictions or their target times.
Near-zero means are labeled weak, never cosmetically changed to wavy trajectories.

The clean Git notebook has no trained states. To create its release, install the
pinned requirements and run:

```text
python files/prepare_research_forecast_release.py --output forecast_demo_artifacts
```

This performs the nine-model retrospective August benchmark, train-only OOF,
fresh 42-day release preparation (35 train / 4 tune / 3 calibration), and actual
future-mutation, closed-prefix and serialization-equivalence assertions on all
27 model/asset pairs. The full CPU benchmark and OOF preparation can take over
an hour; the delivered notebook loads prepared weights for current inference.
New weights do NOT
inherit historical benchmark scores. Added-model August results are retrospective
because the period had already been viewed; independent confirmation needs future
data. Live inference only needs three days of input context.

Historical price plots now show ONE origin and all 15 horizons, never a line
connecting successive h15 forecasts anchored to already known market prices.
Set SHOW_HISTORICAL=True to view these separately from the current UTC comparison.
TFT predicts minute-return quantiles and has test pinball/crossing diagnostics;
point paths sum minute medians. Marginal quantiles are not summed into alleged
price quantiles. All price-path intervals use separate residual calibration.

The following describes the original v3.1 protocol retained as historical context.

Open `Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb` with a dedicated Python
3.11 kernel. It embeds the reviewed research module and contains Russian answers
to all ten interview questions. Install the pinned dependencies shown in its
first code cell, restart the kernel and run all. Binance public data needs no key.

The source notebook has no saved outputs. A completed run writes input hashes,
versions, all model failures, selection decisions and final metrics into
`forecast_demo_artifacts/benchmark.json`. Keep that folder out of Git. The frozen
default snapshot is [2026-05-04,2026-09-01) UTC; three assets use the same
120-day minute grid: 60 train / 15 tune / 15 calibration / 30 test days.
Checksum-verified monthly spot archives are normalized from microseconds.
Persistence, Ridge, XGBoost, ARIMA, SARIMA and SARIMAX are all default models.
Seasonal AR-only models use explicit train-only conditional least squares and
calendar-only SARIMAX exog. DL/Prophet are optional before a new run.

The notebook includes a complete comparison table, familywise uncertainty,
per-method real-price/forecast plots for train OOF, test and frozen inference,
and an ipywidgets panel. In an active kernel, create a timestamped immutable live
research forecast, then manually refresh closed candles to overlay new actuals.
Mature-point counts and metrics update; predictions and trained weights do not.
The panel is research-only and does not override production release expiry.

The original v2 notebook remains unchanged in Downloads. Its saved execution was
partial; it cannot establish a model-quality or profitability result.

## Local service

Set `FORECAST_API_TOKEN` to a separate secret of at least 24 characters in your
process environment. Never put it into the notebook or a command stored in Git.

```text
python -m pip install -r research/requirements-forecast.txt
python files/research_forecast.py --serve --end-utc <preregistered-fresh-UTC-cutoff>
```

Training runs once before serving. No automatic refitting or model selection uses
online test outcomes. `GET /forecast/BTCUSDT` requires an `X-API-Key` header.
`GET /health` reports readiness, drift diagnostics and mature-label coverage.
Unknown model advantage retains persistence. Stale input, release expiry and
missed first-horizon deadlines fail closed. Stop the process to roll back.

For an isolated container, build from the repository root:

```text
docker build -f research/Dockerfile.forecast -t forecast-demo .
docker run --rm -p 127.0.0.1:8000:8000 -e FORECAST_API_TOKEN forecast-demo --serve --host 0.0.0.0 --end-utc <preregistered-fresh-UTC-cutoff>
```

The Docker recipe has not been built here because Docker is unavailable. Supply
TLS, rate limiting, persistent log collection, load checks and a prospective
rollout before external production exposure. The seven-day release expiry
requires an explicitly revalidated release with a new preregistered cutoff.

## Evidence boundary

Forecast MAE and empirical intervals are research diagnostics, not net portfolio
alpha. The extended run uses 30 complete test days and familywise-corrected
day-block intervals with three-day block sensitivity. A confidence interval
crossing zero is INCONCLUSIVE; more rows cannot guarantee significance.
Tests enforce causal feature
construction and serving contracts; they do not establish exact historical
arrival times or stable market alpha. The v3.2 release verifies both DL libraries
and numeric/native model export/import, without pickle in the distributed release.

The bot's full Truth Harness independently returned FAIL TH-11 on 2026-10-04:
the existing portfolio artifact did not match the current replay source hash.
This demo does not repair or waive that production-evidence finding.
