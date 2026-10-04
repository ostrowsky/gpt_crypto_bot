# ML Researcher interview demo

Open `Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb` with a dedicated Python
3.11 kernel. It embeds the reviewed research module and contains Russian answers
to all ten interview questions. Install the pinned dependencies shown in its
first code cell, restart the kernel and run all. Binance public data needs no key.

The source notebook has no saved outputs. A completed run writes input hashes,
versions, all model failures, selection decisions and final metrics into
`forecast_demo_artifacts/benchmark.json`. Keep that folder out of Git. The frozen
default snapshot ends exclusively at `2026-10-04T00:00:00Z`; three assets use the
same 30-day minute grid. Optional models are explicitly enabled before a new run.

The original v2 notebook remains unchanged in Downloads. Its saved execution was
partial; it cannot establish a model-quality or profitability result.

## Local service

Set `FORECAST_API_TOKEN` to a separate secret of at least 24 characters in your
process environment. Never put it into the notebook or a command stored in Git.

```text
python -m pip install -r research/requirements-forecast.txt
python files/research_forecast.py --serve --end-utc 2026-10-04T00:00:00Z
```

Training runs once before serving. No automatic refitting or model selection uses
online test outcomes. `GET /forecast/BTCUSDT` requires an `X-API-Key` header.
`GET /health` reports readiness, drift diagnostics and mature-label coverage.
Unknown model advantage retains persistence. Stale input, release expiry and
missed first-horizon deadlines fail closed. Stop the process to roll back.

For an isolated container, build from the repository root:

```text
docker build -f research/Dockerfile.forecast -t forecast-demo .
docker run --rm -p 127.0.0.1:8000:8000 -e FORECAST_API_TOKEN forecast-demo
```

The Docker recipe has not been built here because Docker is unavailable. Supply
TLS, rate limiting, persistent log collection, load checks and a prospective
rollout before external production exposure. The seven-day release expiry
requires an explicitly revalidated release with a new preregistered cutoff.

## Evidence boundary

Forecast MAE and empirical intervals are research diagnostics, not net portfolio
alpha. A 30-day run has only four complete test-days and cannot meet the minimum
ten-day support for a paired uncertainty claim. Tests enforce causal feature
construction and serving contracts; they do not establish exact historical
arrival times, stable market alpha or optional DL-library compatibility.

The bot's full Truth Harness independently returned FAIL TH-11 on 2026-10-04:
the existing portfolio artifact did not match the current replay source hash.
This demo does not repair or waive that production-evidence finding.
