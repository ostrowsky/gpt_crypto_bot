# V2 Market Breadth Observation Store

Status: research-only  
Last updated: 2026-05-19

## Purpose

The environment-belief track has shown that adaptive policy switching is valuable
under oracle labels, but current causal classifiers cannot infer the winning
policy family reliably enough.

The next hypothesis is that the bot is missing true external-environment
observations. Existing features are mostly aggregated from symbol lifecycle rows
and projected v1 structure. This spec adds a canonical market-breadth observation
layer derived directly from continuous OHLCV history.

## Hypothesis

If we build causal market-breadth snapshots from the canonical history store,
then horizon policy-advantage classification should improve versus the prior
observation feature audit.

The important distinction:

```text
symbol lifecycle / candidate rows -> what individual coins are doing
market breadth snapshots        -> what the external game is doing
```

## Feature Families

For each anchor bar, using only bars at or before that anchor:

- coverage:
  - tracked symbol count;
  - available symbol count;
  - available share.
- breadth:
  - share with positive 1-bar / 4-bar / 8-bar / same-day return;
  - share above EMA20 / EMA50;
  - share with volume above rolling mean.
- dispersion:
  - mean / standard deviation of 4-bar, 8-bar, and same-day returns;
  - top-decile same-day return minus median same-day return.
- risk proxies:
  - BTC 4-bar and same-day return;
  - BTC price-vs-EMA20;
  - ETH 4-bar and same-day return;
  - ETH price-vs-EMA20.

## Diagnostic Protocol

Use the existing horizon policy-advantage samples from:

```text
.runtime/reports/v2_market_observation_features_15m.json
```

For each sample anchor:

1. load canonical `15m` history from `files/.runtime/v2_history`;
2. compute breadth features causally at the anchor timestamp;
3. evaluate nearest-centroid horizon classification chronologically for:
   - breadth-only features;
   - existing features plus breadth features.

## Acceptance Criteria

This audit does not authorize switched policy usage. It only decides whether the
market-breadth observation layer is worth keeping.

A feature set is promising only if it beats the majority baseline by at least
`3pp` on at least one target horizon while keeping meaningful prediction
coverage.

## Promotion Rule

Even if the audit passes, the result only authorizes the next research gate:

```text
market-breadth belief model -> switched replay
```

It does not authorize production BUY / SELL changes.


## Audit Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_market_breadth_observation_store.py --json
```

Coverage:

- tracked symbols: `99`;
- anchors: `179`;
- anchors with data: `179`;
- anchor data share: `1.0`.

| Horizon | Feature set | Accuracy | Majority | Edge | Verdict |
|---|---|---:|---:|---:|---|
| `1h` | breadth only | `0.581921` | `0.608939` | `-0.027018` | below majority |
| `1h` | existing + breadth | `0.621469` | `0.608939` | `+0.012530` | near majority |
| `2h` | breadth only | `0.617143` | `0.636872` | `-0.019729` | near majority |
| `2h` | existing + breadth | `0.605714` | `0.636872` | `-0.031157` | below majority |

## Interpretation

Market breadth improves the `1h` combined feature set and eliminates
wrong-confident errors there, but the edge is only `+1.25pp`, below the required
`+3pp` gate. The `2h` result does not improve.

So this layer is useful as an observation primitive, but not sufficient as the
basis for a switched policy or RL policy selector.

## Decision

Keep the market-breadth observation store as research infrastructure, but do not
promote a market-environment belief model from it yet.

The next research gate should test feature selection / regularization and
conditional use of breadth features, rather than blindly adding all breadth
features to the classifier.
