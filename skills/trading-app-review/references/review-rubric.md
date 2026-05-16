# Review Rubric

Score each axis as:
- `strong`
- `adequate`
- `weak`
- `inconclusive`

## 1. Goal Fit

Check whether the reviewed system is optimized for:
- early trend discovery;
- short-horizon alpha;
- mean reversion;
- market making;
- portfolio allocation;
- discretionary decision support.

Do not treat a strong system for another objective as directly superior to this bot.

## 2. Architecture

Look for:
- separate live, research, evaluator, and control-plane paths;
- bounded queues and backpressure;
- event schemas;
- restart recovery;
- idempotent actions;
- latency isolation for UI/control paths;
- minimal coupling between heavy analysis and user interaction.

## 3. Data And Features

Look for:
- multi-timeframe joins;
- survivorship and universe control;
- missing-data handling;
- feature versioning;
- freshness monitoring;
- leakage prevention;
- market-regime/context features;
- symbol-relative and cross-sectional features.

## 4. Signal And Ranking

Look for:
- candidate generation separated from action selection;
- explicit ranking under scarcity;
- uncertainty or calibration;
- candidate recall versus action precision split;
- event-level explainability;
- ability to catch slow trends and fast impulses without collapsing them into one threshold.

## 5. Exit And Risk

Look for:
- MFE/MAE tracking;
- exit attribution;
- reversal detection;
- trailing logic versus regime;
- portfolio constraints;
- re-entry policy;
- cooldown harm measurement;
- symbol and cluster exposure control.

## 6. Learning Loop

Look for:
- objective-aligned labels;
- delayed feedback;
- hard negatives;
- replayable training sets;
- drift checks;
- promotion criteria;
- separation of teacher metrics from realized PnL;
- avoidance of reward hacking.

## 7. Observability

Look for:
- why-signal / why-no-signal traces;
- block chains;
- daily quality reports;
- model diagnostics;
- live-versus-replay diffing;
- latency and failure dashboards.

## 8. Experiment Discipline

Look for:
- walk-forward splits;
- holdout periods;
- transaction costs;
- stable benchmark definitions;
- ablations;
- canary/shadow deployment;
- rollback plan.

## 9. SOTA Comparison

Compare against current professional practice in:
- event-driven architecture;
- real-time feature pipelines;
- portfolio-aware ranking;
- calibration and uncertainty;
- model monitoring;
- shadow evaluation;
- post-trade analytics;
- execution/risk separation;
- operator observability.

Use fresh external sources when making these claims.
