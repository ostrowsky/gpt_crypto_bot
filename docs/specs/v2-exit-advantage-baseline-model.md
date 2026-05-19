# V2 Exit Advantage Baseline Model

Status: research-only  
Last updated: 2026-05-19

## Purpose

Train the first simple supervised baseline on the action-level exit advantage
dataset.

The goal is not to promote a new SELL rule. The goal is to answer a narrower
research question:

```text
Can causal in-position features predict when SELL now has positive advantage over
continuing with the fixed temporal candidate hold/exit path?
```

## Input

- `.runtime/reports/v2_action_level_exit_advantage_15m.jsonl`

Each row represents one in-position frame and contains:

- current symbol belief masses;
- projected v1 structural features;
- short temporal deltas;
- position context through reward labels;
- `sell_advantage` target.

## Model

Use a deliberately transparent baseline:

- chronological train / holdout split by timestamp;
- train-only feature standardization;
- ridge linear regression predicting `sell_advantage`;
- threshold sweep over predicted advantage to convert scores into SELL decisions.

This is intentionally simpler than RL. It tests whether the label is learnable
before adding policy complexity.

## Metrics

Report:

- train / holdout row counts and timestamp ranges;
- regression error and directional accuracy;
- precision / recall for `sell_advantage_positive`;
- precision / recall for `sell_advantage_strong`;
- predicted SELL rate for each threshold;
- selected threshold by holdout reward proxy.

## Acceptance Criteria

This package passes as a research artifact if:

- the chronological split is explicit;
- model coefficients are saved with feature names;
- holdout metrics are reported;
- the selected threshold is based on holdout metrics, not in-sample fit.

It does not authorize live SELL changes or Telegram signals. A model can only
advance after a full offline replay proves improvement over the fixed temporal
candidate policy.


## Training Result

Latest run on the corrected action-level dataset:

- rows: `99,935`;
- train rows: `69,954`;
- holdout rows: `29,981`;
- train average `sell_advantage`: `-0.902032`;
- holdout average `sell_advantage`: `+0.757862`;
- holdout directional accuracy: `0.438344`;
- holdout predicted-positive rate: `0.216570` vs actual positive rate `0.573296`;
- best threshold by captured holdout advantage: `-2.0`;
- captured advantage at selected threshold: `20,469.654760`;
- always-sell holdout proxy: `22,721.463582`;
- oracle-positive-sell proxy: `65,048.375943`.

## Interpretation

The baseline trained successfully, but the first linear model is **not yet a
credible exit policy**. It generalizes poorly across the chronological split: the
training slice is hold-favorable on average, while the holdout slice is
sell-favorable on average. The selected low threshold effectively sells most
frames (`91.73%`) and still underperforms a naive always-sell proxy on the
holdout advantage labels.

This is useful evidence, not a failure of the architecture. It says the target is
learnable enough to generate non-random high-precision tiny slices, but a simple
linear model is not sufficient for the full decision surface.

## Decision

Do not advance this model to live shadow or Telegram.

Next research gate:

1. run a model-family comparison on the same chronological split:
   - linear ridge;
   - feature-binned lookup / decision table;
   - small kNN or nearest-neighbor model over standardized causal features;
2. require improvement over `always_sell` and `never_sell` proxy baselines;
3. only then run full offline replay of the best candidate against the fixed
   temporal candidate policy.
