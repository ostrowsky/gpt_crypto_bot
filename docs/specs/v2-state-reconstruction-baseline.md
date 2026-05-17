# V2 State Reconstruction Baseline

Status: research-only  
Last updated: 2026-05-18

## Purpose

Test whether lifecycle states can be reconstructed from information available at time
`t` before introducing probabilistic filtering or RL.

## Core Rule

Teacher labels may use hindsight. Reconstruction features may not.

Every row therefore keeps:

- `label_time`
- `observation_cutoff_time`
- causal features computed only from bars up to the current bar.

## Baselines

The first package compares:

1. `majority_class`
   - always predicts the most frequent training label;
2. `nearest_centroid`
   - confidence-weighted class centroids over causal features;
3. `provisional_shadow_rules`
   - the existing deterministic shadow observer mapped to lifecycle labels.

## Features V1

- close return over 1 and 4 bars;
- price vs EMA20;
- EMA20 slope;
- ADX;
- RSI;
- volume multiple;
- MACD histogram normalized by price;
- daily range so far.

These are all available on the current closed bar.

## Split

- chronological split by local day;
- first `70%` of days train;
- final `30%` of days test;
- no random row shuffling.

## Metrics

- confidence-weighted accuracy;
- macro F1;
- per-state recall;
- rows and days per split.

## Acceptance Criteria

1. No future-dependent feature enters the dataset.
2. OOS split is chronological by day.
3. Reconstruction baseline beats `majority_class` on macro F1.
4. Report explicitly states whether `emerging_move` recall is useful or still inadequate.
5. No live trading import.

