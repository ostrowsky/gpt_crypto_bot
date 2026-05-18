# V2 Entry Admission Dataset

Status: research-only  
Last updated: 2026-05-18

## Purpose

Reuse the useful maturity of v1 when building the next v2 layer instead of
re-learning admission from zero.

The dataset joins:

1. calibrated v2 belief trajectories;
2. v1 candidate / blocker decision features;
3. v1 temporal evidence such as scout / wake-up history;
4. hindsight lifecycle labels.

## Why This Exists

The current v2 policy-gap audit found that the best threshold policy still opened:

- `1529` entries in true `noise`;
- only `373` in true `emerging_move`.

v1 has already evolved around a related question: which apparently interesting
candidates are not safe enough to allocate capital to. That knowledge should become
features, weak priors, and baselines for v2 admission rather than being discarded.

## Feature Groups

### V2 belief features

- belief mass for every lifecycle state;
- dominant state;
- max probability;
- entropy.

### V1 structural features

Pulled from `critic_dataset.jsonl` when the same symbol / timeframe / bar exists:

- `candidate_score`;
- `base_score`;
- `score_floor`;
- `forecast_return_pct`;
- `today_change_pct`;
- `ml_proba`;
- `mtf_soft_penalty`;
- `fresh_priority`;
- `catchup`;
- `continuation_profile`;
- `near_miss`;
- `signal_flags`.

### V1 temporal evidence

Pulled from `bot_events.jsonl`:

- whether a prior structural `scout_shadow` exists;
- whether a prior wake-up `scout_shadow` exists;
- minutes since first structural scout;
- minutes since latest wake-up.

## Explicit Non-Goals

- Do not copy v1 BUY gates into v2 as ground truth.
- Do not use v1 mode names as hidden-state labels.
- Do not train admission in this first package.

## Acceptance Criteria

1. Rows are keyed by the same OOS v2 symbol / bar contract used by prior audits.
2. v1 joins are optional and coverage is reported honestly.
3. Temporal evidence uses only prior events, never future events.
4. Output distinguishes:
   - belief-only rows;
   - rows enriched by v1 structural data;
   - rows enriched by v1 temporal data.
5. First audit reports whether v1 enrichment is available at useful scale before any
   model is trained.

## First Audit Result

On the current OOS slice:

- total rows: `163305`;
- exact v1 structural matches from `critic_dataset.jsonl`: `1565` (`0.958%`);
- rows with prior structural scout evidence: `78753` (`48.224%`);
- rows with prior wake-up scout evidence: `5716` (`3.500%`).

Interpretation:

- v1 temporal evidence is already reusable at useful scale;
- exact historical v1 decision-feature joins are too sparse to support the main
  admission model by themselves;
- the next admission package must either:
  - project v1-style structural features directly from canonical bars, or
  - build a broader historical v1 feature replay,
  before comparing `belief_only` versus `belief_plus_v1`.

## Next Gate

After this dataset exists:

1. compare `belief_only`, `v1_features_only`, and `belief_plus_v1` admission baselines;
2. measure noise-admission reduction and emerging-move retention;
3. only then decide whether admission should be thresholded, supervised, or contextual.
