# Exit Failure Discriminator

Date: 2026-09-08
Status: online shadow learning, no live SELL changes

## Problem

Recent exit experiments showed that simple protected-hold rules make results worse in causal candle-path replay. The failure is not just timing; the bot needs to distinguish two different states at SELL time:

1. the exit is correct because the trend is actually ending;
2. the exit is probably wrong because price continues favorably after the SELL.

## Goal

Build a research-only discriminator over historical signal-quality cases that estimates whether a SELL was a likely exit failure.

The first implementation must not change live exits. It should produce:

- labeled exit cases;
- feature buckets available at or near SELL time;
- train/test day split;
- baseline wrong-exit rate;
- top-risk precision versus baseline;
- high-risk feature segments ranked by downside/opportunity loss.

The headless worker refreshes this learner every training interval from all
mature `signal_quality_*_final.json` trade cases.  A new final report therefore
becomes learning evidence without waiting for a fixed cohort size.  The report
and model are persisted separately under `.runtime`; neither is a production
model.

## Ground Truth Label

A case is labeled `wrong_exit_continuation` when post-exit/future favorable movement exceeds the already-seen favorable movement by a configured margin.

This is not perfect causal truth, but it is a better research label than subjective chart review because it directly asks:

> after this SELL, did the market provide enough additional favorable movement that a smarter exit policy might have captured?

The production default margin is `0.75` percentage points beyond the greater
of PnL at exit and maximum favorable excursion already observed by SELL. The
label matures only after the post-exit evaluation window is available.

## Causal Feature Contract

The online shadow model may use only fields known when SELL is emitted:

- normalized exit reason, signal source, mode and timeframe;
- realized PnL bucket at SELL;
- maximum favorable excursion, giveback and exit-efficiency buckets calculated
  over the already completed holding path.

Retrospective `entry_timing`, `exit_timing`, final-day capture ratio and final
top-mover rank are forbidden model features. They may appear in diagnostics or
labels but cannot affect the risk score.

## Maximum-period validation (2026-09-08)

All available mature final reports produced 4,273 labeled exits. The
chronological split used 2,730 training rows and 1,543 later test rows. On the
test period the causal learner selected 309 high-risk exits (top 20%); 167 were
wrong-exit continuations (54.0%) versus 436/1,543 (28.3%) overall, lift 1.91x.

This is a terminal `promising_shadow_segments_only` result. It proves useful
classification on a later time window, not portfolio improvement and not the
safety of delaying exits. Live SELL and cooldown behavior remain unchanged.

## Guardrails

- Online learning is shadow-only (`runtime_eligible=false`,
  `production_effect=none_shadow_only`).
- Model features are restricted to values observable at SELL: normalized exit
  reason, source, mode, timeframe, realized PnL bucket, MFE bucket, giveback
  bucket and exit-efficiency bucket.
- Retrospective `entry_timing`, `exit_timing`, end-of-day capture ratio and
  final top-mover rank are prohibited model features.
- The label matures only after exit and is never used as an input feature.
- Evaluation is a chronological day holdout and publishes numerator,
  denominator, base rate and lift for the highest-risk 20%.
- No production SELL changes.
- No rule adoption without candle-path replay.
- Do not optimize only for fewer exits; late loss and giveback must be monitored separately.
- Treat partial case coverage as weaker evidence.

## Canary

The first deployment is artifact-only. Confirm that the headless worker writes
a fresh report/model pair, that the morning report renders exact OOS counts,
base rate and lift, and that both artifacts continue to declare production
effect `none_shadow_only`. No SELL, cooldown or re-entry code consumes the
model during this canary.

## Rollback

Set `EXIT_FAILURE_ONLINE_LEARNING_ENABLED=False` and restart the headless
worker. The last shadow artifacts may remain for audit but become stale; live
trading behavior is unaffected because the model is not production-eligible.

## Promotion Gate

A future SELL policy may be proposed only if the discriminator identifies stable high-risk segments out-of-sample and a replayed policy improves PnL/exit efficiency without materially increasing giveback or downside.

## Morning Report Contract

The daily learning report must expose the post-exit learner separately from the
entry ranker: data-through day, train/test counts, OOS high-risk wrong-exit
count and precision, full-test base count/rate, lift, and explicit production
OFF state.  Missing, stale or non-causal evidence must not be described as
successful learning.
