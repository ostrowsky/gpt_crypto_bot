# Exit Failure Discriminator

Date: 2026-05-29
Status: research-only, no live SELL changes

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

## Ground Truth Label

A case is labeled `wrong_exit_continuation` when post-exit/future favorable movement exceeds the already-seen favorable movement by a configured margin.

This is not perfect causal truth, but it is a better research label than subjective chart review because it directly asks:

> after this SELL, did the market provide enough additional favorable movement that a smarter exit policy might have captured?

## Guardrails

- Research-only report/dataset.
- No production SELL changes.
- No rule adoption without candle-path replay.
- Do not optimize only for fewer exits; late loss and giveback must be monitored separately.
- Treat partial case coverage as weaker evidence.

## Promotion Gate

A future SELL policy may be proposed only if the discriminator identifies stable high-risk segments out-of-sample and a replayed policy improves PnL/exit efficiency without materially increasing giveback or downside.
