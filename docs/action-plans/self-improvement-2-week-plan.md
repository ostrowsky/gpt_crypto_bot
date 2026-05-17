# Two-Week Self-Improvement Action Plan

Last updated: 2026-05-17

## Week 1 — Restore Measurement Trust

### Days 1-2
- Ship P0 measurement hardening.
- Baseline the longest available history:
  - progress report;
  - trend-start funnel;
  - exit-quality audit.
- Freeze any BUY-gate relaxation until the new funnel is reviewed.

### Days 3-4
- Review the first-loss funnel for final top movers.
- Rank loss points by:
  - number of missed winners;
  - opportunity from first block;
  - whether wake-up / structural evidence existed before the block.
- Form at most two narrow hypotheses:
  1. one recall hypothesis;
  2. one exit-lifecycle hypothesis.

### Days 5-7
- Replay those hypotheses on 7d and widest feasible holdout window.
- Reject any change that improves recall while damaging 30d PnL / precision.
- Publish a one-page decision note for each hypothesis.

## Week 2 — Convert Diagnosis Into Learning

### Days 8-10
- Add blocker-harm decomposition by first-loss point and reason.
- Add post-deploy comparison rows to the progress report:
  - baseline window;
  - after-change window;
  - objective delta.

### Days 11-12
- Run the portfolio replacement grid only on windows where the portfolio actually reached 10/10.
- Keep EV/ranker experiments shadow-only unless they improve top-gainer alignment, not only return.

### Days 13-14
- Produce a two-week review:
  - did `capture_rate` improve;
  - did `early_captures` improve;
  - did `giveback_pct` fall;
  - which hypothesis became durable knowledge.
- Promote only replay-confirmed changes; archive failed hypotheses with evidence.

## Success Criteria After 14 Days

- No long-window report failures.
- Every day has an explicit coverage status.
- Every missed final top mover has a first-loss funnel classification.
- Exit audit is available daily and rolling.
- At least one hypothesis has completed the full loop:
  `metric -> hypothesis -> replay -> decision -> post-deploy review`.
