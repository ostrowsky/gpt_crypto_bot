# Learning Loop Architecture Roadmap

Date: 2026-08-03
Status: active, reprioritized after the 2026-08-02 daily report

## Principle

Every major target metric should have three layers:

1. measurement: daily labels and attribution;
2. shadow learner: model/policy recommendation without trading side effects;
3. replay/promotion gate: production adoption only after robust evidence.

## 2026-08-03 Evidence Checkpoint

The 2026-08-02 daily report is diagnostic-only, not a production-policy
decision report:

- the final top-gainer critic is absent, so the daily watchlist-top denominator
  is unknown rather than zero;
- the 22:00 watchlist-goal job ran until after 03:00 while it shared one
  sequential scheduler loop with the strict 00:00-00:15 final-critic slot;
  the final slot was therefore never observed;
- signal-quality candle coverage is `184 / 206`, but `20 / 22` missing series
  belong to ten symbols whose exchange status is `BREAK`; only the two missing
  `TONUSDT` series are currently `TRADING`;
- the tail-selector, entry-admission, blocker-reward, and portfolio-replacement
  caches were generated on 2026-06-13, before the current cooldown and guarded
  replacement policies, and cannot evaluate those production policies;
- rolling watchlist-top early capture is `72.73%`, while signal-quality reports
  a `74.47%` broad trend-episode miss rate. These metrics use different
  denominators and must not be treated as complements.

Consequently, no BUY/SELL threshold is relaxed from this checkpoint. The next
cycle restores trustworthy measurement first, then refreshes policy evidence on
the maximum available historical period and chronological forward slices.

## Current Gaps

### Entry / early capture

Current: V1 rules, ML ranker, V2 shadow radar.

Learning loop exists but is incomplete: V2 has high recall and low precision. Next learning loop should focus on calibrated admission precision, not more raw alerts.

### Selection / portfolio cap

Current: score gates and ranker training.

Learning loop exists partially through ML candidate ranker. Missing: explicit EV/risk calibration under 10-slot portfolio cap and replacement uplift attribution.

### Blocked winners

Current: structured blocked logging and critic.

Learning loop is measurement-heavy, not adaptive. Missing: per-blocker harm model that proposes candidate relaxations in shadow/replay only.

### Exit monetization

Current: V1 rule-based SELL, exit auditor, exit discriminator, suspicious re-entry replay.

Learning loop did not affect live behavior until shadow re-entry alerts. Next: daily scoring of shadow re-entry precision and forward returns.

### Risk / exposure

Current: portfolio cap and group cap.

Missing: correlation/exposure learner in shadow mode, then replay-gated cap adjustments.

## Recommended Order

### P0: restore decision-grade measurement

1. Split the watchlist-goal and top-gainer-critic schedulers so a long goal job
   cannot block midday/final critic slots.
2. Add persistent due-slot catch-up: if a final artifact is absent after its
   nominal window, retry it independently until it succeeds or records an
   explicit terminal error. A process restart must not erase the due slot.
3. Make critic-dataset rewrites single-writer and atomic with bounded retry;
   expose permission/rename failures in worker health instead of only logs.
4. Separate active-universe coverage from retired/broken-symbol coverage.
   Retry missing `TRADING` series; report `BREAK` series as excluded legacy
   coverage rather than as equivalent live-data failures.
5. Add freshness budgets for every research cache and report the source-policy
   version/config hash used to build it.

P0 acceptance gate:

- seven consecutive target days have a final critic before the 09:00 learning
  report, including restart recovery;
- no scheduled final slot is lost when the 22:00 goal report runs for more than
  two hours;
- active-universe candle coverage is 100%, or every remaining active miss has a
  named terminal reason and is excluded from decision-grade metrics;
- no critic-dataset permission/rename failure occurs during the observation
  window;
- stale research artifacts cannot produce a current production recommendation.

### P1: refresh the evidence under current production policy

Run every comparison on the maximum available historical period, with
chronological splits and a separate post-deployment forward slice. Include fees,
slippage, portfolio capacity, and the canonical watchlist-top denominator.

1. Re-run cooldown `2` versus the former `8` control. Compare early capture,
   cooldown harm, false positives, precision, realized PnL, and capacity cost.
   Do not relax cooldown further from a before/after comparison alone.
2. Re-run the guarded non-losing replacement policy against no replacement and
   the former replacement behavior. The stale June report must not be used to
   describe the guarded policy shipped in July.
3. Audit the score-34 near-miss band with a frozen control: attribute candidates
   to detection, score admission, other blockers, portfolio capacity, and later
   outcome. Test score `32-33` first as WATCH/shadow and promote to BUY only if
   objective uplift survives all chronological windows without unacceptable
   precision or PnL loss.
4. Refresh observable exit-tail labels and replay the selector against the
   current SELL control. The relevant pressure is `22` early exits, median exit
   efficiency `0.00`, median giveback `70.56%`, and negative median closed PnL;
   lower giveback alone is not sufficient if realized reward falls.
5. Rebuild entry-admission and blocker-reward reports only after final-critic
   backfill is complete. Keep all admission/blocker relaxations shadow-only
   until a targeted replay passes.
6. Continue suspicious re-entry label collection. Zero alerts and zero mature
   labels provide no promotion evidence.

P1 promotion gate:

- the primary decision metric is earlier same-day watchlist-top capture;
- false-positive pressure, realized PnL, drawdown, turnover, and portfolio
  capacity are mandatory guardrails;
- an aggregate improvement that reverses in the latest or a material regime
  window is not promotable;
- production changes require focused automated tests, a rollback switch, and a
  frozen replay artifact with policy/config provenance.

### P2: improve attribution before adding model complexity

1. Decompose broad missed trends into: not observed, observed but not signaled,
   signaled but rejected, blocked by portfolio capacity, entered late, and exited
   early.
2. Publish both metric families without conflation: watchlist-top early capture
   for the north star, and broad trend-episode miss rate for lifecycle coverage.
3. Rank the daily failure casebook by realizable opportunity after the decision
   point, not by hindsight daily gain.
4. Resume calibrated V2 admission, portfolio EV/risk ranking, blocker harm, and
   correlation/exposure learners only after P0 is stable and their training
   labels carry current-policy provenance.
