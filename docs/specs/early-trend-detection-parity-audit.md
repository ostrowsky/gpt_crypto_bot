# Early Trend Detection Parity Audit

Status: shipped bugfix  
Date: 2026-05-28

## Objective

Remove implementation mismatches that can delay early trend detection without
changing production thresholds.

The bot's objective remains early capture of same-day watchlist top movers with
a unified 10-position portfolio and exits that preserve MFE.

## Problems

### 1. V1 today-confirmed mode mismatch

`strategy.analyze_coin()` used a narrower signal set for today's forward
accuracy than the live monitor uses for candidate admission.

The accuracy window counted only:

- `check_entry_conditions()`
- `check_alignment_conditions()`

But the live path can admit:

- trend / strong trend
- breakout
- retest
- trend surge / impulse speed
- impulse
- alignment only when explicitly enabled

This mismatch can produce false `today_confirmed=False` / `accuracy_gate`
pressure for exactly the early modes we care about.

### 2. Replay/live intraday day-boundary mismatch

The live monitor and market signal agent use the Europe/Budapest local day for
intraday movement features. `replay_backtest.py` still used UTC midnight in its
intraday change helper.

This makes top-gainer score-gate replay evidence diverge from live behavior
around local/UTC day boundaries.

## Scope

This is a parity and measurement/replay correctness fix only:

- no threshold relaxation;
- no new BUY mode;
- no SELL logic change;
- no portfolio behavior change.

## Acceptance Criteria

1. `strategy.analyze_coin()` uses one helper for the current live-admissible V1
   mode set.
2. The helper includes breakout/retest/impulse/trend-surge and only counts
   alignment when `ALIGNMENT_BUY_ENABLED=True`.
3. Current-signal detection can emit `impulse_speed` directly from
   `check_trend_surge_conditions()`.
4. Replay intraday movement uses the same Europe/Budapest local-day open
   convention as the live monitor/agent.
5. Unit tests cover:
   - early-mode counting for today's accuracy;
   - alignment exclusion when disabled;
   - trend-surge current signal mode;
   - replay local-day intraday calculation.

## Promotion Gate

Because this can change admission timing, production thresholds remain
unchanged and the next daily learning report must be watched for:

- early capture change;
- false-positive BUY pressure;
- blocked-winner change;
- top-gainer score-gate behavior.
