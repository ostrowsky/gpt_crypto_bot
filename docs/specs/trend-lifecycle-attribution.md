# Trend Lifecycle Attribution

Date: 2026-08-03
Status: implemented, measurement-only

## Objective

Turn the broad trend miss rate into causal failure stages without conflating it
with the canonical watchlist-top early-capture metric. Rank remediation cases by
movement remaining after the bot's actual decision point, not by hindsight day
gain.

## Complete detail contract

New signal-quality reports export every missed-trend, late-entry, early-exit,
and false-positive detail row. `detail_coverage` records total, exported, and
complete counts for each bucket. Historical reports produced under the former
`100 / 50` row caps remain usable but are explicitly `partial_historical_detail`.
They must not be presented as a complete historical denominator.

## Causal joins

Blocked events are stored in incremental 15-minute intervals containing exact
first/last timestamps, first price, reason, source, timeframe, and event count.
An interval is joined only when it overlaps trend start through trend peak.
Research-universe observations and V2 shadow transitions provide independent
observed/signal evidence. Episodes before the observer's deployment window are
`observation_coverage_unavailable`, never silently classified as not observed.

Missed-trend stage precedence is:

1. `blocked_by_portfolio_capacity` when an overlapping capacity blocker exists;
2. `signaled_but_rejected` when an overlapping blocker or shadow signal exists;
3. `observed_but_not_signaled` when observations exist without a signal;
4. `not_observed` only inside the observer coverage window;
5. `observation_coverage_unavailable` outside that window.

Caught-trend failures are reported separately as `entered_late` and
`exited_early`; these are not added to missed-trend counts.

## Opportunity ranking

For blockers and shadow signals, the decision point is the earliest overlapping
event with an observed price. For late entries and early exits it is the actual
entry or exit. Remaining peak opportunity includes a fixed 20 bps round-trip
cost. A case without a causal decision price remains visible in attribution but
is excluded from the ranked casebook.

## Runtime modes and rollback

- `report_trend_lifecycle_attribution.py --lookback-days 0` runs the maximum
  available research audit.
- The worker runs a bounded 14-report nightly window after signal-quality
  completion and reuses the incremental cohort index.
- `TREND_LIFECYCLE_ATTRIBUTION_ENABLED=False` disables the nightly report.
- `TREND_LIFECYCLE_ATTRIBUTION_LOOKBACK_DAYS` changes only measurement scope.

The report cannot alter BUY, SELL, blocker, score, or portfolio behavior.
