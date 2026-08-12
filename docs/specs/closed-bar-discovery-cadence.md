# Closed-bar Discovery Cadence

Date: 2026-08-12
Status: production behavior repair

## Problem

Dynamic discovery was configured for 300 seconds while the normal monitoring
poll is 60 seconds. A newly closed C98USDT 15m breakout was therefore routed
about five minutes after its candle closed. The candidate was evaluated and a
blocked strong-signal alert was delivered, but the avoidable discovery delay
made the signal appear missing.

The routing loop also returned from the complete scan when its first ranked
candidate was already represented by an equal-or-better hot-list report, or
already had an open position. That symbol-local outcome incorrectly prevented
all later candidates from being routed.

## Required behavior

- A symbol outside the current hot list is rescanned at least once per normal
  monitoring poll.
- Discovery continues to use only closed candles and the same BUY gates as the
  main monitoring path.
- Faster discovery does not bypass the top-gainer score floor, objective gate,
  cooldown, portfolio capacity, or any other admission rule.
- An unchanged hot symbol or an already-open symbol skips only itself; it cannot
  abort processing of later candidates from the same discovery scan.
- BTCUSDT without a valid raw trend/breakout/retest/impulse condition remains a
  non-signal; this repair does not manufacture an alert from chart direction.
- Every discovery run logs its duration, number of routed candidates, and final
  hot-list size so the faster cadence can be checked for overlap/load regressions.

## Acceptance criteria

- `DISCOVERY_SCAN_SEC <= POLL_SEC`;
- a deterministic routing test proves that an unchanged first candidate and an
  open second candidate do not suppress a new third candidate;
- the existing discovery-cadence regression test passes;
- the why-no-signal report shows actual runtime blockers when critic writes are
  absent;
- no production BUY/SELL threshold is relaxed.

## C98 score-gate hypothesis

The observed C98 event freezes a separate research-only rescue hypothesis:
15m `breakout`, candidate score `>=120`, live top-gainer score `[28, 34)`, and
volume ratio `>=5`. It must be tested on the maximum available historical
period with chronological holdout and portfolio comparison. It is not enabled
in production unless the target and risk gates pass; the broad 32-34 rescue
remains rejected.

### Maximum-period result

The frozen profile was evaluated over every available mature
`blocked_strong_score_gate` label from 2026-06-28 through 2026-08-11, joined to
final-critic top-mover labels, actual entries, reconstructed portfolio capacity,
and cached forward candles. The 2026-08-12 complaint case is not used to select
or validate the rule because it is not yet mature.

| Segment | Eligible | Top movers | T+10 avg | T+10 median | Positive |
|---|---:|---:|---:|---:|---:|
| all mature | 63 | 1 | -0.21% | -0.41% | 41.3% |
| chronological holdout | 27 | 1 | -0.32% | -0.45% | 37.0% |
| recent 14-day stability | 25 | 1 | -0.38% | -0.45% | 36.0% |

Decision: reject the rescue. Do not lower the production score gate and do not
advance this profile to WATCH. A full-policy portfolio replay is unnecessary
because the causal candidate-level return and precision gates already fail.
The earlier broad 32-34 rescue also remains rejected.

Historical C98 profile cases reinforce the decision: the only capacity-eligible
not-yet-bought case had T+10 net return `-1.54%`; two other C98 cases were already
bought or capacity-blocked and also had negative T+10 outcomes.
