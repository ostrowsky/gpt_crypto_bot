# BTC Benchmark Admission Replay

Status: research-only complete; all profiles rejected or insufficient; production unchanged

Last updated: 2026-07-14

## Problem

On 2026-07-14 BTC produced two strong, operator-visible moves without a BUY:

- a `15m impulse_speed` candidate was rejected because two altcoin positions
  already occupied the `15m_impulse` cluster cap;
- a `1h breakout` candidate scored `33.09` against the universal top-gainer
  floor of `34`, despite strong causal structure before the later overbought
  phase.

BTC is both a tradable instrument and the benchmark used by the strategy's
market context. Treating it exactly like an altcoin inside every momentum cap
and same-day top-gainer gate may suppress useful market-leader entries.

## Objective Fit

- Surface the beginning of a broad-market BTC move before it becomes
  overbought.
- Improve selection under the unified ten-position portfolio without lowering
  the global admission floor for altcoins.
- Preserve causal, explainable rules and the existing exit policy.

## Hypotheses

1. `btc_cluster_exempt`: BTC remains inside the unified portfolio cap, but does
   not consume or compete for an altcoin signal-cluster slot.
2. `btc_1h_leader_admission`: only a BTC `1h breakout` may bypass the universal
   top-gainer floor when all of these decision-time conditions hold:
   - top-gainer score is at least `32`;
   - candidate score is at least `120`;
   - RSI is in `[55, 70]`;
   - ADX is at least `20`;
   - relative volume is at least `2`;
   - intraday change is at least `1%`;
   - daily range is no more than `5%`.
3. `btc_benchmark_combined`: apply both hypotheses together to detect harmful
   interaction effects.
4. `btc_benchmark_rotation`: apply the combined admission path and, only when
   the unified portfolio is full, allow a qualifying BTC `1h breakout` to
   replace the weakest non-BTC position when that position:
   - has been held for at least two bars;
   - is currently losing at least `0.25%`;
   - is selected by worst current return, so profitable positions are never
     sacrificed by this override.

   The BTC rotation profile is frozen at top-gainer score `>=32`, candidate
   score `>=120`, RSI `<=72`, ADX `>=18`, relative volume `>=2`, intraday
   change `>=1%`, and daily range `<=5%`.

The thresholds are frozen before replay. They deliberately reproduce the
observed pre-overbought BTC structure without relaxing any altcoin rule.

## Scope

Now:

- add replay-only variants for the frozen admission, cluster, combined, and
  losing-alt rotation hypotheses;
- keep the current `score_replace_cluster` policy as the comparator;
- run all variants over the maximum project-supported full-watchlist period:
  44 days, 105 symbols, `15m` and `1h`, with `4h` context;
- enforce candle-close causality throughout the replay: a Binance candle may
  affect a decision only at `open_time + interval`, including MTF/4h context,
  exits, cooldowns, and replacements;
- reproduce the live discovery grace path causally: a signal may be admitted
  from either of the prior two closed bars only when current slippage is at most
  `0.45%`, while candidate scoring retains the discovery-time movement and
  forecast bonuses used by live monitoring;
- report whole-window and chronological 70/30 holdout results after a
  conservative 20 bps round-trip fee/slippage deduction.

Not now:

- no live BUY, alert, cluster, score, ranking, replacement, or exit change;
- no dedicated eleventh portfolio slot;
- no global top-gainer threshold reduction;
- no tuning on holdout results.

## Primary Metrics

- summed net trade-return PnL and delta versus `score_replace_cluster`;
- net average trade return and win rate;
- trade count and candidate pressure;
- BTC trade count and net BTC PnL;
- count of actual BTC cluster and score-gate bypass admissions;
- count and outcome of BTC-for-losing-alt benchmark replacements;
- replacement, cluster-cap, and top-gainer skip counts;
- chronological holdout stability.

## Acceptance Criteria

A hypothesis may advance only to shadow consideration when:

- it has at least five actual policy-bypass admissions in the whole window;
- holdout net PnL is no worse than the comparator;
- holdout average net return is no worse than the comparator;
- holdout win rate is no more than one percentage point worse;
- whole-window BTC net PnL is positive;
- the combined variant shows no harmful interaction.

Failure of the sample-size gate means `insufficient_evidence`, not approval.
Failure of a quality gate means rejection. Neither result changes production.

## Risks / Trade-Offs

- BTC and altcoins are correlated, so removing BTC from a signal cluster can
  increase directional concentration even while total positions stay capped.
- A BTC-specific score exception can overfit a single observed rally.
- A 44-day portfolio replay is the maximum period previously established for
  full 105-symbol `15m` parity, but it spans fewer macro regimes than the
  multi-year `4h/1d` archive.
- Backtest fills use candle closes and a fixed cost allowance, not order-book
  execution.

## Backtest / Verification Gate

- `python -m unittest test_btc_benchmark_replay`
- `python files/audit_btc_benchmark_replay.py --days 44 --json`
- relevant replay regression tests;
- explicit candle-close timestamp regression tests;
- `git diff --check`.

The report must preserve the frozen thresholds, exact window, symbol coverage,
cost assumption, split timestamp, and per-variant decision.

## Backtest Evidence

Final causal/live-parity run:

- window: `2026-05-31T17:49:01Z` through `2026-07-14T17:49:01Z`;
- chronological split: `2026-07-01T13:01:01Z` (70/30);
- universe: 105 requested watchlist symbols, 95 with valid Binance `15m`,
  `1h`, and `4h` history;
- unavailable on all requested intervals: `SNTUSDT`, `ACAUSDT`, `MKRUSDT`,
  `LRCUSDT`, `RNDRUSDT`, `MDTUSDT`, `EOSUSDT`, `OXTUSDT`, `BAKEUSDT`,
  `TRUUSDT`;
- causal candidates: 64,481;
- cost: 20 bps per completed trade.

The audit uncovered and corrected two pre-existing replay parity defects before
the final run:

- candle opens had been treated as candle closes, exposing one future bar to
  MTF/context/exit decisions;
- replay omitted the live two-bar discovery catch-up path and used a divergent
  top-gainer score formula.

After correction, the observed 2026-07-14 BTC catch-up is reproduced at the
14:00 UTC decision with top-gainer score `32.96` versus live `33.09`; both are
below the floor of `34` and pass the frozen BTC exception profile.

The PnL totals below are summed per-trade percentage returns after costs, not a
capital-weighted equity curve.

| Variant | Policy admissions | Summed net PnL | Delta vs current | Holdout delta | BTC net PnL | Result |
|---|---:|---:|---:|---:|---:|---|
| Current `score_replace_cluster` | 0 | -851.9851% | - | - | -0.9685% | comparator |
| `btc_cluster_exempt` | 2 | -853.0235% | -1.0384% | +0.0000% | -2.0069% | insufficient and worse |
| `btc_1h_leader_admission` | 0 | -851.9851% | +0.0000% | +0.0000% | -0.9685% | no portfolio admission |
| `btc_benchmark_combined` | 2 | -853.0235% | -1.0384% | +0.0000% | -2.0069% | insufficient and worse |
| `btc_benchmark_rotation` | 1 | -852.1087% | -0.1236% | +0.9148% | -0.8128% | insufficient evidence |

The rotation variant reproduced the current incident: its one holdout BTC
replacement earned `+1.1941%` net and improved holdout portfolio PnL by
`+0.9148` percentage points. It still fails promotion because this is only one
replacement, whole-window PnL is worse, and whole-window BTC net PnL remains
negative.

## Decision

- Do not promote any BTC admission, cluster, or replacement relaxation.
- Keep the live top-gainer, cluster, replacement, and ten-position limits
  unchanged.
- Retain the corrected replay harness because prior results from the divergent
  timestamp/score path must not be used as production evidence.
- If the rotation idea is revisited, collect at least four more independent
  causal cases in shadow and rerun the same frozen profile; do not retune it to
  the single successful 2026-07-14 event.

## Rollback

The variants exist only in `replay_backtest.py` and are never selected by the
live bot. Remove the four CLI choices and their replay helpers to roll back the
research harness.
