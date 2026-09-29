# Rocket capture and rule ablation

Status: research-only; production promotion is fail-closed.

## Objective and population

Audit the maximum available final top-mover reports and raw bot/agent events.
A rocket is an ex-post exchange top-mover with open-to-close return >=10%;
also report thresholds 15% and 20%. Final membership is a label, never an entry
feature. Keep non-watchlist rockets visible. Replay the whole cached watchlist
universe, including losers, not just selected winners.

## Measurement contract

Match entry and exit by source, symbol, timeframe, chronological order, and
entry price. Duplicate or overlapping entries invalidate the chain; unmatched
exits and open positions are unknown, not zero-profit trades. Events are signal
records, not exchange fills; legacy timestamps are emission times.

For a complete local calendar day, sum price differences of matched, same-day,
non-overlapping trades and divide by day_close-day_open. This is one-unit
price-path capture, not capital-weighted PnL. Do not clip negative or >100%
values. Cross-day and overlapping positions make this daily metric unknown.
Missing events cannot establish a missed trade: no-event rows remain unknown.
Report separately gross trade return and after-entry high retention, using
only complete candles starting after entry and ending by the day boundary.
The first-entry/last-exit shortcut is forbidden.

## Registered ablations

Rule-only closed-bar replay: baseline; chase guard off; top-gainer score gate
off; cluster cap off; cooldown off; RSI-overbought exit off; WEAK exit off;
RSI plus WEAK exits off; combined chase/score/cluster/cooldown/RSI/WEAK off.
Keep ATR stops, EMA invalidation, capacity ten, and capital constraints.
Rules are patched only inside the research process and always restored.
Entry-side ablations rebuild candidates; exit-side and capacity ablations
reuse identical candidate inputs. Removing RSI exit must reveal subsequent
EMA/ADX/WEAK rules, not return early and hide another reason.

Do not import the current trained model into old decisions: its training
provenance does not establish historical availability. Disable learned scores
in every research arm. This makes results rule-only diagnostics, not exact
production replay. Agent-only leader/mode restrictions need a distinct causal
agent candidate stream; do not claim the main-bot engine tests them.

## Backtest and acceptance

Use all locally cached periods, explicitly count missing/gapped/conflicting
candle series and keep them outside candidate generation. A 30-day block resets
positions equally in every arm; report this limitation. Use warmup candles,
closed features, and identical costs (7.5 bps fee + 5 bps slippage per side).
Report capital-constrained portfolio return/drawdown and BTC benchmark, not
sum of trade percentages as portfolio profit.

Calendar blocks are assigned chronologically to discovery/validation/holdout
before outcomes are computed. Holdout is retrospective and already discussed
QNT is not a pristine unseen experiment. Require positive paired net-return
improvement in validation and holdout, non-worse drawdown, at least 30 holdout
rocket cases, full candidate/population provenance, and forward shadow/canary
evidence before promotion. Missing parity or coverage means UNKNOWN even if
an ablation looks profitable. Preserve rejected arms and costs/coverage.

## Verification and rollback

Tests: trade pairing, ambiguity, repeated entries, cross-day handling,
denominators, future peaks, patch restoration, ablation isolation, date splits.
Run focused tests, maximum-period research, diff check, staged Truth Harness.
Rollback: remove the standalone research command; no live setting is changed.
Any later promotion requires a separate live switch and measured canary.

## Measurement repair and closed-candle recovery (2026-09-29)

The live critic must never combine first entry with last exit. Match a unique
source/symbol/timeframe chain with positive finite prices, chronological order,
and matching explicit entry price (and position ID when present). Ambiguity,
unmatched/open positions and cross-day chains produce UNKNOWN, not zero.
Publish versioned same-day one-unit realized capture separately from remaining
movement at first entry. Missing verified in-position MFE makes exit efficiency
and giveback null, with an explicit reason and denominator; daily high is not MFE.
Legacy exit-quality numbers must not be compared as the same definition.

Recovery writes a separate research cache, never replaces live/raw files.
Discard each legacy file's final candle: filename end is a requested horizon,
not evidence of collection after candle close. Merge only consistent interior
closed rows; re-fetch conflicts and missing timestamps from public Binance.
Check alignment, OHLCV bounds, exact continuity, and exchange close times before
writing. Record retrieval time, file hashes, gap/conflict counts and provenance.
Locally consistent interior rows remain locally validated, not independently
exchange-verified; distinguish these from exchange-refetched rows. Incomplete
series remain UNKNOWN and cannot approve a policy. No filling gaps synthetically.

Tests cover mismatched positions, missing prices, non-finite data, duplicate
events, unresolved chains, partial final bars, conflicts, gaps, and closed-bar
validation. Re-run existing critic and research tests. Rollback reverts the
measurement change only; recovery and ablations never change live BUY/SELL.
