# Priority Replay Audit — 2026-06-14

Status: research-only; no production BUY/SELL/rotation changes.

## Scope

This audit follows the current priority order:

1. exit monetization;
2. suspicious re-entry shadow coverage;
3. portfolio replacement safety.

All behavior-changing hypotheses remain gated by historical replay evidence. The runs below used the maximum locally available report/candle cache where the corresponding harness could complete.

## Exit Monetization

### Max-window exit-quality baseline

Source: `.runtime/reports/exit_quality_max_2026-06-14.json`

- days loaded: 29
- closed trades: 1113
- case rows loaded: 919
- case coverage: partial
- early exits: 195
- late exits: 384
- false-positive buys: 479
- exit efficiency median: -0.3147
- giveback median: 1.5642%
- pnl median: -0.6308%
- negative exit cases: 603
- negative-after-MFE cases: 532
- high-giveback cases: 477
- post-exit continuation cases: 440
- median visible opportunity loss: 3.0815%

Interpretation: exit monetization is a real bottleneck across the longest available measurement window, not a single-day artifact. Because case coverage is partial, these numbers are suitable for prioritization, not direct production SELL changes.

### Hold-after-weak-sell replay

Source: `.runtime/reports/hold_after_weak_sell_max_2026-06-14.json`

- reports loaded: 28
- cases total: 708
- eligible: 616
- labeled: 320
- pending/missing: 296
- decision: `advance_hold_5_to_partial_exit_replay_before_production`

Best simple hold policy:

- `hold_5`: avg delta +0.3341%, median delta +0.2450%, worse-rate 41.25%, median adverse -0.4887%.

Interpretation: short hold after weak/suspicious exit has positive edge, but harm rate is too high for production use without a selector/stop.

### Partial-exit replay

Source: `.runtime/reports/partial_exit_after_weak_sell_max_2026-06-14.json`

- eligible: 616
- labeled: 320
- decision: `advance_partial_50_hold_5_to_trailing_tail_replay_before_production`

Best partial policy:

- `partial_50_hold_5`: avg delta +0.1670%, median delta +0.1225%, worse-rate 41.25%, median tail adverse -0.2443%.

Interpretation: partial exit preserves some continuation upside while reducing tail adverse versus full hold. It is still not production-ready because the worse-rate remains high.

### Trailing-tail replay

Source: `.runtime/reports/trailing_tail_after_partial_exit_max_2026-06-14.json`

- eligible: 616
- labeled: 320
- decision: `reject_or_refine_trailing_tail_policy_no_safe_edge`

Best tested trailing-tail family still failed:

- `tail50_h10_ema20_cap150`: avg delta +0.0672%, median delta -0.0002%, worse-rate 52.81%.

Interpretation: the current simple trailing-tail stop does not safely monetize the partial-exit edge. Do not change production SELL.

### Observable selector replay

Source: `.runtime/reports/observable_tail_selector_max_2026-06-14.json`

- rows: 320
- train rows: 224
- test rows: 96
- decision: `no_observable_selector_passed_test_gate`

Best-looking selectors had positive average test delta, but median delta was 0.0 and the configured acceptance gate was not met.

Interpretation: there is still no production-safe observable selector. Continue feature search; do not enable tail retention live.

## Suspicious Re-entry

Source: `.runtime/reports/suspicious_reentry_scorecard_2026-06-13_adhoc.json`

- status: complete
- alerts total: 0
- labeled: 0

Interpretation: this was not evidence that re-entry is bad. It was a coverage blind spot: the scorecard could not say whether exits failed the watch-registration gate, whether watches expired, or whether candidate confirmation never arrived.

Action implemented:

- add one-shot `suspicious_reentry_watch_decision` telemetry with decisions:
  - `registered`;
  - `rejected_exit_score`;
  - `rejected_mfe`.

This is measurement-only and does not open positions or alter cooldown.

## Portfolio Replacement

Source: `.runtime/reports/portfolio_replacement_shadow_reward_max_2026-06-14.json`

- replacements: 289
- closed incoming: 281
- avg replacement delta: -0.0745%
- median replacement delta: -0.3740%
- positive delta rate: 38.08%
- incoming watchlist top count: 12
- decision: `replacement_policy_hurting_in_shadow_monitor`

Interpretation: current replacement behavior is hurting in shadow. Do not add new rotation/replacement behavior until replacement candidates can beat the held position on watchlist-filtered top-mover and exit-adjusted outcome metrics.

## Decisions

1. Do not relax BUY gates from this audit.
2. Do not change production SELL from current exit replays.
3. Do not enable production re-entry; collect upstream re-entry funnel telemetry first.
4. Do not expand portfolio replacement; current shadow evidence is negative.
5. Next research priority: observable exit selector feature search using only SELL-time features, with train/test gate and false-positive slice control.

