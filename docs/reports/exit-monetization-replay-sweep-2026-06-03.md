# Exit Monetization Replay Sweep — 2026-06-03

Status: research-only replay result; no live SELL changes

## Trigger

The 2026-06-02 daily learning report showed weak exit monetization:

- exit efficiency median: `-0.2209`;
- giveback median: `2.641`;
- shadow tail selector failed the daily gate.

## Scope

Existing exit replay tools were rerun on the longest locally available practical
window:

- `--days 60`
- reports loaded: 19
- labeled exit rows: 212
- eligible high-MFE/giveback rows: 307

## Results

### Hold after weak sell

Decision: `reject_or_refine_hold_policy_no_safe_edge`

Best broad hold result:

- `hold_5`: avg delta `+0.3868%`, total delta `+82.0084%`
- worse rate: `45.75%`

Interpretation: average improves, but too many cases get worse. Not safe as a
production SELL change.

### Partial exit after weak sell

Decision: `reject_or_refine_partial_exit_policy_no_safe_edge`

Best broad partial result:

- `partial_50_hold_5`: avg delta `+0.1934%`, total delta `+41.0043%`
- worse rate: `45.75%`

Interpretation: lower risk than full hold, but still too noisy.

### Trailing tail after partial exit

Decision: `reject_or_refine_trailing_tail_policy_no_safe_edge`

Best broad trailing tail result:

- `tail50_h10_ema20_cap150`: avg delta `+0.0568%`, total delta `+12.0456%`
- worse rate: `58.49%`

Interpretation: rejected; worse-rate is too high.

### Early-exit gated selector

Decision: `no_selector_passed_observable_shadow_gate`

Oracle-style selectors improve totals, but rely on hindsight `early_exits` labels.
Observable weak-signal selectors are too weak or noisy.

### Observable tail selector

Decision: `no_observable_selector_passed_test_gate`

Top selector:

- `non_ema_mfe150`
- test avg delta `+0.0908%`
- test total delta `+5.8126%`
- worse rate `26.56%`

Interpretation: closest candidate, but still fails the current safety gate because
worse-rate is too high and test coverage lacks early-exit positives.

## Decision

Reject broad exit changes:

- no live hold-after-weak-sell;
- no live partial-exit-after-weak-sell;
- no live trailing-tail selector;
- no live observable tail selector.

The correct next step is not another broad SELL change. It is a narrower failure
slice search, especially around high-MFE non-EMA exits where `non_ema_mfe150`
shows weak but nonzero edge.

## Next Hypothesis

Form a narrow slice:

`non_ema_high_mfe_low_worse_rate`

Candidate constraints to test next:

- non-EMA exit reason;
- MFE >= 1.5%;
- exclude false positives;
- require current PnL >= 0 or giveback >= 1.0%;
- use partial tail only, not full hold.

Promotion requires replay with:

- positive test delta;
- worse rate materially below current `26.56%`;
- no false-positive allowance;
- no production SELL change before replay.
