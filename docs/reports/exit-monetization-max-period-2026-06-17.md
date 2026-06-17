# Exit Monetization Max-Period Gate - 2026-06-17

Status: research-only, no production SELL / BUY changes

## Rule Applied

Every behavior hypothesis must be validated on the maximum feasible
replay/backtest period supported by the tool and local data before adoption.
Shorter windows are triage only.

For this sweep:

- exit-quality report maximum: all 32 local `signal_quality_*_final.json`
  reports, spanning `2026-05-05` through `2026-06-16` with gaps;
- exit-monetization case replay maximum: `--days 0`, all available signal-quality
  reports;
- portfolio replay maximum used here: `--days 30`, the established max-period
  gate for `files/replay_backtest.py` on this universe.

## Exit-Quality Audit

Command:

```powershell
.\pyembed\python.exe .\files\report_exit_quality.py --days 0 --output .\.runtime\reports\exit_quality_audit_max_available.json --text-output .\.runtime\reports\exit_quality_audit_max_available.txt
```

Result:

- days loaded: `32`
- closed trades: `1267`
- visible cases: `1047`, partial case coverage
- median exit efficiency: `-0.2209`
- median giveback: `1.4874%`
- late exits: `428`
- early exits: `223`
- top-mover exit failures: `244`
- negative-after-MFE cases: `601`
- high-giveback cases: `539`
- post-exit continuation cases: `507`

Decision: exit monetization remains the dominant near-term work. Case-level
conclusions are partial until signal-quality reports are regenerated with full
trade rows.

## Case-Level Exit Hypotheses

Commands:

```powershell
.\pyembed\python.exe .\files\replay_hold_after_weak_sell.py --days 0 --output .\.runtime\reports\hold_after_weak_sell_replay_max_available.json --text-output .\.runtime\reports\hold_after_weak_sell_replay_max_available.txt
.\pyembed\python.exe .\files\replay_partial_exit_after_weak_sell.py --days 0 --output .\.runtime\reports\partial_exit_after_weak_sell_replay_max_available.json --text-output .\.runtime\reports\partial_exit_after_weak_sell_replay_max_available.txt
.\pyembed\python.exe .\files\replay_trailing_tail_after_partial_exit.py --days 0 --output .\.runtime\reports\trailing_tail_after_partial_exit_replay_max_available.json --text-output .\.runtime\reports\trailing_tail_after_partial_exit_replay_max_available.txt
.\pyembed\python.exe .\files\replay_observable_tail_selector.py --days 0 --output .\.runtime\reports\observable_tail_selector_replay_max_available.json --text-output .\.runtime\reports\observable_tail_selector_replay_max_available.txt
```

Maximum available case replay coverage:

- cases loaded: `822`
- eligible weak-sell cases: `680`
- labeled cases: `357`
- missing/pending: `323`

Findings:

| Hypothesis | Result | Decision |
|---|---:|---|
| Hold after weak sell, 5 bars | avg delta `+0.3634%`, worse `41.46%` | promising but too broad; use only as input to selector research |
| Partial 50%, hold 5 bars | avg delta `+0.1817%`, worse `41.46%` | promising but still broad; do not promote |
| Trailing tail after partial exit | avg delta `+0.0903%` to `+0.0922%`, worse `51.82%` to `52.38%` | reject or refine; unsafe without selector |
| Observable selector: `non_ema_positive_giveback` | test avg delta `+0.1776%`, worse `9.35%`, false-positive allow `0.0%` | advance to shadow observable tail selector, not production |

## Portfolio Replay Gate

Commands use:

```powershell
.\pyembed\python.exe .\files\replay_backtest.py --days 30 --variant <variant> --top-gainer-score-min 34 --objective-top-n 15 --no-baseline --json
```

Results:

| Variant | Trades | PnL total | PnL avg | Win rate | Capture | Trade precision | Median capture | Exit eff | Giveback | Cooldown harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `score_replace` baseline | 1454 | -422.74% | -0.2907% | 37.55% | 100.00% | 27.44% | 0.0000 | -0.3656 | 1.2250 | 1298.26 |
| `protected_trailing_exit` | 1278 | -464.44% | -0.3634% | 27.07% | 100.00% | 27.93% | 0.0000 | -0.5238 | 1.6787 | 924.29 |
| `protected_weak_only` | 1280 | -460.89% | -0.3601% | 27.58% | 100.00% | 27.89% | 0.0000 | -0.5385 | 1.6871 | 899.83 |
| `partial_profit_take` | 1456 | -431.81% | -0.2966% | 38.39% | 100.00% | 27.40% | 0.0000 | -0.3571 | 1.2614 | 1293.85 |
| `exit_discriminator_shadow_policy` | 1370 | -418.90% | -0.3058% | 34.53% | 100.00% | 27.08% | 0.0250 | -0.4286 | 1.4594 | 1087.32 |
| `suspicious_exit_reentry` | 1507 | -370.69% | -0.2460% | 38.89% | 100.00% | 28.67% | 0.0120 | -0.2500 | 1.2195 | 999.05 |

## Decisions

Rejected for production:

- `protected_trailing_exit`: worsens PnL, exit efficiency, and giveback on 30d.
- `protected_weak_only`: worsens PnL, exit efficiency, and giveback on 30d.
- `partial_profit_take`: worsens PnL and giveback on 30d despite slightly better
  win rate.
- `exit_discriminator_shadow_policy`: small PnL improvement, but worsens
  precision, exit efficiency, and giveback.
- broad trailing-tail policies: positive average case delta but too many harmed
  cases.

Promising but still research-only:

- `suspicious_exit_reentry`: improves 30d PnL, average PnL, win rate, trade
  precision, median capture, exit efficiency, giveback, and cooldown harm without
  reducing capture.
- `non_ema_positive_giveback` observable tail selector: promising case-level
  selector with low worse rate and no false-positive allowance on the test split.

## Next Gate

1. Keep production SELL and BUY behavior unchanged.
2. Regenerate signal-quality reports with full trade rows for the maximum
   available local period, so exit-quality case coverage is no longer partial.
3. Add or run a shadow-only observable tail selector for
   `non_ema_positive_giveback`.
4. Promote `suspicious_exit_reentry` only after a dedicated spec update and a
   second maximum-period validation with regime/day splits and duplicate-entry
   safety checks.

## Operational Note

`claude_crypto_bot` must not be stopped or modified during this work unless the
user explicitly asks for it.
