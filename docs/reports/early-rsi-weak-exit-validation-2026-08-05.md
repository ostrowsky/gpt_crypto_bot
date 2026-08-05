# Early RSI-WEAK Exit Validation — 2026-08-05

Status: complete, all tested production relaxations rejected

## Trigger

UNIUSDT `15m/retest` exited at `4.13` after two bars on an RSI-divergence
`WEAK:` warning, then continued higher. Code inspection confirmed a contract
mismatch: the warning text says that the stop is tightened, while the monitor
performs a full SELL after the minimum grace period.

## Validation design

The new causal replay uses every valid RSI-divergence exit event and cached
OHLCV fragments. It recomputes indicators from candles closed by the decision
time, applies a five-basis-point conservative policy-change penalty, and splits
whole UTC days chronologically into `60%/20%/20%` train/validation/holdout.

Command:

```powershell
.\pyembed\python.exe .\files\replay_early_rsi_weak_exit.py
```

Coverage:

- valid RSI-WEAK exits: `938`;
- causally labeled: `831`;
- missing: `107` (`67` no overlapping cache, `9` decision-price mismatch,
  `4` insufficient indicator warmup, `27` insufficient future candles);
- label period: `2026-05-03` through `2026-08-04`;
- train: `392`, validation: `206`, holdout: `233`.

The baseline RSI-WEAK cohort was strongly profitable at the original exit:
average `+1.44%`, median `+0.95%`, win rate `99.3%`. The question was therefore
incremental continuation capture, not whether those exits had positive PnL.

## Results

| Policy | Validation net avg / median | Holdout net avg / median | Holdout worse | Holdout p10 | Decision |
|---|---:|---:|---:|---:|---|
| tighten all, `k=0.9` | `-0.53 / -0.65pp` | `-0.11 / -0.42pp` | `71.7%` | `-1.33pp` | reject |
| tighten all, `k=1.4` | `-0.55 / -0.68pp` | `-0.10 / -0.48pp` | `70.4%` | `-1.61pp` | reject |
| retest tighten, `k=1.4` | `+0.55 / -0.38pp` | `+0.11 / -0.60pp` | `70.0%` | `-2.85pp` | reject |
| retest grace 4 bars | `+0.34 / +0.02pp` | `-0.57 / +0.15pp` | `42.9%` | `-1.05pp` | reject |
| retest two-WEAK confirmation | `+0.35 / -0.49pp` | `-0.26 / -0.58pp` | `63.3%` | `-2.85pp` | reject |
| retest structure veto, `k=1.4` | `+1.27 / -0.40pp` | `-0.17 / -0.67pp` | `77.3%` | `-5.32pp` | reject |
| 1h-confirmed retest, `k=0.9` | `+0.18 / -0.70pp` | `+1.19 / -0.24pp` | `55.6%` | `-1.25pp` | reject |
| 25% 1h-confirmed tail | `+0.24 / -0.15pp` | `+0.17 / -0.07pp` | `61.1%` | `-0.50pp` | reject |
| 50% 1h-confirmed tail | `+0.49 / -0.31pp` | `+0.34 / -0.14pp` | `61.1%` | `-0.99pp` | reject |

Positive means in the MTF slice are driven by a minority of large winners.
Negative medians and high worse-rates show that the observable rules do not
reliably identify those winners at decision time.

## Decision and roadmap

1. Keep production SELL behavior unchanged. No candidate qualified for the
   portfolio replay gate.
2. Mark broad WEAK suppression, longer retest grace, consecutive-WEAK
   confirmation, 15m structure veto, 1h veto, and static partial tails as
   rejected/mixed. Do not retest the same static rules without new evidence.
3. Add universal mature `T+2/T+5/T+10`, MAE, and MFE labels for every WEAK exit,
   including exits below the current suspicious-reentry MFE/score floors.
4. Use those labels for a causal exit discriminator. Required target is
   continuation net of downside, not rare-winner average return.
5. Validate a learned discriminator with nested temporal selection and an
   untouched later forward window before any portfolio or production test.

The runtime JSON/text report remains under `.runtime/reports` and is not a
versioned model or production artifact.
