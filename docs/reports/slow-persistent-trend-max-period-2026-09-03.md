# Slow Persistent Trend Maximum-Period Result — 2026-09-03

Verdict: **rejected**. Production and shadow behaviour remain unchanged.

## Evidence contract

- Pre-registered specification:
  `docs/specs/slow-persistent-trend-max-period.md`.
- SHA-256 of the exact pre-result specification used by the run:
  `0ecdf916075c6782155317abe37375a26af9db0170e512929ac8054cf752094a`.
  The committed specification only normalizes trailing Markdown whitespace;
  no threshold, population, timing, label, split, selection, or acceptance
  rule changed after the result.
- Machine-readable evidence (runtime, not committed):
  `.runtime/reports/slow_persistent_trend_max_period_latest.json`.
- Audit implementation: `files/audit_slow_persistent_trend_max_period.py`.
- Period requested: `2017-08-17T00:00:00Z` through the last closed-hour
  boundary `2026-09-03T10:00:00Z`; actual bars span
  `2017-08-17T04:00:00Z` through the bar opened
  `2026-09-03T09:00:00Z`.
- Population: 105/105 configured symbols, 4,821,680 closed 1h bars, no fetch
  errors.
- Split: train/validation cut `2023-01-20T15:36:00Z`,
  validation/holdout cut `2024-11-11T12:48:00Z`, 48h embargo.
- Entry and costs: next 1h candle open, 20 bps round trip.

## Result

Validation selected `strict_v1` without reading holdout: useful precision
227/1,038 = 21.8690% versus broad-precursor 4,722/32,351 = 14.5961%
(1.4983x, +7.2728pp). Mean/median net 24h returns were already negative at
-0.1422%/-0.2867%.

Untouched holdout for selected `strict_v1`:

| Metric | Result |
|---|---:|
| useful label | 168/909 = 18.4818% |
| broad-precursor base rate | 4,893/35,412 = 13.8174% |
| lift | 1.3376x; +4.6645pp |
| mean net return at 24h | -0.1118% |
| median net return at 24h | -0.1998% |
| p10 net return at 24h | -3.0840% |
| mean net return at 36h | +0.0406% |
| signal pressure | 1.3796/calendar day |

Non-selected holdout diagnostics were also economically negative:

| Profile | Useful | Mean net 24h | Median net 24h | First TRX decision |
|---|---:|---:|---:|---:|
| `balanced_v1` | 708/3,906 = 18.1260% | -0.2826% | -0.4366% | 2026-09-03 05:00Z |
| `strict_v1` | 168/909 = 18.4818% | -0.1118% | -0.1998% | 2026-09-03 06:00Z |
| `low_vol_persistence_v1` | 444/2,269 = 19.5681% | -0.0976% | -0.2312% | 2026-09-03 05:00Z |

None identified the registered TRX incident by `2026-09-02T10:00:00Z`.
Balanced/low-vol first fired 43 hours after that deadline and strict 44 hours
after it. The proposed rolling persistence detector therefore reproduced the
same practical defect: it confirmed an already mature move instead of finding
its early phase.

Failed acceptance gates: positive validation mean and median net 24h,
positive holdout mean and median net 24h, holdout precision >=30%, holdout
lift >=5pp, holdout p10 >=-2%, and the TRX incident deadline. Sample size,
coverage, holdout duration, 1.25x relative lift, validation-to-holdout
stability, and selected-profile alert pressure passed, but cannot offset the
negative economic outcomes.

## Decision

Rolling 12-36h persistence enriches the useful-label rate relative to a weak
broad precursor, but it is still late and loses money on average after costs.
Do not enable it, do not add it to Telegram, and do not retune it against the
now-open holdout. A successor needs a materially new causal hypothesis, a new
pre-registration, and a fresh untouched holdout.

Repository Truth Harness at validation time: **FAIL**, independently of this
hypothesis, on TH-04 (`not_evaluated_dataset_quality`) and TH-11 (canonical
30d portfolio-alpha artifact missing `current_policy_epoch`). These findings
also prohibit any production-readiness claim.
