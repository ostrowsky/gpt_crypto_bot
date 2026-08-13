# Truth Harness

Status: shipped enforcement; repository compliance may still be `FAIL` or `UNKNOWN`
Last updated: 2026-08-13

## Purpose

Prevent the bot, its reports, and its maintainers from presenting incomplete,
proxy, in-sample, or stale evidence as proven progress toward the trading
objective. The harness is a fail-closed evidence checker. It does not approve a
trading idea and it never turns a diagnostic metric into permission to trade.

## Truth invariants

| ID | Required invariant |
|---|---|
| TH-01 | Every published ratio carries its numerator and denominator. Base rate and lift are required for model/ranker claims; otherwise the metric is marked diagnostic-only. A zero or missing denominator produces `null`/`unknown`, never `0%`. |
| TH-02 | Proxy, training, teacher, and shadow metrics are kept separate from realized objective metrics. They cannot make an overall progress verdict positive. |
| TH-03 | Model evidence records feature time, label time, label definition, and features capable of directly encoding the label. Same-snapshot recognition is not forward prediction. |
| TH-04 | Achievement claims use a chronological out-of-sample holdout. Row-index splits and in-sample scores are diagnostic-only. |
| TH-05 | Compared windows have the same metric definition and comparable denominators. Missing, partial, and downtime periods are `unknown`; they are neither misses nor successes and are excluded from objective aggregation. |
| TH-06 | A production gate is validated on the bot's actual candidate population and maximum available historical period. A market-wide or hand-picked sample cannot approve it. |
| TH-07 | Every live behavior change has a rollback switch, guardrails, and a shadow/canary verification plan. BUY/SELL relaxation additionally requires causal replay evidence before enablement. |
| TH-08 | Rejected hypotheses retain period, population, metrics, and verdict so they cannot silently re-enter the roadmap without new evidence. |
| TH-09 | Current-state statements in MD agree with config and code. Historical decisions are explicitly dated and must not masquerade as current state. |
| TH-10 | Freshness, source coverage, sample size, and unresolved uncertainty are part of every conclusion. Insufficient evidence yields `UNKNOWN`/`inconclusive`. |
| TH-11 | Trading profitability is a unified portfolio result after fees/slippage and against a named benchmark. Per-trade or per-mode PnL is not portfolio alpha. |
| TH-12 | Every material change has a registered spec, focused tests, verification evidence, and staged-scope review. Runtime state, reports, logs, credentials, and model artifacts are not committed. |

## Interfaces

```powershell
pyembed\python.exe files\truth_harness.py full
pyembed\python.exe files\truth_harness.py change --staged
```

Both profiles may write a machine-readable report with `--json PATH`. The
output is runtime evidence and must not be committed.

- `full` audits current repository enforcement and the newest progress/model
  evidence. It may legitimately return `FAIL` while known truth gaps remain.
- `change --staged` checks that the proposed diff has a spec, focused tests,
  rollback/evidence language where applicable, and no staged runtime state.

Exit code `0` means no blocking finding in the selected profile, `1` means a
compliance failure, and `2` means the harness itself could not complete. A
harness failure makes the audited conclusion `UNKNOWN`.

## Enforcement

1. `AGENTS.md` requires the repository skill before audits and before handoff
   of trading, metric, model, report, or learning-loop changes.
2. The tracked `.githooks/pre-commit` runs `change --staged`.
3. The working copy uses `git config core.hooksPath .githooks`.
4. A failed check may be waived only in a spec with an owner, reason, risk,
   expiry date, and explicit waiver ID. Wording a claim more optimistically is
   never a waiver.

## Current known gap policy

The absence of canonical portfolio alpha, immutable historical universe
snapshots, or complete model timing provenance must remain visible as a harness
finding. Shipping the harness does not imply that those gaps have been fixed.
