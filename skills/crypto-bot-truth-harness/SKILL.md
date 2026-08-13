---
name: crypto-bot-truth-harness
description: Audit gpt_crypto_bot for truthful trading metrics, evidence provenance, MD/config compliance, and safe change validation. Use for bot audits, progress/report reviews, model or backtest claims, roadmap decisions, and before completing changes to trading behavior, gates, models, metrics, reports, or the learning loop.
---

# Crypto Bot Truth Harness

Treat a favorable metric as unproven until its denominator, provenance, timing,
comparison window, and target population are verified.

## Workflow

1. Read `AGENTS.md`, `SCOUT_OPTIMIZATION_SPEC.md`,
   `docs/specs/truth-harness.md`, and the affected feature spec.
2. Run the mechanical full profile:

   ```powershell
   pyembed\python.exe files\truth_harness.py full
   ```

3. Inspect every blocker at its source. Distinguish current, stale, partial,
   missing, proxy, in-sample, holdout, and realized evidence.
4. Apply the judgment checks in `references/audit-checklist.md`.
5. For a change, stage only intended source/spec/test files, then run:

   ```powershell
   pyembed\python.exe files\truth_harness.py change --staged
   git diff --check
   ```

6. For any trading-policy relaxation, require a maximum-available-period,
   time-separated replay on the actual candidate population, guardrails,
   rollback, and forward shadow/canary evidence.
7. Report `PASS`, `FAIL`, or `UNKNOWN`. For each violation give its TH-ID,
   direct evidence, impact, and smallest safe remediation.

Never convert missing data to a miss or success. Never convert same-snapshot
label recognition, in-sample score, teacher/proxy uplift, or per-trade PnL into
a business-outcome achievement.
