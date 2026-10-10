# Spec-First Workflow

Last updated: 2026-10-10

## Purpose

Keep every non-trivial change tied to the bot objective before code is written.
This prevents ad-hoc gate tuning, metric drift, and undocumented production behavior.

## When A Spec Is Required

A feature spec is required before implementation for any change that:

- adds or changes BUY / SELL / portfolio behavior;
- adds a new live or shadow decision path;
- changes ranking, replacement, cooldown, or alert behavior;
- introduces new model logic, retraining logic, or promotion logic;
- changes operator-visible reports in a way that could alter decisions;
- spans more than a tiny local refactor.

User-required order: **specifications → tests → implementation code**. Bug fixes
also start from a specified contract and a failing reproduction. A small fix or
behavior-preserving refactor may update an existing canonical spec instead of
creating another document; executable implementation never precedes its tests.
Pure documentation updates require consistency/link checks, not artificial tests.

## Required Spec Fields

Every feature spec must state:

1. **Problem**
   - What observed failure, gap, or opportunity are we addressing?
2. **Objective fit**
   - Which part of the north star this serves:
     - earlier same-day top-mover capture;
     - better selection under the unified 10-position cap;
     - better exit retention near trend exhaustion;
     - safer / more explainable operation.
3. **Scope**
   - What changes now.
   - What explicitly does **not** change.
4. **Primary metrics**
   - Which canonical metrics decide whether the feature helped.
5. **Acceptance criteria**
   - Observable conditions that make the work complete.
6. **Risk / trade-offs**
   - What could get worse if the feature works as designed.
7. **Backtest / verification gate**
   - Replay, walk-forward, regression, or smoke checks required before adoption.
   - For every behavior hypothesis, require the maximum feasible replay/backtest
     period supported by the tool and local data. Short windows are only
     provisional triage unless the spec explains why they are the maximum
     available period.
   - Always propose the maximum-period backtest before recommending adoption.
8. **Rollback switch**
   - How to disable or revert the behavior safely.

## Workflow

1. **Write or update the spec first.**
   - Create the feature spec under `docs/specs/`.
   - Add or update the capability row in `docs/FEATURE_SPEC_INDEX.md`.
2. **Write automated tests before implementation code.**
   - Derive focused cases from the spec's observable acceptance criteria.
   - Include normal cases, boundaries and relevant failure/recovery cases.
   - Run them before implementation; for new/changed behavior establish RED
     for the expected absent or incorrect behavior.
   - Record the command, failed scenario IDs and expected failure reason.
   - Tooling, permissions, credentials and unrelated import failures are not RED.
     An intended missing new public interface may be the expected initial failure.
3. **Implement against the spec only after the test-first gate.**
   - New/changed behavior requires the expected RED; preserving refactors require
     documented pre-implementation contract coverage as described below.
   - Keep code changes inside the stated scope.
   - If the implementation reveals a materially different trade-off, update the spec before continuing.
4. **Verify GREEN and refactor.**
   - Make the spec-driven tests pass; preserve assertions and contract semantics.
   - Do not manufacture GREEN through skips, xfail or implementation-derived expectations.
   - Refactor while tests remain green. For behavior-preserving refactors, establish
     contract coverage first and record why existing tests legitimately stay green;
     do not deliberately break correct behavior merely to claim RED.
   - Run the checks named in the spec.
   - For production behavior changes, replay/backtest evidence is mandatory before enablement.
   - For each hypothesis, run or explicitly propose the maximum-period replay
     gate; do not promote from a shorter window without documenting the blocker.
5. **Document the decision.**
   - Record shipped / rejected / shadow-only status and the next gate.
6. **Prefer measurement-first rollout.**
   - When uncertainty is high, ship diagnostics or shadow logging before changing live behavior.

## Default Promotion Rules

- **Observability-only changes** may ship after regression checks when they do not alter BUY / SELL behavior.
- **Shadow-only changes** may ship before replay if they are strictly non-intervening and have a clear review plan.
- **Production behavior changes** require replay/backtest evidence and an explicit acceptance rule.
- **ML / RL changes** must show objective-level benefit, not only surrogate metrics.

## Review Checklist

Before calling a feature complete, confirm:

- [ ] A registered canonical spec describes the change, including small executable fixes.
- [ ] Tests were written before implementation; applicable RED reason/command and GREEN are recorded.
- [ ] The feature is listed in `docs/FEATURE_SPEC_INDEX.md`.
- [ ] The touched metrics are named.
- [ ] The implementation stayed within scope.
- [ ] Verification named in the spec was run.
- [ ] Rollback or disable path is clear.
- [ ] The final note says whether the feature is shipped, shadow-only, replay-only, or rejected.
