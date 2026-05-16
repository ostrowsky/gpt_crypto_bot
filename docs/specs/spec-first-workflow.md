# Spec-First Workflow

Last updated: 2026-05-16

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

Tiny bug fixes and pure internal refactors may skip a dedicated feature spec only when
they do not change behavior, metrics, outputs, or operator decisions.

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
8. **Rollback switch**
   - How to disable or revert the behavior safely.

## Workflow

1. **Write or update the spec first.**
   - Create the feature spec under `docs/specs/`.
   - Add or update the capability row in `docs/FEATURE_SPEC_INDEX.md`.
2. **Implement against the spec.**
   - Keep code changes inside the stated scope.
   - If the implementation reveals a materially different trade-off, update the spec before continuing.
3. **Verify.**
   - Run the checks named in the spec.
   - For production behavior changes, replay/backtest evidence is mandatory before enablement.
4. **Document the decision.**
   - Record shipped / rejected / shadow-only status and the next gate.
5. **Prefer measurement-first rollout.**
   - When uncertainty is high, ship diagnostics or shadow logging before changing live behavior.

## Default Promotion Rules

- **Observability-only changes** may ship after regression checks when they do not alter BUY / SELL behavior.
- **Shadow-only changes** may ship before replay if they are strictly non-intervening and have a clear review plan.
- **Production behavior changes** require replay/backtest evidence and an explicit acceptance rule.
- **ML / RL changes** must show objective-level benefit, not only surrogate metrics.

## Review Checklist

Before calling a feature complete, confirm:

- [ ] A spec exists or the change is explicitly trivial.
- [ ] The feature is listed in `docs/FEATURE_SPEC_INDEX.md`.
- [ ] The touched metrics are named.
- [ ] The implementation stayed within scope.
- [ ] Verification named in the spec was run.
- [ ] Rollback or disable path is clear.
- [ ] The final note says whether the feature is shipped, shadow-only, replay-only, or rejected.

