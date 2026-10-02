# Certified rule-champion score policy

Date: 2026-10-02. Status: disabled runtime adapter; NOT a winning strategy.
Parent: continuous-signal-improvement.md. TH-01..TH-12 apply.

The legacy ranker is disabled and has an unqualified champion model. It must not
be enabled implicitly to make learning appear effective. A separate policy family
adds only a bounded [-1,+1] candidate score bonus to the actual rule champion.
It does not supply ranker quality/EV values, alter downstream ranker vetoes,
change SELL, cooldown, capacity, sizing or other risk rules. Default bonus is zero.
Monitor and replay must use the exact same causal record builder and bonus formula.

Champion identity is a canonical descriptor binding strategy sources, generic and
legacy model bytes and safe uppercase runtime config (excluding only this kill
switch). A release ticket binds this descriptor, exact candidate, evaluator source,
expiry and assignment. Legacy model-champion tickets cannot authorize this family.
Candidate quality probability produces bonus = clamp(2*(p-0.5), -1, +1).
Missing/nonfinite/out-of-range probability, invalid/expired ticket or unavailable
model returns zero. It never silently turns on legacy ranker scoring.

Offline paired comparisons freeze the descriptor/models/sources and rebuild full
admission populations independently in each arm. No production enablement until
maximum-available causal history, independent sealed/shadow/canary account evidence
and full Harness pass. Synthetic tests prove wiring/isolation only, not uplift.
The runtime flag remains false; unsigned offline selection is process-local only.
Rollback: disable the kill switch or clear the signed pointer through existing CAS.
Actual consumption receipts and full prospective portfolio producer remain separate
acceptance requirements; this adapter alone does not close the learning loop.

Maximum historical verification command uses `paired_full_policy_replay.py`
with `--policy-family rule-score`, the frozen 181-day archive and exact frozen
candidate. Both arms retain unchanged legacy scorer state; rule-score no longer
requires turning on legacy ranking or manufacturing runtime_eligible on its model.
Numerical success on reconstructed history is still UNKNOWN for release without
raw/PIT certification and separate prospective full portfolio evidence.
