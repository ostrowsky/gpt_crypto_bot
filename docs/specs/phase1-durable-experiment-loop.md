# Phase 1 Durable Deterministic Experiment Loop

Date: 2026-08-22
Status: implemented; Phase 1 exit gate complete 2026-08-22; research-only

## Purpose

Phase 1 turns the Phase -1 protocol skeleton and Phase 0 evidence contracts
into one restart-safe path for a real, power-feasible hypothesis. It proves
that a manually selected hypothesis can reach a truthful terminal outcome
without an LLM choosing its data, metric, validator, retries, or verdict.

The first registered experiment is a WATCH-layer ranking comparison:

> At 12:15 Europe/Budapest, for watchlist symbols outside the current Binance
> spot Top-50, ranking by return against the fixed 23:00-minus-24h denominator
> (`static_target`) improves same-day 23:00 Top-50 Top-10 precision over ranking
> by the rolling observation-time 24h return (`current_rank`).

This is the previously validated static-target research mechanism. Phase 1
does not reuse its summary as evidence. It freezes the normalized maximum-period
day/candidate rows and recomputes the comparison through a new validator and a
separate verifier.

## Scope

In scope:

- immutable experiment snapshot, manifest, typed hypothesis contract, and
  validator attestation key under an uncommitted runtime attempt directory;
- closed capability and validator registries;
- power/feasibility precheck before validator execution;
- append-only, idempotent attempt ledger with hashes and prior state;
- per-attempt lease, bounded validator subprocess timeout, explicit retry links,
  dead-letter evidence, restart reconciliation, queue age, and last-transition
  telemetry;
- deterministic validator result followed by independent raw-snapshot
  verification;
- deterministic `SUPPORTED | REFUTED | UNDERPOWERED | INVALID_RESULT` outcome;
- one real maximum-period execution over the actual cached Binance population.

Out of scope:

- LLM, RAG, MCP experiment mutation, multi-agent research, or provider APIs;
- production Telegram messages, WATCH routing, BUY, SELL, replacement, sizing,
  release configuration, or automatic promotion;
- reuse of final labels at observation time;
- hidden retries or selecting a different period after seeing a result.

Executable modules:

- `files/phase1_experiment_loop.py`: durable state, registries, precheck,
  leases, timeout, ledger, dead letters, governor, status, and CLI;
- `files/phase1_static_target_snapshot.py`: orchestrator-owned snapshot builder;
- `files/phase1_static_target_validator.py`: allowlisted validator subprocess;
- `files/phase1_static_target_verifier.py`: independent verifier over the raw
  snapshot.

## Typed hypothesis contract

The immutable contract contains every field required by Section 10 of
`continuous-improvement-control-plane.md`:

- identity: `hypothesis_id`, `version`, `parent_hypothesis_id`, `attempt_id`;
- scope: `action_layer=WATCH`, objective and guardrail metric versions;
- evidence: incident/cohort references and `evidence_snapshot_id`;
- mechanism: causal mechanism, competing explanation, and falsifier;
- change: target capability, proposed change, affected population, expected
  effect/horizon, and rollback flag;
- validation: minimum practical effect, registered validator ID, frozen
  protocol, seed, minimum eligible days, and snapshot hash.

Unknown capabilities return terminal `contract_rejected`. A known capability
without a registered validator returns `WAITING/needs_validator`; it is not
silently discarded or executed by a fallback.

## Frozen real snapshot

`phase1_static_target_snapshot.py` is the orchestrator-owned builder. It may
reuse the existing causal cache parser, but it writes a self-contained
normalized snapshot before validation. Each eligible local day records:

- local day and market/watchlist population counts;
- every eligible watchlist candidate outside observation-time Top-50;
- only causal observation fields needed by the two registered policies;
- later 23:00 target rank and target membership as labels;
- the target-entrant denominator.

The manifest binds the raw snapshot SHA-256, cache-content hash, watchlist hash,
requested/eligible/rejected days, policy epoch, metric version, source builder
hash, and current git commit. Validator and verifier receive read-only paths;
neither can select or rewrite the snapshot.

## Power precheck

The independent unit is the local decision day. Before validation, the loop
computes the paired daily precision difference, its day-cluster standard
deviation, effective sample size, two-sided 95%/80%-power MDE, registered
`SESOI=2pp`, and coverage. The experiment is feasible only when:

- at least 30 eligible days exist;
- at least one target entrant and one candidate exist;
- `MDE <= SESOI`;
- snapshot coverage and hashes are valid.

Failure reaches terminal `UNDERPOWERED` or `INVALID_RESULT`; the loop does not
weaken the metric or rerun on a favorable slice.

## Independent validation and verification

`phase1_static_target_validator.py` reads only the frozen snapshot and contract,
sorts the same candidates under both policies, publishes day-level and aggregate
traces, and signs its bundle. It has no network or production-state access.

`phase1_static_target_verifier.py` must not import the validator module or trust
its aggregate metrics/trace. It independently reads the raw snapshot and
contract, reconstructs both selections, recomputes denominators, base rate,
Top-1/Top-10 precision, entrant recall, paired day-cluster bootstrap interval,
lift, power fields, and hashes, then compares its payload with the signed
bundle. Any mismatch yields `INVALID_RESULT`.

The deterministic governor consumes only `VERIFIED_RESULT`. `SUPPORTED`
requires:

- at least 30 eligible days;
- candidate Top-10 precision delta at least `2pp`;
- paired bootstrap 95% lower bound above zero;
- candidate entrant recall no worse than control;
- candidate precision lift over the frozen base rate above `1.0`;
- complete verifier agreement and no contract/coverage violation.

Otherwise the outcome is `REFUTED`, `UNDERPOWERED`, or `INVALID_RESULT`.

## Durable state and operations

Stages and statuses use the architecture's closed enums. Outcome reasons use
registry v1. Every transition key is idempotent and records prior state plus
input/output hashes. An identical attempt resumes from immutable artifacts and
does not duplicate transitions. A conflicting snapshot or contract is terminal
invalid evidence.

An active unexpired lease prevents concurrent execution. An expired lease is
reconciled and recorded before continuation. Validator execution has a fixed
timeout; timeout or other non-transient invalid evidence writes a dead-letter
artifact. A retry requires a new attempt ID, `retry_of`, and retry reason.

The status report exposes attempt count, terminal/waiting counts, terminal
reason distribution, oldest queue age, last transition, dead-letter count, and
program mode. It must never treat no attempts as `0% success`.

## First real experiment and decision boundary

The registered experiment uses the maximum locally available cached Binance
spot population, the frozen 12:15/23:00 Europe/Budapest clocks, Top-50 target,
Top-10 selection, minimum 200 market and 50 watchlist symbols, and deterministic
day bootstrap seed.

A `SUPPORTED` result authorizes only continued or separately reviewed silent
forward shadow. It cannot write production config, send Telegram, open/close a
position, or claim trading improvement. A later production proposal requires a
new spec and its own forward evidence.

## Acceptance tests

1. Typed contracts reject an invented capability and surface a missing
   validator as `needs_validator`.
2. Snapshot/contract reuse is immutable; restart produces no duplicate ledger
   events.
3. Active leases block a second runner; expired leases reconcile visibly.
4. Power-infeasible evidence terminates without invoking the validator.
5. A valid fixture reaches a verified deterministic terminal result.
6. Corrupted validator aggregates or attestation reach `invalid_result` and a
   dead letter.
7. Validator/verifier import boundaries prohibit shared aggregation code.
8. Status telemetry preserves unknown ratios for an empty ledger and reports
   queue age, terminal reasons, retries, and dead letters when present.
9. Maximum-period real execution records its period, actual population,
   denominators, coverage, effect, uncertainty, power, provenance, and one
   terminal verdict.
10. Focused tests, full Truth Harness, staged-change Harness, and
    `git diff --check` pass; production behavior remains unchanged.

## Rollback

Remove the Phase 1 research modules and their untracked runtime directory.
There is no position migration or release rollback because this phase has no
production write path.

## 2026-08-22 real terminal result

The first invocation reached terminal `INVALID_RESULT` because a relative
validator-key path was resolved under the validator subprocess working
directory. The failure and dead letter remain in the append-only ledger. A
regression test now covers relative state directories. The next attempt used a
new ID with `retry_of` and the same frozen snapshot. A final verifier-hardening
attempt also bound and independently checked the snapshot manifest.

The final decision-grade attempt
`static-target-top50-v1-20260822-r2` reached `SUPPORTED`:

- requested period `2025-06-21 .. 2026-08-21`; eligible period
  `2025-06-22 .. 2026-08-20`;
- eligible decision days `425 / 427` (`99.53%` coverage);
- actual candidates `37,060`; target entrants `2,336`;
- current-rank Top-10 precision `648 / 4,250 = 15.25%`;
- static-target Top-10 precision `1,037 / 4,250 = 24.40%`;
- paired daily precision delta `+9.15pp`, day-cluster bootstrap 95% interval
  `[+7.95pp, +10.33pp]`;
- entrant recall `27.74% -> 44.39%` and lift over base `2.42x -> 3.87x`;
- `ESS=425`, `MDE=1.67pp`, registered `SESOI=2pp`, therefore power-feasible;
- validator attestation, snapshot/manifest/contract hashes, and independently
  recomputed metrics agree; verifier errors are empty.

The governor authorizes only continued or separately reviewed silent shadow.
Automatic production promotion is `false`; Telegram, WATCH routing, BUY, SELL,
replacement, and portfolio behavior are unchanged.
