# Phase 0 Evidence Capacity and Canonical Labels

Date: 2026-08-14
Status: implementation contract; measurement-only; production trading policy unchanged

## Purpose

Phase 0 creates the deterministic evidence substrate required before a durable
experiment loop or an LLM research agent can be admitted. It must make sparse,
late, partial, stale, and legacy evidence explicit. It must not turn missing
data into a miss, a success, or a zero-valued denominator.

This slice implements the Phase 0 contracts in
`continuous-improvement-control-plane.md`. It does not authorize an agent,
promotion, Telegram canary, or any WATCH/BUY/SELL/portfolio behavior change.

## Executable surface

`files/evidence_capacity.py` is a stdlib-only measurement module and CLI. It
does not import production trading policy, send messages, mutate runtime model
artifacts, or write release configuration. Its public contracts are:

- `build_move_event_labels(...)`;
- `build_top_mover_labels(...)`;
- `ImmutableLabelLedger`;
- `build_power_report(...)`;
- `build_evidence_throughput_report(...)`;
- `action_layer_metric_registry()`;
- `build_harness_remediation_ledger(...)`;
- `migrate_legacy_research_inventory(...)`;
- `verify_objective_report_contract(...)`.

`files/phase0_canonical_audit.py` is the repository adapter for the maximum
locally available final-critic history. It is read-only and accepts only exact
`top_gainer_critic_YYYY-MM-DD_final.json` names. Manual, midday, temporary, and
similarly named artifacts are outside the population.

Generated audits are runtime evidence and are not committed. The source
contracts, tests, and versioned registries are committed.

## Canonical input boundary

The builders consume an orchestrator-frozen normalized snapshot. Every
symbol-day row contains:

- `symbol`, `objective_day`, and `event_day_timezone`;
- `universe_snapshot_hash` and `source_snapshot_hash`;
- `reference_time`, `reference_price`, and fixed `label_cutoff`;
- `coverage_status=complete` for decision-grade labeling;
- closed bars with `close_time`, `high`, and `close`.

The builders reject a row when its identity, hashes, prices, ordering, or
timezone-aware timestamps are invalid. A partial row is returned as
`NOT_MATURE`/`PARTIAL` diagnostic evidence and never enters a confirmed-event
denominator. A bar after the cutoff cannot affect the label. A label is mature
only when `as_of >= label_cutoff` and the input explicitly declares complete
coverage.

## `MoveEvent` v1

The initial version is `move5_v1`:

- event threshold: `+5%` from the fixed reference price;
- midpoint threshold: `+2.5%`;
- unique identity:
  `sha256(symbol|objective_day|event_version|universe_snapshot_hash)`;
- `first_midpoint_crossing_time` and `first_event_crossing_time` use only the
  first closed bar whose high crosses the registered threshold;
- `label_available_at` equals the fixed cutoff, never an earlier crossing;
- the result publishes `raw_bar_count`, included-bar count, coverage status,
  cutoff, threshold definition, and both source hashes.

The first crossing is a hindsight label, not a claim that the move was
predictable at that time. `Coverage@Move5` may compare a decision-time alert
with that crossing only in a separate paired evaluator.

## Canonical top-mover later label v1

`watchlist_top_close_v1` ranks mature, complete symbol-days within one frozen
universe/day by return from the fixed reference price to the last closed bar at
or before the cutoff. The builder records population size, configured `top_k`,
effective `top_k=min(top_k, population_size)`, rank, return, cutoff, label
availability, and snapshot hashes for every member.

This contract intentionally does not claim equivalence with a different
rolling-24h exchange statistic. A consumer must name the metric version. A
partial member makes the whole day non-decision-grade; it cannot silently
shrink the top-mover denominator.

### Existing mission-objective bridge

The production mission currently defines a top mover as an exchange top-N row
filtered to symbols that were in the watchlist. The maximum-history adapter
therefore emits `exchange_top_filtered_watchlist_v1` labels from the
`exchange_top_gainers` detail rows, not from the stored summary and not by
ranking only the watchlist. Each label binds:

- target day, exchange rank, symbol, and `in_watchlist=true`;
- bought/early-captured outcomes recomputed from row status, capture ratio, and
  the report's registered early threshold;
- source file SHA-256 and a deterministic population-snapshot hash;
- final cutoff and label availability time;
- exchange-top population size and filtered watchlist denominator.

The adapter validates target day, final phase, unique symbols, open/close/change
consistency, exchange-top count, and every summary numerator/rate against the
detail rows. A mismatch excludes the entire day with a named integrity reason.
It never repairs a report by trusting whichever side is more favorable.

Calendar coverage uses every day between the first and last exact final report.
Missing days remain named exclusions. For example, 118 final files inside a
135-day span are coverage `118/135`, not a complete 118-day history. A valid
zero-event day contributes to coverage but not to the event denominator; it is
neither a miss nor a success.

The aggregate objective audit publishes exact numerator/denominator, coverage,
deterministic day-cluster bootstrap interval, `SESOI=5pp`, conservative power
report, expected horizon, source manifest hash, and contract-verification
errors. It uses `baseline_only_no_directional_verdict`: a historical baseline
cannot label itself `IMPROVING` without a registered paired candidate/control
comparison. Partial calendar coverage can be measurement evidence but cannot
be promotion-grade evidence.

## Immutability ledger

`ImmutableLabelLedger` is append-only JSONL with idempotency by `label_id`.
Re-appending byte-canonically equivalent evidence is a no-op. Reusing an
existing `label_id` with different content raises `LabelConflictError` and
leaves the file unchanged. It never updates an older label in place.

## Dependence-aware power report

The initial conservative planning method treats the objective day as the
independent cluster:

- `raw_event_count` is descriptive;
- `complete_objective_days` and `effective_sample_size` are both explicit;
- `effective_sample_size <= complete_objective_days <= raw_event_count`;
- base rate and Bernoulli variance include numerator and denominator;
- the two-sided 95% interval width and two-arm MDE use the day-clustered
  effective sample size;
- `SESOI`, coverage, downtime, expected inconclusive probability, and earliest
  maturity date are mandatory;
- `UNDERPOWERED` is returned before validation when the estimated probability
  of an inconclusive result exceeds the registered `0.80` budget.

The initial estimator deliberately prefers underclaiming power. It does not
pretend that symbols from the same market day are independent. A later
hierarchical/bootstrap estimator requires a new method version and overlap
report; it may not rewrite results produced by `day_cluster_binary_v1`.

## Evidence-throughput baseline

The throughput report groups append-only attempt events by `attempt_id` and
publishes:

- total attempts and terminal attempts;
- terminal-rate numerator and denominator;
- terminal reason distribution;
- median and p90 time to terminal only for attempts with both timestamps;
- missing-duration count;
- power-feasible precheck numerator and denominator;
- label/logging loss numerator and denominator;
- evidence-reuse numerator and denominator.

An empty ledger reports `status=NO_EVIDENCE` and `null` ratios. It never reports
`0%` progress from an unknown denominator.

## Action-layer metric registry

The machine-readable registry binds each metric version to exactly one action
layer: `OBSERVATION`, `WATCH`, `BUY`, `SELL`, or `PORTFOLIO`. Each entry records
its numerator, denominator, label version, availability rule, decision use, and
guardrails. `Move5` is a steering metric only. WATCH metrics cannot approve a
BUY change; per-trade PnL cannot stand in for unified portfolio alpha; model
proxy metrics remain diagnostic.

## Harness remediation ledger

Every current blocking or warning finding is materialized with its TH/finding
ID, severity, scope, blocked actions, allowed work, one accountable owner
(`repository maintainer`), category, repair task, verification command,
`OPEN` state, acknowledgement target, review date, and source evidence hash.
Missing owner or remediation makes the Phase 0 audit fail closed.

The ledger does not waive a failing Harness. A stale model/RL artifact continues
to block current achievement/promotion claims while allowing measurement repair.

### Fresh blocked-training evidence

Freshness describes when readiness was checked, not proof that a model trained
or improved. When provenance-verified rows are below the configured training
minimum, the worker writes a fresh
`blocked_insufficient_provenance` readiness report instead of retaining a stale
successful-looking training report or retraining on legacy rows. The report:

- records the dataset watermark and labeled/verified/legacy denominators;
- declares `achievement_claimed=false` and `runtime_eligible=false`;
- publishes feature-time, label-time, label-definition, and evaluation-scope
  contracts, with `evaluation_scope=not_evaluated_insufficient_provenance`;
- does not overwrite the live model payload;
- remains a promotion blocker until enough forward verified rows mature.

The Truth Harness treats this as an honest warning, not model achievement and
not a provenance violation. It still fails if a blocked report claims runtime
eligibility, uses a holdout claim that never ran, or omits the evidence
denominators. Rewriting a timestamp alone is forbidden: each readiness refresh
must rescan the current dataset and bind its size, mtime, and provenance counts.
`ml_candidate_ranker.py --readiness-only` is the manual recovery path; it writes
only the readiness report and must not create or replace a model artifact.

## Legacy research migration

The bootstrap migrator enumerates the declared roots (`docs/reports` plus
research/rejected entries in `docs/FEATURE_SPEC_INDEX.md`). Every discovered
source receives exactly one state:

- `CONFIRMED_NEGATIVE` only when the source exposes recoverable period,
  population, metric, and verdict fields;
- `LEGACY_UNVERIFIED` when an attempt/result exists but decision provenance is
  incomplete;
- `DUPLICATE` when its content hash matches an earlier canonical item;
- `MIGRATION_ERROR` when the source is unreadable, with a named repair task.

The first implementation defaults ambiguous prose to `LEGACY_UNVERIFIED`; it
never infers a denominator or promotes prose into decision-grade evidence.
The inventory publishes discovered, migrated, and per-state counts so omissions
cannot look like completion.

Human-reviewed upgrades live in
`docs/specs/legacy-negative-results-registry.json`. A
`CONFIRMED_NEGATIVE` entry is accepted only when it has an ISO period,
population, canonical metric, explicit rejected verdict, evidence summary, and
the exact SHA-256 of the reviewed source. A later source edit invalidates the
upgrade back to `LEGACY_UNVERIFIED` until it is reviewed again. Registry entries
outside the discovered inventory or with duplicate IDs are reported as registry
errors; they cannot silently increase the confirmed count.

## Objective report completeness gate

Every objective row must publish:

- metric/action/label/method versions;
- numerator and denominator;
- coverage numerator, denominator, status, and exclusions;
- feature cutoff and label availability/cutoff;
- estimate, interval, SESOI, MDE, effective sample size;
- expected decision horizon, evidence status, registered verdict rule, and the
  deterministic result of that rule.

The verifier rejects missing, contradictory, non-finite, or out-of-range fields.
`UNKNOWN`, `PARTIAL`, and `UNDERPOWERED` are valid honest outcomes; they cannot
be rewritten as `IMPROVING`.

## Acceptance tests for this slice

1. A mature complete fixture produces stable Move5 and top-mover labels.
2. Forming, partial, post-cutoff, and timezone-naive evidence cannot enter a
   decision-grade denominator.
3. Conflicting reuse of a label ID is rejected without changing the ledger.
4. Power uses day clusters and emits `UNDERPOWERED` for a sparse fixture.
5. Empty throughput ratios are `null`; known attempts retain exact numerators
   and denominators.
6. Every registry metric has one action layer and cannot authorize another.
7. Every Harness finding has owner, blocked/allowed scope, remediation, review,
   and verification evidence.
8. Every discovered legacy artifact has one migration state; ambiguous prose
   remains `LEGACY_UNVERIFIED`.
9. Focused tests, architecture conformance tests, Truth Harness staged-change
   profile, and `git diff --check` pass.

`test_evidence_capacity_phase0` is part of the curated release suite so an
operational restart cannot silently bypass the Phase 0 evidence invariants.

10. The maximum-history critic adapter excludes manual/midday names, detects
    missing calendar days, recomputes summary metrics, and reports a baseline
    without a directional improvement claim.

## Phase 0 exit status

Passing this slice means the Phase 0 measurement foundation exists. Full Phase 0 remains open
until a maximum-available local audit proves that every reported
objective satisfies the completeness gate, current model/RL findings are
repaired or visibly owned, and the discovered legacy inventory is complete.
No uplift or production-readiness claim follows from this implementation.
