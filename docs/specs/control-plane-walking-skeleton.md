# Control-Plane Walking Skeleton

Date: 2026-08-14 · Status: Phase -1 implemented

Owner: repository maintainer

Related: [Continuous Improvement Control Plane](continuous-improvement-control-plane.md),
[Learning Loop Architecture Roadmap](learning-loop-architecture-roadmap.md), and
[Truth Harness](truth-harness.md).

## 1. Problem

The continuous-improvement architecture has a precise Phase -1 exit gate but no
executable vertical slice. The repository therefore cannot yet prove that one
frozen experiment can move from observation through independent validation to
an immutable terminal result. Adding more registries, agents, market validators,
or promotion logic before proving that path would repeat the failure mode the
architecture is intended to remove.

## 2. Success metric

The primary success metric is a deterministic protocol smoke run that completes
both cases below in less than ten seconds on the bundled runtime:

1. a valid synthetic attempt reaches
   `stage=CLOSED`, `status=TERMINAL`, `outcome_reason=protocol_verified`;
2. a deliberately corrupted but correctly attested validator bundle reaches
   `stage=CLOSED`, `status=TERMINAL`, `outcome_reason=invalid_result`.

Secondary acceptance criteria:

- rerunning the same `attempt_id` is idempotent and adds no duplicate transition;
- the verifier reconstructs decisions and metrics from the frozen raw snapshot
  and registered contract without importing validator aggregation code;
- the adapter processes at most 64 checked-in rows, uses only the Python
  standard library, performs no network access, imports no trading/replay code,
  and writes no production, release, promotion, config, position, or model state;
- every terminal record says `decision_grade=false` and
  `trading_conclusion_allowed=false`.

Passing this metric proves protocol conductance only. It is not a backtest, a
market result, or evidence that any trading metric improved.

## 3. Scope

### In scope

- `files/testdata/control_plane_smoke_fixture.json`: immutable raw fixture with
  no policy outputs and no more than 64 rows;
- `files/improvement_fixture_validator.py`: `FixtureDeltaValidatorAdapter`;
- `files/improvement_fixture_verifier.py`: independent raw-snapshot verifier;
- `files/run_control_plane_smoke.py`: minimal restart-safe orchestrator and
  append-only attempt ledger;
- focused unit and CLI tests for valid, corrupted, repeat, and isolation paths;
- a versioned, closed Phase -1 outcome-reason registry.

### Out of scope

- LLM, RAG, MCP, multi-agent roles, hypothesis generation, or provider APIs;
- market data, `replay_backtest.py`, portfolio simulation, model training, or
  production trading policy;
- capability/metric registries beyond the one fixed smoke contract;
- production governor, release store, shadow/canary, scheduler, or auto-promotion;
- Phase 0 labels, power expansion, or any BUY/SELL/WATCH behavior change.

## 4. Contracts and behavior

### 4.1 Raw fixture

The checked-in fixture is a JSON object:

```text
schema_version = 1
fixture_id      = control-plane-smoke-v1
rows[]          = {row_id, score, label}
```

Rows are ordered and `row_id` is unique. `score` is finite in `[0, 1]` and
`label` is `0` or `1`. The orchestrator copies the bytes into an attempt-local
snapshot before invoking either implementation and records its SHA-256 hash.

### 4.2 Registered smoke contract

The orchestrator writes an immutable attempt-local contract containing:

```text
contract_version
attempt_id
hypothesis_id
validator_id
fixture_id
metric = accuracy
baseline_policy = threshold(score >= 0.75)
candidate_policy = threshold(score >= 0.55)
snapshot_sha256
decision_grade = false
trading_conclusion_allowed = false
```

Changing the snapshot or contract under an existing `attempt_id` is a terminal
integrity error; it never creates a silent retry.

### 4.3 Validator result

`FixtureDeltaValidatorAdapter` independently loads the snapshot and contract,
reconstructs baseline/candidate predictions, and emits a minimal result bundle:

- contract and snapshot hashes;
- denominator;
- baseline accuracy, candidate accuracy, and paired delta;
- row-level validator trace for comparison only;
- validator identity and schema version;
- HMAC-SHA256 test attestation over the canonical result payload.

The corruption mode changes the signed summary after the validator's genuine
calculation. It intentionally models a validator that confidently emits a wrong
derived artifact; signature verification alone must not accept it.

### 4.4 Independent verifier

The verifier treats validator metrics and trace as untrusted. It:

1. reads the orchestrator-frozen raw snapshot and registered contract;
2. verifies hashes and the test attestation;
3. reconstructs both policy traces with its own implementation;
4. recomputes denominator, metrics, and delta;
5. compares them with the bundle and emits either `protocol_verified` or
   `invalid_result`.

`improvement_fixture_verifier.py` must not import
`improvement_fixture_validator.py`, replay, monitor, strategy, config, model, or
portfolio modules. Validator trace is comparison evidence, never verifier input.

### 4.5 State and ledger

Phase -1 uses the architecture's factorized state model. Allowed stage/status
values are closed enums. The Phase -1 outcome-reason registry is version `1` and
contains exactly:

```text
observed
validation_started
protocol_verified
invalid_result
```

Every ledger transition has an idempotency key
`{attempt_id}:{stage}:{outcome_reason}`. Repeating an identical transition is a
no-op; conflicting reuse of a key is an error. The ledger is append-only JSONL
inside the caller-supplied smoke state directory. No API accepts a release-store
or production-state path.

## 5. Safety, rollback, and applicable Truth Harness rules

There is no production feature flag because the slice is not imported by the
bot and has no scheduler integration. Rollback is deletion/reversion of these
isolated Phase -1 source files; immutable smoke ledgers may be retained as test
evidence.

Applicable integrity rules:

- **TH-07:** no behavior change, production flag, canary, or promotion path;
- **TH-08:** valid, corrupted, failed, and repeated attempts remain visible;
- **TH-09:** implementation status and named files must match this specification
  and the architecture index;
- **TH-10:** synthetic evidence is labelled synthetic and cannot support a
  market conclusion;
- **TH-12:** specification, focused tests, implementation, verification output,
  commit, and push travel together.

Risks and mitigations:

| Risk | Mitigation |
|---|---|
| Validator and verifier share a defect | separate modules, no shared aggregation imports, corruption test |
| Smoke result is mistaken for trading evidence | terminal records carry two explicit false capability flags |
| Retry rewrites history | append-only ledger plus idempotency-key conflict check |
| Skeleton imports the monolith | AST/import tests and explicit deny-list |
| Fixture grows into a hidden backtest | hard 64-row limit and fixed accuracy contract |

## 6. Test-first verification plan

Tests are written and observed failing before implementation. Required checks:

- fixture schema, size, ordering, and expected fixed delta;
- valid bundle independently verifies;
- correctly attested corrupted bundle is rejected as `invalid_result`;
- same attempt resumes without duplicate ledger entries;
- conflicting attempt snapshot/contract is rejected;
- source import graph contains no validator/verifier coupling or banned modules;
- smoke suite finishes under ten seconds and creates no release/promotion state;
- existing architecture conformance tests remain green;
- `git diff --check` passes.

No trading-policy backtest is required because this change has no market inputs,
candidate population, or trading behavior. Treating the fixture as a backtest
would itself violate TH-10.

### Verification results

Verified 2026-08-14 on the bundled Python runtime:

- test-first red state: `test_control_plane_walking_skeleton.py` failed with
  `ModuleNotFoundError: improvement_fixture_validator` before implementation;
- focused implementation suite: **10/10 PASS** in 1.82 seconds;
- architecture conformance suite: **10/10 PASS**;
- one-command valid + deliberately corrupt smoke suite: **PASS** in 0.203
  seconds; valid ended `protocol_verified`, corrupt ended `invalid_result` with
  a valid attestation and independently detected metric mismatch;
- `compileall` and `git diff --check`: **PASS**;
- full Truth Harness: honest **FAIL** from existing model-evidence state
  (`TH-03/TH-04` missing timing/holdout provenance, `TH-03` zero
  provenance-verified rows; stale RL evidence warning). The smoke result is not
  used to waive or improve that verdict;
- curated release suite: 130/131 tests passed; unrelated
  `test_rl_headless_worker.TestDailyCriticSchedulerRecovery` expected one ranker
  row and observed zero. No control-plane file is in that failure path;
- full legacy discovery: 837 tests, 122 failures and 54 errors on historical
  drift. This result is recorded rather than presented as green.

The Phase -1 exit gate is satisfied. This is protocol evidence only and does
not authorize Phase 0 completion, an agent, a market experiment, or a trading
behavior change.

## 7. Follow-ups

- Phase 0 evidence-capacity and canonical-label work remains blocked until this
  specification's exit gate passes.
- Production-grade service identity and asymmetric result signing belong to the
  later validation-service phase; the Phase -1 HMAC is a local protocol
  attestation, not a production trust boundary.
- The full architecture's outcome-reason registry will extend the closed Phase
  -1 registry through a separately versioned contract.
