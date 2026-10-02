# Certified learning lifecycle

2026-10-02. TH-01..TH-12. No evidence thresholds or production policy relaxed.

1. Automatic issuer verifies evaluator-signed full-policy validation receipts,
not trainer booleans. Exact bundle/model/evaluator hashes, source freshness,
maximum period, raw closed provenance, point-in-time population, complete replay
and candidate-generation + live execution parity must be independently attested.
Execution-state-only parity is insufficient. Only logical_same_user certificates
are issued by this accepted local runtime; no Windows isolation assertion.
Negative numerical outcomes can be certified but cannot authorize a strategy.
Missing/expired/mismatched proof remains BLOCKED, never manufactured by issuer.

2. Sealed, shadow, canary have distinct immutable phase registrations and fresh
flat paper portfolios, without invented SELLs in the previous cohort. Advance
only after verified numerical PASS; failed cohorts remain REJECTED. Canary also
requires a valid signed CANARY release ticket. Actual canary assignment receipts
are required to certify canary; hypothetical full-candidate paper is not canary.
Minimum 30 days / 100 completed trades, chronology and embargo remain unchanged.

3. Actual runtime admission receipts bind the selected score receipt, signed
ticket/model, symbol/bar and persisted position. These prove use in a paper BUY
decision, NOT exchange fills, causal uplift or realized profitability. Independent
verification must check durable position/admission evidence, not just a score.
No closed_loop=true or growth verdict merely because one receipt exists.

4. Controller rollback is independently checked at active pointer plus a later
runtime fallback observation. Pointer-only rollback is authorized/applied, not
verified runtime consumption. Expired ticket/model/source mismatch must fall back
to unchanged champion. No tests create production success receipts.

Validation: issuer wrong scope/signature/hash/expiry; certified rejection;
non-overlapping phase registrations, restart/idempotence and failed-canary refusal;
admission vs score/fill distinctions; stale/forged receipt and rollback checks.
Maximum-period validation precedes enablement. Config remains disabled until
genuine certified historical/forward gates pass. Source changes invalidate older
registrations and require new replays/cohorts. Runtime proof/model/key artifacts
are not committed. Rollback: stop lifecycle and retain disabled champion path.

Current implementation boundary: the issuer consumes independently signed
full-population/live-policy validation; it does not manufacture that validation.
The existing maximum stream execution verifier has narrower scope and cannot
produce an eligible full-policy proof. Canary collection intentionally reports
WAITING_ACTUAL_CANARY_COLLECTOR rather than labeling hypothetical full-candidate
portfolios as assigned production canary. These remain explicit blockers.
Admissions are current only with unchanged active authorization and a receipt
within 600 seconds. Rollback fallback must concern a symbol assigned to the
previous signed candidate; repeated controller failures preserve the original
rollback request and its verified observation. Neither status proves profits.
