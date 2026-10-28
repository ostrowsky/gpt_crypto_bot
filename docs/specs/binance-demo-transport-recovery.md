# Demo clock/private transport recovery and truthful operational state

Date:2026-10-28. SPEC before tests/code. Parent: bounded execution pilot,
all-USDT scope, P1 account adapter, interaction audit; TH-01..12 apply.
Scope: independent Demo application, unchanged original seed/expiry/risk.

TRN-01: retain existing clock uncertainty<=250ms, RTT<=500ms for a valid
sample, wall/monotonic jump guards,60s sample TTL and bounded3-sample sync.
Do not reuse rejected/stale samples or increase recvWindow. A transient clock
failure before any new action in flat IDLE is durable WAIT, not a permanent
HALTED. Keep existing entry kill unchanged; clear only the data-wait marker
after a subsequent valid snapshot. Errors after a possible send remain recovery
required with entry kill; never resend an unknown order.

TRN-02: private heartbeat loss is independent of quote monitoring. A flat
IDLE worker supervises its own reader, joins/closes the previous reader, opens
a new authenticated subscription and keeps every incident/session/gap. At most
one reconnect attempt/minute; auth/protocol/audit/namespace errors are not
automatically downgraded. A stuck old reader prevents a second reader. Critical
protection/exit/recovery runs first; no slow reconnect on an active position.
Healthy transport is not private economic-delivery or financial certification.
If durable reconnect logging fails, close the newly opened reader before
propagating the failure. A connected socket with a dead reader thread is not
entry-ready. Warm catalogue/source loading precedes the three reviewed clock
probes; never accept an expired probe group as evidence of current readiness.

TRN-03: an existing flat clock HALTED state can be reviewed only with an
explicit --review-clock-halt flag (user-authorized repair). Require verified
journal origin: prior IDLE had entry kill false, same attempt count and no
protection. Require all action outcomes known, last attempted cycle CLOSED with
matched private delivery (or zero attempts),3 freshly accepted clock probes,
healthy private reader, fresh exact account/owned-fill reconciliation, no open
orders/lists globally, all owned dust<=2USDT, original expiry/max8 and1USDT
loss/drawdown stop. Refuse every failed/missing condition. Preserve cadence,
client IDs, original binding/seed, attempts and incidents; log operator review.
Never resume loss stops, unknown sends or a preexisting independent entry kill.

TRN-04: publish separate monitoring and trading status. Fresh519 quotes can
coexist with BLOCKED/HALTED/WAIT_CLOCK/WAIT_PRIVATE. Overall RUNNING must not
hide a halted controller or dead private reader. Entry-enabled requires a fresh
eligible snapshot and healthy transport; no financial/P1/alpha PASS is implied.

## Verification / rollout / rollback

Synthetic own real-journal tests: flat clock WAIT/recovery; prior kill retained;
active/unknown send stays blocked; reader reconnect/backoff/auth/stuck/no active
reconnect; reviewed resume origin/account/dust/foreign/expiry/actions/delivery/
clock conditions; current status reflects both planes. SPEC -> RED -> code ->
focused/full regressions -> diff review -> staged Truth Harness -> commit/push.
Native read-only inspection confirms pre-send flat halt. Request own worker's
graceful flat stop; never stop legacy/history/research processes. Reopen SAME
journals/profile and use reviewed resume after fresh sources; verify a bounded
protected cycle without changing the original budget/expiry. No profit-policy
relaxation, so strategy replay remains independently waiting on full history.
Rollback blocks new entries and preserves owned protection/exit/query recovery.
Keys and runtime are not committed. Full Harness FAIL TH-11 remains FAIL.

Verification2026-10-28:19 focused GREEN;1050 full GREEN(245.543s). First full
run failed one existing group-writer admission timeout; unchanged24-test suite
and full repeat passed. Retain failed evidence. RED/GREEN are own runtime
records; native rollout is a separate operational check, not financial proof.

TRN-03 native rollout extension, before test/code: first reviewed rollout
refused PILOT_REVIEW_NOT_READY after a slow existing market-store verification
outlasted private heartbeat health. Preserve that failed run. Supervise the
private reader AFTER catalogue/storage warmup, BEFORE reviewed clock probes.
Only existing transient reconnect rules apply; a warmup or source failure must
not clear HALTED. No sends/limit changes are introduced by startup supervision.

Startup follow-up verification:2 new RED ->21 transport GREEN(2.535s), final
full1069 GREEN(160.551s). No threshold/budget/expiry change; failed native first
run remains an incident. Reviewed-startup reconnect is not financial coverage.
