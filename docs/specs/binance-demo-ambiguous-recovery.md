# Demo ambiguous-send recovery and cycle-scoped status

Date: 2026-10-29. Independent Demo app only. SPEC -> RED tests -> code.
Parents: execution-qualification pilot, all-USDT universe, transport recovery.
TH-01..TH-12 apply. This repairs safety/observability, not the trading policy.

## Contracts and named acceptance scenarios

REC-01: Starting a new BUY cycle removes the previous cycle's fill_price,
protected_quantity, take_profit, stop_loss, deadline_ms,
unprotected_duration_ms, protection_action and recovery diagnostics. Preserve
the immutable binding/opening SHA, attempt count, qualification, cadence,
portfolio dust/accounting, all historical actions, outcomes and incidents.
An ambiguous new BUY cannot inherit apparent protection from another symbol.

REC-02: Order recovery uses signed GET with the original symbol/client ID (or
owned order/list ID). Propagate clock, auth, rate, audit and transport errors.
The adapter may attach the safe numeric API error code; only exact -2013 from
an order/list lookup becomes PILOT_ORDER_NOT_FOUND. Generic API_REJECTED is
not absence evidence. Do not log exchange messages, credentials or signatures.
Not-found, malformed/foreign responses and unavailable queries retain the
action's QUERY_REQUIRED, never REJECTED, ACK or proof of never sent. No resend.

REC-03: Each recovery pass performs at most one logical action lookup and
durably records its attempt/check time and next
query time before querying. Retry delay is 10,20,40,80,160,300 seconds, then
capped at 300; rate failures wait at least 60 seconds. Logical lookups are at
least 10 seconds apart (at most six in a half-open rolling minute). Existing
bounded adapter HTTP retries remain separately audited; this clarifies the
pilot's six-read recovery bound as logical lookups, not physical TCP attempts.
An ACK with other unresolved actions leaves recovery blocked for the next pass.
Early calls/restarts
respect persisted next-query time and issue no GET/mutation. Save the last
error reason separately from the original incident and keep entry kill set.
The worker sleep uses this deadline. Queries continue at the bounded cadence
after pilot expiry: missing exposure evidence cannot certify flatness.
Both BTC and all-USDT worker loops keep recovery scheduled rather than running
a one-second busy loop and exiting before the persisted retry deadline.

REC-04: Resume only an explicit BUY_PENDING/PROTECTION_PENDING/OPEN recovery
phase after all unknown actions resolve. A pending entry requires the current
cycle's ACKNOWLEDGED BUY with the same symbol before any continuation; absence
of a prepared/sent BUY must not create a new BUY during recovery. An entry
without an existing action also checks expiry, entry kill and stop-entries
immediately before sending. Keep protection/exit recovery for acknowledged
positions after expiry, with existing ownership/account/price/risk checks.
Never default a missing recovery phase to OPEN. An invalid phase remains
blocked with an explicit diagnostic; no automatic entry resume.

REC-05: Both worker status writers project current-cycle position fields from
the journal's action ownership. Inactive/unknown entry hides fill/protection
fields; fill display requires a same-cycle/same-symbol BUY ACK. TP/SL/quantity,
deadline and protection duration require the same cycle's protection ACK and
OPEN (or recovery of OPEN). A legacy stale protection reference is hidden.
Expose a distinct UNVERIFIED/CLOSED/PROTECTION_PENDING/PROTECTION_ACKNOWLEDGED
position-status label. An ACK is ownership evidence, not fresh exchange
verification. Preserve the raw historical journal rather than rewriting it.
RECOVERY_REQUIRED is operationally BLOCKED even if old state lacks entry_killed.
Monitoring freshness, expiry, private health and recovery uncertainty remain
separate. Reported current exposure remains UNKNOWN unless independently
reconciled; hidden fields must never be interpreted as a flat account.

## Verification, rollout and rollback

Synthetic app-owned journal/exchange tests reproduce stale fields across two
symbols, unknown not-found BUY, clock/auth/rate/read failures, persisted retry,
restart, invalid recovery phase, crash before send, expired unsent entry,
accepted BUY recovered and protected after expiry, unknown protection,
same numeric IDs across symbols and status projection without raw mutation.
Tests make no network calls or credential reads. Retain RED/GREEN evidence in
own runtime; run focused/full regressions, diff review and staged Truth Harness.

No live journal migration, worker restart, new interval, risk increase, seed
reset, strategy promotion or trading-policy relaxation is part of this patch.
Existing processes retain their loaded version until a separately reviewed
deployment. Preview legacy status read-only; do not claim a live fix from tests.
Rollback: block new entries; preserve owned protection, exits and GET recovery;
revert source only without deleting actions, deadlines, fills or original seed.
No alpha/revenue claim: full Truth Harness FAIL TH-11 remains FAIL; tests do not
certify current exposure, private delivery, daily PnL or model improvement.

Verification 2026-10-29: genuine RED preceded implementation for REC-01..05:
stale position fields, unsent BUY after kill/expiry, swallowed GET failures,
missing diagnostics/cadence/status interface; subsequent worker RED reproduced
exit before the second scheduled query and invalid BUY action acceptance.
Final focused GREEN: 16 lifecycle + 9 query scenarios; complete app regression
1145/1145 GREEN (226.652s); Truth Harness unit regression 13/13 GREEN.
Read-only legacy-journal preview hides seven old position fields and retains
buy-5 uncertainty; no live journal/worker mutation or financial certification.
Evidence: own .runtime/recovery-repair/ RED/GREEN/full-regression/preview files.
Full Harness is FAIL TH-11 (stale replay_backtest.py source hash in canonical
portfolio alpha); no waiver, policy promotion or earnings claim is made.
