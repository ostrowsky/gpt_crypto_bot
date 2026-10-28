# Native Demo financial boundary observations (FBO)

2026-10-28. SPEC before tests and code. Independent app only; TH-01..12.
Parent: P0 financial account, actual financial bridge, interaction audit.

FBO-01: bind to the existing immutable, valid provisional opening candidate,
account HMAC, namespace and Europe/Budapest timezone. Do not create another
seed, financial episode, ledger or order authority. Reject a different opening
on reopen. Keep observations in an own hash-verified durable audit journal.
Observation IDs are immutable/idempotent, conflicting repeats fail. A failed
commit cannot return success. Public asset identifiers including TOKEN/COOKIE
must round-trip without relaxing secret redaction; store encoded row fields.

FBO-02: reuse the opening source validator for actual GET account-before,
orders, order lists, metadata, book tickers and account-after, including hashes,
audit references, HMAC, namespace, same credential generation, monotonic/wall
quality, <=250ms uncertainty, account bracketing and fresh <=5s quote. Retain
free AND locked inventory and all valuation reasons. No synthetic snapshots or
forward-filled marks. Preserve source pins, original opening SHA and observed
response time basis. Do not save hypothetical new seed lots.

FBO-03: only a Budapest local midnight is a boundary; DST days may have23/25h.
Choose the last eligible observation at/before the boundary and first after,
with account receipts within30s and pre-boundary quote receipt within5s. A
future quote can never become the opening mark for that instant. Missing,
stale, changed holdings, blocked or incomplete valuation remains PENDING with
reasons. An unchanged valid bracket is PROVISIONAL_BRACKET only: two REST
responses do not certify the exact economic cut, global fill/flow population
or private delivery. Daily net PnL stays null and financial status PENDING.

FBO-04: a finite GET-only observer collects the next two local midnights, then
exits. Warm metadata before the capture window; sync the unchanged guarded
clock and collect native brackets near -6s,+1s,+10s. Late/failed sources remain
explicit gaps; no fabricated backfill. Persist waiting/capture/error/completed
status and source files under own runtime, with bounded own network audit.
Check original account binding at capture. Never invoke mutation APIs, restart
trading, extend its expiry/budget, train or promote. An own stop file exits the
observer without affecting trading. A restart skips past capture offsets;
it never manufactures missing historical data. Native observations have source
timestamps; unit fixtures are synthetic and do not prove account performance.

## Verification and rollout

SPEC -> focused tests -> expected RED -> implementation -> GREEN/regressions,
diff review, staged Truth Harness, commit and push. Test real own SQLite/reopen,
faulted commits, source hashes/namespace/clocks, exact cutoff selection, missing
before/after, stale/future quotes, holdings changes, sensitive asset names and
DST. Observer scheduling uses an injected clock/client in tests: no keys or
network. Deploy the finite read-only worker hidden; verify its current binding
and WAITING status. Rollback: own stop file, retain all evidence. First capture
does not recover the missed start of2026-10-28 or certify daily PnL. Full Harness
TH-11 remains FAIL separately from implementation correctness.

FBO-01 schedule extension before tests/code: freeze the next two midnight
boundaries once in the own durable journal. Reopen returns that same schedule;
never roll a missed capture into a newly invented original day. FBO-04 injected
scheduler tests verify finite completion, stopped waiting, failed native capture
and skipped past slots; native capture calls GET-only sources in bracket order.

FBO-02 boundary extension before tests/code: accepted public asset/symbol
names follow existing source validation (up to64 word characters), including
sensitive substrings and underscores. Missing conversion reasons must survive
as encoded asset + reason rows; redaction must not erase financial gaps.

FBO-01 concurrency extension before tests/code: the waiting observer must not
hold the original opening manifest's writer lease. Freeze verified immutable
attestation/namespace at bind, share the same pure source validator, and close
the original handle before waiting. Existing financial reconciliation must be
able to open its original manifest while boundary storage remains active.

Final implementation verification:17 focused GREEN(3.113s), all1069 app tests
GREEN(160.551s), root13 harness tests GREEN. Source validator moved without
changing its existing opening/financial contracts. No frozen financial.py,
journal.py or isolation.py change. Native deployment remains a separate check.
