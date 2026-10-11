# Independent Demo bot: P0-A tests

Specifications: [financial API](../../../docs/specs/binance-demo-phase0-financial-contract.md#p0-a-публичный-контракт-первой-партии-unit-тестов),
[isolation kernel](../../../docs/specs/binance-demo-application-isolation.md).
Required order: **SPEC → TESTS → CODE** in [app rules](../AGENTS.md).

`test_market_store_concurrency.py` adds P2B2-06: actual concurrent WAL reads/
writes, atomic/pinned prefixes, prepared groups/dependencies/revisions, query
plans, receiver lifecycle and detached cache integrity. Current app: 434 GREEN.
`test_market_checkpoint.py` adds 22 P2B2-07 cases: pinned market-only engine,
explicit transaction/Row/error/trace/file-format compatibility, PASSIVE worker
without writer lock, version/WAL/FULL guards, byte pressure under the writer,
busy/partial/failure diagnostics, restricted maintenance SQL and owned lifecycle.
`test_market_writer_latency.py` adds 12 P2B2-08 cases: actual cache writes/spills,
no SQL/reset telemetry, bounded multi-value inserts, rollback in a later chunk,
overlapping retry, unlocked/pinned result read, causal dependency cache and
three-closed-bar warmup with a current open bar.

| Test file | Coverage in this batch |
|---|---|
| `test_financial_valuation.py` | Decimal, free/locked identity, whole inventory, dust/stablecoins, conversion routes, stale/future quote rejection, historical clock protocol, DST/day boundaries. |
| `test_financial_ledger.py` | Partial fills/FIFO, base/quote/BNB fees, fee-asset disposal, non-USDT quote, duplicate/conflicting trades, atomic failure, seed/foreign/unattributed ownership. |
| `test_daily_performance.py` | Equity reward, flows, no-trade MTM, reconciliation/completeness, TWR, extra-account costs, calendar gaps and corrected revisions. |
| `test_financial_edge_contracts.py` | FIFO vs operational ownership, ambient Decimal context, malformed fills/statuses, refusal of late arrival-order fills until ordered replay. |
| `test_application_isolation.py` | Own root/CWD independence, traversal and real junction/symlink escape, parent-env rejection, explicit-only demo keys, duplicate/missing values and redacted diagnostics. |
| `test_offline_startup.py` | Installed own-venv package starts without credentials/trading; inherited PYTHONPATH rejected. |
| `test_durable_journal.py` | 29 integration cases: WAL/FULL, actual process deaths, single writer, fill/report retry, outbox integrity/ACK, late FIFO/PIT receipt cutoffs, revisions, namespace/schema/cache and provenance. |
| `test_adapter.py` | 56 P1-A cases: exact REST HMAC bytes, Demo host/GET/TLS guards, persistent connection, clock and actual receipts, redaction, Decimal/schema/UID tokens, permissions, read-only orders, fill pagination, deadline/retry/auth/rolling and shared-IP budget. |
| `test_private_stream.py` | 31 P1-B cases: WS signing/host/TLS/quiet logger, ACK/session, buffering, normalization, fees/flows/lists, heartbeat/worker stop, shared limits and redaction. |
| `test_account_evidence.py` | 38 P1-B cases: 36 real durable/reconcile scenarios plus 2 open-list adapter tests; atomic recovery/cursors, actual crashes, hashes/namespace/envelopes, duplicates/conflicts, journal/BNB fees, unknown coverage and report freshness. |
| `test_public_market_client.py` | 7 P2-A cases: credential-free reads, private refusal, exact raw-body SHA, 32MiB metadata/8MiB private caps, bounded gzip/corruption and narrow header allowlist. |
| `test_universe.py` | 20 P2-A cases: complete permission query, AND/OR/unknown rights, public/null denominator, scope/freshness/token and persisted Decimal provenance, non-USDT/status/filter population, transitions, stale failures, immutable own WAL/lease/hash/dependencies. |
| `test_market_data.py` | 30 P2-A cases: books/omissions/invalid/stale responses, reserve/fair rotation, closed/open bars/gaps/revisions/late receipts, immutable cutoff prefixes, incident recovery, exact features and direct/inverse/multileg/stable/dust marks. |
| `test_market_stream.py` | 46 P2-B1 cases: Demo transport/ACK/TLS/topic guards, sharding/non-USDT/UTF-8, depth weights/reserve, bounded atomic L2 bridge/revisions/gaps/age, actual reader threads, restart/quarantine, heartbeat, quota/backoff/stop barrier and durable receipts. |

Run from the app directory with its own installed environment:

```powershell
.venv\Scripts\python.exe -I -m unittest discover -s tests -p 'test_*.py' -v
```

All money fixtures and expected numbers are synthetic and independent of the
implementation. No keys/private payloads, real orders, skips or xfail cases.
The link-escape test uses a Windows junction without admin privileges; it removes
only that link before cleaning up its bounded app-owned fixture directory.

Evidence sequence on 2026-10-10:

1. Original 64 tests: RED, intended missing financial module, exit 1 (commit `7d98413`).
2. Added isolation tests: 74 discovered / 74 missing-interface errors before
   the new modules. Four additional financial boundaries also verified RED
   before the financial implementation (40 selected financial tests).
3. 78 financial/isolation tests passed in own venv; two startup tests RED with
   missing `binance_demo_bot.__main__`. An initial uncommitted startup stub had
   been created too early and was withdrawn for this correction; that initial
   stub is not claimed as compliant test-first history.
4. Late-fill reproduction: five edge tests / one failure (no AccountingError),
   then the chronological guard was implemented. Final full run: **81 GREEN**.

Raw RED/GREEN output belongs in `../.runtime/test-evidence/` and is not committed.
P0-B sequence: 24 missing-module errors before journal implementation; explicit
SQLite-fixture connection cleanup corrected; additional outbox/provenance and
report-retry failures reproduced before fixes. Final full run **110 GREEN**.
The crash tests use own-venv subprocesses and `os._exit`, not simulated mocks.
The old bundled interpreter was an external engineering bootstrap runner for
the initial RED only; it is not a runtime dependency of this application.

P1-A sequence on 2026-10-11: spec registered; **41 intended missing-module
errors RED** in own venv, then 41 GREEN. Added boundary reproductions yielded
10 failures RED before fixes, then 52 GREEN. Connection reuse and quality-sample
contract yielded 2 failures/2 errors RED before code, then **56 adapter GREEN**.
Final URL-review assertions produced 5 failures/1 error RED before canonical
base-URL and path-type guards; total count remained 56.
Full app run: **166 GREEN**. No assertions weakened/skipped. Protocol size-test
fixture gained `read1` support to match the transport interface while retaining
the exact size guard. Scripted transport responses prove request contracts, not
actual account permissions or profit. Public Demo smoke separately verified
HTTP 200 ping/time and 289–296ms warm RTT; private smoke remains pending.

P1-B on 2026-10-11: 21+26 intended missing-module RED errors and one missing
REST symbol field. Open-list tests were isolated from an unrelated missing
fixture dependency and reproduced two missing-method errors before code. New
boundary reproductions preceded fixes (money/commission, cursor/schema,
terminal state, trust/coverage, generation races, WS/REST quota, redaction and
envelope). Fixture corrections retained all assertions: stale request clock
moves with receipt, repeated comparisons reuse their own journal writer lease.
Final **235 app tests GREEN**, including 69 new P1-B cases. Before/after-COMMIT
crashes use own-venv subprocesses and real SQLite. Public WSS handshake/time/Pong
passed separately; no authenticated subscription or orders were attempted.

P2-A on 2026-10-11: 17 universe and 26 market tests failed for the missing
intended module; 5 public tests failed for missing client/public paths. Review
reproductions then preceded fixes for hashes, incident readiness, mutated/future
membership, episode account token/provenance and schema failures. Two explicit
gzip cases reproduced missing request header/allowlist before code. The inverse
fixture changed to finite 1/4 (expected equity assertion unchanged); rounded
50-digit 1/3 times 3 is not algebraically 1. A fixture queues the real clock-probe
response after the synthetic 60-second lease expires; no clock guard is skipped.
Persisted P1 Decimal hash mismatch was reproduced before its fixed-point
provenance serialization repair. Final **292 app tests GREEN**, 57 new P2-A
cases, no skips/xfail.
Public smoke separately verified metadata/books/closed bars/features and a
hypothetical inventory, not actual permissions/balances or latency readiness.

P2-B1 on 2026-10-11: 31 missing intended stream-module/depth-method errors RED
before implementation. Four review failures reproduced future snapshot receipt,
heartbeat under continuous data, quote sequence recovery and runtime backoff;
missing lag diagnostics reproduced separately before repair. Actual concurrent
side-replacement barrier yielded an empty side before lock repair; UTF-8/mixed
source names failed before parser/mapping repair. Clock health and stop barrier
failures preceded guards. Final **338 total app GREEN**, 46 new cases.
Public stream smoke separately used an explicitly labelled BTC/ETH/SOL sample:
all three reached READY, first final point 2/3 and repeated final point 3/3.
No account keys/orders or full-market sustained qualification were involved.

Remaining acceptance gates: actual exchange reconciliation/anchors and PIT
valuation/seed availability, reset evidence, cross-arm sum,
account ownership, full process/resource/port isolation and phases P1–P7.
This is partial FIN-01..12/ISO coverage, not completion of those whole scenarios
or proof of profitability. The offline entrypoint never enables order sending.

P2-B2 on 2026-10-11: initial 22 cases had 2 failures/19 missing-interface
errors RED; canonical hash parity was correctly GREEN. Five review failures
preceded fixes for fixed-context timestamps/future receipts, rolled-back candle
incident durability, tombstone provenance and current denominators. Three live
hot-path/degraded-state failures preceded the compact projection/refactor.
Standalone lock contention and missing FIFO lock were reproduced before fixes;
four failure-index/clock-recovery/percentile failures preceded repair. Final
Index-copy contention and delayed quarantine of a rejected future receipt were
also RED before fixes. 36 new cases / **374 app tests GREEN**. No skip/xfail or weakened assertions.
Public full-market qualification remains FAIL: 3727 discovered, 1375 active,
6 shards, all failed processing-lag guards. A complete REST point snapshot
does not certify sustained stream availability; unsuccessful trials retained.

P2-B2 contention repair: 26 additional tests / **400 app GREEN**. Initial
12 scenarios had 10 failures/1 missing-connection error before implementation;
the owner-candle contract remained GREEN. Count/time cap and actual query-plan
reproductions preceded fixes. Missing prepared batch, reader contention/cache
interleaving, exact candle guard diagnostics and connection cleanup were RED
before code. Pinned WAL snapshot coverage was already GREEN after the first
read-connection implementation, correctly retained as preserving coverage.
Full-public trials still FAIL; latest SQL profile isolated commit up to 400.63ms.
No runtime artifacts, account keys or orders were included in test fixtures.

P2B2-07 checkpoint: initial 15 tests yielded 14 intended missing-interface
errors RED; existing SQLite file-format compatibility was correctly GREEN.
After installing the pinned app-only dependency and implementing the adapter/
worker, 15 GREEN. Five failure-boundary reproductions yielded 3 failures/5
errors including cleanup consequences before repair; then 20 GREEN. Two
pressure/busy reproductions yielded 1 failure/1 missing-category error before
repair. Final **22 focused / 422 app GREEN**, no skip/xfail or weaker assertions.
One busy case holds a real SQLite write transaction against a second connection.
The background maintenance worker never proves account permission, earnings
or full-market SLA. Preserve the separate unsuccessful public observations.

P2B2-08 sequence: 3 missing cache_stats errors RED, then actual native-counter
GREEN; 3 failing bulk/chunk/unlocked-return scenarios RED before repair. Existing
late-chunk rollback and overlapping retry contracts correctly stayed GREEN.
Dependency query-count reproduction showed 8 queries instead of 1 before repair;
the missing/staged dependency case remained GREEN. Pinned-return regression
reproduced a null envelope before repair; latest-N warmup reproduced limit3
instead of4 before repair. The fake supervisor stays patched through startup/
stop and an explicit network deny probe remains uncalled. Final **12 new / 434 app
GREEN**, with actual traced SQL and a real SQLite abort trigger; no skip/xfail
or weakened assertions. Keep public market verdict separate from unit success.
