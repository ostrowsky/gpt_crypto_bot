# Independent Demo bot: P0-A tests

Specifications: [financial API](../../../docs/specs/binance-demo-phase0-financial-contract.md#p0-a-публичный-контракт-первой-партии-unit-тестов),
[isolation kernel](../../../docs/specs/binance-demo-application-isolation.md).
Required order: **SPEC → TESTS → CODE** in [app rules](../AGENTS.md).

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

Remaining acceptance gates: actual exchange reconciliation/anchors and PIT
valuation/seed availability, reset evidence, cross-arm sum,
account ownership, full process/resource/port isolation and phases P1–P7.
This is partial FIN-01..12/ISO coverage, not completion of those whole scenarios
or proof of profitability. The offline entrypoint never enables order sending.
