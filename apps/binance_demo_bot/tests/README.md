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

Remaining acceptance gates: actual exchange reconciliation/anchors and PIT
valuation/seed availability, reset evidence, cross-arm sum,
account ownership, full process/resource/port isolation and phases P1–P7.
This is partial FIN-01..12/ISO coverage, not completion of those whole scenarios
or proof of profitability. The offline entrypoint never enables order sending.
