# Independent Binance Demo bot — P0-A/B financial core and journal

This separate application implements offline Decimal accounting, valuation,
daily equity PnL, FIFO/fees, SQLite journal/outbox/recovery and initial isolation. It does
not send orders, run learning workers or connect to Binance. Actual
exchange reconciliation adapters and the P1–P7 services remain planned.

Specifications: [financial contract](../../docs/specs/binance-demo-phase0-financial-contract.md),
[application isolation](../../docs/specs/binance-demo-application-isolation.md).
Development order: **SPEC → TESTS → CODE**; see [AGENTS.md](AGENTS.md).

From `apps/binance_demo_bot` on this workstation:

```powershell
uv venv --python C:\Python314\python.exe --no-config --cache-dir .runtime/uv-cache .venv
uv sync --locked --no-config --cache-dir .runtime/uv-cache
.venv\Scripts\python.exe -I -m unittest discover -s tests -p 'test_*.py' -v
.venv\Scripts\python.exe -I -m binance_demo_bot
```

Use a separately installed Python 3.14 on other machines. The lock freezes
runtime dependencies; the build backend is explicitly pinned. All cache,
fixtures and runtime evidence belong under this app's ignored `.runtime`.
The existing bot's Python, environment, state and workers are not dependencies.

`-I` prevents CWD/PYTHONPATH from substituting an installed package. The offline
entrypoint also rejects any nonempty inherited PYTHONPATH and returns JSON
with `order_send_enabled=false`. It reads no credentials and starts no services.
The current entrypoint supports this editable project layout, not a deployed
wheel/service. Always test future deployment packaging separately.

When the account adapter is implemented, provision a dedicated demo pair in
this app's ignored `.env`, based on `.env.example`, or pass the complete pair
explicitly. No parent-env discovery, old-config imports or legacy key fallback.
Real credentials have not been copied or validated in this milestone.

`FinancialLedger` is the pure in-memory projection. `JournalStore` persists its
immutable fill events, separate receipt clocks, projection and outbox in SQLite
WAL/FULL with an app-local OS writer lease. Late fills rebuild chronological
FIFO; receipt-cutoff queries exclude fills not yet available. Reports retain
all revisions; changed economic sources mark earlier revisions stale. Exact
fill/report retries and outbox ACKs are idempotent; delivery is at-least-once.

All journal/report evidence remains `LOCAL_PROJECTION`, `exchange_reconciled=false`.
Callers supply marks and certified daily snapshots. Historical availability
of seed anchors/marks, actual exchange coverage and account ownership must be
established by P1/P2/reconcile. No production/account permission follows from
storage correctness. The journal freezes its accounting code fingerprint;
changing accounting artifacts requires an explicit versioned migration.

Verified: **110 tests GREEN**, including 29 durable integration cases and real
process deaths before/after commit. Trading awaits account adapter, exclusive
ownership, actual anchors, OMS/risk and release evidence.
