# Independent Binance Demo bot — P0-A/B and P1-A REST foundation

This separate application implements offline Decimal accounting, valuation,
daily equity PnL, FIFO/fees, SQLite journal/outbox/recovery and initial isolation.
P1-A adds an explicitly invoked read-only Demo REST adapter. The offline
entrypoint starts no account client. No order sending or learning workers exist.
Private events, account reconciliation and the remaining P1–P7 services are pending.

Specifications: [financial contract](../../docs/specs/binance-demo-phase0-financial-contract.md),
[application isolation](../../docs/specs/binance-demo-application-isolation.md),
[account adapter](../../docs/specs/binance-demo-phase1-account-adapter.md#p1-a-first-read-only-implementation-contract-2026-10-11).
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

For authenticated adapter calls, provision a dedicated demo pair in
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

Verified: **166 tests GREEN**, including 56 REST contract tests, 29 durable integration cases and real
process deaths before/after commit. Trading awaits account adapter, exclusive
ownership, actual anchors, OMS/risk and release evidence.

`BinanceDemoAdapter` exposes `ping`, `sync_clock`, `read_account`,
`read_open_orders`, `read_order` and explicit-cursor `recover_fills`. Construct
it with `load_demo_credentials(AppPaths(own_absolute_root))`, opaque local
account/generation/episode/scope IDs and a separately persisted app-owned random
identity key (>=32 bytes). Keep this identity key stable across credential
rotations to compare sanitized account tokens. Do not log credentials, requests
or raw private responses. No existing bot configuration is imported.

Own `DemoHttpTransport` reuses verified HTTPS; call `transport.close()` when
finished. It rejects redirects, production/Testnet targets and all mutations.
Read deadlines/retries, clock quality and rolling/shared-IP budget fail closed.
Use `enabled=False` to disable reads. The adapter is single-threaded; concurrent
services and persisted budget coordination are future work.

Account and fill observations remain `ACCOUNT_UNRECONCILED`; orders/fills stay
UNATTRIBUTED. Symbol tail exhaustion is a cursor observation, not global history
coverage. P1-A neither writes fills to the journal nor certifies midnight
anchors. Real public Demo ping/time were verified on 2026-10-11 (HTTP 200, warm
RTT 289–296ms, valid clock samples). Authenticated read and private stream smoke
are pending; ORDER_SEND is always disabled and key TRADE remains UNVERIFIED.
