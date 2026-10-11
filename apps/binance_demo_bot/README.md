# Independent Binance Demo bot — P0/P1 and P2-A read-only foundations

This separate application implements offline Decimal accounting, valuation,
daily equity PnL, FIFO/fees, SQLite journal/outbox/recovery and initial isolation.
P1-A adds an explicitly invoked read-only Demo REST adapter. The offline
entrypoint starts no account client. No order sending or learning workers exist.
P1-B implements private event collection, durable fill/cursor recovery and
diagnostic comparison with the journal. Authenticated runtime verification,
certified account readiness and the remaining P2-B–P7 services are pending.
P2-A implements all-category public discovery, immutable received-as-of snapshots,
REST quotes/closed minute features and whole-inventory midpoint valuation.

Specifications: [financial contract](../../docs/specs/binance-demo-phase0-financial-contract.md),
[application isolation](../../docs/specs/binance-demo-application-isolation.md),
[account adapter](../../docs/specs/binance-demo-phase1-account-adapter.md#p1-a-first-read-only-implementation-contract-2026-10-11),
[universe/market data](../../docs/specs/binance-demo-phase2-universe-market-data.md#p2-a-discovery-rest-observations-and-portfolio-marks-2026-10-11).
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

Verified: **292 tests GREEN**, including 56 REST, 69 private/evidence cases,
57 public market/universe/valuation cases,
29 original durable integration cases and real
process deaths before/after commit. Trading awaits account adapter, exclusive
ownership, actual anchors, OMS/risk and release evidence.

`BinanceDemoAdapter` exposes `ping`, `sync_clock`, `read_account`,
`read_open_orders`, `read_open_order_lists`, `read_order` and explicit-cursor `recover_fills`. Construct
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

P1-B library sequence:

1. Explicitly load the new app's credentials and stable identity key; construct
   the adapter and read its sanitized account token. Open `AccountEvidenceStore`
   with that account/episode/scope token under the own `.runtime`.
2. Construct `PrivateEventReader(adapter, store, process_instance_id=...)` and
   call `start()`. It verifies account identity, subscribes once and starts only
   its own reader thread. Production/Testnet targets, order mutations, unknown
   subscriptions/events, payload logging and blind subscription retries are blocked.
3. Use `AccountReconciler(store).recover(adapter, explicit_symbols)` to persist
   every explicit symbol's REST evidence and overlapping cursor atomically.
   Discover the complete permitted population through P2; held symbols alone
   cannot establish this population. A stream fill never moves the REST cursor.
4. Take fresh account/orders/lists snapshots and call `compare(journal, marks,
   ..., journal_anchor_ms=...)`. Supply the actual opening-anchor provenance;
   current quantities cannot prove historical seed basis. Check reasons and
   `comparison_stale(id, journal=journal, marks=marks)` before using diagnostics.
5. Close reader, store and transport explicitly. Reader stop leaves a durable
   gap incident and joins only its own thread. No exchange orders are canceled.

The evidence store preserves one economic fill across REST/WS plus separate
receipt clocks, partial balance facts, unclassified flows/lists and unresolved
incidents. Reopen verifies hashes/namespace and reconstructs cursors; conflicting
economics or tampering fails closed. It never silently rewrites financial lots.
Diagnostic READ_CHECK_MATCH still has `exchange_reconciled=false`; actual as-of
cuts, ownership, P2 valuation and resolution evidence are required for readiness.
The public Demo WSS handshake/time/Pong smoke passed on 2026-10-11 without keys.
No authenticated subscription was attempted; startup remains offline.

P2-A library sequence (manual read-only calls, no scheduler):

1. Construct `DemoMarketClient()` without credentials, or explicitly share
   the new app's `BinanceDemoAdapter` to share its clock and weight reserve.
   Open `MarketSnapshotStore(AppPaths(own_absolute_root), account_alias=...,
   episode_id=..., scope_id=...)` under the own runtime; it is separate from
   the private evidence store and financial journal.
2. Call `UniverseService(store, client).refresh(account=None)` for
   PUBLIC_REFERENCE, or pass the new adapter's fresh sanitized account snapshot.
   All 27 registered permission categories are queried; non-USDT pairs remain
   monitored. No account means permitted N=null. Execution is never certified.
3. Use `MarketDataService(store, client).poll_quotes(universe)` for all public
   books, or explicit symbol batches. `collect_candles(symbol, limit=...)`
   stores UTC closed/open 1m bars; `poll_candles` rotates through explicitly
   permitted TRADING pairs within the reserved local budget.
4. After refreshing quotes, freeze `cutoff_ms=client.clock.wall_ms()`; obtain
   `feature(symbol, cutoff_ms)` and `availability(universe, cutoff_ms)`.
   Missing/conflicting/stale bars close readiness. Past reconstruction is
   REPLAY with its actual computation/publication receipt, never backdated.
5. `QuoteConversionGraph(store).value(balances, cutoff_ms)` values every free
   and locked asset using fresh direct/inverse/multileg midpoint routes.
   The caller must supply/classify balances; these marks cannot certify an
   account anchor. USDC/fees/dust need prices; one absent leg means null equity.
   bookTicker event time stays null: OBSERVED_RESPONSE qualifies observation
   age only, EXCHANGE_EVENT consumers cannot use it. Close store and client.

Public smoke verified 3727 metadata rows, 2010 positive books, BTC/ETH/SOL closed
bars/features and a hypothetical midpoint valuation. Whole manual run took
173.67s; continuously bounded processing/lag, market streams, L2 and certified
account reconciliation remain P2-B. No active trading service was started.
