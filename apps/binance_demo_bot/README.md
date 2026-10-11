# Independent Binance Demo bot — P0/P1 and P2-A/B2 read-only foundations

This separate application implements offline Decimal accounting, valuation,
daily equity PnL, FIFO/fees, SQLite journal/outbox/recovery and initial isolation.
P1-A adds an explicitly invoked read-only Demo REST adapter. The offline
entrypoint starts no account client. No order sending or learning workers exist.
P1-B implements private event collection, durable fill/cursor recovery and
diagnostic comparison with the journal. Authenticated runtime verification,
certified account readiness and the remaining P2-C–P7 services are pending.
P2-A implements all-category public discovery, immutable received-as-of snapshots,
REST quotes/closed minute features and whole-inventory midpoint valuation.
P2-B1 adds owned combined-stream readers, deterministic shards, reconnect/quota
supervision and bounded incremental L2. These are explicitly invoked libraries.
P2-B2 adds metadata refresh, all-pair closed-candle topics, bounded bootstrap,
atomic store batches and a separate read-only coordinator. Full-market stream
latency verification is FAIL; this runtime is diagnostic, not production-ready.

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

Verified: **461 tests GREEN**, including 56 REST, 69 private/evidence cases,
57 public market/universe/valuation cases,
46 stream/L2/supervision cases,
36 runtime/refresh/closed-stream/storage cases,
26 WAL concurrency/prepared-write/lifecycle cases,
22 pinned SQLite/checkpoint/pressure/failure cases,
12 writer attribution/bulk/dependency/receipt/warmup cases,
17 durable delivery/evidence/failure cases,
10 preparation read-prefix/handoff/lease cases,
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
173.67s; continuously bounded full-universe processing/lag and certified
account reconciliation remain P2-B2/P2-C. No active trading service was started.

P2-B1 stream sequence:

1. Use a fresh immutable own `UniverseSnapshot` and dedicated credential-free
   `DemoMarketClient`. `stream_plan(universe, deep_symbols=[...])` assigns every
   active permitted or public-reference Spot pair, including non-USDT/UTF-8,
   to <=512-topic shards. Keep explicit diagnostics/sample coverage separate
   from account-permitted coverage; a public source has N=null.
2. `MarketStreamSupervisor(store, client, universe, deep_symbols=[...])` warms
   the clock and starts only owned readers. Default coordinator is manual:
   call `start()`, regularly `tick()`, then `stop()`. `background=True` explicitly
   opts into its own coordinator. No CLI/service auto-start is added.
3. Let the coordinator exclusively own REST/clock probes; do not concurrently
   use the same HTTP client elsewhere. Fresh diff buffers precede REST depth
   snapshots; 1..1000 levels, weighted reserve and bounded retries are enforced.
   A standalone `MarketStreamReader` instead needs an already validated clock
   and explicit `install_depth(symbol, client.read_depth(symbol))` calls.
4. Inspect `health()`: own subscribed n/N, permitted N, shard failures, clock,
   receipt/processing age and depth states. READY requires diff continuity and
   <=1s event/receipt age including clock uncertainty. Snapshot alone or gap
   cannot qualify. BookTicker remains OBSERVED_RESPONSE/null event clock.
5. Stop the supervisor/readers before closing store/client. Stop/reconnect
   invalidates old quote rows; a new session starts without old readiness.
   Disable supervisor with `enabled=False`; no connection is attempted.

Public 20s sample: BTC/ETH/SOL reached READY, final repeated sample 3/3 fresh
quotes and 3/3 READY books. The first sample ended 2/3 READY; this is preserved
as a partial point, not rewritten as sustained availability. Neither sample
qualifies all-market/account readiness or profit. Metadata plan expires after
15min; P2-B2 adds refresh/scheduling, but full-scale lag qualification is FAIL.
P2-C retains financial cuts and account/reference-price filters. Incremental books are bounded observed
levels, never a complete exchange-depth claim. Order sending stays disabled.

P2-B2 diagnostic sequence:

```python
from binance_demo_bot.market_runtime import MarketDataRuntime

runtime = MarketDataRuntime(store, client, deep_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
try:
    runtime.start()  # full public universe; no credential/account fallback
    runtime.tick()  # repeat while observing; the coordinator exclusively owns REST
    diagnostic = runtime.health()
finally:
    runtime.stop()  # stop these readers before closing store/client
```

Metadata refreshes every 10min; all active pairs receive book/kline topics,
clock samples swap atomically, quote fallback is REST OBSERVED_RESPONSE and
candle bootstrap rotates <=2 symbols/tick. Missing history stays unavailable.
Lossless `universe_source(snapshot)` reconstructs the exact original API source;
compact rows and FIFO atomic batches preserve primary hashes/receipt prefixes.
Past cutoff reconstruction excludes future receipts and later failed inputs
cannot resurrect old readiness. Public callers still validate the full envelope.

Full public trials discovered 3727 pairs, with 1375 active Spot pairs over 6
shards. Source reconstruction hashes matched. Several REST points reached
1375/1375, but the initial five stream trials ended 0/6 connected, with
WS_PROCESSING_LAG; failure evidence is retained under own runtime. Do not
interpret those point snapshots as continuous coverage. `STREAMS_DEGRADED`
and per-symbol candle failures close readiness. Sustained throughput/lag must
be fixed before production qualification; own account keys/permission cut,
balances/anchors and OMS are separate pending gates. No orders are enabled.

P2-B2 contention repair adds per-thread read-only WAL snapshots, indexed candle
lookup, <=8-frame/2ms groups and preparation outside the writer. A short atomic
insert follows validation; concurrent candle revision aborts with
`CANDLE_SOURCE_CHANGED`. Reader stop releases its owned connection; shared cache
updates are serialized independently of source reads. Normal snapshot IDs,
canonical money, source hashes and original receipts are preserved; FULL stays.
26 new tests bring the app to 400 GREEN. Full-public latency is still FAIL:
latest observation 46.23s, a point reached 4/6 connected, final 0/6; quotes
1375/1375 and history 14/1375. SQL profiling isolated observed commit pauses up
to 400.63ms. Resolve storage/checkpoint latency before qualification. Concurrent
checkpoint work first requires a fixed SQLite runtime (current 3.50.4), as
documented in the canonical spec. The following checkpoint slice supersedes
that storage implementation; the failed trial remains historical evidence.

P2B2-07 uses pinned APSW 3.53.4.0 / SQLite 3.53.4 for market storage only.
Financial journal keeps its existing engine and accounting fingerprint.
`runtime.start()` explicitly starts an owned PASSIVE checkpoint worker;
store construction, imports and offline startup create no maintenance threads.
Use `checkpoint_background=False` for the baseline autocheckpoint policy.
Worker checks physical WAL size every second, attempts PASSIVE above 4MiB and
leaves WAL/FULL durability unchanged. Its connection cannot mutate evidence
or storage policy. `health()['storage_maintenance']` reports version, actual
bytes, completed/partial frames, busy/failure counts and recent latency.
At 256MiB physical WAL, writes fail with MARKET_STORAGE_BACKPRESSURE; stat
failure gives MARKET_STORAGE_UNAVAILABLE. These gates cover prepared commits
and recheck under the writer lock. Runtime stop/store close join only this
worker and restore the writer's previous autocheckpoint policy.
Tests establish storage contracts, not continuous market availability.
P2B2-07 full-public observation 48.25s: 3727 discovered / 1375 active, 2/6
connected, quotes 151/1375, closed history 1/1375 — FAIL. Checkpoint recorded
61 attempts, 3 partial and zero errors; four shards still failed processing lag.
All owned smoke workers stopped. Remaining gate: SQL writer/group contention
and receiver latency, then authenticated financial cuts. See the canonical
P2 spec for all failed trials; no profitable/production deployment is claimed.

P2B2-08 writes prepared groups with one parameterized multi-value INSERT per
<=64 rows, keeping all chunks in the same WAL/FULL transaction. A normal eight-
frame group uses one INSERT. Failed later chunks roll back the whole group;
immutable retries retain the first primary receipt envelope. Standalone append
copies/deserializes its returned source after releasing the writer. Prepared
groups cache only successful immutable dependency availability in that group;
staged/missing/future dependencies keep their original causal guards.
`Connection.cache_stats()` exposes connection-local SQLite cache diagnostics
without SQL or reset. It is for an owner-path profiler; runtime health never
reads the active writer's status. Counter profiling adds instrumentation cost
and does not itself qualify an SLA; keep minimally instrumented trials separate.
An older pinned caller read keeps its prefix while standalone append returns
the committed envelope through a temporary readonly result connection. Exact
retries retain the first envelope. Candle bootstrap requests four latest bars
to obtain three closed bars plus the current open bar; only closed bars qualify.
Final P2B2-08 observation49.55s: source SHA matched,3727 discovered/1375 active,
1/6 connected,quotes50/1375,history6/1375 — FAIL. Remaining stalls include
BEGIN346.22ms and writer hold426.10ms. No controlled speedup or SLA is claimed.
Next gate: receive/validation/writer-wait/commit latency and bounded handoffs.

P2B2-09 measures steady-state delivery from the oldest actual group receipt
through durable completion, with collection/reader wait/preparation/writer wait/
BEGIN/INSERT/commit stages. `reader.health()['latency']` reports explicit scope,
attempted/durable/passed/failed/late counts and a bounded recent window (default
4096; diagnostic-only `latency_window=1..4096`). Startup buffer latency is
explicitly unmeasured. Idle calls create no zero samples; unavailable stages
are null. A post-commit duration above250ms invalidates/closes the session and
returns False, preserving primary rows and exact original receipts/hashes.
Lag incidents retain all in-flight frames, canonical decoded JSON and exact
wire bytes/hash when observed. Evidence/invalidation storage failures are
visible, and transport is closed even on storage failure. Failure evidence,
quote invalidation and incidents are prepared outside the writer and committed
in one atomic FULL transaction. A failed row rolls back the whole failure group;
memory still invalidates and at most one volatile evidence group remains.
Health exposes no payloads. These diagnostics and stricter guards do not qualify production;
future policy must use current session health and causal source evidence.
Final P2B2-09 public trial1791737943969095300:52.9277s after35.260s startup,
0/6 connected,quotes0/1375,closed history8/1375 — FAIL. All six last sessions
retained lag evidence/invalidation with zero evidence-storage failures. Their
last-group completion was307.5481..460.2281ms. Timing stages include scheduling;
no controlled performance uplift or sustained-readiness claim is established.

P2B2-10 keeps one readonly WAL cut during preparation, releasing its lease
before writer handoff. Eight new rows sharing a persisted dependency now use
one read BEGIN/COMMIT pair instead of nine. Exact retries, original receipts/
hashes, causal dependency availability and atomic primary writes are retained.
The cut pins on its first SELECT; foreign writes remain possible, and a foreign
candle revision is still rejected by the writer's current-sequence guard.
Caller read cuts and overlapping prepared contexts survive non-LIFO release;
the last active lease closes the shared read transaction. A failed lease marks
it for rollback on last release. No queue/engine/financial-code changes or
FULL/250ms relaxation. Ten new tests followed intended RED; app461 GREEN.
This verifies resource/correctness contracts, not full-market latency or PnL.
P2B2-10 public trial1791740709548236200:47.9992632s after41.861s startup,
1/6 connected,quotes21/1375,closed history10/1375 — FAIL. Five current failed
sessions retained evidence/invalidation; zero evidence-storage failures.
