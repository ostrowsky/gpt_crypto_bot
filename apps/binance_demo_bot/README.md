# Independent Binance Demo bot — financial/data foundations and local OMS intents

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

P3-A implements the [durable intent journal](../../docs/specs/binance-demo-p3a-durable-intents.md):
local per-arm reservations and immutable pending outbox identity, canonical
Decimal amounts, idempotency/tombstones, epoch/CAS, persistent entry kill and
uncertain-attempt recovery. Its dedicated `oms-intents.sqlite` is audit-backed
WAL/FULL. Supplied allocations are uncertified seed budgets; pending BUYs cannot
fund SELLs. Neither a reservation nor an attempt annotation authorizes I/O.
No sender, exchange TP/SL or terminal fill/reconciliation integration exists yet.
39 focused tests pass; actual trading remains disabled and no profit is claimed.

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

Verified: **746 tests GREEN**, including 39 local OMS intent/reservation/recovery
cases, 56 REST, 69 private/evidence cases,
57 public market/universe/valuation cases,
46 stream/L2/supervision cases,
36 runtime/refresh/closed-stream/storage cases,
26 WAL concurrency/prepared-write/lifecycle cases,
22 pinned SQLite/checkpoint/pressure/failure cases,
12 writer attribution/bulk/dependency/receipt/warmup cases,
17 durable delivery/evidence/failure cases,
10 preparation read-prefix/handoff/lease cases,
10 owned TLS reuse/verification/failure cases,
24 common writer/durable ACK/backpressure/uncertainty/lifecycle cases,
27 REST/decision audit/redaction/crash/coverage cases,
25 WSS audit/receipt/linkage/failure cases,
28 operational-decision/frozen-prefix/cleanup cases,
16 parsed-control/native-socket/failure cases,
38 archive/export/integrity/crash/retention cases,
36 sealed-rollover/ancestry/limits/restart cases,
26 full sealed-backup/WAL/schema/restore cases,
16 retention-ledger/history/idempotency/coverage cases,
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

P2B2-11 prepares one verified client TLS context before default supervisor
stream startup, reuses it across its shards/reconnects/stop-start, and keeps a
separate context per supervisor. Standalone transports retain their own context
after the first connect. Custom factories and disabled startup load no default
trust store. Optional `ssl_context` is checked for client protocol, CERT_REQUIRED,
hostname verification and minimum TLS1.2 at construction and each handshake.
Creation failure stops the owner before connection attempts; health contains
no TLS material/dependency exception text. Demo host/path, proxy/size/queue/
logging guards, actual receipts, FULL and250ms remain.10 new tests after RED,
471 app GREEN; mocked TLS checks are distinct from the public network smoke.
Public trial1791741563868852000:1 REST plus1 market TLS build, startup6/6
connected, first post-tick0/6; final48.0524598s,0/6 connected,REST quotes1375/1375,
history14/1375 — FAIL. TLS reuse works; sustained stream latency remains open.

P2B2-12 adds a store-owned common writer for prepared groups and standalone
appends (explicit owner transactions remain direct under the same FIFO lock).
The runtime starts it before metadata/streams and drains it after producers.
Queue+active work is capped at8 groups/64MiB, each request at256 rows. Collect
nominally <=2ms after writer acquisition, <=64 rows/4MiB per common transaction;
a larger head runs alone, SQL chunks <=64 rows. Guard-free groups bulk insert
and share FULL commit. Guarded groups isolate candle conflicts with savepoints;
confirmed pre-commit constraint rollback allows one bounded isolation fallback.
ACK follows commit. Commit uncertainty stops the service, returns no success
and never silently retries. Already ACKed fallback commits retain their outcome.
Health counters/limits are detached and payload-free. Overflow/failure keeps
the session invalid and its original raw evidence; uncertain commit is null
with candidate IDs. Runtime `group_writer_enabled=False` rolls back on a fresh
store; stopped/failed services are never bypassed. FULL/250ms stay unchanged.
24 new tests followed RED,495 app GREEN; full-market readiness remains separate.

Full [Binance interaction/decision audit](../../docs/specs/binance-demo-interaction-decision-audit.md)
requires all REST/WSS attempts/errors and decisions, correlation, redaction,
completeness and decision->fill->daily net PnL analysis. AUD-A now implements
explicit opt-in durable REST attempts and a causal decision-recording API.
AUD-B1 adds automatic WSS application capture for readers sharing that journal.
AUD-B2 covers coordinator/supervisor branch decisions and recorded-prefix gaps.
Live maintenance/rebinding, certified reclamation and future OMS/policy decisions remain required before
autonomous execution/learning; snapshots/incidents do not constitute full audit.
P2B2-12 public trial1791744230720272600:47.555206s, startup6/6 then0/6;
final0/6 connected,REST quotes1375/1375,history18/1375 — FAIL. Writer accepted/
committed1914 groups in1166 commit ACKs,zero rejected/failed/unconfirmed/pending.
Coalescing works; last failed groups still took269.6732..453.4159ms.

AUD-A uses a separate own `audit.sqlite`, append-only hash chain, WAL/FULL and
writer lease. Create `AuditJournal` explicitly, pass it as `audit=` to
`DemoReadClient` or a same-namespace `BinanceDemoAdapter`, then close the client
before closing its shared journal. A public wrapper can use
`DemoMarketClient(client=DemoReadClient(audit=audit))`. No `.env` is read by this
public path. The offline entrypoint remains offline. Example from the app root:

```python
from pathlib import Path
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.adapter import DemoReadClient

audit = AuditJournal(AppPaths(Path.cwd().resolve()), account_alias="demo",
                     episode_id="research-1", scope_id="whole-account")
client = DemoReadClient(audit=audit)
try:
    data, provenance = client.request_public("/api/v3/time", {}, weight=1)
    print(client.audit_health())
    print(audit.manifest())
finally:
    client.close()
    audit.close()
```

Each attempt writes intent before transport, then response/error separately;
BEGIN is not proof of send, END is not a fill. Responses are captured before
status/JSON validation, including retry/timeout/429/5xx/cancellation. Private
payloads are sanitized, public bytes exact only when safe. Secrets, signed
queries, request auth headers and dependency exception messages are not stored.
`record_decision` accepts explicit policy/model/risk versions, reasons,
candidate population and sources available by the cutoff, including WAIT and
NO_ACTION; it does not invent policy decisions or place orders. `read` paginates
at most1000 records; `manifest` freezes a verified chain tip and count. Admission
is synchronous, capped at8 submissions,48MiB/event and512MiB DB+WAL by default;
budget exhaustion stops auditing rather than deleting evidence. FAILED/unknown
commit blocks further audited reads; never silently fall back to unaudited mode.
Health keeps process expected/ACKed/gap counts and persistent crash unknowns
separate. `full_audit_complete` is always false at this stage. A client with no
explicit journal exposes UNCONFIGURED and cannot qualify autonomous operation.
No sustained-latency, account certification or PnL result follows from these tests.

AUD-B1: a journal on the wrapped `DemoReadClient` or private adapter is also
bound to the readers' default/injected WebSocket transport, after checking its
store namespace. CONNECT/SEND/RECV/PING/CLOSE each get correlated durable
BEGIN/END and independent physical-call counters. Open candles, malformed
messages, subscribe ACKs, timeouts and negative Pong results are retained.
Public message bytes are exact when safe; private messages and signed subscribe
requests are sanitized, including a later text echo of the generated signature.
Receipt clocks are captured before audit writes and reused by normalization and
the unchanged250ms guard. A slow audit write can therefore invalidate a session.
Processing records link raw receipt IDs to confirmed snapshot IDs or private
evidence record sequence/namespace. Raw and normalized databases have separate
commits: missing processing markers remain unknown; no cross-database atomicity
is claimed. Journal failure closes and invalidates the reader, including after
normalized commit. Reader health exposes audit state/counters without payloads.
AUD-B3 adds per-connection parsed PING/PONG/CLOSE hooks on the pinned default
transports before native receive threads start. Automatic Pong and close echo
have durable INTENT/QUEUED and socket WRITE_ATTEMPT/WRITE_RETURNED/WRITE_UNKNOWN
records. Local socket return is never a Binance ACK. Pending control intents
are bounded at64; audit failure shuts down the raw socket without re-entering
the native protocol mutex. Private payloads remain sanitized/omitted.
Only actual hook installation sets `protocol_controls_observed=True`; injected
transports without hooks and unused mocked factories remain false. This is
parsed-control observation, not serialized TCP/TLS capture: frames rejected
before the parser hook remain unobserved and capture coverage stays UNKNOWN.
Full audit, future policy/OMS integrations and maintenance/reclamation gates remain required.

AUD-B2 logs decisions of the current read-only market coordinator and stream
supervisor before network/scheduling effects: metadata/clock/quote refresh,
not-due waits, history-ready/cooldown/backfill and work limits, reconnect/backoff/
rotation/quota, connected sessions and L2 ready/buffer/snapshot decisions.
Records include actual store-resolved source availability, considered symbols,
explicit unknown initial population, cutoff/component/versions and independent
decision expected/ACKed/gap counts. These are OPERATIONAL decisions, not BUY/SELL
signals. Journal failure marks AUDIT_FAILED even before another tick; abort
preserves its first reason and closes owned resources without joining a
coordinator blocked on the same lock. External stop then joins normally.

`audit.processing_report()` or `audit.processing_report(through_seq=N)` scans a
verified frozen prefix with bounded Python memory. It partitions durable RECV
records into processed/rejected/control-ACK/unknown/conflict counts, checks bad
references and returns a prefix chain tip and report hash. Zero messages stay
UNKNOWN; late markers cannot change an older frozen report. The denominator is
explicitly DURABLE_COMPLETED_RECV. Capture coverage remains UNKNOWN: successful
saved rows cannot reconstruct all physical traffic. Compare transport and audit
submission counters separately. No report sets `full_audit_complete=True`.
Default journal disk cap remains finite; certified rollover/reclamation is required
before sustained deployment. The previous full-market latency FAIL is unchanged.

AUD-C1 supplies an opt-in archive/export API in `audit_archive.py`. It pins an
independent read-only WAL snapshot, releases the journal lock before streaming,
and writes original seq1..cutoff envelopes to deterministic gzip JSONL. A hashed
manifest freezes identity, IDs/cutoff, source state/fingerprints, chain tip,
kind counts, time range and compressed/uncompressed sizes/checksums. This is a
DURABLE_EXPORTED_PREFIX denominator; capture coverage remains UNKNOWN.

Use it with the existing own `AuditJournal` object (no additional source writer):

```python
from binance_demo_bot.audit_archive import AuditArchive

archives = AuditArchive(paths)
result = archives.export(audit)  # Or through_seq=an_explicit_cutoff.
archive_id = result["archive_id"]
manifest_pin = result["manifest"]["manifest_sha256"]
verified = archives.verify(archive_id, expected_identity=audit.identity,
                           expected_manifest_sha256=manifest_pin)
print(archives.inventory(protected_archive_ids=(archive_id,)))
```

Retain the manifest pin independently if authenticity matters. A checksum stored
beside its own data proves internal consistency only; without an external pin,
the verifier returns `authenticity=UNVERIFIED`. Export is manual and off the
receive hot path; a pinned WAL reader can retain WAL pages. This does not qualify
market latency or change the250ms guard.

Archives live only in own `.runtime/audit-archives`. Both files are flushed and
fsynced, verified in `.pending-<id>`, then published by directory rename and
verified again. There is no Windows directory-fsync/power-loss claim. A process
death before rename leaves INCOMPLETE evidence; after rename the published
copy can be verified on restart. An exception after rename is explicitly
`AUDIT_ARCHIVE_PUBLICATION_UNKNOWN`, never a false success. Existing IDs cannot
be overwritten. Error codes omit exception text and paths.

Defaults:64MiB compressed,512MiB uncompressed per export,512MiB archive storage
including incomplete copies,64 directories. Exceeding limits refuses new work;
the source journal and existing archives are preserved. Verification bounds
inflation/line/manifest size and checks CRC, schema, identity and the entire
event hash chain. Inventory keeps valid/invalid/incomplete counts separate,
reports protected missing IDs as UNKNOWN, and never invents complete byte
accounting when unexpected directory entries are present.

Retention is explicitly `PRESERVE_ALL_V1`, with zero deletion-eligible archives
and no delete API. Active exposure, unfinalized days and current research remain
protected because this slice deletes nothing. Rollover/restart mechanics are
implemented in AUD-C2a below; full-tail restoration/reclamation remains C2b.
Sustained storage readiness and future OMS/policy audit remain open.
Archive correctness is not a trading/PnL claim.

AUD-C2a adds explicit maintenance rollover in `audit_rollover.py`. Stop/drain
readers before calling it and rebind them to the returned journal afterward:

```python
from binance_demo_bot.audit_rollover import AuditRollover

identity = audit.identity
rollover = AuditRollover(paths, account_alias=identity["account_alias"],
                        episode_id=identity["account_episode_id"],
                        scope_id=identity["portfolio_scope_id"])
audit = rollover.rotate(audit, quiesced=True)
# At a later restart, construct rollover with the same explicit identity:
# audit = rollover.recover(database="audit.sqlite")
```

Only the default `audit.sqlite` root and managed `audit-segments/<id>.sqlite`
paths participate. Recovery never creates a missing root. The source must be
READY with zero process ACK gaps, outstanding REST/WSS calls, unresolved prior
runs or unresolved/conflicting processing markers. `quiesced=True` declares
that the caller drained producers; this API does not stop an existing service.
There is no automatic switch of live transport references or trading enablement.

Rollover pins/verifies the current archive, fsyncs an exclusive empty child
reservation, then stores terminal RUN_END and `meta.sealed` together in one
FULL transaction. The old handle becomes SEALED and cannot accept new records.
The successor starts local sequence1 with frozen parent/archive anchors in both
metadata and its first BEGIN; no global sequence is invented. The parent's
immutable activation receipt pins that first record before the handle returns.
Ancestral max_bytes/max_inflight ceilings survive rotation, recovery and normal
child reopen; tighter settings may be used. Recovery depth defaults to64 and
rotation checks its budget before sealing.

Recovery verifies original rows, archive/manifest pins, parent seals and child
activation receipts. A reserved empty file may be initialized only without an
activation receipt. Missing children, emptied activated files, removed seal/
parent metadata and inconsistent hashes prevent startup. Process-crash BEGINs
and incomplete-run evidence are retained; recovery does not reset their metrics.
An uncertain seal commit or maintenance exception returns an explicit fixed
error; no unverified handle is exposed. Recover from anchors rather than retrying
an old sealed handle. Windows power-loss durability remains unclaimed.

Original segment evidence, reservations and archives remain preserved; normal
SQLite checkpoint/close may merge and retire WAL files. The
C1 export precedes the terminal seal: final seal/activation evidence still lives
in the original DB, so that DB cannot yet be reclaimed from the C1 copy alone.
C2b1 below supplies full-tail backup/restoration and hold history; authoritative
exposure/day/research certification remains required before deletion.
Live maintenance/rebinding, cross-segment analytics and
aggregate storage qualification remain gates; current reports stay per segment.

AUD-C2b1 adds `SealedAuditBackup` for a complete sealed segment. Its SQLite
online backup includes committed WAL pages, terminal RUN_END, immutable parent/
seal metadata, activation receipt, the linked C1 gzip archive and a successor
first-record witness. Verification uses an immutable read-only snapshot and
the app's fixed SQL schema; it does not add WAL/SHM files to the bundle.

```python
from binance_demo_bot.audit_backup import SealedAuditBackup

copies = SealedAuditBackup(paths, account_alias=identity["account_alias"],
                          episode_id=identity["account_episode_id"],
                          scope_id=identity["portfolio_scope_id"])
copy = copies.create(database="audit.sqlite")  # Source must be sealed/activated.
backup_id = copy["backup_id"]
manifest_pin = copy["manifest"]["manifest_sha256"]
# Store this pin independently. After loss of the original SQLite file set:
# receipt = copies.restore(backup_id, expected_manifest_sha256=manifest_pin,
#                          quiesced=True)
```

Bundles under `.runtime/audit-sealed-backups` are published only after fsync and
verification. Defaults:512MiB per SQLite copy,1GiB aggregate,16 bundles including
incomplete attempts. Logical pages, prefix bytes and manifest reserve are
checked before staging. Restore staging has separate count/byte bounds under
the same ceilings. Original files and failed staging evidence are preserved.

Restore requires a matching manifest pin, explicit maintenance quiescence and
an absent target DB/WAL/SHM. It reuses an existing prefix only if its pin matches,
checks a present successor against the activation witness, and never overwrites
or deletes a target. A partial prefix publication is explicitly UNKNOWN and may
be reused on retry. Restoring copies files and verifies evidence; it does not
append a new BEGIN or start a trading runtime. C2a recovery can then open an
existing valid active head. This is one sealed-segment copy, not a whole-chain,
active-head or hold-ledger disaster backup; Windows power-loss is not certified.

`RetentionLedger` stores append-only holds/releases in separate own
`audit-retention.sqlite` with FULL commits, identity, hash chain and writer
ownership. Holds bind to a seal digest across all copies. Categories are
ACTIVE_EXPOSURE, UNFINALIZED_DAY, RESEARCH and INTEGRITY. Opaque references and
request IDs are32 hex; evidence digests are64 hex. For example:

```python
from binance_demo_bot.audit_retention import RetentionLedger

holds = RetentionLedger(paths, account_alias=identity["account_alias"],
                        episode_id=identity["account_episode_id"],
                        scope_id=identity["portfolio_scope_id"])
try:
    # hold_id = holds.place(seal_sha256, category="RESEARCH",
    #                       reference_id=research_id, evidence_sha256=evidence_hash,
    #                       request_id=request_id)
    # holds.release(hold_id, evidence_sha256=release_evidence_hash,
    #               request_id=release_request_id)
    print(holds.status(copy["manifest"]["source_metadata"]["sealed"]["seal_sha256"]))
finally:
    holds.close()
```

Equal request+payload retries return the original event; conflicting reuse or
independent duplicate release is denied. Unknown persistence cannot turn into a
successful retry until verified reopen. Status exposes frozen-prefix mutation/
placed/active/released/invalid counts, up to100 active IDs, and separate live
ledger failure state. Malformed releases never clear valid holds. Missing
authoritative account/day/research certification stays UNAVAILABLE; even an
empty ledger or released holds always has `deletion_eligible=False`. There is
no deletion/reclamation API. Certifying those sources and backing up the hold
ledger/full chain remain release gates before any evidence can be reclaimed.
