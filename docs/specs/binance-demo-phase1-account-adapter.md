# Binance Demo phase 1: account adapter and capability evidence

Registered: 2026-10-10, Europe/Budapest. Updated: 2026-10-27.
Status: **P1-A/B + order-test foundations; REST/order-test/signed WSS control checks PASS 2026-10-27; full P1 incomplete**.
Priority: P1, after [phase 0](binance-demo-phase0-financial-contract.md).
Program: [autonomous demo trading](binance-demo-autonomous-trading-program.md).
Application boundary: [independent app isolation](binance-demo-application-isolation.md).
Objective: `daily_net_equity_pnl_v1`; environment: `BINANCE_SPOT_DEMO`.
Explicit library calls can read Demo; the offline entrypoint starts no account
client. Order sending and existing signal integration remain disabled.

Current authenticated verification (2026-10-27): the user-provided standalone
Demo env file was moved into the app's ignored `.env`, without legacy fallback.
Existing pinned Demo adapter passed public ping, clock sync, signed account,
all open orders and open order lists:5/5 read-only checks. Account Spot permission
was reported allowed; this does not validate the key's TRADE endpoint. All5 REST
BEGIN/END pairs persisted, with0 unresolved attempts. Raw UID/user identifiers
are removed before audit append (AUD-03 repair);6 diagnostic JSON files and12
final audit records passed a local credential/identifier absence check. These
runtime artifacts, balances and identity keys are not committed.
At that read-only milestone KEY_TRADE_VALIDATED and PRIVATE_EVENTS were
UNVERIFIED; ORDER_SEND DENIED. Subsequently the opt-in
[order-test validator](binance-demo-order-test-validation.md) passed two distinct
source-version diagnostics, one accepted BTCUSDT BUY LIMIT test per run (2/2).
It sends no matching-engine orders. Each run retained11/11 REST attempts and
terminal responses plus2 decisions;4/4 checks passed with unchanged observed
balances/open orders/lists. KEY_TRADE_VALIDATED was SUPPORTED for the tested
request/generation with30s expiry, not indefinitely renewed by saved reports.
39 new focused tests cover default-off/single-POST/audit/clock/rate/auth/fences.
PRIVATE_EVENTS remains UNVERIFIED and ORDER_SEND DENIED.
At that diagnostic milestone exclusive ownership had not yet been declared.
The user has now explicitly confirmed exclusive use by the new bot; the
[opening manifest](binance-demo-account-opening-manifest.md) registers this
against the HMAC account identity. Actual history/flows, certified as-of
account/valuation anchors, private-event delivery and execution/protection
remain pending.
Earlier dated pending-smoke paragraphs below describe their original milestones.

Signed private transport diagnostic (2026-10-27): existing PrivateEventReader,
native pinned websockets17.1 and protocol audit were exercised with the dedicated
Demo pair. Two separate connections/subscriptions accepted (2/2), distinct
session IDs with per-connection subscriptionId0, four requested Ping/Pong
heartbeats completed (4/4). Each session was manually polled for15s; background
service mode was not qualified. All5 checks passed: both subscriptions/heartbeats,
distinct sessions, returned balances agreement, unchanged open orders and lists.
REST snapshots agreed on866/866 union assets, with no missing rows. This is
diagnostic two-snapshot agreement, not comparison to a financial journal.
No opening lots, basis, ownership, account episode or profit were manufactured.

Zero private economic events and fills were observed; the delivery denominator
is unknown, not0% or100%. Only subscription ACKs were received as application
messages (2/2 durable receipts classified CONTROL_ACK). Thirty receive timeouts
were retained as outcomes of quiet polling, not hidden or treated as lost fills.
Capture coverage remains UNKNOWN. Each deliberate close recorded STREAM_GAP;
both incidents remain after reconnect (2), requiring future gap/backfill/anchor
qualification. A new healthy connection does not resolve old history gaps.
PRIVATE_EVENTS stays UNVERIFIED, exchange_reconciled=false, ORDER_SEND DENIED.

Runtime audit:10/10 REST BEGIN and END,42/42 WS BEGIN and END,2/2 processing
markers and30/30 parsed-control records ACKed by local storage;0 unresolved
REST/WS operations and0 process-submission gaps. These are app-local counters,
not a server wire-capture denominator. Privacy scan covered14 diagnostic JSON
files and138 final audit records, with0 key values/raw identifier fields found.
No runtime/account data was committed. Existing synthetic regressions31 private,
38 account evidence,25 WS audit and16 protocol tests pass (110/110); no code
change was needed and no artificial RED was introduced for this verification.
Full Harness remains FAIL TH-11 (historical alpha replay source hash stale).

Dedicated-account exclusivity was explicitly confirmed by the user on
2026-10-27. P0-C1 adds immutable registration and provisional opening
inventory/basis/valuation. This is a human declaration plus source observations,
not independent certification that no external activity ever occurred.
Certified cut/recovery and financial anchors are still prerequisites.

P0-C2 (2026-10-27): [financial bridge](binance-demo-financial-account-bridge.md)
now binds the original C1 seed to the own JournalStore and imports supported
available fills without assigning OMS ownership. A later signed read matched
2/2 required nonzero assets (866 returned), financial status PENDING. No actual
economic fills were observed; global recovery/delivery coverage remains unknown.
The background reader was healthy at the comparison; closure retained incidents
and invalidated snapshot freshness. This is not completion of P1 reconciliation.

## Problem

The current bot creates virtual positions; public market clients are not an
authenticated broker. The previous user-authorized check reported HTTP 200 for
signed account reading. That historical observation proves neither current key
availability nor `TRADE` authority. No secrets are read to write this spec.

## Objective fit

Daily net equity PnL requires actual demo balances, orders, fills and commissions.
This phase establishes trustworthy evidence and environment boundaries. It
cannot establish positive PnL or turn integration readiness into trading approval.

## Scope and planned components

Implement `BinanceDemoAdapter`, `DemoClock`, capability registry and a private
event reader in the new app `apps/binance_demo_bot/`. Planned implementation:
`src/binance_demo_bot/adapter.py` and `tests/test_adapter.py` relative to that
app root, with its own `.venv`, dependencies/lock, configuration and lifecycle.
Current source may be studied read-only as a reference; no runtime imports of
`files/*`, shared current-app clients, configuration or globals are allowed.
Do not repoint existing endpoints or start/stop old collectors or trading workers.
P1-A implements REST; P1-B implements private observation, durable recovery and
diagnostic journal comparison. No authenticated production service has been
started or verified; certified account readiness and execution remain pending.

Allowed operations: public ping/time/metadata; signed account, order/list,
fill, commission and account-filter reads; current private stream subscriptions.
`POST /api/v3/order/test` is an optional separately enabled validation capability.
The phase-1 adapter rejects every matching-engine order mutation, including
placement, cancellation, amendment and cancel-replace. No withdrawals, transfers,
margin, futures, production/testnet fallback, SOR execution or account reset.
[Phase 3](binance-demo-phase3-order-management-risk.md) introduces a distinct
durable OMS execution capability; passing this phase never grants it implicitly.

## Transport and credentials contract

Exact service allowlist:

| Service | Allowed endpoint |
|---|---|
| REST | `https://demo-api.binance.com/api/v3/` |
| Private WebSocket API | `wss://demo-ws-api.binance.com/ws-api/v3` |
| Market streams, owned by phase 2 | `wss://demo-stream.binance.com/ws` or `/stream`, port 443 or 9443 |

These are the documented Spot Demo services, a separate environment from Spot
Testnet. Demo balance reset is user-controlled; demo outcomes are virtual.
[Binance Demo general information](https://github.com/binance/binance-spot-api-docs/blob/master/demo-mode/general-info.md).

Load only `BINANCE_DEMO_API_KEY` and `BINANCE_DEMO_API_SECRET` from
`apps/binance_demo_bot/.env` or an explicitly injected, app-owned process pair.
No upward `.env` search, `files/.env` read, inherited legacy-global credential
fallback or `PYTHONPATH` dependency is allowed. Missing,
empty, whitespace-corrupted or key-type-incompatible values fail closed with a
redacted reason. Never use legacy production variables as a fallback. The
credential pair is loaded once per explicit credential generation; rotation
invalidates prior capability evidence and starts reconciliation again.

The earlier check and keys saved in the old app do not provision the new app.
Migration is a separate user-opt-in action; this spec neither copies secrets
nor imports old capability/runtime evidence. Use app-owned account aliases,
credential generations, manifests, logs, cursors and locks under its `.runtime/`.
Resolved writable paths must remain inside the new app, including symlink checks.
No current `positions`, models, logs, runtime stores or process controls are read
or modified. Dependency installation targets only the app's `.venv`.

A different key does not create a different account. Register exclusive demo
account/OMS ownership before execution; foreign/manual orders and fills cause
reconciliation/attribution incidents, not silently adopted bot profit. Binance
IP budgets and account-wide filters/order counts are not isolated by app/key.
The new adapter uses its own bounded limiter and aggregate exchange evidence;
it cannot reserve capacity by stopping or reconfiguring current workers.

Validate parsed scheme, exact hostname, port and path before serialization and
at dispatch; reject userinfo, unexpected ports, fragments, alternate hosts and
dynamic arbitrary URLs. TLS validation stays enabled and redirects disabled.
Private HTTP/WS request bodies, API headers, query signatures, keys, secrets and
raw account UID never appear in logs, exception strings, reports or fixtures.
Persistent evidence uses a local opaque account alias and account episode ID.
Sanitize private responses before persistence; unredacted debug dumps are forbidden.

REST signing serializes deterministic parameter pairs to one percent-encoded
UTF-8 query, signs those exact bytes with HMAC-SHA256 and sends the same bytes
with `X-MBX-APIKEY`. Do not re-encode after signing or mix body/query signing.
WS signing is a separate serializer: alphabetical parameter names excluding
`signature`, documented UTF-8 value encoding, exact Decimal/time formatting.
[REST request security](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#request-security),
[WebSocket signing](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-api.md#signed-request-example-hmac).

Use server-time samples plus monotonic RTT to estimate offset and uncertainty.
Initial versioned limits: sample age <=60s, RTT <=1s, clock uncertainty <=250ms,
`recvWindow=5000ms`. A clock jump invalidates samples; widening recvWindow is not
an automatic repair. Values are preregistered pilot limits, not latency guarantees.
Resync once for a signed GET timestamp rejection; do not repeatedly retry bad
signatures or permissions. Tests cover ms/us conversion and non-ASCII symbols.

Initial request budget: connect 5s, overall 15s, at most three attempts for
idempotent reads with bounded backoff. Obey `Retry-After` and the shared rate
budget; ban/auth/schema failures enter blocked states. A timeout/5xx is not
evidence of an empty account or failed exchange mutation. Future mutations
return `QUERY_REQUIRED` and are never automatically resent by transport.
[REST limits and unknown execution](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#http-return-codes).

## Capability contract

Each capability has `SUPPORTED | DENIED | UNVERIFIED | UNAVAILABLE`, source,
checked/expiry clocks, environment, account alias, credential generation and
sanitized response hash. Missing fields/failed probes remain `UNVERIFIED` or
`UNAVAILABLE`; absent is not `false` and does not imply permission.

| Capability | Evidence and limitation |
|---|---|
| `USER_DATA_READ` | Fresh authenticated account/read success; historical HTTP 200 is not renewable proof |
| `ACCOUNT_SPOT_ALLOWED` | Explicit `canTrade` and Spot account permissions, preserving absent fields |
| `KEY_TRADE_VALIDATED` | A valid explicitly enabled `/order/test` request; unsupported endpoint is not key denial |
| `PRIVATE_EVENTS` | Verified subscription plus event/reconciliation health; quiet socket alone is insufficient |
| `SYMBOL_EXECUTION_TYPES` | Current metadata and account restrictions; actual protection tested only in phase 3 |
| `ORDER_SEND` | Always disabled in this phase, regardless of all other probe results |

`canTrade=true` describes account capability, not the key's `TRADE` permission.
The order-test endpoint is `TRADE` secured and validates without matching-engine
submission. An invalid-size/filter test cannot prove authority. Register a
filter-valid symbol/quantity from fresh data, evidence expiry and a bounded
explicit `allow_order_validation=false` default. No loop places test orders for
all symbols. Success proves the tested validation request only.
[Test new order](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#test-new-order-trade).

## Account snapshots, events and reconciliation interface

Use the program `EventEnvelope`: `schema_version=1`, `application_id=binance_demo_bot`, environment,
`account_episode_id`, `portfolio_scope_id`, `event_id`, `event_type`,
`causation_id`, nullable `exchange_event_time_ms`, actual UTC `received_at_ms`,
`received_monotonic_ns`, `process_instance_id`, `recorded_at_ms`; optional `decision_id`, `arm_id`,
`policy_version`, `model_id`, `source_sha256`, `risk_contract_id`.
Adapter snapshots add `snapshot_id`, `objective_contract_id`, source endpoint/method,
request start/end, payload hash, clock quality, completeness and capability IDs.
Absent model/policy identity is null, never a fabricated fitted artifact.

Balances preserve every asset's Decimal `free` and `locked`, including zero
records where returned; use explicit response coverage, not a presumption that
omitted assets are zero. Orders retain exchange/client/list IDs, symbol, status,
quantities and transaction time. Fills retain `(application_id, environment, account_episode_id, symbol, exchange_trade_id)`
identity, price/base/quote quantity, order ID, commission/asset and event time.
Unknown status/schema is quarantined; no conversion into virtual BUY/SELL fills.
An all-zero response or unexpected balance jump is not automatically a reset;
phase 0 classifies flows/resets and bounds every affected result.

For the current HMAC pair use `userDataStream.subscribe.signature`; an Ed25519
session may instead use `userDataStream.subscribe`. Persist `subscriptionId` to
account mapping locally and reject foreign/unmapped events. Legacy listenKey
start/ping/stop is not the new design. Stream subscriptions have their own
signature/event schema checks, not REST signature reuse.
[Current private subscription methods](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-api/user-data-stream).

Normalize `executionReport`, order-list and account/balance changes into the
durable ledger interface. Deduplicate fills by exchange identity and updates by
validated event identity; a duplicate never changes money twice. Preserve
out-of-order events; compute order state from consistent cumulative/fill facts,
not from arrival order alone. Unknown money-affecting events stop account readiness.
[Private event formats](https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md).

Startup/reconnect: begin buffering events, take account/open-order/list anchors,
query fills using overlapping stored per-symbol cursors and reconcile; then
apply buffered deltas idempotently. Paginate to documented limits and checkpoint
only persisted evidence. Never infer all symbols' fills from currently open
positions. Private events do not provide a certified global gap-free sequence;
socket uptime alone cannot prove completeness. Unrecoverable history remains
`ACCOUNT_UNRECONCILED`; missing changes are not erased by a newer snapshot.

Initial entry-readiness limit: full account snapshot <=30s plus reconciled
changes and healthy private stream; stricter OMS requirements may apply.
This is not proof of midnight balances: phase 0 reconstructs boundary state from
certified fills/flows. Reconciliation must compare balances, pending orders,
lists and new fills, report every nonzero discrepancy and preserve incidents.

## Primary metrics and acceptance criteria

Business target remains daily net equity PnL; this phase reports integration
metrics separately: reconciled assets `n/N`, recovered fills `n/N` where the
exchange denominator is known, private-stream lag, capability age, signed-read
success `n/N`, auth failures and query-required counts. Unknown denominator is
null with a reason. Do not report financial improvement or zero discrepancy when
history is incomplete. `ACCOUNT_READ_READY` and `TRADE_VALIDATED` are separate.

Completion requires deterministic offline tests, redaction/host/capability
guards, complete paginated recovery and documented current demo read evidence.
Optional validation evidence is explicitly scoped and cannot grant `ORDER_SEND`.
No existing signal producer bypasses the planned OMS by calling this adapter.
Missing required capability creates a concrete recovery/probe action, not PASS.

## Backtest / verification gate and planned focused tests

No trading-policy hypothesis or new historical financial result is claimed.
Read-only integration may proceed after fixtures and authorized demo read
smoke checks. Financial enablement still needs the maximum available historical
replay and new forward gates in phases 4/6/7; fixtures are not exchange evidence.
Implement and run these scenario IDs before calling the phase complete:

| ID | Planned scenario and expected result |
|---|---|
| ADP-01 | Exact REST/WS signature bytes, Unicode, Decimal and time units; independent expected signature matches |
| ADP-02 | Production/testnet/redirect/userinfo/path injection; zero private requests to rejected targets |
| ADP-03 | Missing credentials, `canTrade` absent/false/true, test endpoint rejection; capabilities remain separate |
| ADP-04 | Valid optional order validation; matching-engine send/cancel stays blocked |
| ADP-05 | Credential rotation, timeout, skew, 429/418 and auth failure; bounded recovery and explicit blocked evidence |
| ADP-06 | Exceptions, WS payload and reports with synthetic credentials/UID; no sensitive value escapes |
| ADP-07 | Paginated fills, duplicate/reordered events and reconnect fills; exact balances/IDs, no duplicate ledger debit |
| ADP-08 | Omitted balances, unknown schema, stream gap/unrecoverable history; reconciliation never invents completeness |
| ADP-09 | Restart with durable cursors, preexisting orders and reset ambiguity; no orders sent, no result silently reset |
| ADP-10 | Old `.env`, globals/PYTHONPATH, models/tickets and runtime path supplied; rejected without reading old stores |
| ADP-11 | Separate key/same account, foreign orders and shared rate exhaustion; ownership uncertain/entry blocked, no old-process stop |

## Risk / trade-offs and rollback switch

Read latency and pagination can delay readiness; this is preferable to supplying
uncertified state to an OMS. Demo response freshness and feature support are
observed, not inferred from HTTP 200. Suspend adapter subscriptions/reads via
`BINANCE_DEMO_ADAPTER_ENABLED=false`; retain evidence/cursors and last state as
stale. In later execution phases this switch also closes entry readiness, but
never cancels exchange protection or deletes positions. Resume only after fresh
reconciliation. TH-01..TH-12 and the complete engineering cycle apply when code
is implemented; planned scenarios are not tests already run.

## P1-A: first read-only implementation contract (2026-10-11)

Implement the REST foundation first. This milestone covers ping/server time,
account balances/permissions, open orders, individual order lookup and per-symbol
fill recovery. Private WS, order lists, commission/filter probes, durable cursor
checkpoints, journal ingestion, complete reconnect reconciliation and optional
order validation remain P1-B/P2 work. No phase-complete or account-reconciled
claim follows from P1-A. Its switch is an explicit constructor `enabled=False`;
disabled adapters perform no transport calls. No environment auto-discovery.

Verified sources on 2026-10-11:
[Demo endpoints](https://developers.binance.com/en/docs/products/spot/demo-mode/general-info),
[REST signing and limits](https://developers.binance.com/en/docs/products/spot/rest-api),
[account/order/fill endpoints](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/account).
`myTrades` permits `symbol+fromId`, not `fromId+endTime`; the maximum page is
1000. A terminal page certifies only exhaustion of the queried symbol/cursor
at that observation. It cannot certify all history, all symbols, socket health,
financial ownership or balances. Demo is distinct from Testnet.

Public APIs in own `adapter.py`:

* `AdapterError(reason)` exposes only a stable reason code, never raw URLs,
  queries, headers, response bodies, UID or underlying exception messages.
* `sign_rest(parameters, secret)` returns the exact sorted percent-encoded
  query plus HMAC-SHA256 signature. Values: strings, finite Decimal, integer,
  boolean; no floats/null/containers, duplicate names or reserved `signature`.
  UTF-8 percent encoding and Decimal fixed-point are explicit. Time inputs are
  already integers in the declared unit; `time_ms(value, unit='ms'|'us')`
  converts explicitly, rejects fractional millisecond truncation and booleans.
* `DemoClock(wall_ms=..., monotonic_ns=...)`, `observe(server_time_ms,
  started_wall_ms, ended_wall_ms, started_monotonic_ns, ended_monotonic_ns)`,
  `timestamp_ms()`: midpoint offset, half-RTT uncertainty, <=60s age, <=1s RTT,
  <=250ms uncertainty/jump. Negative/nonmonotonic clocks fail closed. No sample
  means `CLOCK_UNSYNCED`; rejection invalidates the old sample.
* `HttpResponse(status, body, headers={})`, `DemoHttpTransport(base_url=...)
  and `request(method, path, query, headers, timeout_s)` use HTTPS only, exact
  Demo hostname/443, no redirects or inherited proxy, TLS verification, fixed
  `/api/v3/` read-path allowlist, GET only. Invalid path/host/method is rejected
  before sending keys. Private request/response fields have redacted repr.
  Connect <=5s, remaining operation budget <=15s; response cap 8MiB. Network
  exception content is not propagated. Dispatch guards apply even when called
  directly; a caller cannot bypass them with mutation endpoints.
  REST base accepts exactly the canonical URL or its explicit `:443` form;
  empty userinfo/port, control characters and empty query/fragment markers are
  rejected too. Non-string dispatch paths fail before opening a socket.
  One owned HTTPS connection is reused after a fully consumed response; request
  headers never carry over. `close()` is idempotent and releases it. Transport errors,
  redirects, size/deadline failures discard the connection; the next read may
  open a new verified connection. No cross-host pool or redirect is introduced.
* `BinanceDemoAdapter(credentials, *, account_alias, credential_generation,
  episode_id, scope_id, identity_key, transport=None, clock=None,
  enabled=True, sleep=...)`: all identity strings are nonempty opaque local
  IDs; `identity_key` is explicit app-owned bytes >=32 bytes, hidden in repr.
  HMAC of the exchange UID with this stable local key allows same-account
  recognition across API-key generations without persisting raw UID. A missing
  UID remains identity-unverified. No ownership is granted by this token.
* `sync_clock()`, `ping()`, `read_account()`, `read_open_orders(symbol=None)`,
  `read_order(symbol, order_id)`, `recover_fills(symbol, *, from_id,
  page_size=1000, max_pages=100)`; no arbitrary private-request API.
  Credentials are validated at construction (nonempty ASCII without whitespace,
  no PEM/RSA/Ed25519 material); the caller uses the existing app-only loader.
  `capabilities()` returns fresh scoped evidence and always disables ORDER_SEND,
  KEY_TRADE_VALIDATED and PRIVATE_EVENTS. Account `canTrade` is distinct from
  key authority. A stale/failed authenticated read invalidates prior readiness.

Normalized result dictionaries contain only whitelisted data, actual receipt
and request clocks, monotonic receipt, endpoint, SHA256 of sanitized canonical
data, application/environment/objective/episode/scope/account/generation,
clock uncertainty, `exchange_reconciled=False`, and ACCOUNT_UNRECONCILED.
Account response `updateTime` is preserved as a mutation timestamp; freshness
uses receipt clocks, never a claim that updateTime equals snapshot time.
`omitZeroBalances=false`; preserve returned zero rows, `free` and `locked` as
Decimal. Omitted assets remain outside returned coverage, not invented zeroes.
Absent canTrade/permissions remain null; malformed booleans/money, duplicate
assets, unknown account type/order enum or future exchange time fail closed.
UID, API secrets and arbitrary response fields never enter normalized data/hash.
Orders preserve exchange/client/list IDs, side/type/status, price, original,
executed and cumulative quote quantities, time/updateTime. Fills preserve symbol,
trade/order/list IDs, price/qty/quoteQty, commission/asset, buyer/maker and time;
ownership stays UNATTRIBUTED and never becomes a virtual APPLICATION fill.
Monetary response strings use ASCII digits with an optional fractional part;
whitespace, underscores and exponent notation are schema errors. Every signed
read failure, including order/fill parsing, invalidates previous read capability.
A clock jump also closes capability readiness. A failed explicit clock sync
invalidates the prior sample, even if failure occurs before `observe`.
`sync_clock` may take at most three quality samples on the same transport within
one 15s budget, discarding high-RTT samples. TLS setup can make the first sample
too slow; a warm connection must meet the same original limits. Do not relax
RTT/uncertainty limits, use a failed sample or synthesize a successful clock.

Retries: maximum three attempts per logical GET within total 15s, fresh signature
each attempt, .25/.5s transient backoff. A -1021 timestamp error triggers at most
one server-time resync within that budget. Auth/signature errors block the adapter
until a new credential-generation instance; 418 blocks until explicit recovery.
429 closes reads until Retry-After expires (monotonic clock); missing/malformed
Retry-After blocks for 60s. No immediate rate-limit retry. Observe shared-IP used
weight; a local conservative rolling-minute budget of 1200 weight reserves before
each attempt, with no claim of exclusive IP capacity. Unknown headers/schema
cannot certify global rate availability. Persisted limiter/restart and concurrent
worker budgets remain future services; P1-A is single-threaded.
The rolling budget retains each reservation for a full 60s; it must not reset all
requests at an arbitrary minute boundary. A shared-IP weight observation is a
conservative floor for 60s after receipt. Expiration of a rate-limit block does
not restore invalidated authentication evidence. Unknown dependency exception
reasons and malformed response headers are redacted stable failures. A response
after the 15s logical deadline cannot establish success. Evidence hashes remain
independent of ambient Decimal precision.

Fill recovery requires an explicit starting ID, retains immutable observations,
sorts by trade ID, deduplicates identical economic identity, rejects conflicts,
foreign symbols and nonprogressing pages. More than max_pages yields an explicit
partial result with resume ID; no implicit durable checkpoint or missing-history
success. A failed page raises with no complete result. No inference from held
assets/open orders to the all-symbol fill population.

Focused offline tests, before implementation:

| ID | Observable P1-A contract |
|---|---|
| REST-01 | Independent exact signature vector, Unicode/Decimal/booleans, explicit ms/us, malformed/reserved params. |
| REST-02 | Production/Testnet/userinfo/port/path/method/redirect and response-size guards; TLS verified, no proxy fallback. |
| REST-03 | Clock midpoint/age/RTT/uncertainty/jumps, initial sync, stale sync, one timestamp retry. |
| REST-04 | Balance precision/zeros/omissions and actual receipt, permissions missing/false/true, generation and UID sanitization. |
| REST-05 | Read-only orders with complete identity, UNATTRIBUTED fills/fees; unknown enums/schema/money rejected. |
| REST-06 | Pagination/tail/overlap duplicates/conflicts/nonprogress/partial resume; no global reconciliation or checkpoint. |
| REST-07 | 429/418/auth/timeout/5xx, total attempts/deadline, used-weight/local budget, disabled adapter, stale capability. |
| REST-08 | Synthetic key/secret/UID/error/query never in repr, diagnostics or normalized evidence; old credentials never read. |

Injected scripted transports test request bytes and failure handling; they are
unit evidence, not authenticated Demo observations. Standard HTTPS transport
requires focused dispatch/TLS/redirection tests and an authorized public Demo
ping/time smoke where networking permits. Real private smoke remains pending
until a dedicated pair is provisioned explicitly in this application's own .env.

### Recorded verification (2026-10-11)

SPEC -> 41 missing-adapter RED errors -> implementation -> 41 GREEN. Subsequent
boundary tests reproduced 10 failures before fixes (money grammar, invalidated
read evidence, clock/sample failure, rolling/shared rate budgets, dependency
redaction and Decimal precision); 52 GREEN. Persistent-connection/clock-quality
contract tests then produced 2 failures + 2 missing-interface/behavior errors
before implementation; final **56 adapter tests GREEN**, no skip/xfail.
Final URL review strengthened two existing tests: 5 failures/1 type error RED
for empty userinfo/port/query/fragment, stripped control characters and non-string
paths, then the exact canonical URL and dispatch type guards were implemented.
Protocol response-size fixture was corrected to implement the actual `read1`
interface; the size assertion was retained. All fixtures use synthetic data.

Real public Demo HTTPS smoke: ping HTTP 200; three time HTTP 200 responses on
one warmed connection, RTT 295.631/289.361/293.302ms, every clock sample SYNCED
under unchanged limits. Earlier fresh-connection RTT 1211–1857ms correctly failed
CLOCK_QUALITY. These are dated connectivity/clock observations, not an SLA.
Private requests: 0; matching-engine orders: 0. Unit tests are not authenticated
account permission evidence. No new-app real credential pair was provisioned or
read. Authenticated read smoke, PRIVATE_EVENTS, account ownership and full
ADP-01..11/reconciliation remain pending. Raw logs stay app-local ignored runtime.

## P1-B: private observation and reconciliation contract (2026-10-11)

Implement `private_stream.py` and `account_evidence.py` in the independent app.
This is a read-only evidence service, not an OMS or automatic financial writer.
The existing accounting code/frozen journal identity is unchanged. No current
bot state, process or credential file is consulted. Real authenticated smoke
still requires explicit provisioning in the new app. A service supervisor/CLI,
actual account ownership, certified P2 population/valuation and optional order
validation are later gates; full P1 is not called complete without them.

Sources: [signature subscription](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-api.md#user-data-signature),
[event schema](https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md),
[open lists](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/account),
[websockets sync client](https://websockets.readthedocs.io/en/stable/reference/sync/client.html).
Pin the WS dependency in the own project/lock; no global installation.

### Private reader public API

* `sign_ws(parameters, secret)` returns an HMAC hex digest of alphabetical
  `name=value` pairs joined with `&`, UTF-8 without REST percent encoding.
  Same scalar formatting as REST; reserved signature/duplicate names rejected.
* `DemoWsTransport(url=canonical)` opens only
  `wss://demo-ws-api.binance.com/ws-api/v3` (or explicit :443), verified TLS,
  no inherited proxies, private headers, redirects or arbitrary connect kwargs.
  Verify the completed handshake Host/path before sending any private payload;
  reject alternate URI even if the library followed a public handshake redirect.
  Opening timeout 5s, close 2s, message limit 1MiB, bounded queue 64; disable the
  library's payload logger with an app-owned logger, without changing global
  logging. `connect()`, `send(dict)`, `recv(timeout_s)`, `ping(timeout_s)`,
  `close()`; send allowlist: time, ping, session.subscriptions,
  userDataStream.subscribe.signature. No order mutations or test orders.
* `PrivateEventReader(adapter, store, *, transport=None, process_instance_id,
  background=True)`: `start()`, `poll_once(timeout_s=1)`, `heartbeat()`,
  `health()`, `close()`. Validate a fresh REST account identity against the
  store before subscribing. Reserve WS connect/subscription weight in the own
  REST limiter; one signed subscribe with a unique request ID, recvWindow 5000.
  No blind resend after missing ACK. Persist the subscription mapping only
  after matching success ACK (status 200 and integer subscriptionId).
  Buffer <=64 pre-ACK events with original receipt clocks, then normalize them
  against that exact session/subscription. Mismatched ACK, auth/ban/rate error,
  unknown/foreign event, schema failure, overflow or disconnect records an
  incident, closes transport and invalidates readiness. Never echo wire bodies
  or error messages. A manual new session retains the prior gap incident.
  After start, an own bounded reader thread (optional false in deterministic
  tests) persists received observations while REST recovery runs. SQLite access
  is serialized within the single store writer lease. Close joins its own
  thread only; no other bot worker is touched. Heartbeat is an explicit Pong
  check every 15s, <=5s wait. Silence/timeouts alone are not missing events.

Normalized events: executionReport -> order update plus optional TRADE fill;
outboundAccountPosition -> partial absolute balances (u), never full snapshot;
balanceUpdate -> unclassified signed flow, never automatic trading PnL;
listStatus -> list/order identities; eventStreamTerminated/externalLockUpdate
-> explicit blocking incident. Unknown types are quarantined as reason/receipt
metadata only. Preserve E/T/u and actual UTC/monotonic receipt separately.
No fictitious minute order, inferred trade or arrival-order overwrite.
TRADE requires positive l/L/Y, nonnegative trade ID, exact commission n/N,
side and maker flag; non-TRADE updates create no financial fill. Fill money
identities match REST fields; zero commission has null economic commission
asset. No raw UID, API values or arbitrary payload fields are persisted.
Positive commission requires an explicit commission asset; null is permitted
only for zero economic commission. Normalized event kinds have separate exact
field sets: a balance/list/flow event cannot smuggle in a TRADE fill. Equal-time
terminal order conflicts block rather than using arrival order as authority.

### Durable evidence public API

`AccountEvidenceStore(paths, *, account_alias, account_identity, episode_id,
scope_id, database='account-evidence.sqlite', fault_hook=None)` uses own SQLite
WAL/FULL/FK and OS writer lease; account_identity is the sanitized 64-hex HMAC
token, not UID. Frozen metadata contains schema/app/demo/objective/account/
episode/scope; credential generation belongs to observations/session bindings,
not an irreversible account reset. Different account/scope/schema fails closed.
`close()` is idempotent, context manager supported. Append-only source records
are hashed and verified on reopen; derived cursors are recomputed from immutable
recovery receipts, not trusted as an unchecked mutable cache. No auto repair.

* `bind_session(session_id, subscription_id, credential_generation)` freezes
  mapping for that session; reuse/conflict rejected. Persist received/recorded
  clocks and process provenance; restarting never revives old session health.
* `record_event(session_id, subscription_id, normalized_event,
  received_at_ms, received_monotonic_ns, process_instance_id)` saves an immutable
  event/receipt; same economic trade across REST/WS is one fill, different
  receipts survive. Conflicting economics block with durable incident. Event
  dedup is normalized identity/hash; preserve out-of-order facts. No financial
  projection is modified here. `fills()`, `events()`, `incidents()` expose only
  sanitized stored facts. Unknown/flow/lock/gap incidents remain unresolved;
  no later snapshot silently clears them.
* `record_recovery(result)` accepts only namespace-matching, correctly hashed
  P1-A myTrades observations for one explicit symbol/cursor. Atomically persist
  the explicit `symbol` field added to recovery results, including empty tails;
  deduplicated fills, all page evidence and a recovery receipt; cursor advances
  only in that same transaction. `cursor(symbol)` is next recovery ID, while
  `recovery_start(symbol, overlap=100)` replays IDs below it without assuming
  IDs are contiguous. WS fills never advance a REST recovery cursor. A partial
  page result is retained as PARTIAL and cannot certify tail exhaustion. A
  replay starting ahead of the durable cursor records HISTORY_GAP, not success.
  Initial origin must be explicit ID zero for available-history recovery;
  arbitrary starting ID remains a history limitation. Unknown global denominator
  stays null. `before_commit`/`after_commit` fault hooks test real crash recovery.
* `record_incident(reason)` stores only a recognized stable code/provenance;
  no raw response or secret text. Append-only and retained across reopen.

`BinanceDemoAdapter.read_open_order_lists()` reads signed GET
`/api/v3/openOrderList` (weight 6), validates known list enums, complete member
identity and transaction time. All list IDs/client IDs remain UNATTRIBUTED.
No optional mutations are added to the transport allowlist.

### Reconciliation public API

`AccountReconciler(store).compare(journal, marks, *, account, orders, lists,
population_symbols, population_complete=False, owned_client_ids=(),
stream_health=None, now_ms=None, journal_anchor_ms=...)` produces a diagnostic immutable evidence
result: every expected/nonzero returned asset's free+locked versus actual
JournalStore projection, exact Decimal deltas, missing coverage rows, matched
assets n/N, order/list observations and foreign identities, every post-anchor
recovered fill absent/conflicting in the journal, per-symbol recovery tails and
global fill denominator null. The caller must supply an explicit
`journal_anchor_ms`; fills at/before the opening anchor are not replayed twice
but cannot certify seed history. Compare journal episode/scope, account token,
snapshot source hash/endpoint, nonfuture <=30s receipt and all supplied evidence
namespaces. Stale, incomplete, unknown history/population, unhealthy stream,
foreign/manual activity, flows or any discrepancy yields ACCOUNT_UNRECONCILED.
Matching quantities alone yield BALANCES_MATCH diagnostic, never ORDER_SEND
or certified money. Even a fully matching observation is READ_CHECK_MATCH,
exchange_reconciled=false: certified as-of cut/ownership and P2 evidence must be
implemented before readiness. Record input hashes and evidence-store generation;
new source events make the saved comparison stale. No clearing incidents,
inventing missing zeroes, overwriting financial opening lots or claiming profit.
Invalid/untrusted account evidence is quarantined: it cannot yield BALANCES_MATCH
or a favorable matching ratio. Duplicate balance rows and malformed normalized
snapshots are schema failures. Freeze evidence generation at check start; new
events during a check make its report stale. `comparison_stale(id, journal=None,
marks=None)` checks account-store generation and, when an explicit journal/marks
pair is provided, the current financial projection hash. Without that pair it
cannot certify external journal freshness. Recovery next ID must equal the
maximum observed trade ID +1 (or the starting ID for an empty result); a caller
cannot skip unobserved IDs by supplying an arbitrary checkpoint.
Future journal anchors are invalid input. Malformed order/list observations
yield unknown foreign-order counts, not zero. Unknown dependency error reasons
are redacted even when an injected transport raises AdapterError; handshake
cleanup failures cannot expose dependency messages. WS 429/418 closes the shared
own limiter using the reported retryAfter (absolute UTC ms) or a conservative
60s fallback; validated WS minute weight counters feed the same conservative
IP budget. Signed flow quantities preserve exact Decimal input independently
of the ambient arithmetic context.
Every durable P1-B record includes the program EventEnvelope namespace,
event_id, event_type=`PrivateAccountEvidenceRecorded`, actual receipt/recorded
clocks, process instance, source SHA and nullable causation/model/policy/arm/risk
identity. A RECOVERY/COMPARISON/SESSION has no invented exchange event time.
Envelope hashes and namespace are checked on reopen. These observed events
are distinct from the financial journal's `ExchangeFillRecorded` publication.

`AccountReconciler.recover(adapter, symbols, *, page_size=1000,max_pages=100)`
queries the explicit complete caller population in sorted order, overlapping
durable cursors and recording each symbol atomically. It does not infer the
population from held/open assets. Empty/partial/failed population remains
uncertified. Caller starts reader before recovery and compares fresh REST
anchors afterward; concurrent deltas are retained and close readiness rather
than silently consumed. Automated global bootstrap/entry readiness remains a
later service until certified population and P2 as-of valuation are available.

### Focused tests before implementation

| ID | P1-B scenarios |
|---|---|
| PB-01 | Exact independent WS signing; all host/mutation/logger/TLS/handshake and size guards. |
| PB-02 | Correct/wrong/missing subscription ACK, events before ACK, clock/rate/auth failure, mapped session only, no automatic resend. |
| PB-03 | Execution/fill/fees, non-TRADE, partial balances, signed flows/lists, unknown/termination/lock/overflow, receipt clocks and redaction. |
| PB-04 | Durable identity, single writer, immutable events, duplicate/conflicting REST/WS economics, out-of-order receipts, restart gap. |
| PB-05 | WAL/FULL, real process crash before/after fill+cursor commit, overlapping cursors, partial pages, omitted origin, tamper failure. |
| PB-06 | Balance/locked/missing/dust delta, journal fill omission/conflict, foreign orders/lists, stale/future/wrong-scope evidence, n/N/null. |
| PB-07 | Actual journal integration without writes, empty population, all-symbol recovery, stale comparisons on later evidence and heartbeat limitations. |

Full/changed Truth Harness and regressions remain mandatory. Unit/thread/crash
tests prove these mechanics only. Public WS handshake/time smoke can establish
connectivity without sending keys; it cannot establish PRIVATE_EVENTS support.

### P1-B recorded verification (2026-10-11)

SPEC -> 21 private_stream and 26 account_evidence intended missing-module RED
errors, plus missing symbol reproduction in existing REST tests. Two open-list
tests independently reproduced the missing REST method after correcting an
unrelated fixture dependency; implementation followed those RED runs. Added
boundaries reproduced failures before fixes: cross-kind fill injection, exact
commission/flow values, terminal-state conflicts, skipped cursor, mixed
generation pages, untrusted snapshot scores, report source races, dependency
redaction, shared WS/REST limits and missing program envelopes. No skips/xfail
or weakened assertions. Stale-snapshot fixture now moves its request-start clock
along with receipt; its repeated comparison reuses the own journal lease.

Final **235 app tests GREEN**: existing 166 plus 69 new P1-B tests (31 private
stream, 38 evidence/open-list). Actual own-venv subprocess deaths before/after
recovery COMMIT prove atomic fill+cursor durability; WAL/FULL, immutable-row and
tamper/reopen tests use real files. Journal integration includes BNB fee matching;
the comparison does not write financial events. Root regressions: 27 PASS.
Full Truth Harness PASS; staged gate remains required before commit.

Public Demo WSS smoke: verified handshake (1560.641ms), time response status
200 and verified Pong. Private requests/subscriptions: 0; orders sent: 0.
This proves public transport connectivity only. Credentials were not provisioned
or read; real signed private smoke, exclusive account ownership, certified
population/valuation/as-of cut, incident resolutions and OMS remain pending.
The offline entrypoint is unchanged and starts no service. ORDER_SEND stays
disabled; PRIVATE_EVENTS cannot become certified through a quiet socket or tests.
