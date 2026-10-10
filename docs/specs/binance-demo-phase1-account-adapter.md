# Binance Demo phase 1: account adapter and capability evidence

Registered: 2026-10-10, Europe/Budapest. Status: **PLANNED; specification only**.
Priority: P1, after [phase 0](binance-demo-phase0-financial-contract.md).
Program: [autonomous demo trading](binance-demo-autonomous-trading-program.md).
Application boundary: [independent app isolation](binance-demo-application-isolation.md).
Objective: `daily_net_equity_pnl_v1`; environment: `BINANCE_SPOT_DEMO`.
This document does not enable account access, order sending or existing signals.

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
This specification creates no executable app code.

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
