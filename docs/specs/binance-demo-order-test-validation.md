# Binance Demo: isolated TRADE permission validation

Registered:2026-10-27. Status: implemented;39 focused /790 app GREEN; matching-engine execution disabled.
Parent:[account adapter](binance-demo-phase1-account-adapter.md).
Objective:`daily_net_equity_pnl_v1`. Environment:`BINANCE_SPOT_DEMO`.

## Purpose and authority

Current signed GET smoke proves account reading, not the key's TRADE permission.
Implement a deliberately separate non-matching validation path. This is technical
integration evidence, not a trading strategy, live-order authority or profit.
No changes to legacy bot, financial fingerprints, OMS, policy or risk gates.
The full-market latency and full Truth Harness TH-11 FAIL remain separate.

Reviewed2026-10-27 against primary Binance docs:
[test new order](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/trade#test-new-order-trade),
[Demo URLs](https://developers.binance.com/en/docs/products/spot/demo-mode/general-info).
`POST /api/v3/order/test` is TRADE-secured, validates without entering the
matching engine, and weighs1 without commission calculation. Only this path
is enabled by the new transport. No fallback to `/order`, SOR, OCO or mainnet.

## Interfaces

`DemoOrderTestTransport(base_url=BASE_URL)` shares the existing bounded verified
HTTPS implementation but public `request` accepts exactly POST plus
`/api/v3/order/test`. It rejects GET, order placement/cancel/list paths, URL/path
injection, production/Testnet hosts, redirects and bad headers before I/O.
`DemoHttpTransport.request` remains GET-only with the existing read allowlist.
No existing read adapter method can invoke the new POST path. The common TLS
helper independently rejects any method/path outside GET/read or POST/test,
even when called directly; the attempt helper requires a true boolean signed
flag for POST. HTTPS serializer
sends the exact signed query once, retains TLS verification and15s deadline,
5s connect,8MiB response bound, redacted failure and connection cleanup.

`DemoTradeValidator(client, *, allow_order_validation=False, transport=None)`
requires the existing own `BinanceDemoAdapter` with a READY same-namespace
AuditJournal. Credentials/clock/identity/IP budget stay with that adapter;
no credential file or inherited environment discovery. True must be explicit
and boolean. `disable()` is the immediate rollback switch, revokes earlier
validation evidence and prevents another POST. `close()` closes only its own
transport. It does not own/close the read adapter/journal or stop foreign workers.
Disable records an operational BLOCK when audit is healthy; storage failure
must still leave the local switch disabled and evidence revoked.

`validate_limit(symbol, *, side, quantity, price)` accepts USDT pairs, BUY/SELL,
positive decimal text or Decimal (<=28 significant digits, exponent magnitude
<=18), never binary floats, bool, NaN/infinity, zero/negative or extra parameters.
It constructs only LIMIT GTC, with unique36-char `bdv_` client ID. No arbitrary
type/path, commission calculation, market order, stop price or TP/SL fields.
Caller obtains fresh exchangeInfo/reference data and registers filter-valid
quantity/price; definitive server success validates this exact tested request.
An invalid-filter test proves no permission. Full pre-send filter/risk engine
and actual exchange-side protection belong to phase3, not this interface.

Before POST require explicit enablement, fresh SUPPORTED USER_DATA_READ and
ACCOUNT_SPOT_ALLOWED, READY audit, bounded shared rate budget and healthy clock.
Clock sync may use bounded existing GETs; every validate call makes at most
one POST, with `recvWindow=5000` and15s overall deadline. No automatic POST
retry after timestamp rejection, timeout,5xx,429,418 or ambiguous outcome.
Do not widen clock limits or silently switch target to repair a failure.
Honor Retry-After/ban and used-weight headers on the shared read-client budget.

Each call first invalidates prior KEY_TRADE_VALIDATED evidence, including bad
arguments or disabled operation. Read/account capability does not become TRADE
proof. Only HTTP200 with exact empty object and durable terminal audit ACK
sets KEY_TRADE_VALIDATED SUPPORTED, with source P1_ORDER_TEST, sanitized
request/response digest, tested symbol/side/type and existing30s evidence expiry.
This claim is scoped to this request/generation/account, not all symbols/types,
OCO, protection, remote settlement, ownership or permission at a later time.
ORDER_SEND remains DENIED and no order/position/fill/PnL object is fabricated.

Auth rejection (`-2014/-2015/-1022`, or HTTP401) is DENIED. Invalid parameters/
filters and unsupported endpoint remain UNVERIFIED, not key denial or success.
WAF/HTTP403 without explicit auth code, transport/clock/audit uncertainty,
5xx/429/418 are UNAVAILABLE. A malformed/nonempty200 response is UNVERIFIED.
Every failure supersedes old validation evidence; exceptions expose only stable
reason codes. KeyboardInterrupt/SystemExit retain cancellation audit and revoke
evidence before propagation.

REST_BEGIN (method POST, exact allowed endpoint, safe params, no credentials or
signature) must commit before network; REST_END records status/outcome, primary
receipt clocks, response redaction and local request correlation before success.
Reuse the existing independent attempt/response/error counters. Audit failure
before BEGIN means0 transport calls; after network means no validated result.
Default client read audit behavior stays unchanged. The internal common attempt
helper checks method/path pairs as well; a caller cannot select a placement path.
Operational WAIT before the probe and NO_ACTION/BLOCK after its outcome are
durable causal decision records, including safe tested parameters/source digest.
Only ACKed final NO_ACTION may precede SUPPORTED. Invalid/disabled calls also
record BLOCK when storage is healthy. Failed audit never permits another POST.
This service retains its parent's single-threaded ownership contract.
The15s deadline is checked again after final decision ACK; slow audit cannot
turn an expired validation into fresh capability evidence.

Return an observation with same namespace/credential generation, POST source,
exact tested safe parameters and digest, request/receipt clocks and audit IDs,
`state=VALIDATED_NOT_SUBMITTED`, `order_send_enabled=false`, `submitted=false`.
Private UID/key/signature redaction remains mandatory. Persist only app runtime
evidence, never account balances, keys or live snapshots into Git.

## Test-first scenarios and verification

| ID | Expected result |
|---|---|
| VLD-01 | Defaults disabled; invalid constructor/audit/account inputs; no POST or inherited credentials |
| VLD-02 | Valid exact signed LIMIT/GTC, schema/Decimal/USDT bounds, deterministic independent HMAC check |
| VLD-03 | Both transport allowlists: POST test only vs GET read only, exact HTTPS/TLS/request bytes |
| VLD-04 | Durable BEGIN before real foreign DB read, terminal ACK before SUPPORTED; no secrets/UID |
| VLD-05 | Empty200 success only; invalid/filter/unsupported/auth/WAF mapped separately |
| VLD-06 | Timeout/5xx/timestamp/cancel single POST; rate/ban/cooldown/clock/deadline failures |
| VLD-07 | Old success revoked on later failure/disable;30s expiry/generation scope; ORDER_SEND denied |
| VLD-08 | Native SQL BEGIN/END abort:0 sends or unknown validation; no fabricated fills |

Use synthetic app-owned fixtures, mocked HTTPS sockets and native SQLite, no
real credentials/network in tests. Record RED, then focused/regression GREEN,
diff review, staged Truth Harness, commit/push. TH-01..12 apply; no policy
relaxation means no new financial backtest claim. Future BUY enablement still
requires maximum-available causal policy replay and separate forward/canary.

After tests, explicitly load own `.env` for a bounded Demo diagnostic: public
metadata/quote, signed account and pre-orders/lists; one filter-valid BTCUSDT
BUY LIMIT test; post-orders/lists and balances. Preserve raw sanitized evidence,
all attempted checks and before/after differences. No observed change is a
point-in-time check, not exclusive ownership or complete account history.
If limits/filters/endpoint fail, record the exact failure and retain disabled
ORDER_SEND; no blind retries or live tiny-order fallback. Check runtime files
for secret/UID absence. Report technical validation separately from profitability.

Recorded test-first evidence:32 missing-module RED before implementation.
The constructor-override fixture had duplicate Python keyword arguments; fixing
that fixture did not change assertions or count as behavior RED. Five additional
boundaries reproduced missing disable audit and a deadline bypass during final
decision ACK. After repair37 GREEN,788 intermediate full app GREEN (157.647s).
Two final defense tests reproduced common TLS helper dispatch outside the allowed
union and integer1 accepted as signed=True (2 failures/1 callback error). Both
repaired before39 focused GREEN (9.345s), without skips/xfail or weaker checks.
Final790 app tests GREEN (145.254s),14 root spec checks GREEN. Both diagnostics'
runtime privacy scans passed12 JSON files/26 final audit records each, with0
credential values/raw account identifiers found. Staged Truth Harness required
before commit; full profile's existing TH-11 FAIL remains visible.

Live technical evidence2026-10-27: initial diagnostic PASS, then a distinct
source-bound diagnostic after tightening internal fences PASS. Each run made
exactly1 non-matching BTCUSDT BUY LIMIT test POST;2/2 tests accepted across the
two recorded source versions. No matching-engine order was submitted. Each run
preserved all11 REST BEGIN/END pairs and2 explicit decisions, with0 unresolved
attempts;4/4 checks (validation and three before/after comparisons) passed.
Balances/open orders/lists had no observed change at the comparison instants.
That is not certification of exclusive ownership, history completeness or PnL.
KEY_TRADE_VALIDATED was SUPPORTED at each check; the30s TTL prohibits treating
these saved reports as current reusable permission. PRIVATE_EVENTS UNVERIFIED,
ORDER_SEND DENIED. All live quantities/prices/balances/keys/UID/runtime evidence
stay outside Git. Full Harness remains FAIL TH-11 (stale historical replay hash).
