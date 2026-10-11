# Binance Demo phase 2: complete universe and causal market data

Registered: 2026-10-10, Europe/Budapest. Updated: 2026-10-11.
Status: **P2-A/B2 foundations GREEN; 165 focused tests / 400 app GREEN; full-market stream latency FAIL; full P2 incomplete**.
Priority: P2; may be developed alongside [phase 1](binance-demo-phase1-account-adapter.md).
Program: [autonomous demo trading](binance-demo-autonomous-trading-program.md).
Application boundary: [independent app isolation](binance-demo-application-isolation.md).
Objective: `daily_net_equity_pnl_v1`; environment: `BINANCE_SPOT_DEMO`.
This document does not change the current watchlist, feed or trading policy.

## Problem and objective fit

Existing public clients and manually limited watchlists cannot establish all
instruments permitted to this demo account. Selecting survivors retrospectively,
stale features or incorrect quote conversion invalidates daily portfolio PnL.
Provide complete as-of monitoring, credible valuations and decision-ready inputs;
coverage is a prerequisite, not evidence that more instruments increase profit.

## Scope and planned components

Implement `UniverseService`, `MarketDataService`, `FeatureSnapshot` and
`QuoteConversionGraph` in the independent `apps/binance_demo_bot/` application.
Planned app-relative paths: `src/binance_demo_bot/universe.py`,
`src/binance_demo_bot/market_data.py`, `tests/test_universe.py`,
`tests/test_market_data.py`; own `.venv`, dependency lock and lifecycle.
Old algorithms may be studied read-only and reimplemented with new-app tests;
no `files/*` imports, current clients/config/global state or old runtime/data
stores are used. Own snapshots/caches/cursors live only under app `.runtime/`.
The old partial-depth client is not proof of a valid incremental L2 algorithm.

Monitor every account-permitted Spot pair, including non-USDT quote assets;
maintain separate assets/pairs, actual holdings, execution scope and shortlist.
Initial execution scope is explicitly USDT-quoted pilot pairs in phase 3/4.
This restriction never redefines the all-universe monitoring denominator.
No margin/leverage, synthetic L2, forced purchase, autonomous capital conversion
or retrospective replacement of the historical eligible population.

## Universe discovery and completeness

Fetch Demo `exchangeInfo` with permission sets included, account permissions
from phase 1, and a versioned recognized permission registry. The default public
response can filter permission categories. Explicitly request the currently
recognized permission categories, including categories required by the account;
union and deduplicate responses where needed. Record request parameters,
permission/schema revision and discovery completeness. Unrecognized account
permission, omitted category or truncated response yields
`UNIVERSE_INCOMPLETE`, not a smaller complete universe.
[Exchange information](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#exchange-information),
[permission definitions](https://github.com/binance/binance-spot-api-docs/blob/master/enums.md#account-and-symbol-permissions).

Evaluate symbol `permissionSets` as AND across sets, inclusive OR within each:

```text
symbol_permission_ok = all(any(p in account_permissions for p in group)
                           for group in permissionSets)
```

An absent/malformed permission field or unknown required enum is uncertified;
do not accept it through the mathematical empty-array identity. If the documented
schema explicitly permits an empty list, preserve that documented interpretation
in the parser version and tests. Legacy empty `permissions` is not a substitute.
[Official permission-set semantics](https://github.com/binance/binance-spot-api-docs/blob/master/CHANGELOG.md#2024-04-02).

Snapshot every discovered pair's base/quote, status, Spot allowance, evaluated
permissions, order types/list support, exchange/symbol/account filters and reason
codes. `TRADING` is required for a new entry. `HALT`, `BREAK`, `CANCEL_ONLY`,
removed/unknown statuses stay recorded and block entry; metadata absence is not
permission to sell or buy. Track holdings/orders in stopped symbols for incident
handling even when new market data cannot be obtained. Availability of quotes
does not imply permission to execute.
[Spot symbol-status definitions](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/spot_glossary.md).

Refresh discovery at startup, at most 15min apart initially, and immediately on
unknown symbol, permission/status/filter event or relevant API rejection. Commit
an immutable `UniverseSnapshot` with `snapshot_id`, payload/source SHA, metadata
clocks, registry version, account capability ID and complete exclusion registry.
Record first seen/listing/status/removal transitions as received. Never infer
an asset traded in the past because it is listed today. Failed refresh retains
the prior snapshot as stale; current discovery cannot certify historical PIT.

Define separate counts: discovered pairs; certified permitted Spot pairs;
currently entry-eligible pairs; all monitored assets; fresh feature-ready pairs;
shortlisted L2 pairs; all held assets with fresh valuation. Every discovered pair
has one availability row with stable sorted reason codes. Permission-denied
pairs remain visible outside the permitted denominator. A quote shortage or
pilot restriction blocks an entry, not monitoring membership.

## Filters and quote-capital contract

Use Decimal strings and explicit tick/step rounding, never asset precision as
an executable step. Preserve all applicable symbol/exchange filters, order-count
limits, percent-price bounds, market/notional constraints, trailing/STP support
and account-specific restrictions. Fetch signed relevant filters (`myFilters`)
for prospective executable symbols; mark unsupported/unknown account restrictions
explicitly. Unknown required filters block execution certification.
[Binance filters](https://github.com/binance/binance-spot-api-docs/blob/master/filters.md),
[account-relevant filters](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#myfilters).

Eligibility previews use known `free` quote funds net of phase-3 reservations,
not total free+locked equity. Phase 3 owns the final account/risk/filter recheck
before send. No automatic BUY route assumes USDT can pay for BTC/ETH-quoted
pairs without actual quote funds and a separately approved conversion action.

Valuation graph covers every nonzero `free+locked` asset, commissions and quote
currencies. USDT is the sole identity conversion; other stablecoins have market
prices. Versioned deterministic route choice: fresh same-environment direct
USDT pair first, then a previously registered shortest path (initial maximum
three edges), with stable lexical tie breaking. Freeze route ID/leg snapshot
IDs as-of valuation; route selection never optimizes on later realized prices.
Inverse midpoint uses reciprocal mid, while liquidation estimates use correct
bid/ask direction plus depth/fees. Missing fresh path makes valuation incomplete;
it does not set the asset to zero or assume stablecoin parity.

The financial boundary contract belongs to
[phase 0](binance-demo-phase0-financial-contract.md): every route/leg stores
preregistered `time_basis=EXCHANGE_EVENT|OBSERVED_RESPONSE`. In EXCHANGE_EVENT
mode the real event time must be at/before the boundary; in OBSERVED_RESPONSE
mode the completed response/actual receipt must precede the boundary, with
unknown exchange tick age disclosed. Both apply the registered age limit.
Later receipt in EXCHANGE_EVENT mode can repair
accounting only using an actual historical mark; it cannot repair decision
availability or use a current price for a past boundary.

## Data tiers, scheduling and causal features

Use Demo REST/streams from the phase-1 allowlist. Production data may exist as
explicit external features in a later registered policy, but never silently
substitutes for demo quotes, fills, fees or account valuation. Demo maintenance
may make responses stale even when they succeed.
[Demo data services](https://github.com/binance/binance-spot-api-docs/blob/master/demo-mode/general-info.md),
[Demo maintenance changelog](https://github.com/binance/binance-spot-api-docs/blob/master/demo-mode/CHANGELOG.md).

| Tier | Population | Scheduled data |
|---|---|---|
| A | Entire permitted universe | Metadata/status, aggregate volume/returns, bid/ask freshness and closed-bar features |
| B | Shortlist and all open/pending exposure | Trades, bounded L2, volatility/liquidity and order-flow features |
| C | Every holding, quote/fee asset and conversion leg | High-priority valuation quotes and account protection inputs |

Tier A must expose every pair's data status; all-ticker responses that emit
changed symbols only cannot by themselves certify per-pair freshness. Schedule
fair sharded polling/streams; record queue lag, retries, missing symbols and
intentional throttle. Deep shortlist is chosen by as-of inputs and frozen rules,
with `selected/not_selected/reason`, not by later winners. If L2 is a required
feature, unselected/failed pairs cannot receive invented zeros; the policy must
explicitly wait or use a separately evaluated non-L2 fallback.

One shared IP request-weight and connection/control-message budget serves all
services. Reserve headroom for account reconciliation and protection; market
backfills are lower priority. Read current limits and endpoint weights, obey
429/418 and bounded reconnect/backoff. Initial scheduling caps: <=800 streams
per connection, <=3 control messages/s including reserved ping/pong headroom,
bounded restart attempts below the documented IP limits. These conservative
values are configuration, not a promise that full L2 for all pairs is cheap.
[Market-stream limits](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#websocket-limits).

The limiter owns only new-app requests, not the existing bot's scheduling.
IP/connection and account order/filter budgets may be shared despite separate
keys/processes. Inspect aggregate exchange headers and conservatively account
for external traffic; exhausted/uncertain headroom blocks entries or lowers data
readiness. Never stop/reconfigure old workers, reuse their sockets/cache/locks,
or promise guaranteed protection capacity from independent local rate counters.
Own service/UI ports/tokens, reconnection state and source manifests remain
distinct; a port/path/namespace collision fails without taking over another app.

Raw samples and `FeatureSnapshot` use program `EventEnvelope` with
`application_id=binance_demo_bot` and add
`snapshot_id`, `objective_contract_id`, `process_instance_id`, source/method, symbol, interval, exchange event
time, actual receipt/monotonic time, closed-bar flag, schema version, raw hash,
metadata/universe/route IDs, feature definition SHA and validity reasons.
`policy_version`, `model_id`, `source_sha256`, `risk_contract_id` are attached
when applicable; absent values are null. Persist before publishing to decisions.

At decision cutoff, every input must have `received_at_ms <= decision_at_ms`
and event time consistent with validated clock uncertainty. Feature availability
is the maximum receipt time of dependencies; a downloaded historical candle's
close time is not proof of historical receipt. Admit closed candles only; store
in-progress bars separately. Sort/deduplicate validated keys, quarantine
conflicting duplicates and schema changes; no future interpolation or backward
fill. Gaps reset dependent rolling/sequence features unless a separately
registered missingness-aware definition explicitly handles them.

Some quote payloads, including the documented individual book-ticker stream,
omit exchange event time. Preserve null and `TIMESTAMP_UNVERIFIED`; receipt time
must not be renamed exchange time. Prefer timestamp-capable ticker bid/ask or
certified diff-book marks for the strict event-age/boundary contract. An explicit
versioned observed-time convention may use the actual completed response
interval before the cutoff, with quote-age uncertainty disclosed and phase-0
agreement; it cannot synthesize a past exchange timestamp or repair a past
decision. Unknown event age never silently satisfies the five-second check.

Initial versioned freshness registry:

| Dependency | Limit at consumer cutoff |
|---|---|
| Entry/valuation bid and ask | <=5s age by preregistered time_basis: event age for EXCHANGE_EVENT, observation age for OBSERVED_RESPONSE; actual receipt before decision |
| Account readiness | <=30s full snapshot plus certified reconciliation, phase 1 |
| Universe/filter metadata | <=15min; forced refresh on relevant change/rejection |
| Required L2 state | <=1s plus certified sequence continuity |
| Latest closed 1m candle | <=90s after its closing boundary; other intervals registered explicitly |

Required inputs exceeding limits reject new decisions with the exact reason;
data loss never disables exchange protection or counts as zero-price loss.
Every quote/route leg includes `time_basis`; null event time cannot pass an
EXCHANGE_EVENT check. Clock uncertainty/event-age and receipt-age checks are separate. Initial limits
may be tightened before pilot; weakening them is a versioned hypothesis/review.

Incremental L2 uses a buffered diff stream plus REST snapshot and verifies
`U/u/lastUpdateId` bridging and continuity. A sequence gap invalidates the book
until rebuilt; zero quantity deletes a level, not an order-flow trade. Partial
depth snapshots are labelled bounded states and never replayed as full-book diffs.
No complete depth beyond the observed levels is claimed.
[Local order-book reconstruction](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#how-to-manage-a-local-order-book-correctly).

## Primary metrics and acceptance criteria

Daily net equity PnL is the business target. Infrastructure publishes monitoring
coverage `fresh_ready/permitted_total`, exclusion/reason counts, feature receipt
lag, clock uncertainty, L2 gaps, request budget and holdings valuation coverage
`fresh_valued/nonzero_held`; zero/unknown denominators remain null. A short deep
list does not appear as 100% all-universe coverage. Every pair is fresh or has
an explicit unavailable/pending reason; unresolved failures block readiness.

Completion: full permission/status discovery verified, non-USDT pairs/assets
included, deterministic as-of snapshots and capital/valuation routes, bounded
scheduling, observable losses/gaps and exact feature provenance. At least one
listing/status/reconnect fault is exercised without shrinking denominators.
No successful API call alone grants `MARKET_DATA_READY` for all dependencies.

## Backtest / verification gate and planned focused tests

Implement scenarios below with synthetic faults and pinned public fixtures;
then run an authorized bounded demo monitoring smoke. Fixtures do not certify
historical PIT or exchange execution. Any universe/filter/feature hypothesis
that changes decisions requires maximum available historical policy replay,
coverage audit and a new forward cohort before promotion (phases 4/6/7).

| ID | Planned scenario and expected result |
|---|---|
| UNI-01 | AND/OR permission groups, missing fields/unknown enums; exact eligibility and incomplete discovery reason |
| UNI-02 | Non-USDT pairs, duplicate assets, disabled/pilot pairs and unseen permission category; no silent denominator shrink |
| UNI-03 | Listing/delisting/HALT/BREAK/CANCEL_ONLY and failed refresh; immutable transitions, blocked new entry |
| UNI-04 | Tick/step/notional/order/account filters and Decimal boundaries; correct previews without rounding overspend |
| UNI-05 | Fee/stablecoin/inverse/multileg/dust valuation with one stale leg; exact route or incomplete result, never zero |
| DATA-01 | Closed/in-progress candles, late receipts and future mutation; unchanged prefix decisions and correct availability |
| DATA-02 | Duplicates/conflicts/gaps/unknown schema and clock jump; explicit quarantine/invalid feature |
| DATA-03 | L2 snapshot/diff bridge, zero levels, overlap/out-of-order and gap; reset until certified rebuild |
| DATA-04 | Changed-only ticker omissions, many symbols, 429/ban/reconnect; explicit freshness and protected priority budget |
| DATA-05 | Future-chosen shortlist or conversion route; rejected leakage, paired as-of population preserved |
| DATA-06 | Midnight mark arrives late versus current-price repair; financial repair cannot alter prior decisions |
| DATA-07 | Old cache/import/PYTHONPATH, path traversal/symlink escape or worker namespace; isolated startup rejects, current files untouched |
| DATA-08 | Shared IP quota/external account orders and port collision; own readiness degrades, no foreign-process control |

## Risk / trade-offs and rollback switch

All-pair ingestion increases compute and missing-data exposure; fair coarse
monitoring plus selective deep data bounds cost. Small pilot execution scope
reduces opportunity while coverage remains broad and explicit. Disable new
feature/shortlist releases via `BINANCE_DEMO_MARKET_DATA_ENABLED=false` and
freeze entry readiness; retain snapshots and mark caches stale. OMS/account
reconciliation and existing exchange protection continue independently. Restore
with fresh universe and rebuilt required streams; never reuse stale L2 silently.
TH-01..TH-12 apply; source implementation must add/run focused tests, full and
staged Truth Harness, diff review, commit and push. Scenario IDs are planned,
not a claim that the functionality or tests already exist.

## P2-A: discovery, REST observations and portfolio marks (2026-10-11)

Deliver the foundations first: full registered-permission discovery, immutable
as-of membership, whole-universe bid/ask availability, bounded/fair closed-1m
polling/features, and deterministic midpoint valuation of all free+locked assets.
P2-B will add continuously supervised market streams, full Tier-A scheduling,
incremental L2/sequence recovery, signed myFilters/complete execution previews,
timestamp-certified marks and full UNI/DATA acceptance. This phase does not
enable decisions, orders, learning or account reconciliation certification.
Actual account permissions remain unavailable until the separate app's keys
are provisioned; public discovery is PUBLIC_REFERENCE, never invented account
authority. Own paths/store/processes only; no old bot/cache/key fallback.

Sources checked 2026-10-11: [REST metadata/klines/book ticker](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md),
[permission/status registry](https://github.com/binance/binance-spot-api-docs/blob/master/enums.md),
[filters](https://github.com/binance/binance-spot-api-docs/blob/master/filters.md).
The frozen permission registry is SPOT, MARGIN, LEVERAGED and TRD_GRP_002..025.
Ask explicitly for all 27 categories, showPermissionSets=true and no status,
symbol or pilot filter. Do not use account-only/default SPOT discovery. Empty
permissionSets has no documented unrestricted interpretation in this parser
revision; keep it uncertified. New enums trigger a versioned registry update.

Real public preflight: all-category Demo exchangeInfo returned HTTP 200,
17,769,069 bytes, 3727 symbols, BREAK/TRADING, no empty permissionSets. This
exceeds P1's 8MiB cap. Retain 8MiB for private/other replies; allow <=32MiB only
on public /api/v3/exchangeInfo. Gzip on this public endpoint is optional and
bounded before/after decompression (no zip-bomb or corrupt/trailing fallback).
Do not weaken host, TLS, timeout, redirect or mutation guards.

### Public interfaces (own package)

* Extract `DemoReadClient(transport=None, clock=None, enabled=True, sleep=None)`
  from the already-tested P1 read/clock/limiter implementation.
  `request_public(path, parameters, weight=...)` allows only ping/time,
  exchangeInfo, bookTicker and klines; it has no credentials and rejects signed
  account access before dispatch. BinanceDemoAdapter retains exactly its P1
  contracts as a subclass using the shared implementation. Constructor/method
  refactoring is behavior-preserving; existing P1 tests remain authoritative.
* `DemoMarketClient(client=None)` uses a credential-free DemoReadClient by
  default, or explicitly shares the new app's P1 adapter/clock/budget.
  `read_exchange_info()`, `read_book_tickers(symbols=None)`,
  `read_klines(symbol, start_ms=None, end_ms=None, limit=1000)`. GET only,
  no API headers/signatures for public calls. Weights 20/4/2; initial klines
  support UTC 1m only. Explicit symbol lists are sorted/deduplicated, <=128 per
  batch. Record exact request params, actual UTC/monotonic response interval,
  environment, raw-body/canonical-data hashes and clock quality. Invalid JSON,
  duplicated keys, inconsistent params or oversized/schema replies fail closed.
  Reserve 200 of the local 1200 rolling weight for account/protection by checking
  worst-case three market attempts against <=1000 before dispatch. Aggregate
  exchange weight may exhaust it; this does not guarantee shared-IP capacity.
* `MarketSnapshotStore(paths, *, account_alias, episode_id, scope_id,
  database='market.sqlite')`: own WAL/FULL/FK and OS writer lease, frozen
  schema/app/demo/objective/account/episode/scope identity. `append(kind, data,
  received_at_ms, received_monotonic_ns, dependencies=[])`, `as_of(kind,
  cutoff_ms)`, `records(kind=None)`, `close()`. Kinds UNIVERSE, QUOTES, CANDLES,
  FEATURE, VALUATION, INCIDENT. Every saved result has EventEnvelope,
  snapshot/event IDs, source SHA, dependency IDs, actual recorded/receipt clocks,
  process ID and nullable model/policy/arm/risk provenance. Validate namespace,
  hashes, dependency existence and availability; resolve own paths before I/O.
  Rows are immutable and verified on reopen; no automatic tamper repair.
  Never expose a snapshot before the maximum actual receipt of dependencies.
  Copies returned to a caller cannot mutate stored or earlier snapshots.
* `UniverseService(store, client).refresh(account=None)` fetches all categories,
  validates a fresh <=30s P1 account snapshot when supplied, and persists one
  row per discovered pair. `current(cutoff_ms)` retrieves the received-as-of
  snapshot, not today's survivors. Preserve base/quote, all statuses, Spot flag,
  permission sets, order/list flags, symbol/exchange filters and explicit
  exclusion reasons. AND across groups, OR within group. Non-USDT quotes stay
  monitored. Unknown account/permission/schema means UNIVERSE_INCOMPLETE with
  permitted_total=null; PUBLIC_REFERENCE without account still exposes every
  public row and discovered n, but cannot claim any permitted denominator.
  Unknown statuses/filter/order enums block metadata entry previews, not erase
  rows. Initial preview scope is USDT only; it never shrinks monitoring.
  Certified execution is always false until P3/account filters/risk.
  Record ADDED/STATUS_CHANGED/REMOVED as received. Removed pairs remain explicit
  tombstones for exposure handling; current permitted total excludes only a
  confirmed removal. Failed refresh records an incident and retains last-good
  membership as stale without changing its original snapshot or denominator.
* `MarketDataService(store, client).poll_quotes(universe, symbols=None)` persists
  per-pair rows, including zero/crossed/malformed/missing/unknown-metadata states.
  A bad pair never makes the remaining pairs disappear. bookTicker has no event
  time: retain null and TIMESTAMP_UNVERIFIED; registered time_basis is
  `OBSERVED_RESPONSE` with convention `observed-book-v1`. This qualifies only
  observation age, not a fabricated exchange-tick age. EXCHANGE_EVENT consumers
  cannot use these marks. Explicit omitted expected rows invalidate those
  rows; stale/invalid newer observations never resurrect earlier healthy data.
* `collect_candles(symbol, ...)` stores sorted, deduplicated 12-field UTC 1m
  bars with precise OHLCV/counts and actual receipt. Positive finite prices,
  consistent high/low, nonnegative volumes/counts, open alignment and exact
  close=open+59999. Closed iff close < validated server cutoff; in-progress bars
  are separate observations. Conflicting closed revisions are quarantined;
  evolving open bars can become a closed bar without false conflict.
  Downloading a past close does not backdate availability. No interpolation.
* `feature(symbol, cutoff_ms, lookback=3)` uses only snapshots received by the
  cutoff; last lookback>=2 closed consecutive minutes, latest close <=90s old,
  quote observation <=5s and metadata <=15min. Return exact
  `close_return=(last_close/first_close)-1`, quote_volume_sum and trades_sum,
  dependency IDs, definition SHA, maximum dependency receipt and reasons.
  Gaps/conflicts/short history yield FEATURE_UNAVAILABLE/null values, not zeros.
  Freeze the versioned feature definition. Persist before return with actual
  compute/publication time; historical prefix reconstruction is labelled replay
  and cannot masquerade as an earlier real decision. Later data does not change
  the cutoff's selected values/dependencies.
* `availability(universe, cutoff_ms)` returns every discovered row's quote/bar
  status and separate discovered/permitted/quote-ready/feature-ready counts n/N.
  Missing or zero denominators stay null. quote-ready is not feature-ready.
  `poll_candles(universe, max_symbols=50)` rotates fairly through explicit
  permitted TRADING pairs; it never claims unscheduled bars exist. Exceeding
  budget leaves remaining rows pending, protects account headroom and starts
  no worker/supervisor. `enabled=False` closes new market collection; stored
  observations remain available only within their original freshness limits.
* `QuoteConversionGraph(store).value(balances, cutoff_ms,
  time_basis='OBSERVED_RESPONSE', max_age_ms=5000)` chooses only metadata/quotes
  available at cutoff: direct USDT first, then <=3-edge shortest fresh path,
  lexical tie break, correct inverse orientation. No price-optimizing or future
  route. Call the unchanged P0 value_portfolio for exact free+locked Decimal
  midpoint equity/marks. USDT is the sole identity; stable/fee/dust assets need
  real routes. One missing/stale leg makes overall valuation PENDING/equity=null.
  Preserve route IDs, universe and leg snapshot IDs, clock convention, age
  uncertainty, held-assets n/N and reasons. This is midpoint mark evidence,
  not a liquidation guarantee, historical account anchor or profitable result.

P2-A is read-only and manual/library-driven; the P2-B1 stream/L2 foundation is
described below. P2-B2 retains certified marks, sustained all-pair scheduling/lag, reference-price/account filters,
execution previews and formal as-of financial reconciliation. No active CLI
service or new trading policy is introduced by this implementation.

### Focused acceptance tests before code

| ID | P2-A observable contract |
|---|---|
| P2A-01 | Credential-free/public transport, exact 27-permission query, bounded gzip/32MiB metadata vs 8MiB private, no mutation/production fallback. |
| P2A-02 | AND/OR, empty/missing/unknown permission, account scope/hash/freshness, public reference/null denominator and non-USDT population. |
| P2A-03 | Listing/status/removal, failed refresh, as-of immutable membership, filters/unknown enums and own store/lease/tamper/dependencies. |
| P2A-04 | Book omissions/zero/crossed/malformed rows, null event age, explicit observation basis, cutoff/staleness and fair budget/disabled collection. |
| P2A-05 | Closed/open bars, sorting/duplicates/conflicts, gaps/late download, prefix invariance, no interpolation, exact math and feature provenance. |
| P2A-06 | Direct/inverse/multileg/stable/fee/dust valuation, old/future/wrong-venue quote, one stale leg, no false zero or historical repair. |

Record RED/GREEN evidence under own ignored runtime; run P1/P0 regressions,
full and staged Truth Harness, review/stage only intended source/spec/tests,
commit and push. Public real smoke validates discovery/quotes/closed bars and
public hypothetical valuation only; it does not substitute for account rights.

### P2-A review boundaries

Account provenance is retained as a whitelisted P1 snapshot (no arbitrary extra
fields); the first valid account token binds subsequent snapshots in the same
episode, including after restart. Unknown token/generation/schema or future
update time is not a valid permission denominator.
P1 data Decimal values retain their fixed-point spelling when stored inside the
provenance object, so its original source SHA verifies after persistence (the
market store's own normalized hash remains separate). Every consumer verifies
the supplied universe against its immutable own-store source and receipt cutoff.
Refresh failures close metadata readiness until a newer successful refresh;
prior membership/denominators remain visible as stale evidence. Candle schema
or clock failures leave a per-symbol incident and close feature readiness until
a newer valid receipt; malformed input cannot silently reuse a prior feature.
Market budget protects the 200-weight reserve before any network attempt; public
request metadata includes the actual raw-body SHA and exact parameter convention.
Arithmetic follows P0's fixed 50-significant-digit Decimal context; recurring
inverse fractions are rounded at that precision, not claimed algebraically exact.
Public metadata requests explicitly advertise Accept-Encoding: gzip; this sole
additional header is allowed only on exchangeInfo with the exact value gzip.
Private endpoints and other public reads do not inherit it. The 15-second read
deadline and endpoint-specific decompression caps remain enforced.

### Delivered and verified on 2026-10-11

P2-A is implemented in the own package; 57 focused tests and **292 total app
tests GREEN** after recorded RED. P0/P1 behavior is retained; 27 repository
spec/harness regressions and mechanical full Truth Harness PASS. Staged/change
release verification is required before commit. No trading-policy relaxation.
Initial 48 cases reproduced missing intended modules/client/public paths. Review
reproductions preceded fixes for raw-body SHA, stale incident readiness, mutated
or future membership, account provenance/token, persisted Decimal source hashes
and schema failure. Gzip tests
reproduced both missing header and transport allowlist before implementation.
The inverse fixture uses finite 1/4 with the same expected equity assertion;
it does not incorrectly demand algebraic exactness from rounded P0 1/3.

Real public Demo smoke: 3727 discovered instruments, 2010/3727 positive valid
book observations (this is public observation coverage, not account coverage).
BTCUSDT/ETHUSDT/SOLUSDT each returned 10 minute bars, 9 closed; all three causal
features were ready at the frozen observation cutoff. A clearly hypothetical
BTC+USDT inventory had complete midpoint valuation; no actual balance, rights,
private subscription or orders were used. Initial uncompressed attempt failed
READ_UNAVAILABLE; bounded gzip succeeded. End-to-end manual run took 173.67s,
metadata stage 86.53s including validation/persistence. This is not a real-time
latency qualification. P2-B must profile/bound processing and publication lag,
implement supervised streams, all-pair scheduling, L2, certified timestamps,
account/reference-price filters and actual reconciliation. Order sending stays
disabled; account-permitted N stays null without fresh own account evidence.

## P2-B1: public stream and bounded incremental L2 (2026-10-11)

Implement the next read-only vertical slice; full P2-B remains incomplete.
No trading decisions/orders/learning, automatic key provisioning or current-bot
services. P2-B2 retains the complete market scheduler, closed-candle streams,
metadata/account-filter/reference-price refresh, certified financial as-of cuts,
publication-lag optimization and sustained full-universe qualification.

Protocol sources checked today: [Demo hosts](https://github.com/binance/binance-spot-api-docs/blob/master/demo-mode/general-info.md),
[streams, limits and L2 bridge](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md),
[depth weights](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#order-book).
Only `wss://demo-stream.binance.com/stream` (explicit :443 equivalent) is used;
TLS verification, no proxy/redirect/fallback, own quiet logger. Existing pinned
websockets package handles server Ping/Pong. No credentials or signed methods.
Combined stream subscriptions use lowercase symbols, <=512 topics per shard,
<=32 explicit deep symbols. Auto-client Ping disabled; proactive health Ping
at most every 20s. JSON controls <=2/s, leaving room for server Pong; connection
attempts <=20/rolling 5min locally, exponential reconnect 1..60s and 23h rotation.
This is not a guarantee about aggregate shared-IP quota or unbounded server pings.

### Interfaces and invariants

* `stream_plan(universe, deep_symbols=())`: every discovered TRADING Spot pair
  with permitted=true or null public-reference goes to bookTicker shards;
  denied/removed/stopped pairs stay in the coverage rows. Non-USDT remains.
  Deep symbols must be explicitly in that received population; plan includes
  universe ID, registry/counts, deterministic shard IDs/topics and deep list.
  <=512 topics/shard (official ceiling 1024), <=32 deep symbols. No future
  membership/shortlist is accepted by a reader; null permission N remains null.
* `DemoMarketWsTransport`: only combined /stream, credentials absent; SUBSCRIBE
  <=512 topics and non-null local id, only bookTicker/depth@100ms allowed.
  No order/private methods. Validated exact handshake path/Host, max_size 1MiB,
  max_queue 16; static redacted dependency errors. `recv`, `ping`, `close`.
* `DemoMarketClient.read_depth(symbol, limit=100)`: GET /api/v3/depth only,
  limit 1..1000, weights 5/25/50 with unchanged 200-weight local reserve,
  exact parameters/source/receipts/hash; no inferred snapshot exchange time.
  No 5000-level expensive request in this slice.
* `IncrementalBook(symbol, clock, max_buffer=1024, max_levels=20000)`:
  `on_diff(payload, received_at_ms, received_monotonic_ns)` buffers until REST
  snapshot installed. `install_snapshot(observation)` verifies Demo endpoint,
  exact symbol/limit params, hash, receipt interval and valid positive/unique
  price levels/nonnegative quantities. Snapshot alone is BUFFERING, not READY.
  Discard u<=snapshot.lastUpdateId; first applied diff bridges next expected
  ID with U<=lastUpdateId+1<=u, then same rule for continuity/overlap. A gap,
  buffer/level overflow, schema/crossed/empty book, clock jump or unhealthy age
  clears the usable book and requires a new snapshot. Exact duplicate/stale
  u<=last_id does not refresh age. Same u with conflicting payload is rejected.
  Zero quantity deletes, positive quantity replaces (never adds to prior qty).
  Open partial-depth messages are rejected as diffs. Book levels are bounded
  observations, not a certified complete exchange book. Top-level exhaustion
  cannot invent unseen prices. Clock must be valid; event and receipt age <=1s
  with clock uncertainty; no future event beyond measured uncertainty. Event
  older than previous accepted event cannot make freshness move backwards.
  `view()` returns copies, state/reasons/IDs/top bid/ask/known sorted levels and
  age/uncertainty; never resurrect READY after a gap without rebuild.
* Market store accepts new immutable kind DEPTH, no destructive DB migration;
  append wire snapshot/diff plus small derived top-of-book state, never a full
  book on every diff. Existing P2-A snapshots/hash/schema remain unchanged.
* `MarketStreamReader(store, client, universe, topics, transport=None,
  background=True)` verifies own stored immutable universe and selected topics
  before network. `start()` validates clock, sends one SUBSCRIBE, matches ACK
  and retains <=64 pre-ACK events with their original receipt clocks. No blind
  subscribe retry. Every event must match its combined stream/symbol. BookTicker
  retains null event time and OBSERVED_RESPONSE; strictly increasing u within
  session, duplicates do not refresh receipt and conflicts/regression close the
  symbol. Depth uses the bounded book above; persist received diffs before return.
  Per-event processing lag >250ms prevents readiness. Emit explicit invalid quote
  rows at session start/disconnect/stop, with universe/session provenance so
  P2-A feature/valuation consumers cannot resurrect earlier good quotes.
  Unchanged symbols retain their own receipt, not another symbol's freshness.
  `install_depth(symbol, observation)` is explicitly caller/supervisor driven;
  snapshot is persisted before derived state. `read_once()`, `health()`, `stop()`.
  Session rotation/reconnect resets quote update IDs/L2. Health is connectivity, per-pair
  quote freshness and per-deep-book readiness; it never certifies account/entry.
* `MarketStreamSupervisor(store, client, universe, deep_symbols=(),
  transport_factory=None, background=False)`: own library lifecycle, default
  manual tick; background opt-in launches only its own coordinator/readers.
  `start/tick/stop/health`, shared local connection quota/backoff/rotation and
  explicit per-shard failures. Rest snapshots/clock probes owned by one
  coordinator, with shared client reserve; no concurrent external REST use.
  Warm clock before readers; resync on lease expiry. A failed reader invalidates
  its own subscribed rows and rebuilds depth from new diffs+REST. Snapshot retries
  bounded by backoff and quota. Stop joins owned threads only; nothing controls
  old bot workers. Reopen never inherits live socket/L2 readiness from SQLite.
  This slice freezes a source-bound universe; stale/new metadata requires a new
  supervisor/plan, not an automatic permission override. No permanently deployed
  daemon or endpoint/port is introduced. `enabled=False` starts no I/O.

### P2-B1 acceptance before code

| ID | Required focused scenario |
|---|---|
| P2B1-01 | Demo/TLS/header/path/method guards, topic caps, credential-free depth/weights/reserve and combined ACK. |
| P2B1-02 | Full/non-USDT/public-null/denied membership, deterministic shards, future/tampered source and deep outside source rejected. |
| P2B1-03 | Snapshot+buffer bridge, stale/duplicate/overlap, zero deletion, exact replacement, gap/conflict/overflow/crossed/empty rebuild; partial-depth rejection. |
| P2B1-04 | Valid clock/event+receipt age/uncertainty, future/stale/out-of-order, null book time, processing lag; no freshness renewal from duplicates or other symbols. |
| P2B1-05 | Durable receipts/dependencies, pre-ACK receipt preservation, unmapped events, disconnect/stop invalidation, real owned thread shutdown/restart without inherited readiness. |
| P2B1-06 | Shard supervision, retry budget/backoff/rotation, disable/no I/O, depth recovery and explicit missing coverage. |

Verification: focused RED before implementation, full own-app regressions,
27 repository regressions, full/staged Truth Harness and selective commit/push.
Bounded public Demo smoke uses BTC/ETH/SOL streaming+L2 only, no account keys,
orders or profitability claim. Report coverage/latency and any FAIL explicitly.
Rollback: stop/disable only the new library readers; persisted rows invalidate
their own readiness, financial journal/account protection remain separate.

P2-B1 review cases: quote sequence/schema failures quarantine that symbol for
the remainder of the session (not reset the high watermark). A supervisor sees
this as reconnect-needed. Runtime disconnections get the same exponential
backoff as handshake failures, applied once per failed session. Quota includes
every connection attempt. Snapshot receipt UTC/monotonic values must not be
future or reversed; clock/metadata failures are not papered over by fresh Pong.
Health exposes bounded processing/event/receipt age diagnostics. Heartbeat is
checked even under continuous data, not only on recv timeout. A restored source
restarts BUFFERING and cannot inherit sequence certification from persisted data.
Book mutation/snapshot/view are serialized atomically, including when a producer
temporarily deletes a side's last level before inserting its replacement in the
same valid diff. A consumer cannot see that transient half-applied state.
Documented UTF-8 symbol names are accepted without silently shrinking discovery;
letters/numbers/underscore only, explicit source spelling and lowercase topic
mapping. Separators/control characters remain forbidden. Mixed-case symbols use
the actual source symbol, not reconstruction by uppercasing a topic.
Coordinator stop is checked before each connection/snapshot and immediately
after a bounded REST request; pending stop prevents subsequent network work.

### P2-B1 delivery evidence

Implemented after 31 intended missing-module/method errors RED. Additional
reproductions preceded repairs for sequence quarantine, runtime backoff,
continuous-message heartbeat, future monotonic snapshot receipt, processing lag
diagnostics, atomic side replacement, UTF-8 source mapping, clock-health gating
and pending-stop I/O barrier. **46 focused stream tests / 338 total app GREEN**;
27 repository spec/harness regressions and full Truth Harness PASS. Real
threads, barriers and immutable SQLite receipts are exercised; no skips/xfail.

Two bounded public 20s Demo stream samples used BTC/ETH/SOL only, with an
explicitly stored PUBLIC_DIAGNOSTIC_SAMPLE and permitted N=null. All three
books reached READY in both runs; first final point was 2/3 READY, second 3/3
READY and 3/3 fresh observed quotes. The last sample had valid measured clock
uncertainty 148.522ms, receipt ages 85/282/298ms, and last local post-recv
processing lag 0.1672ms. This last local measurement is not network/event latency
or a percentile/SLA. One initial engineering sample request was API_REJECTED
because JSON symbols contained spaces; compact JSON corrected the diagnostic
request, consistent with the existing client. No account keys/balances/orders.
327 accumulated DEPTH records across the two sessions remained in own ignored
runtime. Readers/store/transport were explicitly closed after each bounded run.

No full-market sustained coverage, account certification, historical anchor,
profitability or automatic trading release is established. Frozen metadata
expires after 15min; P2-B2 must add versioned refresh/candle scheduling, bound
full-universe CPU/publication lag, account/reference-price filters and certified
valuation/reconciliation before P3 enablement. Offline entrypoint remains
credential-free and starts no service. Staged verification/commit/push complete
the release cycle; runtime evidence is not a source artifact.

## P2-B2: refresh, closed candles and full-market observation runtime

Registered 2026-10-11. Read-only implementation slice. Signed myFilters,
reference-price execution previews and certified account/financial as-of cuts
remain the next P2-C/P1 integration gate; they cannot be established by public
data or synthetic balances. No decisions/orders/learning/daemon auto-start.

The current full catalog costs 86.53s validation/storage; repeated copies and
per-message WAL commits prevent credible all-market latency claims. Reduce
these costs without shrinking registered discovery, losing provenance, changing
money arithmetic or silently reusing stale inputs. Official UTC 1m kline
protocol rechecked on 2026-10-11 in the stream source linked above (x closure,
t/T/E, OHLCV/counts). Keep 512 topics/shard and 20 local attempts/5min.

### Source/performance contracts

* Universe format `universe-v2`: normalized row fields plus `metadata_extra`
  and original mapped-field presence; no triple copies of permission arrays.
  `symbol_source(row)` reconstructs the exact original symbol object, including
  unknown flags/fields and legacy v1 raw rows. Observation retains raw-body and
  canonical response hashes, all non-symbol response fields and original symbol
  order (including identical duplicates). `universe_source(snapshot)` reconstructs
  the original canonical response; removal tombstones are excluded from that
  source order. Reconstructed hash must equal recorded source SHA. Every filter,
  flag and timestamp remains accessible; price/step precision is unchanged.
* Market store canonicalization happens once per append; resulting ID/source/
  content hashes match the original journal canonical JSON convention exactly.
  Financial/journal/isolation code fingerprints remain untouched. No destructive
  migration, modified historical payloads or dropped verification. `as_of`
  queries only the latest eligible row. `validate_snapshot(value,kind,cutoff)`
  checks own immutable full envelope/content SHA without reparsing all history;
  mutation/future/wrong kind/namespace closes it. Stale view compatibility remains
  only in P2-A availability, not permission to start a new current runtime.
* `store.batch()` serializes an atomic group of <=32 received frames/reader
  iteration in one WAL/FULL transaction. A failed group rolls back all new rows;
  no live quote/book health is exposed before commit (reader/store locks held).
  Per-frame actual receipts/hashes/dependencies remain distinct. Explicit group
  rollback is recorded as a stream failure; no counterfeit READY or lost event
  reported as success. Reopen verifies all primary snapshots. Batch groups are
  bounded, not silent sampling/drop/thinning. Historical publication certification
  still belongs to P2-C; raw receipt convention is kept explicit.
* Readers validate the full source once, then hold copied lightweight membership
  and immutable source ID/receipt references. Start verifies that own stored
  reference again; caller mutation cannot alter reader membership. Supervisors
  retain one verified own full source for constructing readers, no full metadata
  deepcopy per shard. Source cache never supplies a future snapshot.

### Closed-candle streams

* `stream_plan(..., include_candles=False)` and supervisor flag preserve B1
  defaults. True adds `<symbol>@kline_1m` for every subscribed pair. No 24h
  ticker or in-progress bar is substituted for a closed minute.
* `MarketDataService.record_closed_kline(payload, received_at_ms,
  received_monotonic_ns)` validates source symbol, interval=1m, bool x, aligned
  t/T, E, OHLCV/counts and valid clock. x=false yields no feature candle; it
  cannot refresh the latest close. x=true requires E>T and conservative server
  cutoff>T. Future/malformed/unknown schema leaves a per-symbol incident and
  unavailable feature until valid recovery; no blind receipt backdating.
  REST and WS candle revisions compare normalized economic fields (unused wire
  field B/spelling of trailing zeros cannot cause false conflict). A genuinely
  changed closed bar remains quarantined. Every saved close has its wire source,
  actual receipts, symbol/interval/event timestamp and original hash.
* Reader batches bounded recv frames and retains real receipts before parsing.
  Pre-ACK buffering retains B1 <=64/lag guard. Closed bars persist in CANDLES;
  open packet counts remain diagnostics and are not PIT forecasting inputs.
  Ready features still need 3 consecutive closed minutes and fresh quotes.

### Coordinator and evidence

* `MarketDataRuntime(store,client, account_provider=None, deep_symbols=(),
  transport_factory=None, enabled=True, background=False)`, own lifecycle
  `start/tick/stop/health`. One coordinator exclusively owns REST/clock. Startup
  fetches all categories, warms clock, creates book+closed-candle shards; default
  manual tick, background explicitly opt-in. Offline entrypoint stays offline.
* Refresh metadata every 10min (hard freshness 15min), backoff failed refresh
  >=30s. Account provider, when explicitly supplied, is own fresh P1 observation
  only; absent provider gives public reference/permitted N=null. On successful
  refresh record immutable new source, stop old readers and rebuild source-bound
  shards under the SAME rolling connection-attempt quota. Listings/status/removal
  change future membership only. Failed refresh preserves prior denominator as
  stale and closes qualification; no blind permission expansion.
* Refresh clock every 30s using a separate candidate DemoClock and bounded public
  probes; publish valid sample atomically without exposing transient None to
  healthy readers. A failed refresh is reported, valid old clock remains usable
  only within its original 60s lease; runtime qualification degrades until recovery.
* Every 3s bounded REST all-book poll supplies an explicit OBSERVED_RESPONSE
  fallback for unchanged books; its actual response interval is not fabricated
  stream/event freshness. Market weight reserve remains 200. Closed-history
  bootstrap/backfill rotates fairly, <=2 symbols/tick and <=3 bars per request;
  stream closes naturally warm other pairs. Missing history/gaps stay pending.
  No quiet coin is discarded to make a coverage percentage look good.
* Evidence contains discovered/active/subscribed/permitted/quote-ready/candle-
  ready n/N, status exclusions, session/gap faults, processing-lag count/limit
  and p50/p95/max (bounded recent window marked as such), source/clock conventions,
  elapsed observation window and pipeline status. Exact current public denominator
  is explicit; public readiness is distinct from permitted-account readiness.
  Quote lookup can incrementally index committed own receipts; past cutoffs
  always reconstruct the eligible prefix and newer invalid input never resurrects
  an older good quote. Snapshot/namespace/hash verification remains authoritative.
* Bounded full-public-market smoke: complete all-category discovery, all active
  book+kline topics, no keys, 30–60s observation after startup, report actual
  coverage/faults/lag and any incomplete warmup. A short smoke cannot qualify a
  sustained production SLA or account/financial readiness. Keep unsuccessful
  trials and causes. Stop only new workers and retain own evidence.

### P2-B2 focused acceptance before code

| ID | Required scenario |
|---|---|
| P2B2-01 | Lossless compact metadata, original source order/duplicates/unknown fields, tombstones, legacy hash/reopen parity and tamper/future reference refusal. |
| P2B2-02 | Atomic batch commit/rollback/receipt separation, cache/past-prefix correctness and invalid newer input. |
| P2B2-03 | All-pair kline topics/limits/non-USDT, valid/open/future/malformed/duplicate/revised closed bars and REST/WS economic parity. |
| P2B2-04 | Metadata rotation/failure/expiry/status/listing, shared connection budget, clock candidate swap/failure and disabled/stop lifecycle. |
| P2B2-05 | Fair bounded backfill, explicit REST fallback, quota/staleness, separate unknown account denominator, lag/fault/window evidence. |

SPEC → RED → implementation → full own-app/regressions → public smoke →
full/staged Truth Harness → selective commit/push. Rollback disables/stops only
the new market runtime, invalidates current readiness, preserves immutable data
and never cancels exchange protection or changes the old bot. This slice makes
no trading-policy or profit claim; maximum-history alpha gates remain mandatory
for later policy changes.

P2-B2 review boundaries: tombstones preserve the last actual symbol source status;
current removed-row counts are separate from the current discovered denominator.
UTC/event/receipt comparisons use a fixed Decimal context; future receipts and
future events are rejected even under a caller's precision=1. A malformed close
inside a failed batch leaves a durable per-symbol failure after rollback, so old
closed history cannot silently stay qualified. Clock failures/backoff do not make
account readiness true. Failed stop/replan never enables orders.

Full-market trial `1791722289478026500` failed: 0/6 connected shards, all
WS_PROCESSING_LAG, despite pipeline OBSERVING. Keep that failure evidence.
Supervisor validates the full caller envelope once and copies a private compact
membership projection. Its readers use that owned projection and revalidate the
stored source reference; direct public Reader callers still require full envelope
validation. Runtime REST polling uses its owned projection after reference/TTL
validation, not a repeated multi-megabyte hash while readers are receiving.
Stream coverage below planned shards yields STREAMS_DEGRADED (even if REST
quotes are complete), recovers only when all planned shards connect; stopped
runtime reports STOPPED. Clock/metadata failures take priority. Lag threshold
remains 250ms; full population and original receipts remain unchanged. Internal
projection helpers are private and cannot accept caller-supplied unverified data
through the public polling/reader interfaces.

Standalone append canonicalization/hash work occurs outside the writer lock;
only immutable dependency lookup and SQLite insert/commit hold it. Existing
canonical hashes, duplicate insert results and batch atomicity remain exact.
A concurrent reference lookup must complete while standalone serialization is
paused. Batch serialization stays within its bounded atomic group. Public
diagnostic quote polling preserves P2-A's independent metadata/quote age contract;
runtime TTL prevents subscribing/using expired membership for qualification.

Trial `1791723226477284600` also failed with 0/6 shards; latency maxima
251..400ms. A writer cannot repeatedly reacquire ahead of already waiting
readers. Store uses a FIFO reentrant lock: nested owned operations keep atomic
groups, contenders acquire in arrival order. Verify deterministic queue order,
ownership and reentrancy; this fairness contract does not itself prove a SLA.

Runtime history counts exclude any symbol whose latest eligible candle attempt
failed, until a newer valid collection supersedes it. Bulk failure index uses
committed receipt-prefix ordering, reconstructs past cutoffs, never publishes
uncommitted data and resets on rollback. Successful clock refresh clears only
CLOCK_REFRESH_FAILED; it cannot clear metadata/runtime failure. Reader and tick
recent latency evidence includes p50/p95/max and n/4096, never a sustained SLA.

Trial `1791723785457912400` retained FAIL and measured writer-lock waits up
to 1528ms. Deepcopy of current quote/candle indices happens after the lock is
released: take an atomic shallow cut of internally immutable record objects
(copy each candle-time map), then give callers a detached deep copy. Caller
mutation cannot corrupt the cache. Prefix fallback remains under the original
atomic read contract. Tests pause deepcopy and require concurrent source lookup
to finish; benchmark still must validate full-market latency, not this proxy.
Rejected future/invalid receipt incidents use the observer's actual current
wall/monotonic clocks; an invalid claimed future time cannot postpone quarantine.

### P2-B2 delivery and remaining qualification

36 focused cases / 374 total app GREEN, with recorded intended RED before
each interface/failure repair. Root spec/harness regressions and mechanical
Truth Harness are separate gates, not proof of profitable/production operation.

Full public trials (own runtime evidence, no keys/orders):

| Trial ID | Discovered | Active public | Final connected / planned | Result |
|---|---:|---:|---:|---|
| 1791722289478026500 | 3727 | 1375 | 0/6 | FAIL, WS_PROCESSING_LAG; original false OBSERVING reproduced |
| 1791723226477284600 | 3727 | 1375 | 0/6 | FAIL after compact-source/hot-path repair; status corrected |
| 1791723538395186000 | 3727 | 1375 | 0/6 | FAIL after FIFO lock |
| 1791723785457912400 | 3727 | 1375 | 0/6 | FAIL with lock timing diagnostics |
| 1791724080766903200 | 3727 | 1375 | 0/6 | FAIL after detached index copying; window 48.47s |

All reconstructed original source hashes matched. Point REST readiness reached
1375/1375, but final last trial quote/history/depth readiness was 0/1375,
0/1375 and 0/3. This is no sustained coverage claim. Actual shard lag maxima
crossed the unchanged 250ms guard. Lock profiling measured >1s waits/holds;
remaining parsing/commit contention needs operation-level profiling and a
bounded publication/writer design before qualification, without dropping events,
thinning the population or backdating receipts. Short trials also cannot prove
a continuous SLA. All own workers stopped; failed evidence remains local.
P2-B2 is a diagnostic foundation, not completed production enablement. P2-C
account/financial integration, actual own credentials, filters/anchors and
reconcile stay pending; ORDER_SEND remains disabled. No policy/alpha change.

## P2-B2 read/write contention repair (2026-10-11)

Baseline full-public trial `1791725443819122200` retained FAIL, 0/6 shards.
Operation-level profiling found `quote_index` holding the writer lock for
1013ms in the observation phase. Separate diagnostic reads from the single
WAL/FULL writer; do not relax the 250ms guard, change prices/hashes, sample
events, shrink full discovery or weaken validation. No order/policy change.

* Open an app-local SQLite read-only/query-only connection after primary
  identity/hash verification. Serialize readers with a separate reentrant lock
  and pin one WAL read transaction per public read operation. JSON parsing,
  current caches, historical prefix reconstruction and full-envelope hashing
  must not hold the writer lock or block an unrelated append. No credentials,
  external database access, second writer or financial fingerprint changes.
* Outside the owner transaction, get/records/as_of/reference/index queries see
  only committed data, including while another thread has a pending write or
  rollback. Cache cursors advance only across committed primary sequences;
  rollback cannot publish uncommitted cache entries. Future available_ms still
  excludes inputs from an earlier cutoff. Current invalid inputs supersede
  earlier good ones; detached caller mutation remains harmless.
* The actual writer owner retains read-your-own-writes for candle-conflict and
  dependency checks inside an atomic batch; those reads use the writer connection
  and never acquire the diagnostic reader lock. Own quote/failure indices reject
  publication inside that batch. There must be no writer→reader lock cycle.
* Add derived `(kind,seq)` and partial candle-symbol/sequence indexes, including
  on reopening existing stores, without modifying historical payloads/IDs.
  Within a batch, symbol-specific candle revision checks query only that symbol
  through the partial index. Primary hash verification still runs on reopen.
* Closing stops both connections under own lifecycle; double close is harmless.
  Database remains WAL/FULL, one own writer lease, and immutable evidence.

Acceptance P2B2-06 before code: paused read parsing/hashing allows concurrent
append; read while a writer is paused excludes uncommitted data; rollback/cache
and future-prefix correctness; read-only connection rejects writes; owner sees
its staged candle revision; symbol query uses the derived index; reopen adds
indexes and retains old snapshots/hashes; namespace/tamper and lifecycle guards.
Then full app/regression/harness checks and a fresh full-public network trial.
Report actual denominator, observation duration, lag and faults. A short success
still does not certify continuous SLA/account/PnL. Rollback stops only this own
runtime; failed trials and immutable sources remain. Commit/push intended files.

Trial `1791725909959519500` retained FAIL (0/6). WAL separation removed
diagnostic `quote_index` from the writer-lock holds, but reader batches still
held it for 331ms. Bound each reader transaction to <=8 frames and recv draining
to <=2ms (previous 32/20ms). This changes grouping, not topics, receipts, validation
or event retention; excess queued frames are consumed next iteration, never
discarded. Confirm all packet IDs persist across successive bounded groups and
the time cap works independently of the count cap. Candle-symbol lookup uses an
explicit derived-index hint, verified from the actual traced query plan rather
than an unrelated manually written SQL query; cutoff/source checks remain.

Trial `1791726183491404800` retained FAIL (final 1/6 connected). Small groups
still held the writer during parsing/validation/serialization. Add a prepared
reader group: process/normalize/hash <=8 packets under the reader's health lock,
without holding the SQLite writer. Commit all immutable prepared snapshots in
one short WAL/FULL transaction before health is exposed. Failed preparation or
commit publishes none of the group; existing durable failure handling remains.
Every original receipt/hash/dependency is retained, no premature READY or drops.
Use one serialized read-only WAL connection per thread, registered and closed
by this store, so slow diagnostic parsing cannot hold up the receiver's source
checks. Keep writer-owner read-your-own-writes separate. Pinned read snapshots
and primary hash verification remain mandatory. Never access external DBs.

Prepared append must match normal snapshot IDs/source SHA/canonical money,
resolve preceding prepared dependencies, give callers detached values and
return existing records on exact retries. Candle checks overlay earlier prepared
bars on the symbol's committed prefix. Capture the committed candle sequence
in that same read snapshot and recheck it under the writer at commit: a
concurrent candle write aborts the group as CANDLE_SOURCE_CHANGED and records a
symbol failure instead of silently overwriting a revision. No nested prepare
inside an owned write batch. Rollback preserves primary rows and caches.

Additional P2B2-06 tests before code: concurrent readers do not serialize behind
another thread's parsing; pinned snapshot excludes a mid-read commit; prepared
serialization permits unrelated append; failed group/dependency/hash/retry and
caller-mutation parity; staged candle conflict and concurrent-source guard;
actual stream uses preparation outside writer; all owned read connections close.

Per-thread read transactions share derived cache cursors only under a separate
index lock; one reader cannot interleave cache mutation with another's pinned
cut. This index lock never blocks normal source reads or the prepared receiver
path. A stream commit candle guard failure is explicitly CANDLE_SOURCE_CHANGED
in both session diagnostics and the durable per-symbol failure, not a misleading
schema-error label. Validate both with reproductions before repair.
An exiting owned background receiver releases its thread's read connection and
registry entry; reconnects cannot accumulate obsolete reader connections. This
release is idempotent, does not close other threads' readers or the writer, and
store close still owns final cleanup. Test both explicit release and real reader
thread stop with a processed packet before wiring lifecycle code.

### P2-B2 contention repair delivery

26 new focused cases / 400 total app GREEN. Initial WAL scenarios reproduced
10 failures/1 missing-connection error before code (one existing owner-candle
contract correctly stayed GREEN). Group count/time cap failures, missing prepared
API, shared-reader contention, cache interleaving, exact candle guard reason and
connection cleanup were RED before their repairs. The pinned snapshot scenario
was legitimately GREEN after the first WAL implementation; no manufactured RED.
The index-plan test now inspects the actual traced query rather than an unrelated
manual SQL; the same expected partial-index assertion first remained RED and
passed after the explicit query hint. No skips/xfail or weakened assertions.

Own full-public trials retain 3727 discovered / 1375 active / 6 planned shards
and matched original source hashes; no keys or orders:

| Trial | Change/evidence | Final connected | Verdict |
|---|---|---:|---|
| 1791725443819122200 | Baseline: quote_index writer hold 1013ms | 0/6 | FAIL |
| 1791725909959519500 | WAL reads remove diagnostic writer holds; batch hold 331ms | 0/6 | FAIL |
| 1791726183491404800 | <=8 frames/2ms groups and indexed candle check | 1/6 | FAIL |
| 1791729042494407500 | Prepared writes/per-thread reads/cache and lifecycle repair | 1/6 | FAIL |
| 1791729295847478400 | SQL profiling: observed commit up to 400.6265ms; sample reached 4/6 | 0/6 | FAIL |

Last final quote coverage was 1375/1375, closed history 14/1375; this is a point
snapshot with incomplete warmup and failed continuous streams, not an SLA.
Evidence is retained in app-local ignored runtime and all own workers stopped.
Separate actual SQL time from mutex time and startup from observation; timing
trials are diagnostics, not a controlled uplift/profit comparison.

Next P2-B2 gate is storage/commit/checkpoint latency. Keep synchronous=FULL and
250ms; do not conceal pauses, drop packets or use async uncommitted READY. Moving
checkpoint off the writer is a new spec/test-first slice, not part of this delivery.
The workstation Python reports SQLite **3.50.4**. Current code has one writable
connection and read-only readers. A concurrent write/checkpoint design must first
use a fixed SQLite runtime: [official WAL-reset advisory](https://www.sqlite.org/wal.html#the_wal_reset_bug)
identifies fixes in 3.50.7/3.44.6 backports and 3.51.3+. Do not add a concurrent
checkpointer on 3.50.4. [Official WAL performance documentation](https://www.sqlite.org/wal.html#performance_considerations)
explains automatic checkpoint pauses; the local measured commit latency alone
does not prove that every pause was checkpoint rather than disk sync/scheduling.
P2-C account/financial cuts and trading remain pending/disabled.
