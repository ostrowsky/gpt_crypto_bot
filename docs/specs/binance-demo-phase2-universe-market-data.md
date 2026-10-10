# Binance Demo phase 2: complete universe and causal market data

Registered: 2026-10-10, Europe/Budapest. Status: **PLANNED; specification only**.
Priority: P2; may be developed alongside [phase 1](binance-demo-phase1-account-adapter.md).
Program: [autonomous demo trading](binance-demo-autonomous-trading-program.md).
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
`QuoteConversionGraph`. New planned paths: `files/binance_demo_universe.py`,
`files/binance_demo_market_data.py`, their focused test files. Reuse appropriate
collector/feature helpers after same-environment and receipt-clock verification.
`files/ws/binance_stream.py` currently defaults to production public streams;
do not assume its partial-depth messages are a reconstructable incremental book.

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

Raw samples and `FeatureSnapshot` use program `EventEnvelope` and add
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
