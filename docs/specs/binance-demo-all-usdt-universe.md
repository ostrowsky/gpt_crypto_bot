# All-USDT Demo monitoring and execution scope

Date:2026-10-28. Status: implemented; live expanded-scope canary pending.
User authorizes monitoring and trading all Binance Spot Demo coins against
USDT in the independent app. Production Binance, legacy workers, leverage and
separate per-symbol budgets remain outside scope. Parent:P0/P1/P2/P3/P5 and
the execution-qualification pilot; TH-01..12 apply.

## Contract

USD-01: discover full exchangeInfo with all registered permission categories;
intersect each permissionSets AND-of-ORs with fresh account permissions.
Monitor every account-permitted TRADING Spot quoteAsset=USDT pair; never infer
the quote by suffix alone. Preserve excluded/removed pairs and reason counts.
Refresh metadata15min and all-market bookTicker snapshot30s, explicit receipt
clocks/coverage. Missing/bad/zero/crossed quotes block the affected pair.
Native metadata contains newer permission alternatives absent from the local
enum registry. An OR group already satisfied by a verified account permission
is TRUE regardless of irrelevant unknown alternatives. An unresolved unknown
AND group remains UNKNOWN/blocked; do not add guessed permission values or
grant account permissions by observing symbol metadata. Record unknown extras.
Compare native account updateTime against the validated exchange clock, while
receipt freshness uses local monotonic/wall receipt clocks. A measured server
offset must not make a freshly received account look future-dated or stale.
Dynamic listings enter monitoring; delistings block new entries but owned exits
remain possible. No static BTC/ETH shortlist masquerades as all-market coverage.
Retrieving an already acknowledged immutable action is reconciliation, not a
new entry: return its cached source response even after scope revocation/expiry.
Reusing an action with changed symbol/parameters still fails; an unknown action
still never resends. Scope checks apply before every new mutation.
Native Unicode alphanumeric instrument identifiers are supported (including
CJK listings); do not silently restrict the universe to ASCII tickers. Require
exact metadata baseAsset+USDT identity, uppercase where applicable, no separators
or control characters, native signing/URL UTF-8 and per-instrument permission.

USD-02: a single portfolio owns all (symbol,order_id)/(symbol,trade_id) keys;
exchange numeric IDs may collide between symbols. Asset deltas and commissions
include every owned symbol; selling seed inventory or charging fees twice is
forbidden. Preserve old BTC order IDs, seed, balances, attempt count, loss stop,
cadence and pilot expiry. Native per-pair filters, MARKET/OCO capabilities,
notional/lot/percent bounds, healthy private delivery and source freshness gate
each new order. Unsupported pairs stay monitored with explicit entry blockers.
Only one live position across the account;30 USDT/order,100 pilot budget,
32 gross exposure,2 aggregate dust,1 loss/drawdown entry stop and8 total
attempts per original24h window are account-wide. Every supported USDT pair
can become an entry candidate; no simultaneous blanket purchases.

USD-03: use cheap full-market quotes and bounded cached candles, not signed
account/fill queries per monitored pair. Cache terminal order fills by composite
key. Refresh owned positions before scanning/backfilling and reserve request
headroom for protection/reconciliation. A missing scan candidate is WAIT,
not a malformed BTC snapshot or application crash. Preserve audit records and
per-pair selection/block reasons, mode and exact monitored/eligible counts.
Use symbol-scoped openOrders in the hot position snapshot; the mandatory
preflight still checks whole-account openOrders/order lists before every BUY.
Do not spend the80-weight all-account query repeatedly between BUY and OCO.
When exposure is pending/open, serve owned reconciliation/protection before
bulk monitoring. Its selected/owned quote reads use the existing1200 transport
ceiling and reserved headroom; bulk monitoring keeps its1000 threshold. Do not
raise either ceiling or let full-catalog/candle refresh block an owned exit.

USD-04: fix the sticky native-weight floor without raising1000 market/1200
transport caps. Each native weight observation lives at most60s from its OWN
receipt. Keep a conservative envelope of unexpired observed weights plus
own reservations after each receipt. Lower/new headers cannot extend an old
high floor forever; within a receipt minute the floor must not disappear early.
REST reads, order-test and matching calls share this mechanism. Unknown
capacity/429/418 prevents entries and pauses lower-priority collection. A rate
pause in flat IDLE is recoverable and does not permanently kill the worker;
unknown mutations/actual exposure still require reconciliation.

USD-05: register candidate selection/replay before inspecting results. Before
enabling expanded execution collect the maximum available15m history for the
actual allowed universe from startTime0 to a frozen closed-bar cutoff, resuming
pages with hashes, gaps and exact per-symbol coverage. Do not call a recent
sample or current-universe history point-in-time certified. Keep historical
membership/receipt/order-book unknowns visible; reference OHLC outcomes are
SIMULATION and cannot prove profit. Preserve negative results and financial
PENDING. Ordinary strategy promotion remains disabled. Any still-unvalidated
pair stays monitored but cannot enter. Test all declared gates, recovery,
symbol-ID collisions and multiple assets before actual bounded canary.
This extension retains EXECUTION_QUALIFICATION mode: deterministic round-robin
among source-validated, filter/quote-ready USDT instruments, sorted by previous
attempt count then symbol. It does not pretend this is a learned profitable
selector. No momentum/alpha policy is silently introduced. Each instrument's
earliest-to-frozen-cutoff reference must finish before its first canary entry;
all-market historical collection progresses without starving owned exits.
First actual probe of each newly supported instrument uses60s, remaining pilot
holds remain900s; old/global attempt and expiry limits never reset.
Archive uses own SQLite WAL/FULL with zlib-compressed JSON pages, per-page
checksums/coverage and resume cursors; partial history is never labelled
MAXIMUM_AVAILABLE_FETCH_COMPLETE. Interrupted transactions roll back and
resume from the last durable page; no silent rewrite of historical evidence.
Native shortened historical candles (e.g. exchange interruptions) remain in the
hashed raw archive with their actual close timestamps. Count/quarantine them
from replay and all dependent feature windows; never invent a full candle or
discard an entire instrument because one historical candle was shortened.

USD-02 identifier detail: persist public instrument/fee-asset identifiers as
explicit codepoint arrays when recording the scoped pilot journal, then decode
losslessly for operation. This avoids broad credential-redaction words in valid
symbols (e.g. COOKIEUSDT) without relaxing secret redaction or encoding arbitrary
payloads/credentials. Composite ownership also applies to cancel/list recovery.

USD-06: write-ahead audit, scoped permission probe and unknown no-resend apply
to any pair; immutable client IDs encode symbol scope. Matching endpoint/host
allowlist remains exact Demo-only. Rollback stops new entries, keeps protection
and all historical evidence, and reconciles owned exposure. No budget/seed
reset by changing symbols or restarting. Candidate/export includes symbol and
all feature/decision/label timestamps. Only mature actual outcomes enter actual
learning handoff; no automatic promotion or learned risk increase.
Match private and REST fills by symbol/trade/order/side/commission asset as
well as exact quantities. Store feature receipt availability and reject a
feature received after its decision even if the candle close is earlier.
Once exposure exists, reuse its frozen features for protection/exit snapshots;
irrelevant new candle availability must not block a source-backed owned exit.
Read failures publish fresh PAUSED/UNKNOWN monitoring status, not a stale
apparently running report. Preserve last native evidence and block entry.
Recheck spread<=20bps and both displayed sides>=30USDT after the fresh
preflight quote, not only in the30s scan. Before any mutation, pair-local
filter/feature/quote failures quarantine that pair for60s and allow scanning
others. Never clear unknown sends, balance/ownership failures, risk stops or
errors on an open position by treating them as a pair-local data pause.

## Verification

SPEC -> focused RED -> minimal code -> GREEN/regressions -> final diff review,
git diff --check -> staged Truth Harness -> commit/push -> live read-only
coverage and one bounded protected execution if all prerequisites pass.
Runtime credentials, balances, native logs, reports and history stay uncommitted.
The live native audit has an8GiB bounded storage allowance for full metadata
and quote payloads across the24h pilot; this does not change any trading/API
limit or authorize deleting evidence. The pilot authorization ledger stores
only monitored USDT instruments and a hash reference to the complete catalogue,
avoiding repeated copies of unrelated markets on every archive-ready update.
Full Harness currently FAIL TH-11 (legacy replay source fingerprint); do not
translate it into a profit/production pass. Public sources:official Binance
exchangeInfo, permissions, klines, filters and rate-limit documentation.

## Test-first and native evidence

Initial modules reproduced15 scope,14 archive and7 controller missing-interface
RED scenarios before implementation. Added genuine RED for sticky weight TTL,
native unknown OR alternatives, exchange-clock updateTime, Unicode signing,
shortened native candles, selected-history verification/tamper, received-after-
decision features, private fee/order/side mismatch, refreshed spread, pair
quarantine, native ACK identifiers, revoked-scope cached ACK, critical request
headroom and authorization-ledger size. Fixture corrections (closed audit,
missing native symbol/market client and checking valuation at its proper API)
were not counted as implementation RED and did not weaken assertions.

Final54 all-USDT focused GREEN (3.710s),57 pilot regression GREEN and986 full
application GREEN (262.814s); root14 specification tests GREEN. Actual fresh
Demo account/catalog verified519 monitored and structurally supported USDT
instruments. Full historical source collection is in progress, per-pair verified
references only; no all519-complete or historical/PIT/alpha pass is claimed.
Native archive identified shortened2021 candles: retain raw/source hashes,
quarantine their replay windows and resume durable cursors without resetting
the frozen cohort/cutoff. Full Harness remains FAIL TH-11 due the unchanged
legacy replay source fingerprint. Staged change validation and live all-USDT
monitoring/protected-canary evidence are separate release steps.
