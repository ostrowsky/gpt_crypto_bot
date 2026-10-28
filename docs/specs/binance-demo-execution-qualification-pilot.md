# Demo execution qualification pilot and learning handoff

Date: 2026-10-28. Status: actual protected probe closed; bounded collection only. User priority: start Demo
execution today and obtain first actual results for a learning integration test
on 2026-10-29 (Europe/Budapest). Parent: P0/P1/P3/P5 program contracts.

Scope update2026-10-28: the subsequent user-authorized
[all-USDT extension](binance-demo-all-usdt-universe.md) replaces the initial
BTC-only candidate universe below while preserving the same original seed,
account-wide budget/attempt limits, stop and expiry. BTC-only statements below
describe the initial registration and evidence, not a second parallel budget.

## Scope and authority

Separate application only, dedicated Binance Spot Demo account already attested.
Never production/testnet fallback or other-app processes/state. This is a bounded
EXECUTION_QUALIFICATION arm, not a released profit strategy, policy uplift or
permission to relax ordinary entry/promotion gates. Existing PENDING financial
history, old stream gaps, all-market latency FAIL and legacy TH-11 FAIL remain.
The user has authorized actual Demo purchases, sales and TP/SL. One bootstrap
probe may establish actual event delivery; it is limited by the controls below.
No normal-strategy entries become permitted from this technical probe.

Register before results: BTCUSDT only; budget100 USDT from existing free USDT,
max BUY quote30 USDT, one position, max gross owned exposure32 USDT including
dust, aggregate dust cap2 USDT, max8 cycles/24h, cadence3h, daily/interval loss
entry stop1 USDT and drawdown1 USDT. Position max hold15min; the first technical
probe has max hold60s and is explicitly excluded from strategy-performance
claims. TP +0.5%, SL -0.5%; use native SELL OCO with LIMIT_MAKER TP and STOP_LOSS
market leg. No leverage, top-ups, seed USDC sales or arbitrary symbols.
Risk limits are not learned or increased automatically.

Max unprotected owned exposure32 USDT and duration10s. Protection failure or
missing delivery blocks further entries and starts bounded ownership-aware
recovery/emergency sale. Max6 recovery reads within60s; never promise an exit
while Binance is unavailable. Emergency quote slippage bound1%, one mutation per
logical action, max fee reserve0.2% base, qty floored to native lot step; excess
or unresolved dust stops entries. Exchange stops survive local process exit.

## State, transport and ownership

Persist app-owned WAL/FULL hash-chained pilot state and exclusive OS writer
lease. Bind full account HMAC, namespace, original opening manifest, pilot
configuration/code hashes, start/expiry UTC and initial balances. Restart uses
that binding, never a new budget or timestamp. Unknown prior attempts are queried
before any new send. Stop-entry marker is own `.runtime/pilot/stop-entries`;
it does not cancel protection or stop reconciliation.

Default transport remains GET-only (and existing explicit order-test). A new
opt-in Demo matching transport permits only POST order, POST orderList/oco,
DELETE owned order/list, plus required authenticated GET queries. The common
TLS helper may accept those exact pairs only under the explicit matching flag;
no generic endpoint/host passthrough, redirects, retries or credential fallback.
Sign once immediately before send; verify current clock/caps/account identity,
fresh receipts, permissions, native filters and account budget. Persist prepared
parameters/client IDs and SEND_STARTED before network I/O. Durable audit BEGIN/
END and causal decisions are mandatory. Timeout/5xx/-1006/-1007/audit-unknown
means QUERY_REQUIRED; not-found alone never authorizes a repeat mutation.
Before each new BUY use the existing opt-in, nonmatching order-test permission
probe, then refresh the account/quote/features and recheck the 30s capability.
The tested LIMIT permission scope does not certify MARKET/OCO execution; only
the actual protected probe can supply that evidence.

Pre-send filters: current exchangeInfo, status/SPOT/market/OCO types,
PRICE_FILTER, LOT_SIZE, MARKET_LOT_SIZE, NOTIONAL, percent-price-by-side using
the current average price, order/list/algo limits and known nonapplicable filters.
Unknown filters block. Verify after rounding, including both protection legs.
No floats for money. New GET avgPrice/orderList is read-only and explicitly
whitelisted. Native server rejection is preserved, not repaired by widening risk.

Only source-verified client/exchange IDs created under this pilot own fills.
REST/WS key dedup and actual commissions determine net base. Sell only this
owned quantity (minus base fee reserve), never seed inventory. Initial foreign
orders/locks or unrelated balance changes block entries. Protect before waiting
for private-delivery qualification. After first BUY require actual matching
private fill(s) and REST cumulative/fill/balance agreement; quiet socket is not
delivery proof. Missing agreement kills new entries and retains incidents.

Time exit: cancel this owned OCO only, query both legs/order fills and released
balance, account for a concurrent TP/SL fill, then SELL only remaining owned
free base. No broad cancel-all and no blind second SELL. Unknown cancel/send
blocks conflicting mutations pending recovery. Partial fills are exposure;
unsupported/uncertain protection enters recovery and cannot be labelled protected.

## Evidence, replay and learning

Before first matching request collect the maximum available BTCUSDT Demo 15min
history from its actual earliest response through the frozen last closed bar.
Persist pagination bounds/hash/gaps and coverage. Causal reference replay of the
registered cadence/TP/SL/15min hold uses prior-closed features, next-bar entry,
stop-first on ambiguous OHLC, explicit 0.1% fees/side and1bp slippage/side. Report
all years/windows and failure/negative outcome honestly. This is an OHLC
reference with missing historical receipt/order-book/execution evidence; it does
not certify actual execution or alpha. Do not use favourable subsamples or tune
after viewing the result. No ordinary strategy promotion regardless of replay.
The reported aggregate is an independent-cycle reference with a per-day loss
stop, not a continuously funded portfolio or a P0 daily-PnL label.

All emitted WAIT/BLOCK/BUY/HOLD/SELL decisions and HTTP/WS interactions are
durable. Monitor catalogue availability separately; this release executes only
BTCUSDT and makes no all-market readiness claim. Status/report includes exact
mode, expiry, orders/fills/fees, protected/dust/unprotected quantities, incidents,
realized result and whole-account valuation status. Actual and simulated rows
remain distinct; first partial day is not a COMPLETE daily net PnL observation.

On each mature closed cycle export immutable feature/decision/execution/outcome
rows. Features use only closed pre-decision candles; label end/availability must
be <= fit cutoff. Fees debit once. Unknown, open, partial or conflicting rows
are excluded with reasons. Nightly learning integration runs even with too few
rows: validates manifest/provenance, chronological split, purge and no leakage;
insufficient samples -> explicit NOT_TRAINED, never fake fit/uplift. It can train
a diagnostic candidate only with adequate chronological samples. It cannot
write risk/entry configuration or promote a model. Full-day objective rewards
require certified P0 days; canary partial intervals stay integration diagnostics.

## Test-first acceptance and rollout

Rate-window recovery: do not hammer six recovery snapshots one second apart.
Keep entry halt and native protection, wait for the60s local budget window to
expire, then perform bounded ownership-aware reconciliation. OPEN polling is
at most twice per minute, with the last sleep shortened to the position deadline.
After a budget incident, automatic new entries remain killed until operator
review of fills, balance and actual delivery; a protected position is never
treated as closed merely because reads were unavailable. A normal IDLE wait
must not replay every historical myTrades request once terminal fills are cached.
An explicit operator resume may clear only that entry halt after CLOSED actual
outcome/private match, unchanged account, empty orders, no uncertain mutation,
same limits/expiry and fresh healthy snapshot; record the review decision.

PIL-01 transport pinning/default off; PIL-02 exact Decimal/filter/risk bounds;
PIL-03 immutable IDs, write-ahead state and unknown no-resend; PIL-04 filled
BUY -> verified OCO -> cancel/race-safe SELL; PIL-05 foreign seed/order refusal,
fee/dust/partial/limit/restart handling; PIL-06 native audit failures and locks;
PIL-07 causal full-history replay, ambiguous OHLC and cost accounting;
PIL-08 mature-only export, future mutation, missing rows and no auto-promotion.
Require focused RED/GREEN, regressions, root specs, diff/staged Truth Harness,
commit/push before background launch. Verify one actual bounded Demo cycle and
exchange-side protection/closure. Only then leave the same capped worker running
to collect more outcomes until expiry, with a scheduled next-day evidence review.
If technical qualification fails, stop new entries, preserve protection/recovery
and report the actual blocker instead of claiming trading is enabled.

Rollback: own stop-entry marker/config only, preserve ledger/IDs/fills and native
protection, reconcile possible sends; never delete losses or reset seed. No
profit promise; the experiment tests infrastructure and prepares valid learning
inputs. TH-01..12 apply; full Harness FAIL cannot be renamed PASS.

Primary API reference: [Binance official REST specification](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md),
[filters](https://developers.binance.com/en/docs/products/spot/filters).

## Implementation evidence, 2026-10-28

First startup found a local audited-attempt allowlist omission before network
send. Fix acceptance: the common attempt helper permits only the explicit
matching flag plus signed exact matching pairs; ordinary calls still reject.
One operator-reviewed migration may abandon the provably never-sent first BUY,
preserving its ID, attempt/cycle count and all limits/expiry. Proof requires the
known blocked adapter source hash, closed verified native audit manifest with no
matching REST_BEGIN, fresh empty orders/lists and unchanged initial balances.
Not-found or a missing audit row alone is never sufficient. Append proof/reject
events; do not erase/reuse the old ID. A fresh cycle must repeat permission and
snapshot checks and remain the60s technical probe until actual qualification.

48 focused tests GREEN after genuine missing-module and boundary RED runs:
unknown accepted BUY query/recovery, partial BUY cancel, rejected/unknown OCO,
native order count, absent permission, durable audit and causal learning export.
Maximum available archive:321918 BTCUSDT15m bars from2017-08-17,322 pages,
33 gaps, frozen last close1793177099999; SHA256
e711bfe44a276f03a3a5a70bf4b572ebf11665df1fd77f3596ad7131eb5553f8.
26673 independent reference cycles aggregate -1834.8749922054358 USDT:
NOT_ALPHA_EVIDENCE; no profitable-policy enablement.
Chronological simulation diagnostic:257451 train /64364 holdout rows;
Brier0.08904658 vs training-prevalence constant0.09939728, correct57520/64364
vs constant57580/64364. Calibration diagnostic is not portfolio improvement;
promotion_allowed=False. Actual first partial day remains PENDING, and at most8
actual cycles cannot establish reliable training or daily profitability.
Runtime archive, models, journals and balances stay uncommitted.
Full isolated application916 tests GREEN (197.862s); root14 specification tests
GREEN. Full Harness remains FAIL TH-11: stale legacy replay source fingerprint.
This does not approve portfolio profitability or clear financial PENDING gates.

Follow-up native audit repair:4 new expected RED failures before code;
51 pilot +40 native permission tests GREEN, full920 GREEN (207.257s), root14
GREEN. A fixture initially lacked the required prior RUN_END and was corrected
without weakening the closed-audit check. Actual read-only migration confirmed
unchanged seed/empty orders/lists; old buy-1 is REJECTED with durable proof,
attempt count1 and original expiry retained. No matching request was sent in
that refused attempt. Next actual startup still requires protected probe proof.

Actual probe: BUY filled, native OCO verified, owned cancel and SELL completed;
REST/private fills and per-asset balances agreed. Protection took6312ms. The
software time exit exceeded the intended60s because local request budget
prevented reads. This incident is retained; it is not a timing-SLA pass. The
closed technical outcome reached learning_handoff (1/1 eligible,0 excluded),
NOT_TRAINED and daily_objective_eligible=False. Monetary/account details stay
only in own runtime. Operator stopped only the own IDLE entry-halted worker for
update; interrupted WS/run history is not repaired into complete coverage.
No ordinary strategy, financial COMPLETE day or profitable learning is claimed.

Rate repair:3 new missing-interface RED cases,54 focused GREEN (9.237s),
full923 GREEN (215.952s), root14 GREEN. OPEN uses deadline-aware max30s polls,
budget recovery waits60s before bounded queries, and only explicit reviewed
resume can clear the retained entry halt after verified closure. No rate/risk
limit was raised. Next-day native automation reviews evidence and tests mature
learning handoff on2026-10-29 at11:45 Europe/Budapest; no model promotion.
