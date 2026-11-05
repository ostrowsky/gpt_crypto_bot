# MRP-1: full frozen-policy native-minute portfolio replay

2026-11-05. Priority1 after IC-1. SPEC -> RED -> CODE. TH-01..12.
Independent Demo app research; no orders, keys, training or promotion.

## Question and immutable scope

MR01: Reevaluate RDE-1-MAX519-20261029 on the SAME maximum original history,
519 current-survivor source symbols, TUNE/TEST boundaries, 21 original arms,
all five seeds, predictors/thresholds/candidate screen/features, costs, capital
and risk. Start each independent portfolio with100USDT, max debit30/min5,
one position, interval loss/drawdown entry latch1USDT, no midnight resets.
No threshold/model/seed selection from this exposed TEST. SIMULATION /
RETROSPECTIVE_REUSED_HOLDOUT; not a claim of daily income.

MR02: First reproduce all84 original financial portfolios (2segments*2costs*21)
using unchanged original15m execution. Require every original ledger, fill,
equity/day mark, fees, exposure and risk metric to match within1e-8USDT.
Exclude only already documented suppressed post-lock intent diagnostics.
Then produce168 native-minute reports for two UNKNOWN-within-minute conventions:
MINUTE_STOP_FIRST and MINUTE_TARGET_FIRST. Resolve chronological minutes first;
the convention applies ONLY when both levels first touch inside the same1m
with OPEN between them. Never override a previously resolved minute/open.
These are conditional scenarios, NOT mathematical PnL bounds. No aggregate
certified historical execution/PnL if unresolved cases remain.

MR03: Use ONLY native Demo1m for financial replay. IC-1 compatible Spot1s remains
a separately labelled data diagnostic and is NOT a Demo tick/fill identity proof.
Old RDE-2 native1s prerequisites stay incomplete; this new minute-only protocol
does not rename them PASS. Recompute continuous cash/admissions after every
changed exit; never patch historical trade profits. Newly reached execution
windows must be collected and validated, not assumed covered by IC-1.

CASH requires no market lookup. BTC_BUY_HOLD has no intrabar exit/action and
remains an UNMANAGED_15M_VALUATION_BENCHMARK with original open entry/close
marks, costs and terminal exposure. Its unchanged15m marks are not a native
minute execution certification; fetching its otherwise unused minute history
would consume the bounded collector budget without changing this benchmark.

## Account and timing

MR04: Freeze action/observation from past cached15m features BEFORE any selected/
held execution lookup. Validate original15m and every realized selected/held
1m window, including an EXIT completed at open. Missing/malformed/incompatible
fine data invalidates the objective with null PnL, no substitute symbol, skipped
day, favourable scenario or close-price fill. Native candle schema/Decimal OHLC/
counts/rounding compatibility is the IC-1 contract, never retuned after results.

MR05: Preserve next15m-open entry and adverse market slip/fees; resting open
protection precedes discretionary EXIT. For each subsequent minute, OPEN gap
stop fills at worse open, TP capped at target; otherwise apply LOW/HIGH first
touch, recording any unresolved both-touch case and chosen convention.
Stops/trails/targets use original float64 level arithmetic for this execution
ablation; data validation stays exact Decimal. Record this distinction from
IC-1's Decimal(str(level)) diagnostic, never reuse an old ordering label as fill.
Prices within each minute are converted only after complete source validation.

MR06: Decisions, learned EXIT calls, risk/accounting marks and trailing updates
remain at15m cadence. Trailing uses PRIOR15m peak throughout a full15m step;
update peak at its end. Fixed stop/TP .5%, max16bar hold, terminal MTM included.
No minute-level policy, synthetic path or intraminute high-based risk peak.
This preserves the registered risk-mark grid, not a certification of continuous
intraminute drawdown. Execution timestamps are bucket proxies with disclosed
resolution, not measured fill/touch times. Flat+risk-latched cash steps may be
fast-forwarded with SYSTEM_RISK_LOCK and full day coverage, not invented WAITs.

## Data ownership and bounded public collection

MR07: Before replay seal parent RDE/EOS/IC-1 registrations, source/spec/model/
environment/data/split/threshold hashes, full Truth Harness and resource scope.
Verify IC-1 PRC-1 journal and exact response links before importing all613 cached
minute windows. Seal own copies with original/page/receipt references; old
registrations, sources, reports, models and archives remain unchanged.

MR08: Model runtime remains original pinned Python3.11 ML stack. A separate
app-owned3.14 subprocess handles missing native data using existing public
DemoReadClient/PacedReadTransport/PublicResearchJournal. No dependency changes,
private endpoints, credential reads or archive fallback. Query only fixed
GET https://demo-api.binance.com/api/v3/klines,1m,UTC, registered519 symbols,
TUNE/TEST timestamps through original cutoff. Before every GET seal exact query
plan linked to the already-frozen decision, owner registration and requested
execution frame. Batch at most16 parent bars (240minutes), capped at scope end.

MR09: Prefetched future execution candles remain opaque to policy/features.
Do NOT require future-batch completeness to admit/validate the current frame.
Only a subsequently reached selected/held frame needs all15 ordered minutes.
Never use data-presence as a candidate screen. Conflicting overlapping minute
values fail closed; formatting-only differences may compare equal as Decimal.
Successful response bytes/query/receipt are immutable. Retain all errors and
rejected frames. Record parent-cache versus new-native provenance separately.

MR10: One owned collector invocation at a time. PRC-1 commitments before
redaction, begin/end/run/response ownership, raw bytes, closed journal and frame
aggregation are reverified after collection. Scope<=2000 native attempts across
resumes (derive from durable records), >=.3s each native attempt, <=15s native
timeout, max3 read attempts per logical request, owned subprocess timeout90s.
No redirect/alternate-host retry, no unbounded service. Collector scope expires
after7200s; failed/ambiguous worker retains evidence and invalidates replay.
Do not reset the budget/expiry on resume. New market data scope and direct
public parameters are validated before network access.

## Exact acceleration and reporting

MR11: Precompute immutable candidate-only frozen feature rows at every origin
without ANY execution/label lookup. A cached tape must return identical
candidate/forecast/vector/None semantics to the original; held noncandidate
lookups use the original frozen cache. Keep dtype/binary values, masks, hashes.
Stateless deterministic MLP actions may be memoized by exact full observation
bytes and model identity; no rounding, hidden/recurrent state or stochastic
action reuse. Golden84 parity is mandatory before accepting refined results.
Nonfixed-target native results may be reused across the two within-minute
conventions only after tested economic identity; label reuse and source task,
do not pretend an additional independent experiment. All168 reports remain.

MR12: Report native minute coverage, new requests/errors/source mismatches,
resolved versus unresolved fills, all scenario/seed financials, inactive and
partial days, interval net, fees/cash/trade/open-MTM reconciliation, drawdown,
exposure and descriptive cash baseline. Five-seed means, individual ranges;
no best-seed selection or fresh significance claim on reused TEST. Show paired
net/equity graphics on the same calendars. No overall winner if source failure
or unresolved scenario assumptions prevent that conclusion. Full Truth Harness
FAIL remains FAIL even if accounting/source audits pass.

MR13: Own .runtime/minute-portfolio-replay only; <=2 CPU threads total for model
and finite collector, bounded feature/memo caches, <=1GiB working memory goal,
one portfolio in memory, compressed reports, <=4GiB new runtime disk. Resume
only matching completed artifacts; no partial financial state restoration.
Spec and focused RED tests precede source; cover ordering/gaps/causality/new
admissions, nonfixed invariance, reward/cash/fees, cache exactness/memo identity,
scope/budgets/tampering/collector reports and real model integration. Run full
app tests, diff/staged gates, review, commit and push intended code/tests/specs.
Runtime, market data, logs, positions and models stay uncommitted.

## Priority boundary

Complete this execution-method check before interpreting new policy uplift.
Next is native Demo trade/book data quality and fresh forward outcomes, then
calibrated entry/independent exit learning, cross-sectional allocation/regimes
and alternative RL/encoder ablations. No later experiment may silently bypass
an unresolved prerequisite or promote itself through its own evaluator.
