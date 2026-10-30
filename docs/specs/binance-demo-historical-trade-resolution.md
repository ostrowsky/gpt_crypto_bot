# RDE-2C: same-source historical trade availability and TP/SL ordering

Date 2026-10-30. SPEC -> RED -> CODE. Independent Demo research, TH-01..12.
Parent protocols RDE-1 and RDE-2A remain immutable. This extension does not
rewrite their registrations or make their incomplete prerequisites pass.

## Scope and registration

HT01: Probe ALL three remaining ambiguous original15m windows, fixed before
requests: DEXEUSDT1750797000000, GMXUSDT1752068700000,
NXPCUSDT1749808800000. These correspond to four original CatBoost/GRU base-cost
trade bindings. This is a post-hoc source diagnostic, not a policy comparison,
profitability estimate or maximum-period validation. Do not reprice old fills.

Bind the RDE-2A registration, accepted1m response payload/file hashes and native
audit result, original RDE-1 registration/portfolio/trade hashes, these exact
windows, new spec/source hashes and full Truth Harness report at registration.
Verify parent bindings and minute/source MATCH before any GET. Seal registration
and each successful request response; changed resumes or conflicting data fail.
Use own .runtime/intrabar-execution; no original/archive/report/model edits.

## Native source and collection

HT02: The sole new capability is credential-free GET
https://demo-api.binance.com/api/v3/aggTrades. A separate research-only transport
owns this exact fixed hostname/path. Do not change or bypass existing adapter
allowlists. No private/account/order endpoint, keys, signed headers, redirects,
proxy/environment endpoint discovery, live Binance fallback or new worker.
TLS validation, timeout<=15s, raw body<=8MiB, no automatic HTTP-library retries.
Whitelist query keys symbol,startTime,endTime,fromId,limit; Unicode URL encoding;
limit1..1000, nonnegative integer bounds, initial start/end span<=15m. Initial
time query and fromId pagination are distinct modes, never combined.

HT03: Every native attempt is journaled with AuditJournal REST_BEGIN/REST_END,
public=true/private=false, parameters, unique IDs, status, bounded public raw
bytes/SHA, source receipt, permitted rate headers and terminal errors. Responses
return raw rows plus matching metadata (raw_body_sha256, received_at_ms,
audit_request_id/attempt_id/begin/end IDs). Preserve HTTP/API errors as errors,
not empty history. Log no credential/account data or exception messages.
Single writer. >=.3s spacing EVERY attempt, max3 attempts per logical request,
honour Retry-After without retrying an IP ban. If the requested cooldown exceeds
60s or the remaining invocation budget, stop with UNKNOWN instead of retrying
early. Latch an IP ban/long cooldown across all later reads in the invocation.
Allow waits<=60s only; max300 native attempts
and900s per invocation, max100 data pages/window, max100000 rows/window.
Persist results/partial state. Budgets/timeouts/network errors are UNKNOWN.

HT04: First time-bounded query: [origin,origin+899999],limit1000. If it returns
[], record HISTORY_EMPTY_FOR_REQUESTED_WINDOW, preserve exact HTTP receipt,
and stop this window; this is not proof of global nonexistence. For nonempty
data require increasing unique aggregate IDs a, finite positive string p/q,
integer f/l/T (0<=f<=l), boolean m/M, complete schema. Page by fromId=last_a+1
until at least one trade at/after window end. Read fromId=first_a-1,limit1
for an exact preceding guard with T<origin. Missing guards cannot certify
completeness; do not infer no missing data from a short page alone.

## Completeness and price ordering

HT05: Require consecutive aggregate IDs, nondecreasing T and disjoint consecutive
underlying f..l ranges across the guarded sequence. Treat this as an explicit
strict coverage condition, not an undocumented guarantee that the API always
provides such a sequence. Any gap/conflict/truncation gives INCOMPLETE. Previous
guard must be strictly before origin and subsequent guard>=origin+900000.
Records beyond the window are guards only; never count them into its metrics.

HT06: Reaggregate ALL in-window records to EACH of the fifteen accepted1m
candles: first/last price, high/low EXACT Decimal; raw trades=sum(l-f+1) EXACT;
base=sum(q),quote=sum(p*q),taker-buy values where m=false. For additive volumes
use fixed ASSUMED_NEAREST_ROUNDING bound (raw_count+1)/2 times
min(1e-8, original minute field quantum), never widened by wire formatting.
Missing-trade minutes can match only zero counts/volumes and flat OHLC equal to
the last preceding trade price. Keep actual differences/bounds. Any mismatch
gives SOURCE_INCOMPATIBLE; never discard offending rows/minutes.

HT07: Only after guarded full-window coverage AND minute aggregation MATCH may
the first p<=SL or p>=TP determine TRADE_SEQUENCE_SL_FIRST/TP_FIRST, using the
original frozen entry-price float arithmetic (stop=entry*.995,
target=entry*1.005) converted to Decimal for comparison. Report source
trade IDs/times and original bindings. Trade IDs order equal millisecond T;
timestamps alone cannot decide a tie. Aggregated prints are not native order
fills, latency, liquidity or queue-priority certification. If the guard/schema/
aggregation is incomplete, retain ORDER_UNRESOLVED even if a partial path
looks favourable. Never infer flat movement from []. No portfolio PnL changes.

## Audit, reporting and engineering

HT08: Close the native journal and verify its read-only hash chain, paired
GET attempts, closed runs and exact accepted raw-response/cache links. Audit
state is separate from availability/completeness/order. Missing provenance or
unfinalized WAL is UNKNOWN; contradictions are FAIL. A successful HTTP request
with [] can have verified provenance and still leave source/order INCOMPLETE.

HT09: Report all3 windows/4 bindings, requested/received rows, HTTP/API states,
guards, gaps, per-minute comparisons, resolved/unresolved counts and retained
original PnL. Freeze no new model/threshold/seed selection. No production,
promotion, limit relaxation or historical execution claim. Full Truth Harness
FAIL stays FAIL; new green tests do not prove profit. If history unavailable,
preserve the conservative result and identify a forward-data or bound-analysis
next step without pretending future collection repairs past observations.

Focused tests precede code: normal and rejected source parameters; native
errors/empty responses/pacing/body limits and audit links; pagination/guards,
duplicate/gapped IDs/time order, malformed data, minute price/count/volume
mismatch, zero-trade minutes, first-touch/tied timestamps, immutable bindings
and source failures. Run relevant regressions, diff check and staged Truth
Harness. Commit/push only intended spec/source/tests; runtime stays uncommitted.
