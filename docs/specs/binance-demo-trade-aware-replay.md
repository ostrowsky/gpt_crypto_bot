# MRP-2: trade-aware candle validation and execution-model repair

2026-11-18. SPEC -> RED -> CODE. Independent research only, TH-01..12.
Parent MRP-1 remains INCOMPLETE; do not rewrite its failed gate or reports.
No orders, credentials, training, live-policy change or model promotion.

## Frozen experiment

TA01: Same maximum RDE-1 history,519 survivor symbols, frozen predictors,
thresholds/splits/screen and five seeds. Recompute168 portfolios (21 arms,
TUNE/TEST, base/stress, two same-minute conventions). This is exploratory
EXECUTION_MODEL_REPAIR_ABLATION / REUSED_HOLDOUT, not a new untouched test.
No dropping a symbol, time or failing portfolio after seeing its outcome.
Keep100USDT capital,30max/5min debit, one position,1USDT interval risk latch,
no midnight reset; costs and benchmark unchanged. No intraminute risk guarantee.

TA02: Reuse the already completed MRP-1 golden84 as explicit prior evidence:
verify their sealed registration/meta/compressed hashes, parity against ALL84
original references and original executable source hashes/snapshots. Record
source report IDs/SHAs; never relabel these as newly run MRP-2 portfolios.
Their account/candidate-cache code must match current unchanged shared source.

## Source contract (separate from unchanged strict IC-1)

TA03: Preserve original validate_window output including MISMATCH. New validator
requires ALL15 ordered closed native Demo1m rows; original schema/volume rounding
bounds/trade-count checks remain unchanged. Zero-trade minutes must have zero
base/quote/taker volumes and flat positive OHLC. A positive trade count requires
positive base and quote volumes. After the first minute, an empty minute must
carry the immediately preceding close. Reject inconsistent activity, not coerce it.

TA04: Aggregate PRICE OHLC from traded minutes only, exact Decimal against
original15m. Do not delete empty rows from the temporal grid or volume sums.
Return MATCH_TRADE_AWARE only if price/volume/count all agree. If entire15m has
no trades require original zero activity and every fine OHLC equal its flat
carry price; return MATCH_NO_TRADES. Any unexplained difference is MISMATCH;
missing/malformed data is INCOMPLETE, both null the portfolio objective.
No tolerance chosen from results. This deterministic semantic rule was motivated
by exposed ADXUSDT cases; disclose that origin, not a confirmation on fresh data.

TA05: Freeze policy action/observation BEFORE execution lookup. Empty minutes
never create BUY/SELL, SL/TP touches, trailing peaks or alternate candidates.
Orders use next available traded-minute OPEN in the current15m interval with
original adverse slip/fees. This is a FIRST_TRADE_BUCKET_FILL_PROXY, not a quote,
queue, liquidity/capacity or actual fill certification. Timing is a minute bucket.
Market intent in an entirely empty interval remains unfilled and expires at
bar end; retain its count and every calendar mark, do not remove the decision.
A held position remains exposed; its carry-price MTM is explicitly stale/proxy.

TA06: Resting gap protection precedes discretionary EXIT at the first traded
minute; scan only subsequent traded minutes chronologically. Stop gaps use worse
open; TP capped at target. Both touches within SAME traded minute use separate
MINUTE_STOP_FIRST/TARGET_FIRST conventions, unresolved counts retained; no bounds
or aggregate profit certification. Prior15m traded peak drives trailing; never
use this interval's later high to move its earlier stop.

TA07: Record entry_decision_ms separately from minute entry_ms. Frozen policy
age uses completed decision bars as before. A16bar holding deadline is known at
entry: submit forced EXIT at origin>=entry_decision_ms+16*STEP, fill at first
available traded open AFTER deadline, behind resting OPEN protection and before
new policy EXIT/high-low scanning. Never sell retroactively at the last trade
of a bar after learning its close. With no trades retain exposure/overdue intent;
record overdue empty steps. Accounting/risk marks keep15m cadence, explicitly
last-trade/carry proxies; not continuously executable account equity.

## Collector and provenance

TA08: New immutable MRP-2 registration/scope; own .runtime/minute-portfolio-replay
folder. Old attempts/expiry remain untouched. Scope at most2000 new native GET
attempts across resumes,7200s expiry,3 attempts/query,.3s pacing,15s request/90s
owned child deadline. Seal exact public query linked to frozen decision before
GET. Reuse PRC-1 public descriptor commitments; same Demo host, no Spot fallback.
All errors retained; unfinished preexisting query cannot reset attempt budget.

TA09: Import ALL613 IC-1 minute windows after exact wire/journal validation.
Also import all prior MRP-1 prefetched raw windows only after verifying old
registration, query/owner/decision links, exact wire/receipt, raw cache extraction
and original source snapshots. Pin imported files, no new request for existing
wire evidence. Do not require future-prefetch validation for a current frame;
validate each reached selected/held frame against original archive. Future
execution presence cannot affect features or candidate admission. Overlap conflict
fails closed. Bound accepted frame to source response SHA and original page SHA.

TA10: New sources/spec/shared source and frozen model/data/splits/environment,
full Truth Harness, imported records and budgets sealed before first portfolio.
One finite collector;<=2 model CPU threads,<=1GiB working memory goal, compressed
reports and<=4GiB new runtime goal. Exact candidate cache and deterministic
stateless policy memo only; original input recheck and public-data audit at end.
Invalid/incomplete portfolio saved with reasons and null PnL, no partial winner.

TA11: Report all168 financial ledger checks and168 scenario/seed outcomes,
full calendar denominators, partial/inactive days, open/carry exposure, fees,
empty intents/deadline delays, native coverage and unresolved ordering. Five-seed
means/ranges, no best seed. Show equal-window equity/net comparisons, base/stress.
Cash and unmanaged15m BTC benchmark remain descriptive. Old RDE-2 gate remains
incomplete. Full Truth Harness FAIL stays FAIL. No actual Demo income inference.

## Completed PFD-1 input audit

TA12: Audit entire completed public-flow-20261105-priority2 manifest, every chunk
hash/uncompressed hash/JSON/count/receipt/session and stored validation summary.
Bind source version, registration, manifest/status hashes, planned boundaries.
Separate seven elapsed calendar days from uninterrupted coverage; enumerate
stale-receipt pauses, protocol/ID errors, reconnects, late end-tail and per-day
records. Persist read-only diagnostics in own runtime. No training/admission/PnL
conclusion from capture completion or checksum PASS. No worker restart/extension.

TA13: Tests precede code; cover padding/true mismatch/empty activity, delayed
entry/exit/protection/deadline, missing data nulls, state/cash conservation,
causal action freeze and nonfixed convention invariance, cache wire tampering,
golden reuse identity and flow integrity-vs-coverage. Run app regressions and
staged harness, diff review, commit and push intended source/spec/test only.
