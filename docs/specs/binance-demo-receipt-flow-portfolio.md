# Receipt-flow shared-cash execution replay (RFP-1)

Registered 2026-11-19. SPEC -> TESTS -> CODE. Research only; parent RFD-1 and
daily_net_equity_pnl_v1. No keys, account/positions, orders, worker restart,
live risk/expiry changes, actual-fill learning or model promotion. TH-01..12.
Full Truth Harness FAIL TH-11 remains explicit. Rollback stops only own offline
research; preserve incomplete attempts and frozen inputs.

## Frozen cohort / maximum available time

Use completed RFD-1-20261118-v2 frozen predictions, native model identities,
registration, prepared proof and independent artifact audit. Verify seals and
the frozen prediction NPZ hash. Original PUBLIC_DEMO_OBSERVATION BTC/ETH/SOL
capture is not ACTUAL_DEMO account execution. Reuse its completed full raw
validation proof, not a new claim of independent raw revalidation.

Models trained on Nov6..8. Evaluate ALL available out-of-sample origins from
Nov9 00:00 through Nov13 00:00 Europe/Budapest: four complete calendar days.
One continuous portfolio per case: no cash, inventory or seed reset at TUNE,
CAL or TEST. Primary TEST remains Nov11..12 and inherits prior OOS state.
Earlier TRAIN financial outcomes cannot be called out-of-sample. TEST was
already disclosed; no new independent confirmation or significance claim.
Three liquid assets do not satisfy the actual all-USDT population gate.

Use every feature-valid issued row, INCLUDING rows without mature future labels.
Never use the RFD future-label mask, future data availability or correctness to
choose signals, rank symbols, select chart origins or cancel an existing order.
Intervals calibrated on Nov10 are NOT trading inputs for Nov9 or any other day.

## Arms / ex-ante decisions

All P/B/F CatBoost point forecasts are frozen; no refit, search or seed choice.
Add M0 raw microprice deviation and M1 linear microprice return calibration.
For each horizon5/10/30sec, M1 fits y=a+b*micro_bps using only mature TRAIN rows
before original train cutoff, with an intercept and ordinary least squares.
Zero feature variance gives slope0/intercept TRAIN mean. No CAL/TUNE/TEST labels
enter this fit; save coefficients and exact TRAIN feature/label hashes.
M0 is a deliberately simple unscaled control, not a calibrated h-return model.

For every arm/horizon run two fixed admission modes:
direction: predicted gross bp>0;
cost-aware: predicted gross bp > spread_at_origin +2*(fee_bp+slip_bp)+1bp.
No threshold tuning. The latter is an ex-ante heuristic, not guaranteed net edge.
At each5sec origin consider only currently issued rows. Rank admissible scores
descending then symbol; select at most one without looking up execution quotes.
An explicitly future feature-end or mismatched source origin is a fatal input
error, never a silently accepted prediction or excluded opportunity.
Freeze origin, source row/features hash, score, spread and decision identity.
One shared long-only Spot position across all three assets; CASH starts same100.
All5 arms x3 horizons x2 admission modes x2 profiles =60 portfolios, plus CASH.

## Execution / money / risk

Cash100USDT, maximum entry debit30 including fee, minimum5. Decimal prices,
quantity, cash, fees and equity; never floating monetary ledgers. Fractional
paper quantity is allowed; historical LOT_SIZE/tick filters and actual fills
are not certified. No leverage, shorts, external flows or added financing.

Base: fee10bp each side, adverse slip5bp, client latency1000ms.
Stress: fee20bp each side, adverse slip10bp, latency2000ms.
BUY eligible origin+latency; two-second exclusive submission window. While
quote is unavailable/stale, client does not submit; expire at eligible+2000ms.
At first fresh quote, require enough best ask quantity for the whole debit.
Insufficient capacity rejects that BUY; do not substitute another symbol or
invent partial fills. BUY price=ask*(1+slip), fee=notional*fee, qty=debit/(price*(1+fee)).
Actual fill time starts the fixed holding horizon. At entry+h create SELL
intent, eligible after same latency. SELL waits for a fresh bid and sufficient
best bid quantity; no abandonment/oversell, no partial fills. Price=bid*(1-slip).
Fees debited exactly once. This top-of-book/slip model is conditional SIMULATION,
not proof of exchange fillability/queue position. No profit-target/exit fitting.

Cash/inventory carry across midnight. Daily loss and daily observed-peak
drawdown entry latch1USDT reset ONLY at Budapest midnight; whole-interval
drawdown latch5USDT never resets. On breach cancel pending BUY and create an
exit intent if held. Exits continue after latches. These are trigger thresholds,
not guaranteed hard loss bounds: latency, gaps and fees can overshoot them.
Cooldown five seconds after rejected/expired BUY or completed SELL.
Unknown opening equity while held blocks entries for that entire day and
requests an exit; a later quote cannot repair the missing midnight anchor.
No actual pilot risk contract is modified by these separate research settings.

At each integer second while held/pending, and each5sec decision or boundary:
first close/open calendar boundary from PRE-fill state; then process eligible
SELL/BUY; then mark/update risk and create exit intent; then evaluate new entry.
Fill exactly at midnight belongs to next day. End mark is pre-fill at exclusive
interval end; do not force a future liquidation or truncate a holding deadline.

## Receipt quotes and missingness

Source is the SAME original compressed chunks pinned by RFD. Verify capture
metadata and manifest against RFD input hashes. Lookup only requested execution
and valuation times, using manifest bisection and bounded16-chunk LRU. Every used
chunk is compressed/decompressed/count/metadata verified before access and
hashed again at finalization; report exact used source vector. No network.
Prior full original-validator proof is reused explicitly; current quote lookups
independently decode raw standing best prices/quantities.

Only receipts STRICTLY before requested clock; age<=1000ms. Preserve source
ordering for equal clocks. Latest connection/session end/control invalidates
quotes; new session cannot inherit a previous session's book. Duplicate depth
frames do not refresh quote age. Validate positive uncrossed best levels. Do not
interpolate, forward-fill past freshness, sort backward clocks or borrow future.
Support exact exclusive final midnight using pre-cut raw messages, never the
capture's one post-end frame. Exchange-age remains UNKNOWN.
Once a symbol is selected, missing future quotes defer/expire that intent under
the above online policy; they cannot cause a different future-valid selection.
Quote source/receipt/session/chunk/hash is attached to every fill and mark.
At selected origin compare recovered midpoint/spread to frozen RFD features;
mismatch fails the case, not a signal exclusion.

## Financial truth / logs / outputs

Primary equity uses fresh bid/ask MID, consistent with P0; bid liquidation
diagnostic is separate. Daily PnL=closing equity-opening equity, no flows.
Entry/exit fees/slippage already affect cash: never subtract them again.
Every held one-second mark and both boundaries must be known for a COMPLETE
full calendar simulation day. Otherwise primary day PnL=null; preserve any
known boundary delta as DIAGNOSTIC_ONLY, excluded from daily objective mean.
The total unmarked-held counter counts unique clocks, including a shared
midnight only once; both adjacent days may still be incomplete. Observed
drawdown includes the exclusive final pre-fill valuation, not only earlier ticks.
Flat USDT-only state needs no market quote. Unknown days are not zero/profit.
Expose requested/complete/excluded days and paired common-day denominators.
Mean includes complete no-trade days. Partial calendar days are diagnostic only.
No current/future quote repairs a past valuation. Cash/quantity conservation,
closed net + open unrealized = interval equity change and sum of known daily
boundary changes must reconcile. Open exposure and pending exits remain visible.

Persist shared immutable source/forecast references, all policy decisions and
all intents/rejections/expiry/fills/risk events, fee movements, daily boundaries,
marks, ledger final state and complete runtime registration. Compressed logs,
atomic completion, fresh output only; keep incomplete errors. No reports/models/
runtime staged. Record model/source/spec/environment/data/split/risk hashes.

Report all61 cases on identical OOS/TEST calendars: net interval diagnostic,
complete-day mean, profitable n/N, fees, turnover, entries/closed cycles,
observed drawdown, unmarked held ticks, blocked reasons, open exposure and stress.
Provide TEST daily PnL/equity comparison and trade plots at predetermined first
chronological fill per case (never best trade). CASH0 is named benchmark.
Do not promote a point winner; four OOS/two TEST days, exposed TEST, Demo receipt
clock, partial-depth execution and restricted population remain limitations.
Even positive simulated net is not realized income or actual learning progress.

## Focused acceptance scenarios

RFP01 strict as-of, equal-clock order, session reset, duplicate freshness and
exact final cut; source hashes/metadata/tampering fail closed; LRU bounded.
RFP02 future quote/label mutations cannot alter pre-origin selection; no fallback
to a different asset after missing execution; issued-without-label rows retained.
RFP03 debit/fee/spread/slip/quantity conservation, full capacity check, no oversell.
RFP04 BUY latency/TTL, delayed SELL/holding deadline, no forced terminal exit.
RFP05 continuous cash/inventory; midnight pre-fill boundary; DST23/25h calendar.
RFP06 risk latches, cooldown, exits after halt, missing opening cannot be repaired.
RFP07 missing held marks/null day, flat cash known0, denominator and fee nonduplication.
RFP08 TRAIN-only micro calibration and source/prediction immutability; all61 cases.
RFP09 complete journals and independent ledger/daily/source reconstruction.
Require genuine missing-API RED, focused GREEN, appropriate app regressions,
diff review, staged Truth Harness, commit/push and independent runtime audit.
