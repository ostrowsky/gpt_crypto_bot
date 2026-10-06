# Order flow for bounded market execution

Registered 2026-10-06 before new execution outcomes. Research only; TH-01..12.

## Problem and objective fit

Direction accuracy does not establish cheaper execution. Test whether delaying
an already selected order when its predicted signed displacement is favorable
reduces implementation shortfall, without pretending to improve BUY selection.
The ultimate gate remains early watchlist capture and one ten-slot portfolio
after costs. No production code, orders, positions or configuration changes.

## Fixed experiment

Reuse frozen CatBoost_OFI and CatBoost_Book models/predictions from the registered
5/10/30-second study. No retraining, tuning, or test-selected thresholds. For each
issued historical TEST origin, evaluate BUY and SELL separately, equal diagnostic
weight, fixed quantity = 1000 USDT / decision mid. This synthetic population is
not the bot's order population. Primary policy: wait 5 seconds if signed forecast
is below -1 bp; otherwise execute immediately. Secondary fixed waits 10 and 30
seconds use their own frozen horizon forecasts. Controls: immediate, always wait,
book-only model, and microprice. No minimum/maximum future-price oracle.

Common assumed arrival delay is 1 second: immediate at t+1, wait at t+h+1.
Walk first five observed price/quantity levels at arrival. Require the full fixed
quantity; insufficient depth is unknown, not a successful or partial fill. BUY
cost includes 7.5 bp assumed fee; SELL proceeds deduct the same fee. Do not add
the former fixed 5 bp slippage on top of observed book walking. No impact,
queue, maker fill, transport-latency, or exchange fill certification is implied.
Implementation shortfall compares each net unit price to the decision mid;
positive gain means lower BUY cost or higher SELL proceeds than immediate.

At issuance use only frozen past features/predictions. Missing future quotes
cannot change the issued action. Score matched immediate/wait full-depth pairs
separately for each horizon and publish issued, deferred, known/unknown action,
matched and harmed counts. Source session must remain valid through arrival;
exact one-second clocks, quote age <=250ms, original >500ms gap resets. Never
forward-fill gaps. Unknown actions remain in evidence and are not called misses.
Publish pooled and per-asset/side means, p95/p99 shortfall, gain distribution,
and daily paired diagnostics. Three-calendar-day block bootstrap, 5000 draws,
seed42, familywise95% for nine OFI comparisons (immediate/book/micro x 3 horizons).
Calendar-interior days can still contain source holes: disclose coverage.
Require >=30 fully observed candidate days and positive lower bound before a
robust execution claim; this archive cannot satisfy that gate.

## Maximum period and actual orders

Use all 90 pinned BTC/ETH/SOL perpetual book files, all prepared source parts,
and all issued TEST origins (including unknown future paths). Verify raw LFS,
prepared, frozen native/prediction and source hashes. Also audit ALL orders from
the maximum 186-day control replay: entries, exits and partial exits, reconstruct
quantities from verified allocation ledger. Join only exact decision clocks;
missing symbol/date/features remain explicit. This archive is USD-M perpetual,
whereas config.BINANCE_REST selects spot. Transfer of replay order clocks is
diagnostic, not an actual spot fill or a portfolio improvement. Maximum-period
execution replay is blocked by absent same-venue books, receive clocks and fills;
never extrapolate four overlapping trades to the 93-symbol portfolio.

## Fresh data and production gate

Run a finite 120-second public spot depth20@100ms and trade-stream capture for
BTC/ETH/SOL, preserving raw payload, local wall and monotonic receive clocks,
stream name, update ID and exchange trade clock where supplied. This is a
connectivity/provenance pilot, not new model training or forward-profit evidence.
Partial depth payloads need not supply exchange time; do not invent it. No keys,
authenticated endpoints, account data or order placement. Publish failed access
and gaps honestly. No background service or recurring task is created.

Before promotion collect the actual watchlist's same-venue receive-time book and
trade history linked to decisions/submits/acks/fills, train on past only, freeze
the policy, then evaluate >=30 fresh days. Maximum available portfolio replay
must retain early/captured count, precision, drawdown and net/BTC alpha guardrails;
hard exits execute immediately. Reject or keep research-only when gates fail.
Rollback: remove offline scripts; no live behavior is enabled.

## Delivery and verification

Freeze spec, inputs, sources, parameters, issued actions, quote outcomes and
result receipt. Independently recompute scalar book walking on predetermined
samples, all arithmetic/count metrics, native predictions on all issued rows,
actual order extraction, and hashes. Explicitly distinguish full recomputation
from producer-bound prepared-book audit. Plot matched mean gains and shortfall,
with denominators and source limitations adjacent. Focused tests cover causal
selection, side/fee arithmetic, depth failure, missing/session gaps, exact joins,
actual quantities, no-future mutation, denominators, and capture validation.
Run relevant tests, full/staged Harness, diff review, commit and push source,
spec and tests only. Runtime data/reports/models are excluded.

## Completed diagnostic, 2026-10-06

All 90 raw LFS files and prepared parts verified. 253951 issued TEST origins;
507902 equal-weight synthetic BUY/SELL tasks per horizon. Fixed rule unchanged:

| Wait | Deferred / issued | Matched deferred | Mean gain on matched deferred bp | Mean gain on all matched tasks bp | Harmed / matched deferred |
|---|---|---|---|---|---|
| 5 sec | 9 / 507902 | 9 | +0.089555 | +0.000001609 | 5 / 9 |
| 10 sec | 175 / 507902 | 172 | +0.999724 | +0.000346761 | 73 / 172 |
| 30 sec | 778 / 507902 | 765 | +2.371368 | +0.003795221 | 321 / 765 |

Matched all-task denominators: 500951,495882,477995 respectively; selected-action
unknown outcomes:1306,1307,1312 respectively (all issued tasks, including unknown
immediate quotes). Seven interior calendar days, each incomplete. OFI vs immediate
familywise95% daily gain intervals in bp: 5s [-.000102630,.000155518],
10s [.000043908,.000549329],30s [.001830548,.006829546]. These intervals from a
short exposed historical cohort do not certify robustness. OFI vs book-only
intervals include zero at ALL three horizons; incremental OFI benefit unproven.
Fees unchanged; this is price improvement, not fee savings or portfolio alpha.

All11010 control trades /22020 order actions inspected across186 days.21780
actions lack a symbol book archive;235 lack a past eligible forecast;5 transferred
order clocks join. All5 actions execute immediately, zero change under all
horizons. Spot vs perpetual transfer and absent historical receive clocks/fills
block actual execution proof. Verdict NOT_APPROVED, runtime_eligible=false.
The broader hypothesis remains research-only; primary5s policy has no useful
demonstrated effect. Do not silently promote secondary30s diagnostic.

Finite public spot pilot succeeded on network-enabled retry:3123 depth updates,
5776 trade messages over120seconds,0invalid payloads. The initial sandbox access
failure is retained. No orders or keys used. A first historical run was stopped
before outcomes to protect partial risk exits unconditionally; v2 is the valid
run, with identical models/thresholds. Native forecasts and metric audit PASS;
book-walk scalar checks are sampled, prepared books retain producer provenance.

Runtime evidence: `.runtime/order_flow_execution_20261006_v2/`, source/result
receipt, outcomes, actual-order coverage, independent verification, comparison
CSV/PNG and transferred-order example plots. Fresh pilot:
`.runtime/order_flow_execution_spot_20261006_v2/`. Production unchanged.
