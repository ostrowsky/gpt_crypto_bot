# Receipt-time Spot flow direction experiment (RFD-1)

Status: preregistered research, 2026-11-18. SPEC -> TESTS -> CODE.
Independent application only. No credentials, orders, policy promotion, worker
restart or live limits change. Objective of this phase: test incremental forward
return information. It does not establish the application's daily net PnL.

## Frozen source and timing

Use every hashed chunk of own completed public-flow-20261105-priority2 capture:
BTCUSDT, ETHUSDT, SOLUSDT, Demo Spot, seven planned Budapest calendar days.
Warmup may supply context; exclude warmup and receipts at/after planned end from
origins and labels. Pin registration, manifest, status, every compressed chunk,
producer, preparation/model sources, spec, versions and split before fitting.
Verify compressed and decompressed hashes/counts and replay FlowValidator on ALL
raw messages; compare recomputed validation with stored validation. Abort on
invalid frames, receipt regressions, changed input or metadata mismatch. Keep
failed attempts. Use bounded gzip reads, per-symbol rolling queues, partitioned
output; no full raw capture in memory. No new data collection or venue fallback.
Grid emission is lazy even across a long silent receipt interval. Refuse an
existing output before entering error persistence; never overwrite its incident
or status. Persist an INCOMPLETE status only for the current newly created run.

Use local receipt ordering, NOT exchange clocks. At integer second t issue a
snapshot from receipts strictly <t. Quote receipt age <=1000ms, trade receipt
age <=5000ms. Exchange-age remains UNKNOWN. Session/control boundaries and any
recorded gap reset context (per symbol for topic gaps). Duplicates contribute
neither new trades nor OFI. Require eleven consecutive valid one-second states
of the same generation for ten-second features. A silent stale interval is
unknown, never certified zero flow. Zero flow inside a valid continuous receipt
window is an observation conditional on collector coverage, not exchange proof.

Book is standing partial top20, not diff/cancellation order flow. Cont-style
level1 supply-demand change is a snapshot-derived feature, not identified orders.
Aggressor BUY when buyer is not maker; aggregate quantity counted once, not
multiplied by underlying trade count. Feature windows [t-w,t), w=1/5/10sec.

## Frozen ablation and labels

Origin cadence five seconds, all three assets, no future-validity candidate
selection. Store feature-valid issued rows even without a mature valid label.
P: receipt mid log returns1/5/10sec and ten-second realized volatility.
B: P + spread, microprice deviation, standing depth imbalance1/5/20,
log total depth20 and quote age.
F: B + snapshot OFI and aggressor quantity imbalance/count/log quantity for
1/5/10sec. Same origin and target masks for every compared arm and baseline.
OFI sums are normalized by current total standing top20 base quantity;
trade imbalance is signed base quantity divided by total base quantity.
No symbol ID, absolute price, calendar feature, scaling, imputation or search.

Target for h=5/10/30sec: 10000*log(mid(t+h)/mid(t)), using receipt snapshots.
Require all intervening one-second quote/trade freshness and the same generation;
future invalidity masks LABELS only and never changes issued features/predictions.
Both origin and label snapshot must precede exclusive segment end. Common
30-second maturity mask across horizons for fit and matched evaluation. This
also purges the final30 seconds of each split. No backfill or interpolation.

Calendar days1..3 TRAIN, day4 TUNE, day5 CALIBRATION, days6..7 TEST, fixed by
registration boundaries, not row counts. All periods use maximum source coverage
with explicit requested/eligible/issued/scored counts by asset/day and exclusions.
TUNE is diagnostic only: no selection, threshold fitting or early stopping.
Fit nine CatBoostRegressor models: P/B/F x5/10/30sec. Fixed parameters:
iterations300, depth5, learning_rate.05, l2_leaf_reg10, RMSE, seed42,
bootstrap_type No, has_time True, thread_count2, allow_writing_files False.
Use existing isolated Python3.11.9, CatBoost1.2.10, NumPy2.4.4, SciPy1.17.1.
No retraining/refit or outcome-driven run restart. Save native states and verify
reload predictions on deterministic first32 scored rows. Minimum100 TRAIN and
100 calibration rows per horizon, otherwise NOT_TRAINED / INCOMPLETE.

## Evidence and limits

Report TRAIN/TUNE/CAL/TEST MAE, RMSE, correct/nonzero N, three-sign correct/N,
positive/negative/unchanged outcome counts, majority-direction control fitted
on TRAIN, zero-return error control and microprice-direction control. Report
matched F-B deltas by asset, horizon and TEST day. Calibrate symmetric90%
absolute-residual intervals on CAL only using finite-sample conformal rank;
publish TEST covered/N, width and no exchangeability guarantee. No significance
or adoption with two TEST days; minimum seven independent TEST days would be
needed even for a later exploratory day-block diagnostic.

Frozen TEST predictions contain all feature-valid origins (including unmatured
labels), origin/feature-end, model/source hashes and calibration radius. Never
select chart origins by future correctness: first issued origin at/after TEST
start+12h and TEST start+36h, per asset, by feature availability only. Plot actual
receipt mid history and post-origin fact with independent predicted5/10/30sec
anchors for each arm on identical axes. Connecting anchors is visualization,
not a learned second-by-second path. Missing actual points remain gaps.

No portfolio profit computed in this phase; net_daily_pnl=null. A positive
direction score is not financial improvement. Forward receipts are new to
these models, but the source was integrity-audited before registration. Three
liquid Demo assets are not the live all-USDT candidate population. Prior futures
order-flow rejection and full Truth Harness FAIL TH-11 remain separate evidence.
Next gate, only if useful: delayed bid/ask shared-cash execution replay after
costs, maximum actual candidate population and fresh forward shadow validation.
Rollback: stop own offline process; retain artifacts; no live behavior to revert.

## Focused acceptance scenarios

RFD01 strict receipt cutoff/prefix invariance, future edits cannot change features.
RFD02 session/gap/staleness invalidation, warmup and labels never cross gaps.
RFD03 maker-side trade sign, duplicate exclusion, OFI signs and standing semantics.
RFD04 calendar splits and mature-only common mask; missing target retains issuance.
RFD05 matched denominator, zero denominator null, TRAIN-only majority baseline.
RFD06 CAL-only finite-sample interval rank, no TEST fitting; native reload parity.
RFD07 hashed streaming raw revalidation and immutable registration fail closed.
RFD08 deterministic charts choose origins before inspecting labels; gaps preserved.
RFD09 bounded four-process raw validation: main process derives boundary states
from raw last-topic payloads, exact receipt/session ordering and stored counters.
Every raw frame still executes the pinned original FlowValidator in a worker;
compare every stored validation and the entire recomputed ending state. A wrong
boundary/counter/frame fails closed before prepared data is sealed or fit begins.
Feature emission stays serial in recorded receipt order. At most four chunks
are pending; no skipping frames or parallel chronological feature construction.
Technical v2 may restart preparation only, with v1 interrupted evidence and
source snapshot retained; no model has been fitted or TEST score inspected.
Record genuine missing-API RED before implementation, focused GREEN, relevant
regressions, diff review, staged Truth Harness, commit and push. Runtime excluded.
