# RDE-2: native intrabar execution refinement

Date 2026-10-30. SPEC -> RED tests -> CODE. Independent Demo research only.
Parent: [RDE-1](binance-demo-rl-direction-experiment.md), TH-01..12.
This protocol is an execution-resolution ablation, not a new trading policy.

## Question and immutable scope

Can finer candles returned by the SAME Binance Spot Demo endpoint resolve the
TP/SL ordering hidden inside RDE-1 native15m candles and change portfolio results?
Keep RDE-1-MAX519-20261029, its519 current-survivor pairs, maximum source period,
70/15/15 calendar boundaries, cutoff1793197800000, frozen predictors, features,
candidate screen, thresholds, policies, five seeds, costs and risk unchanged.
Never overwrite RDE-1 source, registration, caches, reports or model artifacts.
SIMULATION / RETROSPECTIVE_REUSED_HOLDOUT only; no automatic promotion, orders,
credentials, live-worker changes or claim of native historical Demo fills.

## Stage A: prerequisite source-compatibility experiment

R2A01: Before implementing/relying on a refined portfolio replay, attempt actual
credential-free public GET /api/v3/klines on https://demo-api.binance.com only.
Reuse the app read-only transport; allow intervals1m and1s only, UTC, bounded
historical start/end/limit. No production Binance fallback, signed requests,
new transport permissions, external price provider or secret discovery.

R2A02: Register immutable ordered probe windows BEFORE requests, their original
archive page/content hashes, RDE-1 registration SHA, fine interval and validator
version, and the full Truth Harness report path/hash/status at registration
(UNKNOWN if absent). Never hardcode a historical check as current. Include
every unique previously diagnosed ambiguous TEST window
(TST1749763800000, NXPC1749808800000, NXPC1749809700000,
PROM1750638600000, DEXE1750797000000, GMX1752068700000), plus BTC/ETH/SOL
earliest regular, first regular TEST and latest regular archive windows.
Probe choice is a source diagnostic, never a performance estimate. A pass on
these windows does not certify source compatibility on unrequested history.
Verify the frozen prepared manifest's canonical data SHA against RDE-1 and
frozen-encoder bindings; require its resolved archive path and per-symbol
source-page-vector SHA to match. A self-consistent replacement archive cannot
stand in for the original. Freeze the prepared manifest's file SHA as well.

R2A03: Read the original all-usdt/history.sqlite with SQLite mode=ro, verify
compressed page SHA and canonical UTF-8 source JSON SHA, select exact15m row.
Do not instantiate its writer or mutate archive metadata. Fail closed on an
absent, duplicate, irregular or corrupt original candle. Preserve raw source
numbers as Decimal strings; no float tolerance concealing price differences.

R2A04: A1m window requires exactly15 ordered candles;1s requires900. Every open
equals the expected UTC grid, close=open+interval-1, no duplicates/gaps/open
candles, all twelve fields well formed, finite positive OHLC with consistent
range, nonnegative volumes/counts, taker<=total. Receipt must follow window end.
Aggregate first open, max high, min low and last close EXACTLY as Decimal;
trade counts must sum EXACTLY. For each additive volume, discrepancy may be
at most (number_of_fine_candles+1)/2 times a field quantum frozen BEFORE GET:
min(1e-8, the original value's decimal quantization unit). Fine-response textual
precision cannot widen this bound. This is explicitly ASSUMED_NEAREST_ROUNDING
compatibility, not an exchange precision guarantee or exact equality.
Report actual discrepancy/bound for each field; an accepted rounding bound
is not exact equality. Never retune tolerances from observed mismatches.

R2A05: Store request parameters, endpoint, UTC receipts, response metadata and
canonical response SHA with immutable raw response or bounded error in own
.runtime/intrabar-execution. Hash the registration and reject changed resumes
or a second different response for one successful request. Sequential public
requests, >=.3s spacing, max2000 attempts, max3600s wall time per invocation,
timeout<=15s, max3 attempts per request, bounded rate/unavailable retry.
Persist partial progress; exhausted budget/network errors are UNKNOWN, never
proof that candles do not exist. No credentials or account data in evidence.
Use one exclusive probe writer. Pace EVERY native GET, including adapter
retries, and cap each request by the remaining invocation deadline.
After closing the native AuditJournal, verify its hash chain and paired public
GET begin/end records read-only. Bind every accepted cached response to its
recorded parameters, IDs, raw-body SHA, receipt and decoded rows. Verify closed
runs, no unknown endpoint/private method, and preserve failures. Record audit
state separately; a source MATCH is not complete native provenance. Do not
print a finalized report until journal close and verification have completed.

R2A06: Report requested/success/missing/mismatch/unavailable counts, exact
windows and field differences. Any mismatch means SOURCE_INCOMPATIBLE for
that resolution/window; any missing/error means INCOMPLETE. Do not launch a
refined full replay while the prerequisite has failures. Preserve diagnostic
results without interpreting them as improved PnL or model performance.
Omitting required seconds gives MINUTES_ONLY, never PASS_GATE_ONLY.
If all1m prerequisite windows pass, request1s for ALL six preregistered
ambiguous windows, independently of their returns/order at1m. Validate the
whole15m window against BOTH original15m and EACH of the fifteen
accepted1m candles (each consecutive sixty-second bucket). Equal whole-window
OHLC alone cannot establish that seconds belong to the correct minute.
If both remain touched in one1s candle, label ordering UNRESOLVED and retain
the separate conservative stop-first estimate; never invent a smooth path.

## Stage B: full replay, only after prerequisite succeeds

R2B01: Before execution code, add focused RED scenarios for refined accounting
and full-scope runner. Reuse frozen RDE-1 signals and policy observations at
15m decision boundaries. Finer future execution bars are requested only AFTER
the action is frozen, never used in features, screening or admission filters.
Every realized selected/held bar in every arm is validated, including new bars
reached after changed earlier exits. Do not patch only original losing trades.

R2B02: Preserve next-open entry, entry fee/slip, resting open protection before
discretionary EXIT, adverse gap stop fills, target-capped TP,16bar max hold,
continuous cash, terminal MTM and latched1USDT interval risk kill. Trailing
levels remain fixed from PRIOR15m peak throughout the fifteen minutes; update
the peak only after an entire completed15m holding bar. Refine first-touch
order/time only. No per-minute policy calls or trailing recalculation.

R2B03: Replay every original TUNE/TEST origin,21 portfolios per segment for
base/stress costs (84 reports), all five seeds. Fine-data failures invalidate
that objective; do not substitute another symbol, discard a seed, omit a day
or merge unresolved performance into a certified result. Original conservative
baseline and any unresolved conservative bounds remain separate diagnostics.
Once flat AND risk-latched, later cash marks may be fast-forwarded only with
tested economic equivalence and explicit SYSTEM_RISK_LOCK provenance.

R2B04: Bind original registration/cache/model hashes and new spec/source/data
hashes before replay. Verify original files remain unchanged. Report interval
net first, completed/partial calendar days, active days, actual transactions,
all costs/open exposure, ambiguous/resolved/unresolved counts and coverage.
No new TEST threshold, seed, stopping rule or hyperparameter selection.
Daily zero after risk kill is inactivity, not daily profitability.

## Acceptance and engineering evidence

Tests cover normal aggregation, Unicode canonical hashes, exact window lookup,
corrupt/missing originals, duplicate/gapped/out-of-range/nonfinite fine data,
OHLC/count/volume mismatch, exact rounding boundary, immutable registration/
responses, GET-only parameters and unavailable responses. Record genuine RED
before implementation. Source probe success is distinct from replay completion.
Run focused and relevant app regression tests, git diff --check, selective
staged Truth Harness, commit/push only intended spec/tests/source. Keep runtime
data and reports uncommitted. Full Truth Harness FAIL remains FAIL.
