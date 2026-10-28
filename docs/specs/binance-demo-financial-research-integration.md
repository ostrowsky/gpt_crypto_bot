# Demo financial integration and strategy research sequence

Date:2026-10-28. Status: implemented;45 focused scenarios GREEN after RED;
actual4-fill/4-asset accounting verified; calendar financial certification pending.
Scope: independent apps/binance_demo_bot only. Objective daily_net_equity_pnl_v1.
Parents: P0 financial contract, financial account bridge, phase4 baseline,
phase6 hypothesis evaluation, all-USDT universe. TH-01..12 apply.

## Ordered delivery

1. Source-backed financial imports and whole-account comparison using the
original seed. Native fills remain UNATTRIBUTED in accounting; OMS ownership
is independent. Never rebase the ledger to current balances.
2. Continue existing maximum-available archive for the frozen519-pair cohort.
No partial archive/current-survivor result authorizes a new trading policy.
3. Implement a frozen causal trend baseline and shared-portfolio research,
with explicit entry/rank/exit/holding/cost definitions before results.
4. Evaluate maximum verified history, chronological train/tune/test, stress,
and USDT/no-trade control. Receipt history/PIT limitations remain disclosed.
5. Actual strategy canary and training require prior gates; existing technical
round-robin is not a profitable champion. Insufficient mature data remains
NOT_TRAINED. Do not extend original expiry/risk limits to force sample size.

## FRI-01: historical mapping before actual accounting

FinancialAccount.reconcile accepts optional metadata_history observations.
Validate every observation's public Demo GET exchangeInfo identity, hash,
request/receipt clocks and audit pins before importing any fill. Historical
mapping receipt must precede fill exchange time; a later metadata refresh cannot
provide retroactive mapping. Select the latest available mapping containing the
instrument. Conflicting base/quote at the same receipt is an error. Mapping is
not a historical fee price: base fees use actual execution price, USDT fees one;
third-asset fees remain unsupported until historical conversion is provided.
Persist the selected metadata hash in each immutable import recipe. Accepted
imports reuse their recipe after refresh/restart; native REST/private duplicates
debit once. Historical observations form part of immutable check input identity.

## FRI-02: diagnostic account attribution

After independently comparing fresh account snapshots, produce an interval
report from the bound seed and the journal prefix pinned by the comparison.
Include every seed asset, locked balance and dust. Compare observed equity with
projected equity at the SAME current midpoint marks. Attribute gross realized,
actual fee expenses and unrealized change; never subtract fees twice. Missing
marks, unimported fills, residuals or changed journal prefix block attribution.
This report is diagnostic, not daily profit or exchange reconciliation.

## FRI-03: calendar reporting honesty

Do not manufacture midnight snapshots from a later current quote or the pilot
start. Day boundaries are Europe/Budapest including DST. Missing boundaries,
unclassified flows, complete symbol population or delivery/cut evidence keep
daily_net_pnl_usdt null. Report exact unresolved dependencies. Native technical
outcomes, diagnostic equity change and historical SIMULATION remain separate.
No learning/promotion authority is granted by matching balances or positive PnL.

## FRI-04: frozen research baseline before evaluation

Baseline trend-breakout-v1 is a NEW research challenger, not a migration of
legacy policy. Input:32 contiguous regular closed15m bars; no candle/receipt
after decision. Entry: close above prior20 highs, mean last8 closes above mean
last24, four-bar return>=0.5%, last quote volume>=2x median prior20, close no
more than8% above slow mean. Rank: four-bar return times quote-volume multiple,
lexical tie-break. No future daily top labels. Reject invalid OHLC/volume/gaps.
Holding: fixed1% stop,4% take profit, trailing1.5% after1% gain, max16 bars;
weakening when fast mean below slow mean exits at NEXT open. Cooldown4 bars.
Frozen parameters and source hashes precede any result. No fitting/search here.

## FRI-05: shared-portfolio maximum-history simulation

One shared100USDT cash account, max30USDT cash debit per entry including fee,
one position,1USDT loss/drawdown entry stop per Budapest day; exits continue.
Signals from prior closed bars execute at next regular bar open, adverse5bps
slippage and10bps fee per side; stress10/20bps. Fill is SIMULATION, never actual.
Fixed/trailing stops use only peaks through PRIOR closed bar, never a high
earlier/later within the same current OHLC candle. If both stop and TP touch,
choose stop. Gap-through stop fills at worse open; TP limit receives its limit
(no invented favourable open). Any held missing bar blocks numerical portfolio
verdict; do not value missing exposure at zero or silently skip it. Value equity
after fees at every closed bar. No-trade benchmark is the SAME100USDT scope,
not the actual mixed USDT/USDC seed. Max drawdown, turnover, fees, closed/open
positions and simulated mark-to-market PnL accompany results.
Mean simulated daily PnL and profitable-days numerator/denominator use only
complete Budapest calendar days with every15m slot present (92/96/100 bars
across DST). Partial/missing days are excluded, never counted as zero or as a
successful day. These OHLC simulation days do not certify actual account days.

Evaluate all maximum archived bars with bounded streaming state, shared
chronological epochs and date-based70/15/15 train/tune/test boundaries. Frozen
strategy is unchanged across segments; segments get their own100USDT portfolio
and prior causal warmup. No model is selected on test. Run regular/stress once,
persist manifest/source/cohort/coverage/split hashes. A partial cohort may be
reported as provisional diagnostic but cannot satisfy maximum-population gate.
Every verified symbol archive cutoff must equal the registered cohort cutoff;
a valid shorter archive is not a complete longer experiment.
Historical receipts/PIT/delisted universe and historical filters/fills remain
uncertified; even positive results cannot auto-enable a live strategy. A background
research runner may wait for collection and run once the frozen cohort completes;
it has no credential/network/order APIs. Save only own ignored runtime.

## FRI-06: promotion / learning handoff

Keep existing actual technical canary and its limits unchanged. Research results
must explicitly block live promotion on incomplete P0/P1/P2, full Harness FAIL,
unverified historical PIT/receipts, insufficient new forward financial cohort,
or nonpositive/unstable net results. Actual two-row handoff stays NOT_TRAINED.
Do not manufacture thirty complete days/100 trades, extend expiry or fit on
simulated fills as if actual. Next actual strategy experiment requires a new
frozen bounded admission after dependencies and replay, not automatic enablement.

## Verification / rollback

Named focused cases: historical mapping import, latest pre-fill selection,
future-only mapping exclusion, tampering/conflict before mutation, immutable
restart/dedup; seed USDC revaluation, base fee, same-mark bridge, mismatch and
missing-price diagnostics; absence of certified midnight never reports daily
PnL. SPEC -> tests RED -> code -> focused/regression GREEN -> diff review ->
staged Truth Harness -> commit/push. Native validation is GET/recovery only,
retains all incidents and original seed, no manual orders or worker resets.
Rollback stops new checks/imports and preserves committed recipes/ledger;
existing protective exits remain active. Runtime evidence/keys/logs not committed.
Full Harness FAIL TH-11 (legacy fingerprint) remains FAIL, never financial PASS.

## Verification evidence2026-10-28

Financial14 missing-interface RED preceded implementation; regression plus new
contracts53 GREEN, then the new14 also passed in the full application suite.
Research18 missing-module RED, then2 genuine boundary failures (future current
bar quality and same-open gap loss),3 missing calendar outputs,7 missing runner
contracts and1 cutoff mismatch RED all preceded the corresponding repairs.
Final focused23 trend +8 runner GREEN; fixture clock correction and test-loader
deduplication were not represented as product failures. Tests use synthetic
economic facts, own actual SQLite/failure boundaries and no keys/network.

Native GET-only verification imported4 actual fills into the existing original
ledger;4/4 nonzero balances matched, whole-account FIFO/fee/unrealized bridge
residual0. Seed and prior stream incidents remain unchanged. The first attempted
capture failed FINANCE_SOURCE_INVALID because saved JSON clock uncertainty had
not been restored to Decimal; that attempt and all recovery evidence were retained.
The corrected capture passed independently. Actual outcome ownership remains
UNATTRIBUTED in this accounting slice and no daily financial PASS is claimed.
The archive runner is a separate research process; it cannot promote models,
increase limits, extend pilot expiry or place orders.
Final full application regression1031 GREEN (249.022s), root specification14
GREEN (0.187s), focused runner8 GREEN (2.679s). Privacy verification scanned22
native JSON files,105 audit/evidence records and22 ledger JSON rows:0 credential
literal hits and0 raw UID fields. Full Harness still FAIL TH-11 from unchanged
legacy replay fingerprint. Runtime evidence remains uncommitted.
Final full application regression1031 GREEN (249.022s), root specification14
GREEN (0.187s), focused runner8 GREEN (2.679s). Privacy verification scanned22
native JSON files,105 audit/evidence records and22 ledger JSON rows:0 credential
literal hits and0 raw UID fields. Full Harness still FAIL TH-11 from unchanged
legacy replay fingerprint. Runtime evidence remains uncommitted.
