# Leader accompaniment: causal trend-confirmed soft exit deferral

Registered 2026-10-06 before new diagnostics/policy outcomes. Offline only.
TH-01..12, SCOUT_OPTIMIZATION_SPEC.md and leader-mission-reaudit.md apply.
Use the repository crypto-bot-truth-harness skill. Full Harness PASS at start.

## Question and fixed intervention

Diagnose first exits and actual cooldown skips of captured daily leaders on the
maximum available 186.333-day /93 of105-symbol market. Then test ONE fixed rule
on ALL original rule candidates, never select runtime positions by future leader
rank, final-day capture or future weakening labels. No training or grid search.
This is a distinct causal trend rule, not another fit of the rejected one-bar
cash-advantage model. It is motivated by exposed prior accompaniment diagnostics.

At the first original WEAK SELL, defer only if exact closed15m data show:
current close > recursive EMA20; EMA20 strictly rising; running CLOSE peak since
entry >= entry price +1 entry-time ATR14; current close >= peak -1 current ATR14.
ATR14 is local rolling mean true range, matching the registered raw evaluator.
Apply price tolerance1e-12*entry price. EMA starts at archive start. Require exact
15m entry/current timestamps and complete intervening grid; missing/invalid/stale
data retain immediate SELL. Origin must match an actual closed native timeframe
candle and its execution price. No future trade extrema/report fields as inputs.

Hold at most60minutes from the FIRST deferred proposal, no extensions. Recheck
confirmation at every15m portfolio tick. Loss of confirmation forces SELL; original
ATR/time/fast-loss/EMA/micro hard exits always execute first. Original replacement,
partial exits, position limit10, entry gates, candidate stream and cooldown stay.
Timeout/loss-of-confirmation reason preserves WEAK classification and its cooldown.
For1h positions forced wrapper exits explicitly use the available latest15m close
and current tick, rather than a stale1h price. No intrabar/fill guarantee. Consider
only first soft proposal per entry; no recursive reactivation. Instrument every
first decision, active tick, terminal reason and actual cooldown skip.

## Comparison and interpretation

Re-run original control, require all trade fields/decisions match receipt-verified
prior control. Record actual cooldown skips in both full portfolio simulations.
Fixed challenger vs control, common start/end/raw population/candidate snapshot.
Correct raw local00:00-open/22:00-close Top15 labels; old early annotations are
known TH03/05 FAIL and cannot define winner. TEST boundary1787781600000 (38days),
full186days. TEST is exposed retrospective, never sealed confirmation.

Primary: early leaders, unique leader coverage, leader BUY precision and same-entry
confirmed leader accompaniment (held-time including reentries, retained rise,
first reduction before operational weakening). Original2ATR/two-close/EMA marker,
24h follow-up, unknown/unconfirmed/terminal separate. Diagnostics by exit category
and exact reason, with n/N and later new highs; actual cooldown events mapped to
known leader days and whether within first-entry-to-marker episode. Those counts
are blockers, not proof an alternative entry would succeed or be available.

Single-policy paired3-calendar-day block bootstrap5000seed42. Preserve conservative
familywise24-comparison intervals for continuity; no outcome-selected interval.
Also paired held-time deltas on exact same-entry confirmed nonforced cohorts.
Acceptance: full AND TEST early/coverage/precision not worse, TEST matched
accompaniment mean positive with lowerCI>0 across>=30days, retained movement not
worse and no higher premature first exits. Otherwise descriptive trade-off or no
confirmed gain. Account/cost/DD are safety diagnostics, not mission winner.
No production approval: even a retrospective gain needs fresh forward shadow,
PIT-universe/receive/fill parity and rollback. No live flags/orders changed.

## Evidence and delivery

Register source/spec/raw/parent/candidate hashes before outcomes; immutable runtime
artifacts and source snapshots. Independently reconstruct rule inputs from scalar
raw prefixes, verify all decision/active-state contracts, all mission counts and
scalar episodes; reconcile account against canonical ledger. Tests cover future
mutation/prefix invariance, no future extrema, stale/missing data, hard priority,
deadline/nonextension, loss of confirmation,1h current15m execution, unchanged
WEAK cooldown, actual skip attribution and same-entry uncertainty/denominators.
Generate metric table, category chart and earliest changed same-entry TEST price
examples chosen by entry chronology, never return. Preserve any failures.
Full/staged Harness, focused/regression tests and diff review then commit/push
source/spec/tests only. Reports/data/models/runtime are uncommitted. Rollback:
stop offline runner/remove wrapper; production remains unchanged.

## Runtime diagnosis

V1 completed both simulations, then its metric calculation slowed heavily.
Separate diagnostic reproduced raw challenger counts (TEST243early/473capture)
and episodes. A disposable20second faulthandler timing worker intentionally exited
with timeout in repeated day-key conversion; this is not policy verification.
An attempted Stop-Process on verified own PID failed; a later identity-checked
taskkill request found the worker ALREADY FINISHED and performed no termination.
V1 finished normally with COMPLETE and exit0 before recovery. A proposed V2
finalizer correctly refused the now-complete checkpoint and published no result;
unused recovery sources removed. V1 is the sole final result. No parameter,
decision, source or label rule changed during diagnosis; all failures retained.

First independent audit FAIL: original engine boundary liquidation calls
progression again at len(data)-2 after the last tick. Exactly one GMT15m active
callback is backdated by one bar, then original boundary helper finalizes at end.
No live tick clock waiver: verifier permits only exact inherited len-2 callback,
last clock=end, actual trade reason=open_at_end and final exit=end. Classify this
as terminal legacy recheck, not causal action. Forced episodes excluded from exit
quality, fixed trades/parameters unchanged. Preserve failed log/audit inputs;
repeat full independent audit in fresh verification_inputs_v2 with focused test.

## Completed fixed policy replay

Sole final run `.runtime/leader_trend_continuation_20261006_v1`, result SHA256
db43071c95b8c920e061d5fa44535374191f4ea69caec86936a8f99b5d551205.
Control11010 trades matches pinned parent ALL fields; challenger10439 trades.
3554 first soft proposals,2083 deferrals,6655 trace ticks (one inherited terminal
recheck,6654 causal ticks). Actual cooldown skips8028/7107. No live change.

| Cohort | Control early / leaders | Rule early / leaders | Control capture | Rule capture | Control leader BUY / BUY | Rule leader BUY / BUY |
|---|---:|---:|---:|---:|---:|---:|
| full |1031/2790|1025/2790|2219/2790|2203/2790|5444/10376|5132/9822|
| TEST |247/570|243/570|480/570|473/570|1250/2470|1185/2347|

292 same-entry confirmed TEST episodes over38days: mean held-time gain
.03056705365618735 ratio (+3.0567pp), corrected familywise95%CI
[.00788743880685256,.05276942369231888] (+.7887..5.2769pp). This is a real
retrospective COMPONENT gain, not overall mission improvement. First premature
exit count delta0; retained peak-rise delta-.027617817010152007 ratio, CI
[-.06266508532114773,.008611772765877929]. Rule loses early/coverage/precision
on full and TEST. Verdict MISSION_TRADEOFF_OR_WORSE, runtime_eligible=false.

All-survivor TEST held time.533776control/.558417rule has different populations,
so only292 exact-entry comparison determines component gain. Rule first-retention
.076004 on419 vscontrol+.079749 on430; marker is operational, not ultimate peak.
Control reasons among430confirmed: WEAK238 (229before,205/238laternewhigh),
ATR87 (80before), otherhard92 (87before), replacement11 (9before), time2
(2before). Cooldown756 events inside275/430control confirmed leader episodes;
rule662 inside258/419. These are actual blocked candidates, not viable successful
counterfactual entries. Partial first reductions categorized by final SELL reason.
Costs/account safety diagnostic only: full net-96.256673/-97.250732%,TEST
-44.809011/-43.138735%; not live account metrics or mission winner criterion.

Independent final audit PASS,4422 scalar episodes,2790 raw daily labels,
3554 first soft decisions,6654 causal active ticks+one terminal recheck,15135
actual cooldown candidates; verifier SHA256
52cfe998475143f5effdc882e02860e39ae27c1cbf8289d01812958c805ac101.
46focused/relevant tests PASS, full Harness PASS. First failed boundary audit
retained; its narrowly verified exception never approves live clock regressions.
Publication also validates all326 TEST clock/price-matched pairs agree on all16
immutable entry fields (tf/mode/indices/risk/entry scores), excluding future
annotations. Changed SELL examples include portfolio replacement effects;
ENA's earlier replacement is not represented as direct WEAK deferral.
