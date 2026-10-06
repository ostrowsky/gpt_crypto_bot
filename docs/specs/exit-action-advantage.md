# Causal SELL action advantage: protected one-bar deferral

Registered2026-10-06 before fitting/results; offline only. TH-01..12 apply.
Full Truth Harness PASS. Parent: joint-direction-amplitude.md,
impulse-entry-catboost.md, learned-exit-tail-policy-replay.md,
SCOUT_OPTIMIZATION_SPEC.md. Earlier continuation-classification/partial-tail
family and entry filters remain rejected; no production relaxation.

## Fixed hypothesis and actual decision population

Can expected CASH advantage of one concrete hold action improve monetization,
without losing early leader capture or increasing risk? Population: ALL original
rule-only control trades whose actual SELL reason contains WEAK:, all entry
modes/timeframes, from maximum complete186.333-day recovered93/105-symbol
archive Apr1-Oct4 06UTC. Do not select by future MFE, top rank or profitable
outcome. Use receipt-verified control states/trades/native features and raw market
from prior experiments. Recheck source/input hashes before reuse; controls may
reuse trades but cash/objective recalculated. Actual agent/PIT/receive-time and
fill parity still uncertified; historical exposed data is retrospective.

## Concrete action and causal features

At first WEAK SELL, compare immediate full sale to holding full remaining
position until next timeframe candle closes (15m or1h). During deferral invoke
the ORIGINAL progression kernel: ATR, time limit, fast loss/EMA and micro hard
exits still execute. Suppress only WEAK reasons before deadline; force full sale
at deadline even if no exit signal; no recursive extensions. Replacement and
boundary liquidation remain unchanged. Hard SELLs never overridden. Deferral
state keyed by symbol/timeframe/entry time and installed only after real soft
SELL proposal; unchanged BUY score, candidate stream, cooldown and allocation.
Require exact timeframe-close origin and matching raw close execution price;
stale/nonmatching origins are logged unknown past state and retain immediate
SELL. This avoids claiming edge from an unavailable stale-price fill.
Deadline reason preserves the original WEAK category, so existing weak-exit
cooldown applies unchanged. Hard exits keep their actual hard reason/category.

Label: cash proceeds difference per one unit of remaining asset, normalized
by soft-SELL origin raw price:100*(hold_exit_price/sell_now_price-1)*
(1-fee)*(1-slip). Both actions incur one SELL, so do not invent a second BUY/SELL
cost. Use first actual hard exit or fixed deadline, not future maximum. Label
availability conservatively deadline even if action exited earlier. Incomplete
future path/backdated action exit is unknown, excluded from fitting/diagnostics;
future availability NEVER gates runtime inference. This single-position action
target omits capacity/replacement opportunity cost; full portfolio replay is
required to assess those endogenous effects.

Features: same10 exact17-closed15m market-prefix features as prior experiments,
plus current unrealized PnL, observed-to-origin MFE/giveback, elapsed minutes,
remaining max-hold minutes, distance to current trail stop, and six fixed entry
mode one-hot flags. No final-day capture/rank, future extrema/price, exit outcome
or teacher/agent retrospective fields. Baseline trade state at its actual soft
SELL is causal; direct prediction inputs are reconstructed from as-of market
and entry, rather than final report labels. Runtime uses live replay trade state.

## Fixed learning and evaluation

First30days unchanged, expanding refit every30days. Whole-local-day inner
train80%/validation20%; purge any action label reaching validation boundary,
require all training/validation labels matured strictly before refit. CatBoost
RMSE400/depth3/lr.03/l2=10/seed42/threads2, early stopping40 on validation only;
minimum train300/validation50, otherwise explicit untrained passthrough.
Freeze model before next block. Defer only predicted action advantage>+.05pp;
unknown features retain immediate SELL. Fixed threshold/action, no grid search.
Earlier matured TEST labels can train later blocks: explicitly prequential OOS,
not sealed. Control-distribution training and changed-policy distribution shift
disclosed; log every actual challenger soft decision and its causal features.

One new policy vs original control; unchanged prior joint entry model is optional
descriptive reference only. Re-simulate full portfolio on original ALL candidates
with original entry/exit kernel except local soft-exit deferral wrapper. Every
15m cash mark and total fees/slip must equal canonical account. Report full/TEST
net/BTC alpha/DD/exposure/turnover, unique early/captured leader pairs and
precision counts, MFE retention/giveback with known denominators; frozen-entry
action deltas, affected/worse/p10/mean counts separately from account alpha.
Single-comparison TEST daily log-return3-calendar-day moving-block bootstrap,
5000seed42,95%CI. >=30completeTESTdays, full+TESTreturngain>=1pp,
lowerCI>0, no fewer early/captured pairs full+TEST, precision drop<=.5pp,
DDworse<=1pp. Selected known OOS action mean>0 and hurt-rate<=35% required
as additional action guardrails, never substitutes for full portfolio gates.
Passing numerically permits only a fresh shadow design; runtime_eligible=false.
Do not promote from fewer exits or isolated selected percentage-return sums.

## Verification, delivery and rollback

Tests: future-prefix mutation, observed-MFE causality, label deadline/purge,
one-SELL cash delta, mandatory hard exits, no repeated extensions, timeout,
future-label-independent actual native fit and inference. Independent verifier
reconstructs all dataset features/labels/action paths from raw market and frozen
original kernels, replays all model predictions/actual challenger decision
features and screens, cross-checks canonical cash/objective/gate. Preserve
native state snapshots and every action trace with SHA receipts. Generate
same-period full/TEST account plot, action/giveback diagnostics and report.
Spec/tests/full+staged Harness/diff review/commit/push source/spec/tests only.
Reports/prices/weights/positions never committed. Rollback stops own worker;
no active BUY/SELL/config/orders/positions touched. Future live adoption needs
complete population/kernel/fill parity, new forward/paper/canary and immediate
rollback flag. Retain failures; rejected policy never silently reintroduced.

Integration correction before portfolio results: v1 worker stopped and retained
as SUPERSEDED_BEFORE_POLICY_RESULTS. Artificial timeout reason had changed WEAK
cooldown from its original category; v2 retains original WEAK reason with a
deadline annotation. No model parameter/action/threshold changed or selected
using outcome evidence. Correction has a focused cooldown-equivalence test.

## Reproduction

Embedded Python: prepend `.runtime/forecast_dependencies` and `files` to
sys.path; use runpy.run_path(...,run_name='__main__'). Set numerical-library
thread environment variables to1. Runner:

```
files/run_exit_action_advantage.py
  --parent .runtime/impulse_entry_catboost_20261006_v1
  --output .runtime/exit_action_advantage_20261006_v2
```

Use a NEW output path on reproduction. After completion run
`files/verify_exit_action_advantage.py --run <output>`, then
`files/render_exit_action_advantage.py --run <output>`. Focused tests:
test_exit_action_advantage, test_run_exit_action_advantage,
test_verify_exit_action_advantage, test_render_exit_action_advantage;
also prior causal-entry and turnover/canonical-account regressions.
Original indicator checkpoints remain producer/input/source-hash bound;
audit reconstructs as-of positions and concrete actions but does not claim an
independent full recomputation of all indicator arrays. All runtime files stay
outside Git; only sources/spec/tests committed and pushed.

## Completion outcome (2026-10-06)

Run `.runtime/exit_action_advantage_20261006_v2`, result SHA256
`59864e6db5b151222c15ec90b518705e9f4c9afe0500b697e0d37819ca4fcbdf`.
3817actual control WEAK exits,3816known concrete action labels,one boundary
exit has its next candle outside the frozen archive. Six causal fits;3207OOS
states scored,35positive-threshold control-state actions selected. Actual
changed-policy replay logs3812first soft proposals and36deferrals. Counts refer
to different state distributions and must not be conflated.

| Arm | Full net % | TEST net % | Trades | Early /2790 | Captured /2790 | TEST early /570 | TEST captured /570 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | -96.256673 | -44.809011 | 11010 | 1039 | 2219 | 253 | 480 |
| action model | -96.082419 | -41.364367 | 11017 | 1039 | 2217 | 253 | 478 |

Full account improvement only+.174255pp, below fixed1pp hurdle; TEST gain
+3.444644pp. Full alpha -120.308355/-120.134101pp,DD96.347967/96.177962%.
Full precision5444/10376 vs5445/10387. Early capture unchanged, total capture
loses2pairs full+TEST.38completeTESTdays,paired95%CI[-12.058550,+53.179100]bp
mean daily log-return difference includeszero. Registered gate REJECTED;
runtime_eligible=false. Positive descriptive account delta is not approval.

Control-state OOS selected concrete actions:35,mean+.437235pp,median+.102752,
hurt16/35,p10-1.563134pp. TEST33,mean+.576239pp,median+.609924,
hurt14/33(42.42%),p10-1.347218pp: above pre-registered35%hurt ceiling.
All3207OOS action MAE .705585 vszero-action .703770pp;984TEST MAE
.807108 vs .803875pp. Selected action gains are normalized single-position
cash differences, not executed policy or portfolio return percentages.

Unweighted winning MFE retention: control .660915 on4675/11010 trades vs
model .660241 on4675/11017; mean gross giveback1.668520 on11010/11010
vs1.672691 on11017/11017. These diagnostics do not show better average
profit retention despite positive selected action mean. Risk protected by
original hard rules and one-bar deadline; timeframe-close fills remain idealized.

V1 and corrected v2 have byte-equivalent dataset numeric arrays(clock,
available,x,y,prediction,fold): model design/targets/threshold not retuned;
only timeout exit classification/cooldown integration corrected. V1 retained
with superseded receipt and stopped own PID19352 before account/gate outcomes.
49focused/relevant tests PASS; full Truth Harness PASS. Rejected policy retained
without production change; later evidence requires separately registered work.

Independent audit PASS:3817control soft-exit states rebuilt from entry (final
extrema/stop not reused),all3207native OOS predictions and3812actual first-soft
decision features/predictions checked,36deferrals and timeout/cooldown contracts
verified; all17889marks per account and objectives/gate reconciled. Verifier
SHA256 `aaa52343b2be1639d89328cb0fa9b2ea0fb92af2f899c7838ee01a5e43b86885`.
Raw OHLCV independently JSON-decoded and compared to checkpoint arrays;
indicator arrays inherited by producer/source/input SHA (not fresh full
indicator recomputation). Corrected runner and verifier completed normally.
