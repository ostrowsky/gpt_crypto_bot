# Learning Loop Architecture Roadmap

Date: 2026-08-14
Status: active; Phase -1 walking skeleton complete, Phase 0 evidence capacity in progress

## Principle

Every major target metric should have three layers:

1. measurement: daily labels and attribution;
2. shadow learner: model/policy recommendation without trading side effects;
3. replay/promotion gate: production adoption only after robust evidence.

The program objective is not the number of agents, hypotheses, or promotions.
It is the number of decision-grade terminal results produced per unit of
calendar time, compute, and operator attention without weakening provenance,
power, or guardrails.

## 2026-08-14 Evidence-Throughput Implementation Order

The control-plane design is not permission to build another large research
framework before completing an experiment. Its implementation order is:

1. **Walking skeleton first:** the dedicated
   `FixtureDeltaValidatorAdapter` validates at most 64 immutable fixture rows,
   independent verification reads the frozen raw fixture, and one append-only
   attempt closes in under ten seconds. It does not import the monolithic market
   replay or any production policy. The executable contract is
   [`control-plane-walking-skeleton.md`](control-plane-walking-skeleton.md).
2. **Measure and expand evidence capacity:** canonical labels, power/throughput
   report, logging repair, dependence-aware pooling, continuous outcomes,
   metric/capability bindings, and migration of negative/inconclusive research.
   The executable Phase 0 contract is
   [`phase0-evidence-capacity.md`](phase0-evidence-capacity.md). Its first slice
   is measurement-only and cannot change WATCH/BUY/SELL/portfolio behavior.

### 2026-08-14 Phase 0 evidence checkpoint

The exact-name maximum final-critic audit found `118` artifacts from
`2026-04-01` through `2026-08-13`. Seventeen calendar days are missing and two
additional days (`2026-06-10`, `2026-07-09`) fail the complete top-N integrity
contract, leaving coverage `116/135`. Recomputed mission labels contain `256`
watchlist-filtered exchange top movers and `153` early captures (`59.77%`),
with a deterministic day-cluster bootstrap interval of approximately
`51.04%..67.97%`.

This is a baseline, not an improvement verdict. The conservative power report
has `ESS=97` event-bearing days, `MDE=14.07pp` against `SESOI=5pp`, and estimated
inconclusive probability `89.29%`; status is `UNDERPOWERED`. The nominal
additional horizon is `672` complete event-bearing days, so the next action is
power expansion/continuous-response design and coverage repair, not waiting two
years or weakening the evidence gate.

The maximum critic-dataset provenance audit covers `26,202` total rows and
`22,163` labeled rows but `0` provenance-verified rows. Ranker evidence is now a
fresh `blocked_insufficient_provenance` readiness record with
`achievement_claimed=false` and `runtime_eligible=false`; no model artifact was
rewritten. TH-03 is therefore a visible collection warning rather than a false
training claim. The separate current TH-11 portfolio-alpha artifact still lacks
`current_policy_epoch` and remains blocked in its own scope.

Legacy migration assigns all `109` discovered artifacts a state. Five reviewed
negative results pass the period/population/metric/verdict contract and exact
source hash; `104` remain `LEGACY_UNVERIFIED`. No ambiguous prose was upgraded.

### 2026-08-20 maximum-period replay recovery checkpoint

The network-only validator returned an empty market population for both the
wall-clock and anchored windows. The replay now has a fail-closed `local-only`
adapter over the existing signal-quality candle cache, with deterministic
timestamp merge/deduplication and complete-1h-only 4h aggregation. This is a
measurement-path change; it does not change candidate, BUY, SELL, replacement,
or sizing decisions.

The full-watchlist `2026-07-21T09:00Z .. 2026-08-20T09:00Z` 30-day replay is
decision-grade: period coverage `98.51%`, valuation coverage `100%`, `1,622`
closed trades, zero portfolio-contract violations, and current policy/source
provenance. It restores TH-11 and the full Truth Harness is `PASS` with only the
honest zero-verified-row training warning.

The baseline is economically poor: portfolio return after costs `-47.84%`
versus BTC `+4.46%`, net alpha `-52.30pp`, cost drag `25.995pp`, and maximum
drawdown `51.03%`. Top-15 replay recall is `100%`, trade precision `23.7%`,
median capture ratio `0.0`, median exit efficiency `-0.3043`, and median
giveback `0.9804pp`. These are baseline measurements, not improvement deltas.
The next trading hypothesis must use the identical frozen local population and
report paired objective and portfolio deltas; no production relaxation follows
from this checkpoint.
3. **Prove the durable deterministic loop:** leases, retries and recovery reuse
   the skeleton, then one real manually/deterministically selected hypothesis
   must reach a terminal result within 30 days.
4. **Evaluate whether an LLM is necessary:** frozen-world evaluation compares it
   with the deterministic baseline after LLM cost, latency, and operator time.
   Human comparison is optional when a complete timestamped baseline exists.
5. **Only then allow one agent-selected experiment and alert shadow:** no
   BUY/SELL or portfolio autonomy is introduced by the control plane.

There is one accountable owner (`repository maintainer`) and one triage queue.
Subsystem names categorize work but do not pretend five staffed teams exist.
Blockers have an acknowledgement target at the next weekly triage;
`ACCEPTED_DEBT(review_at=...)` is an honest outcome while the affected scope
remains blocked.

Power expansion is a first-class queue: extend a comparable window, use
dependence-aware partial pooling, adopt a causal continuous response, validate a
higher-volume event proxy, widen the real candidate population, or repair
missing outcomes. If none preserves the question, record `ACCEPTED_UNKNOWN`
rather than repeatedly generating underpowered hypotheses.

Planning assumptions are one evidence pack per week, at most one new primary
historical validation per week, and normally 2–4 weeks for a forward hypothesis
version to mature. Twelve decision-grade forward versions per year is a ceiling,
not a delivery target. The quarterly roadmap is therefore governed by
power-feasible share, terminal results, time-to-terminal, cost per terminal
result, label/logging loss, and evidence reuse. Multi-agent research is outside
the roadmap at this throughput.

Four consecutive infeasible/underpowered primary prechecks, or a
power-feasible share below 25% across the trailing eight, enters
`EVIDENCE_CAPACITY_RECOVERY` and pauses LLM-selected admissions. Failure to
finish the first real durable-loop hypothesis within 30 days enters
`LOOP_RECOVERY`, freezes agent/promotion expansion, and returns work to the
last proven vertical slice.

Every cycle has a versioned compute/token/operator-time envelope; optional LLM
drafts and diagnostic slices are cut before provenance, the primary validator,
raw-snapshot result verification, or mandatory guardrails.

## 2026-08-03 Evidence Checkpoint

The 2026-08-02 daily report is diagnostic-only, not a production-policy
decision report:

- the final top-gainer critic is absent, so the daily watchlist-top denominator
  is unknown rather than zero;
- the 22:00 watchlist-goal job ran until after 03:00 while it shared one
  sequential scheduler loop with the strict 00:00-00:15 final-critic slot;
  the final slot was therefore never observed;
- signal-quality candle coverage is `184 / 206`, but `20 / 22` missing series
  belong to ten symbols whose exchange status is `BREAK`; only the two missing
  `TONUSDT` series are currently `TRADING`;
- the tail-selector, entry-admission, blocker-reward, and portfolio-replacement
  caches were generated on 2026-06-13, before the current cooldown and guarded
  replacement policies, and cannot evaluate those production policies;
- rolling watchlist-top early capture is `72.73%`, while signal-quality reports
  a `74.47%` broad trend-episode miss rate. These metrics use different
  denominators and must not be treated as complements.

Consequently, no BUY/SELL threshold is relaxed from this checkpoint. The next
cycle restores trustworthy measurement first, then refreshes policy evidence on
the maximum available historical period and chronological forward slices.

## 2026-08-03 Accumulation Audit

The policies and shadow contours that were explicitly waiting for forward data
were re-audited through 2026-08-03. Raw event counts were deduplicated into
decision episodes where repeated scans emitted the same candidate.

| Contour | New evidence | Decision |
|---|---|---|
| Base cooldown `2` | equal 16-day before/after windows: false positives `25.08% -> 19.35%`, broad miss rate `81.13% -> 76.67%`, watchlist-top early capture `65.91% -> 70.97%` | keep `2`; no further relaxation; before/after evidence is supportive but not causal |
| Non-losing replacement guard | `118` unique blocked candidate/protected-position pairs; candidate minus protected 12h return averaged `+0.3542pp`, candidate won `57.76%` | forward warning; do not broaden or roll back until the frozen portfolio replay completes |
| Observable tail `non_ema_mfe150` | `147` live candidates and `100` mature T+10 labels; average partial delta `-0.1239%`, median `-0.2881%`, worse rate `59%`, before costs | reject production promotion; retire or replace this shadow selector |
| `regime_start/base_recovery_v1` | `51` mature post-deploy T+5 cases; useful precision `9.80%` vs `35%` gate, median return `-3.2704%`, median MFE `1.9665%` vs `6%` gate | live data confirms rejection; Telegram remains disabled; stop waiting on this profile |
| BTC early-trend WATCH | `18` raw matches / `11` 12h-deduplicated episodes; 6h average `-0.1788%`, median `-0.2082%`, positive `36.36%`; 12h average `+0.0068%` | post-deploy gate failed; disable/recalibrate the dedicated WATCH, retain general V2 shadow data |
| Peak-risk shadow | `40` paired events overall, `9` post-deploy; exit-minus-alert average `-0.1026pp` overall and `-0.1753pp` post-deploy, median approximately zero | no tighter exit or alert promotion; current profile has no demonstrated edge |
| Suspicious re-entry | `798` post-deploy upstream decisions, `43` registered watches, zero final alerts | waiting longer will not create labels; label registered watches counterfactually before considering threshold changes |
| V2 broad upside | `14` complete days, top recall `31/31`, precision `2.60%`, confirmation ratio `13.94%`, handoff `87.10%` | keep detector/shadow role; do not promote broad V2 to BUY |
| Score gate `34` | one blocked winner across `14` final critics; score gate was only one link and final blocker was agent mode/alignment | keep gate; test the complete blocker chain rather than lowering one threshold |
| Research universe | `130,693` valid rows but no recent mature labels; one malformed JSONL row makes the all-or-nothing label rewrite return zero | status is `labeling_broken`, not `collect_more_labels`; repair and backfill before analysis |

The fresh maximum-period tail replay did not complete within five minutes. The
fresh replacement comparison did not complete in a 15-minute parallel attempt
or a 20-minute sequential guard attempt. These runs produced no result and must
not be treated as positive or negative evidence. Replay caching/incremental
execution is now an operational prerequisite for the frozen replacement audit.

## Current Gaps

### Entry / early capture

Current: V1 rules, ML ranker, V2 shadow radar.

Learning loop exists but is incomplete: V2 has high recall and low precision. Next learning loop should focus on calibrated admission precision, not more raw alerts.

### Selection / portfolio cap

Current: score gates and ranker training.

Learning loop exists partially through ML candidate ranker. Missing: explicit EV/risk calibration under 10-slot portfolio cap and replacement uplift attribution.

### Blocked winners

Current: structured blocked logging and critic.

Learning loop is measurement-heavy, not adaptive. Missing: per-blocker harm model that proposes candidate relaxations in shadow/replay only.

### Exit monetization

Current: V1 rule-based SELL, exit auditor, exit discriminator, suspicious re-entry replay.

Learning loop did not affect live behavior until shadow re-entry alerts. Next: daily scoring of shadow re-entry precision and forward returns.

### Risk / exposure

Current: portfolio cap and group cap.

Missing: correlation/exposure learner in shadow mode, then replay-gated cap adjustments.

## Recommended Order

### P0: restore decision-grade measurement

1. Split the watchlist-goal and top-gainer-critic schedulers so a long goal job
   cannot block midday/final critic slots.
2. Add persistent due-slot catch-up: if a final artifact is absent after its
   nominal window, retry it independently until it succeeds or records an
   explicit terminal error. A process restart must not erase the due slot.
3. Make critic-dataset rewrites single-writer and atomic with bounded retry;
   expose permission/rename failures in worker health instead of only logs.
4. Separate active-universe coverage from retired/broken-symbol coverage.
   Retry missing `TRADING` series; report `BREAK` series as excluded legacy
   coverage rather than as equivalent live-data failures.
5. Add freshness budgets for every research cache and report the source-policy
   version/config hash used to build it.
6. Make research-universe labeling tolerant of malformed rows, quarantine bad
   records, and backfill the accumulated dataset. A single invalid row must not
   cancel labels for the entire file.
7. Replace full JSONL rewrites and repeated full-file-per-symbol scans with an
   incremental cohort store. The daily loop must not run maximum-period replay.
8. Label every registered suspicious re-entry watch at T+2/T+5/T+10, including
   watches that never pass final alert confirmation.

P0 acceptance gate:

- seven consecutive target days have a final critic before the 09:00 learning
  report, including restart recovery;
- no scheduled final slot is lost when the 22:00 goal report runs for more than
  two hours;
- active-universe candle coverage is 100%, or every remaining active miss has a
  named terminal reason and is excluded from decision-grade metrics;
- no critic-dataset permission/rename failure occurs during the observation
  window;
- stale research artifacts cannot produce a current production recommendation.
- research-universe scorecards contain recent mature labels and explicitly
  report quarantined rows;
- registered re-entry watches accumulate counterfactual labels even when final
  alert count remains zero.

Coverage checkpoint (2026-08-03): implemented active/inactive separation,
six-hour exchange-status freshness with provenance, conservative handling of a
failed refresh, and retry of empty candle-cache entries. The apparent active
TONUSDT gap was a stale local `TRADING` status; Binance reports `BREAK` and the
series is now an explicit inactive exclusion. The seven-day operational
acceptance window remains open.

Freshness/provenance checkpoint (2026-08-03): the four decision-bearing cached
research replays in the morning report now carry builder, live-policy hash,
research-config hash, generation time, and input watermarks. A stale or
policy-mismatched artifact is diagnostic-only and cannot produce a current
relaxation recommendation. Extending this contract to non-decision telemetry
remains follow-up hardening.

Incremental-cohort checkpoint (2026-08-03): entry-admission, blocker-reward,
and portfolio-replacement research now share a transactional SQLite event
cohort store. The first sync backfills maximum available history; later runs
consume only appended log bytes. Compact blocker cohorts replace repeated
multi-million-row Python groupings. Runtime population completed over 2,373,955
lines / 746.6 MB; the next sync processed only 40 KB / 103 appended lines in
1.29 seconds. The store is operational; tail-label caching remains a separate
performance follow-up.

Exit-selector checkpoint (2026-08-03): `exclude_ema_and_false_cleanup` passed
both the 76-day maximum-period chronological holdout and the recent 14-day
stability window. It replaces the rejected `non_ema_mfe150` profile in the
existing observable-tail collector, shadow-only. Production SELL remains
unchanged until an independent mature forward cohort passes the promotion
guardrails.

Current-policy reward checkpoint (2026-08-03): maximum available blocker/event
history now covers 89 critic days, 1,787,853 blocked events, and 21,203 compact
cohorts. Entry admission remains rejected (`cooldown_only`: 1 top / 18 false,
5.3% precision, `-18%` net reward). No blocker passed the harm gate. Portfolio
replacement is now neutral over 505 closed cases (average delta `+0.043%`,
median `-0.149%`, positive rate `44.0%`); the best causal filter still has
`35.77%` regret and is not promotable. No BUY, blocker, or rotation relaxation
was applied.

### P1: refresh the evidence under current production policy

Run every comparison on the maximum available historical period, with
chronological splits and a separate post-deployment forward slice. Include fees,
slippage, portfolio capacity, and the canonical watchlist-top denominator.

1. Keep cooldown `2`; its first forward window is supportive. Re-run it versus
   the former `8` control only when a causal rollback warning appears. Do not
   relax cooldown further from a before/after comparison alone.
2. Re-run the guarded non-losing replacement policy against no replacement and
   the former replacement behavior after replay caching is fixed. Until both
   maximum-period and fresh stability windows finish, treat the live cohort as
   a warning and keep the current guard unchanged.
   The paired 2026-08-03 rerun is now complete: 30d favors the guard by
   `+10.7653pp` total PnL, while 14d reverses by `-7.7212pp`; Top-20 capture is
   `95%` in both variants/windows. Evidence is conflicted, so keep the current
   guard unchanged and require another stability window before rollback.
3. Audit the score-34 near-miss band with a frozen control: attribute candidates
   to detection, score admission, other blockers, portfolio capacity, and later
   outcome. Test score `32-33` first as WATCH/shadow and promote to BUY only if
   objective uplift survives all chronological windows without unacceptable
   precision or PnL loss.
   The frozen 2026-08-03 audit is complete over all final-critic days from
   `2026-04-01` through `2026-08-02`: `1,259` unique day/symbol candidates and
   `1,179` mature outcomes. After excluding full-capacity and already-bought
   cases, `196` candidates were admission-eligible. Net T+10 was negative over
   all history (average `-0.125%`, median `-0.334%`), chronological holdout
   (average `-0.160%`, median `-0.343%`), and recent 14-day stability (average
   `-0.030%`, median `-0.226%`). Holdout top-mover precision was only `3.82%`.
   Reject the `32-33` WATCH hypothesis and keep the live score floor at `34`.
4. Reject `non_ema_mfe150` for production from its negative independent live
   labels. Search a materially different observable exit hypothesis; do not
   retune thresholds on the same failed cohort.
5. Rebuild entry-admission and blocker-reward reports only after final-critic
   backfill is complete. Keep all admission/blocker relaxations shadow-only
   until a targeted replay passes.
6. Repair suspicious re-entry measurement before more collection. Zero alerts
   after `43` registrations is a zero-conversion funnel, not a request for more
   of the same unlabeled data.
7. Disable or recalibrate the dedicated BTC WATCH after its post-deploy forward
   gate failure; retain V2 shadow observation and use at least 12h episode
   deduplication in the next replay.
8. Retire `base_recovery_v1` as a promotion candidate. Any next regime-start
   hypothesis must be materially different and use the failed live cases as a
   frozen holdout.

P1 promotion gate:

- the primary decision metric is earlier same-day watchlist-top capture;
- false-positive pressure, realized PnL, drawdown, turnover, and portfolio
  capacity are mandatory guardrails;
- an aggregate improvement that reverses in the latest or a material regime
  window is not promotable;
- production changes require focused automated tests, a rollback switch, and a
  frozen replay artifact with policy/config provenance.

### P2: improve attribution before adding model complexity

1. Decompose broad missed trends into: not observed, observed but not signaled,
   signaled but rejected, blocked by portfolio capacity, entered late, and exited
   early.
2. Publish both metric families without conflation: watchlist-top early capture
   for the north star, and broad trend-episode miss rate for lifecycle coverage.
3. Rank the daily failure casebook by realizable opportunity after the decision
   point, not by hindsight daily gain.
4. Resume calibrated V2 admission, portfolio EV/risk ranking, blocker harm, and
   correlation/exposure learners only after P0 is stable and their training
   labels carry current-policy provenance.

P2 attribution checkpoint (2026-08-03): implemented complete failure-detail
export for new signal-quality reports, causal 15-minute blocker intervals, and
a lifecycle report that keeps watchlist-top early capture separate from broad
trend episodes. The maximum available artifact spans 73 signal-quality days and
contains 6,913 unique exported missed episodes, 1,924 late entries, and 598
early exits. Missed-stage attribution is dominated by `signaled_but_rejected`
(`4,731`), followed by `blocked_by_portfolio_capacity` (`985`), `not_observed`
inside observer coverage (`742`), `observation_coverage_unavailable` (`375`),
and `observed_but_not_signaled` (`80`).

The historical aggregate remains `partial_historical_detail`: 60 of 73 old
signal-quality reports were produced under former 100/50-row detail caps. The
latest day is complete and all new reports carry explicit detail coverage.
The nightly worker uses a bounded 14-report window; maximum-period attribution
is an explicit research run and cannot block the daily scheduler. The casebook
ranks only net movement remaining after an actual blocker/shadow/entry/exit
decision price and deduplicates day/symbol/stage. No trading policy changed.

Deferred agent wake-up checkpoint (2026-08-03): the May
`agent_wakeup_rescue` hypothesis now has sufficient temporal coverage and is
rejected. A paired 80-day replay admitted 704 rescue candidates but left Top-15
capture unchanged at 100%, added 112 trades, worsened total PnL by `-15.9121pp`,
and increased portfolio-full skips by 359. The independent recent 14-day window
also left capture unchanged at 93.33%, worsened PnL by `-2.3237pp`, reduced
trade precision `26.04% -> 25.00%`, and added 53 portfolio-full skips. Keep
`AGENT_ALLOWED_MODES` unchanged and do not retune the same profile.

## 2026-08-03 Early-Trend Discriminator Checkpoint

The maximum-period research-universe replay rejected all 648 transparent
ignition-rule combinations at the frozen train gate. A materially different
CatBoost adjunct then passed validation and untouched holdout: holdout primary
precision improved by `+1.50pp`, recall by `+3.74pp`, and average T+10 by
`+0.046pp`; the strict label also improved. On 9 critic days / 26 canonical top
movers, recall was unchanged and three captured movers were surfaced 560
minutes earlier on average, while top-mover precision declined `0.94pp` within
the frozen guardrail.

The profile advances only to independent research-dataset shadow annotation.
No BUY, score gate, blocker, portfolio, V2, or Telegram behavior changes. The
next decision point is a new forward cohort with first-event symbol/day
deduplication; do not reuse replay holdout rows as forward evidence.

## 2026-08-03 Prioritized Hypothesis Backtest Checkpoint

Three next-step hypotheses were evaluated causally before any production
implementation:

1. The current CatBoost EV/opportunity ranker failed its chronological
   capacity-ranking pre-gate. On test top-1/top-3/top-5 competitions it reduced
   teacher top-gainer rate by `2.59-5.11pp` and capture by `0.87-1.70pp`;
   top-3/top-5 also reduced forward return. Do not run the full ten-slot replay
   or enable this frozen model for portfolio allocation.
2. The action-level exit model was corrected to include causal trade-path
   context and use train/validation/untouched-holdout day splits with purges.
   The validation-selected model captured only `62.322330` advantage on
   holdout versus `21,869.736704` for always-sell. Reject it before full
   environment replay; production SELL remains unchanged.
3. Losing-incumbent replacement with a train-selected leader-delta threshold
   passed the recent 14-day slice but failed train median/positive-rate and
   holdout blocked-regret gates. Keep the current guard and require a new
   independent stability window before testing a regime-conditional profile.

The next active evidence gates are therefore forward, not threshold retuning:
the frozen early-trend discriminator cohort, the observable tail-selector
cohort, the seven-day measurement acceptance window, and replacement stability.

## 2026-08-04 Forward-Gate Checkpoint

The next two replay-approved profiles are now evaluated by one provenance-safe
forward report. It excludes every research row at or before the frozen model's
creation time, deduplicates early-trend candidates to the first symbol/local-day
event, matches observable-tail labels to their causal candidate, and joins
canonical top movers only from final critics.

Current independent evidence is not yet large enough for promotion. The early-
trend cohort has 9 mature first candidates over 2 local days versus a minimum
of 30 over 5 days. Primary precision is `11.11%`, but average and median T+5
are `-0.7864%` and `-0.5587%`; its one available final-critic day adds no new
or earlier canonical top mover. The new observable-tail profile has 11
candidates and 6 mature T+10 labels versus a minimum of 30 over 5 days. Its
average T+10 partial delta is currently `-0.0013%` and median is `+0.0552%`.
Both remain shadow-only.

## 2026-08-21 Objective Leader Admission Checkpoint

The objective values in this checkpoint were later superseded by the daily
watchlist objective correction below. The variants remain disabled; their
mission comparison must be rerun before any conclusion is reused.

Three materially different causal profiles were tested over the frozen
maximum 30-day, 105-symbol, ten-slot replay: a quality-bounded continuation
chase bypass, a reserved objective-leader slot, and their interaction. All
retained 15/15 Top-15 recall, but none improved the mission gate. Against the
same-snapshot production control, average capture changed from 16.38% to
16.28%, 16.48%, and 16.41%; median capture remained 0% and median lead remained
570 minutes. The reserve admitted only one leader while suppressing 2,064
ordinary candidates; the combined profile admitted two while suppressing
2,144. Their lower drawdown is inactivity, not evidence of better leader
capture. A same-timestamp boundary-accounting defect was repaired and the four
canonical reports are now decision-grade with complete coverage and no
contract violations. Reject all profiles and do not change production. Next
priority is causal ranking among simultaneous admissible candidates with
explicit incumbent/candidate regret attribution.

## 2026-08-21 Daily Objective Denominator Correction

Replay measurement was found to rank one Top-15 set over the whole 30-day
window and average capture over all trades. This did not represent the bot's
same-day watchlist mission. The corrected decision-grade control covers 29
complete local days and 435 Top-15 day-symbol labels: captured `353/435`
(`81.15%`), early `132/435` (`30.34%`), with trade precision `828/1509`
(`54.87%`). Captured-pair median capture is `21.43%`. Daily available symbols
range down to 91 of 105 requested, so coverage remains explicit. The portfolio
guardrail is still weak: net `-47.2765%`, BTC-relative alpha `-55.1296pp`, and
drawdown `51.0277%`. All later ranking work must use daily pair labels; prior
whole-window `15/15` claims are superseded.

The replacement stability audit was also refreshed over 510 closed cases. Its
recent slice remains favorable, but train still fails median, positive-rate,
and regret gates, while holdout blocked-positive regret is `53.85%`. Keep the
non-losing incumbent guard and reject the targeted relaxation.

## 2026-08-05 Daily Report and Deferred-Evidence Audit

The 2026-08-04 daily report has a valid final critic and safe partial candle
coverage: all 22 missing series are fresh-status `BREAK` exclusions. Its one
watchlist top mover was bought and early-captured, so the daily 100% figures are
correct but weak evidence. Rolling early capture remains above the 25% target
at `56.2%`, while declining from `71.4%`; with medium confidence the correct
verdict is `УХУДШИЛСЯ ПО EARLY-CAPTURE`, not a flat verdict caused by alleged
training failure. Broad trend miss-rate `70.18%` uses a different denominator
and is not the complement of watchlist-top capture.

The ML freshness alarm was false. The last successful run consumed 22,163
ranker-eligible rows, the current eligible count and dataset watermark are
unchanged, and the worker has no error. Training is correctly waiting for new
labels under its data-driven trigger; daily retraining is not required. The
current model nevertheless remains rejected for portfolio allocation because
its untouched test reduced teacher top-gainer rate and capture.

Current-policy refreshes preserve the previous policy decisions:

- entry admission: 91 critic days / 1,842,138 blocked events, best net reward
  `-18%`; reject BUY expansion;
- blocker reward: 21,801 cases, no blocker passed the harm gate; keep blockers;
- portfolio replacement: 510 closed cases, average delta `+0.0462%`, median
  `-0.137%`, positive `44.31%`; causal filters retain `34.89-40.33%` regret and
  are not promotable;
- suspicious re-entry: five registered-watch labels, average T+5 `+1.66%` but
  only `40%` positive and zero final confirmations; continue measurement only.

The tail-selector maximum-period run timed out after five minutes and the
14-day run after three minutes. Neither produced evidence. Incremental
tail-label caching is therefore the next executable P0 engineering item before
another replay. Independent forward evidence is already unfavorable but below
the frozen decision support: observable tail has 13/30 mature T+10 cases,
average/median partial delta `-0.2747%/-0.2096%`, worse rate `53.85%`; early
trend has 15/30 mature cases over 3/5 days, primary precision `6.67%`, average
T+5 `-0.5904%`, and no new or earlier canonical top mover across two critic
days. Keep both profiles shadow-only until their frozen minimum support is met.

P0 operational acceptance is now 2/7 consecutive post-fix target days with a
final critic before the 09:00 report. No production BUY, SELL, blocker,
portfolio, cooldown, or Telegram trade behavior changes follow from this audit.

### Prioritized validation queue after the 2026-08-05 audit

1. **Incremental tail-label cache (P0 engineering).** Cache candle-derived exit
   labels by immutable trade/horizon identity and make maximum-period plus
   recent-window replay incremental. Acceptance: the same replay result as the
   uncached implementation on a frozen fixture, a warm run under 60 seconds,
   and no live SELL change.
2. **Frozen observable-tail forward decision.** Do not retune
   `exclude_ema_and_false_cleanup` on its first 13 cases. At 30 mature T+10
   cases over five days, apply the already frozen average, median, and worse-
   rate gate. If it fails, retire the profile. Only then may a materially
   different exit-reason x market-regime selector be trained on maximum history
   and checked on a new untouched holdout.
3. **Frozen early-trend forward decision.** Continue to 30 mature first
   symbol/day candidates, five local days, five critic days, and five canonical
   top movers. Require proxy returns plus at least one new or earlier canonical
   capture. Failure retires this discriminator; no threshold tuning on its
   forward cohort.
4. **Re-entry confirmation-funnel replay.** Continue registered-watch labels to
   at least 30 mature cases. Then compare the current zero-conversion final
   confirmation with a frozen registered-watch policy using T+5/T+10 return,
   positive rate, drawdown, false-entry penalty, fees, and ten-slot opportunity
   cost. No confirmation relaxation from the current five labels.
5. **Objective-aligned ranker v2.** Retraining freshness is not the problem.
   After new labels arrive, test a materially different model whose primary
   loss directly includes canonical top-mover/capture targets rather than
   promoting the current EV proxy. Require chronological validation, untouched
   test improvement in top-1/top-3/top-5 teacher metrics, and full ten-slot
   portfolio replay before any allocation use.
6. **Conditional replacement only after a valid causal ranker.** The favorable
   losing-incumbent and hindsight incoming-top segments are hypothesis
   generators, not a production rule. Revisit replacement only when a causal
   decision-time incoming-quality estimate passes its own independent gate;
   require blocked-positive regret at most 25% in train, holdout, and recent
   windows.

## 2026-08-13 Truthful Provenance Checkpoint

The next P0/P2 prerequisite is implemented: new ML and critic observations now
carry an immutable policy epoch and causal feature cutoff, while every new
forward/outcome/teacher label carries its definition and availability time.
Candidate-ranker training is fail-closed to provenance-verified rows by
default, split boundaries preserve complete decision-time groups, and runtime
loading rejects model payloads without decision-grade evaluation provenance.

Legacy rows are deliberately `legacy_unknown`; they are not relabeled as the
current policy. Therefore model retraining may wait for a new mature cohort and
old ranker scores remain diagnostic rather than proof of self-improvement. The
next roadmap item remains the canonical unified ten-slot portfolio-alpha
evaluator after fees/slippage and against a named benchmark. Only after that
evaluator and a sufficient verified cohort exist should objective-aligned
ranker v2 be reconsidered.

Entry-admission rescue, blocker relaxation, score 32-33, delayed +120m entry,
agent wake-up rescue, and the current EV/action-exit models remain closed after
failed maximum-period or untouched-holdout gates. They are not active retuning
tasks.

## 2026-08-13 Canonical Portfolio-Alpha Checkpoint

The P0 measurement contract is implemented in
`docs/specs/canonical-portfolio-alpha.md`. Replay profitability is now evaluated
as one symbol-deduplicated ten-slot cash account after fees and slippage against
the named `BTCUSDT` buy-and-hold benchmark. The legacy arithmetic `pnl_total`
remains diagnostic-only and cannot satisfy Truth Harness TH-11.

This checkpoint changes no live entry, exit, replacement, or sizing rule. The
next priority remains collecting a sufficient provenance-verified cohort for an
objective-aligned ranker v2. Frozen exit-tail, early-trend, re-entry, and
replacement hypotheses retain their existing forward-label gates; they are not
unblocked by portfolio measurement alone.

The completed 30d/103-symbol run reports `-49.98%` net portfolio return versus
`-2.49%` for BTC and therefore `-47.49pp` net alpha after costs, with about `50.00%`
maximum drawdown. TH-11 now passes, but the result rejects any claim that the
current profile is profitable. Before ranker v2 can affect allocation, the next
P0 analysis is a frozen-cache paired canonical-alpha attribution of turnover and
the already frozen production alternatives. It must identify an improvement
without relaxing entry/exit gates; otherwise the current policy remains
unchanged while the verified label cohort accumulates.
