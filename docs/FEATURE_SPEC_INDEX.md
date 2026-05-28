# Feature Spec Index

Last updated: 2026-05-28 15:55 Europe/Budapest

## Objective

Keep every material bot capability tied to the same north star:
earlier capture of same-day watchlist top movers with a single unified
10-position portfolio and profitable exits near trend reversal.

## Spec Catalog

| Capability | Status | Canonical spec / source | Primary metrics | Next gate |
|---|---|---|---|---|
| Spec-first workflow | shipped | `docs/specs/spec-first-workflow.md` | regression safety, reproducibility | use for every non-trivial change |
| Scout objective and metric contract | shipped | `SCOUT_OPTIMIZATION_SPEC.md` | `watchlist_top_bought`, `early_captures`, `false_positive_buys` | keep current |
| Signal-quality evaluator | shipped | `skills/signal-quality-evaluator/SKILL.md` | `miss_rate`, `capture_ratio_at_entry`, `exit_efficiency`, `giveback_pct` | feed hypothesis queue |
| Exit-quality auditor | shipped measurement-only | `docs/specs/exit-quality-auditor.md` | `exit_efficiency`, `giveback_pct`, `top_mover_exit_failure_count`, `negative_after_mfe_count` | choose SELL hypotheses for replay; no live SELL change |
| Daily learning progress report | shipped reporting | `docs/specs/daily-learning-progress-report.md` | early capture trend, capture quality, learning freshness, operator actions | send daily at 09:00 local via RL worker |
| Watchlist-filtered top-mover denominator | shipped measurement correction | `docs/specs/watchlist-filtered-top-mover-denominator.md` | `watchlist_top_capture_rate_pct`, `watchlist_top_early_capture_rate_pct`, `exchange_top_in_watchlist` | use filtered denominator for operator reports |
| Main top-gainer intraday feature parity | shipped bugfix | `docs/specs/main-top-gainer-intraday-feature-parity.md` | `top_gainer_score`, `today_change_pct`, `forecast_return_pct`, false-positive pressure | monitor next daily report; thresholds unchanged |
| Watchlist top lifecycle audit | research-only diagnostic | `docs/specs/watchlist-top-lifecycle-audit.md` | `early_failures`, `v2_to_buy_delay_min`, `exit_failures`, `exit_efficiency`, `giveback_pct` | use to choose replay-backed early/exit hypotheses |
| Ranker training freshness | shipped operational fix | `docs/specs/ranker-training-freshness.md` | model freshness, learning-loop reliability | retrain on newer local dataset even after restored row-count rollback |
| Blocked-winner focus audit | shipped diagnostic tooling | `docs/specs/blocked-winner-focus-audit.md` | blocked-winner explainability, operator answer latency | use for specific “why was/wasn't symbol bought?” checks |
| Watchlist rescue admission replay | research-only implementation | `docs/specs/watchlist-rescue-admission-replay.md` | early watchlist capture, ret5 precision, missed-winner rescue candidates | advance only if holdout profile passes gate |
| 1m wake-up scout | shipped shadow-assisted | `SCOUT_OPTIMIZATION_SPEC.md` | `wakeups`, `admitted`, `buy_conversion`, `top_mover_conversion` | live funnel review |
| Early trend-start mode | replay-only research, first profile rejected | `docs/specs/trend-start-mode.md` | `capture_rate`, `capture_ratio_at_entry`, `lead_time_to_final_top_min`, `trade_precision`, `PnL` | next narrower replay profile |
| Agent entry quality gates | shipped | `docs/specs/agent-entry-quality.md` | `trade_precision`, `false_positive_buys`, `blocked_winners` | replay before any relaxation |
| Local control-plane reliability | shipped | `docs/specs/local-control-plane-reliability.md` | process health, duplicate-launch avoidance | keep operational-only |
| Telegram positions freshness | shipped operational fix | `docs/specs/telegram-positions-freshness.md` | operator display correctness, stale-position avoidance | keep cache fallback-only |
| Local context MCP | local tooling | `docs/specs/local-context-mcp.md` | context-read latency, report compactness | keep outside trading path |
| Full-stack restart helper | shipped local tooling | `docs/specs/full-stack-restart-helper.md` | restart success, operator recovery time | keep operational-only |
| Repository skills tooling | shipped local tooling | `docs/specs/repo-skills-tooling.md` | workflow reuse, review consistency | keep outside trading path |
| Market-data / execution scaffold | research-only | `docs/specs/market-data-execution-research-scaffold.md` | replay/live parity groundwork, slippage realism | no production import without later gate |
| Local artifact hygiene | shipped local tooling | `docs/specs/local-artifact-hygiene.md` | status readability, secret hygiene | keep operational-only |
| P0 measurement hardening | shipped measurement-only | `docs/specs/p0-measurement-hardening.md` | report coverage, funnel loss point, exit lifecycle | use before next algorithm change |
| Agent mode rescue replay | research-only | `docs/specs/agent-mode-rescue-replay.md` | `capture_rate`, `trade_precision`, `pnl_total`, entry timing | compare `agent_allowed` vs `agent_mode_rescue` before any live change |
| Wake-up confirmed agent rescue | research-only | `docs/specs/wakeup-confirmed-agent-rescue.md` | `capture_rate`, `trade_precision`, `pnl_total`, temporal rescue admissions | evaluate only on windows with wake-up coverage |
| Belief-state v2 architecture | research-only, dual-belief design | `docs/specs/belief-state-v2-architecture.md` | state reconstruction quality, environment-belief quality, later policy uplift | build environment separability before RL |
| V2 sequence dataset builder | research-only | `docs/specs/v2-sequence-dataset-builder.md` | sequence coverage, transition count, gap diagnostics | label lifecycle states before model fitting |
| V2 sequence coverage audit | research-only | `docs/specs/v2-sequence-coverage-audit.md` | transition density, contiguous coverage, longest usable history | choose data-repair strategy before state modeling |
| V2 canonical market history | research-only | `docs/specs/v2-canonical-market-history.md` | contiguous OHLCV coverage, missing-interval count | choose canonical history source before lifecycle labels |
| V2 canonical history store | research-only | `docs/specs/v2-canonical-history-store.md` | stored slice count, continuity, provenance | define population/backfill policy |
| V2 history coverage plan | research-only planning | `docs/specs/v2-history-coverage-plan.md` | continuous days, valid-symbol ratio, missing-bar count | implement historical population adapter |
| V2 history population adapter | research-only | `docs/specs/v2-history-population-adapter.md` | valid-symbol ratio, contiguous slices, missing intervals | populate canonical store before lifecycle labels |
| V2 hindsight lifecycle labeling | research-only implementation | `docs/specs/v2-hindsight-lifecycle-labeling.md` | label balance, transition quality, leakage control | sensitivity audit before reconstruction |
| V2 lifecycle label sensitivity audit | research-only | `docs/specs/v2-lifecycle-label-sensitivity-audit.md` | teacher-label stability, threshold robustness | complete before reconstruction |
| V2 soft teacher confidence | research-only | `docs/specs/v2-soft-teacher-confidence.md` | teacher uncertainty, confidence-weighted training readiness | use before reconstruction baseline |
| V2 state reconstruction baseline | research-only | `docs/specs/v2-state-reconstruction-baseline.md` | weighted accuracy, macro F1, per-state recall | beat naive baseline before filtering |
| V2 belief filter v1 | research-only | `docs/specs/v2-belief-filter-v1.md` | filtered macro F1, emerging recall, belief normalization | beat isolated baseline before policy work |
| V2 belief calibration audit | research-only complete | `docs/specs/v2-belief-calibration-audit.md` | early recall vs late-state trade-off | balanced filter selected before policy |
| V2 offline decision environment | research-only implementation | `docs/specs/v2-offline-decision-environment.md` | deterministic rewards, legal actions, policy-comparison readiness | add fixed policy baselines before offline RL |
| V2 offline policy baselines | research-only implementation | `docs/specs/v2-offline-policy-baselines.md` | oracle gap, reward decomposition, policy anchors | inspect baseline gap before learned policies |
| V2 belief action calibration | research-only implementation | `docs/specs/v2-belief-action-calibration.md` | thresholded action quality, trade explosion control | tune belief-to-action bridge before RL |
| V2 policy gap audit | research-only implementation | `docs/specs/v2-policy-gap-audit.md` | admission-vs-exit loss attribution | choose next policy architecture before RL |
| V2 entry admission dataset | research-only implementation | `docs/specs/v2-entry-admission-dataset.md` | v1-feature reuse coverage, admission readiness | compare belief-only vs v1-enriched admission baselines |
| V2 v1 structural projection | research-only implementation | `docs/specs/v2-v1-structural-feature-projection.md` | dense projected-v1 coverage, lifecycle separation | compare projected-v1 admission baselines |
| V2 entry admission baselines | research-only implementation | `docs/specs/v2-entry-admission-baselines.md` | noise rejection, emerging recall, admission precision | compare baseline families before reward replay |
| V2 entry admission reward replay | research-only implementation | `docs/specs/v2-entry-admission-reward-replay.md` | total reward, false-buy penalty, trade-count reduction | validate admission in offline environment |
| V2 residual gap decomposition | research-only implementation | `docs/specs/v2-residual-gap-decomposition.md` | residual admission vs exit loss attribution | choose next dominant architecture slice |
| V2 exit quality baselines | research-only implementation | `docs/specs/v2-exit-quality-baselines.md` | giveback, realized PnL, exit reward | choose first better transparent exit profile |
| V2 exhaustion discrimination audit | research-only implementation | `docs/specs/v2-exhaustion-discrimination-audit.md` | mature-vs-exhaustion separability | choose rule baseline vs richer temporal features |
| V2 exhaustion-aware exit baselines | research-only complete, first-pass manual rules rejected | `docs/specs/v2-exhaustion-aware-exit-baselines.md` | replay-tested exit reward, giveback, false early exit | move to temporal / supervised exit modeling |
| V2 temporal exit baselines | research-only first pass complete | `docs/specs/v2-temporal-exit-baselines.md` | replay-tested temporal exit reward, giveback, false early exit | robustness test `mature_decay_late_rise` before promotion |
| V2 temporal exit robustness | research-only complete, locally robust | `docs/specs/v2-temporal-exit-robustness.md` | local-grid reward stability | test window / regime stability before promotion |
| V2 temporal exit window stability | research-only complete, promising but regime-sensitive | `docs/specs/v2-temporal-exit-window-stability.md` | per-window reward stability | explain the losing latest window before promotion |
| V2 temporal exit failure-slice audit | research-only complete, coherent weak-structure slice found | `docs/specs/v2-temporal-exit-failure-slice-audit.md` | winning-vs-losing slice explainability | test conditional temporal-exit hypotheses next |
| V2 market-environment taxonomy | research-only implementation | `docs/specs/v2-market-environment-taxonomy.md` | policy-relevant environment vocabulary | test separability before classifier |
| V2 market-environment separability audit | research-only first pass complete | `docs/specs/v2-market-environment-separability-audit.md` | favorable-vs-unfavorable day separability | build first classifier-switched replay baseline |
| V2 market-environment switch replay | research-only first pass complete | `docs/specs/v2-market-environment-switch-replay.md` | oracle vs causal switched-policy reward | improve causal environment belief; first classifier rejected |
| V2 market-environment belief v1 | research-only first pass rejected | `docs/specs/v2-market-environment-belief-v1.md` | rolling belief quality, switched-policy reward, abstention | diagnose lag / feature / target failure before next belief model |
| V2 market-environment target design | research-only complete | `docs/specs/v2-market-environment-target-design.md` | target disagreement, horizon balance, structural future mix | use 1h/2h future-horizon policy advantage for next belief model |
| V2 market-environment horizon-belief diagnostic | research-only complete, current classifier rejected | `docs/specs/v2-market-environment-horizon-belief-diagnostic.md` | causal horizon-target accuracy, wrong-confidence, coverage | improve market observation features before switched replay |
| V2 market observation feature audit | research-only complete, near-majority only | `docs/specs/v2-market-observation-feature-audit.md` | richer observation separability, horizon-target accuracy | add stronger observations before switched replay |
| V2 market-environment edge target audit | research-only complete, rejected | `docs/specs/v2-market-environment-edge-target-audit.md` | actionable edge separability, no-edge coverage | build stronger market-breadth observations before switched replay |
| V2 market breadth observation store | research-only complete, useful but below promotion gate | `docs/specs/v2-market-breadth-observation-store.md` | causal breadth separability, horizon-target accuracy | feature selection / regularization before switched replay |
| V2 v1 market-structure feature audit | research-only complete, selected features passed diagnostic gate | `docs/specs/v2-v1-market-structure-feature-audit.md` | v1 feature reuse, causal horizon-target accuracy | run selected-feature switched replay |
| V2 selected-feature market switch replay | research-only complete, rejected | `docs/specs/v2-selected-feature-market-switch-replay.md` | switched reward vs fixed policies, wrong-confident loss | build reward-weighted selector before switched replay |
| V2 reward-weighted market selector | research-only partial pass: 2h advances, 1h rejected | `docs/specs/v2-reward-weighted-market-selector.md` | reward-weighted switch reward, downside loss | full offline replay required before telemetry |
| V2 reward-weighted market selector offline replay | research-only complete, rejected | `docs/specs/v2-reward-weighted-market-selector-offline-replay.md` | full environment reward, trade-count safety | decompose selector failure before redesign |
| V2 selector failure decomposition | research-only complete | `docs/specs/v2-selector-failure-decomposition.md` | loss source attribution, stale-choice impact, action mismatch | split entry selector from position-aware exit selector |
| V2 position-aware exit selector | research-only complete, first manual profiles rejected | `docs/specs/v2-position-aware-exit-selector.md` | position-aware exit reward, giveback, trade-count safety | build action-level exit advantage labels |
| V2 action-level exit advantage dataset | research-only complete | `docs/specs/v2-action-level-exit-advantage-dataset.md` | sell-vs-hold advantage labels, feature coverage | train chronological baseline exit-advantage model |
| V2 exit advantage baseline model | research-only complete, first linear model rejected | `docs/specs/v2-exit-advantage-baseline-model.md` | holdout directional accuracy, captured advantage vs naive baselines | compare nonlinear / binned model families |
| V2 exit advantage model-family comparison | research-only complete, binned models rejected | `docs/specs/v2-exit-advantage-model-family-comparison.md` | captured advantage vs always-sell proxy, sell rate, bad sells | add explicit trade-path context features |
| V2 shadow signal observer | expedited shadow-only | `docs/specs/v2-shadow-signal-observer.md` | transition timeliness, alert precision, later outcome alignment | observe tomorrow; never trade |
| V2 shadow explainability | expedited shadow-only | `docs/specs/v2-shadow-explainability.md` | answer latency, why/why-not coverage | required before operator observation |
| V2 shadow alert policy | expedited shadow-only | `docs/specs/v2-shadow-alert-policy.md` | alert precision, operator noise | Telegram only for upside discovery |
| V2 shadow operator notification policy | expedited shadow-only | `docs/specs/v2-shadow-operator-notification-policy.md` | operator noise, summary usefulness | no realtime Telegram until learned policy |
| V2 shadow daily summary | expedited shadow-only | `docs/specs/v2-shadow-daily-summary.md` | daily discovery count, confirmation ratio, operator noise | daily post-factum report |
| V2 daily scorecard | shipped measurement-only | `docs/specs/v2-daily-scorecard.md` | `v2_top_recall_pct`, `v2_top_precision_pct`, `v2_confirmation_ratio`, `v2_handoff_bought_pct` | review day-over-day / week-over-week before any V2 promotion |
| V2 upside precision discriminator | research-only diagnostic | `docs/specs/v2-upside-precision-discriminator.md` | `baseline_precision_pct`, `slice_precision_pct`, `slice_recall_pct`, `false_favorable_reduction_pct` | advance only a replay candidate, never production directly |
| V2 early admission full-candidate backtest | research-only diagnostic | `docs/specs/v2-early-admission-full-backtest.md` | top precision/recall, false-favorable rate, hold-to-close return, MFE | advance only to portfolio-aware replay if gate passes |
| V2 wake-up to V1 bridge replay | research-only diagnostic | `docs/specs/v2-wakeup-v1-bridge-replay.md` | top precision/recall, ret5 precision, candidate pressure, V2→V1 delay | advance only to portfolio-aware replay if gate passes |
| V2 unified runtime integration | planned | `docs/specs/v2-unified-runtime-integration.md` | one-command startup, runtime health, release parity | required before any live-shadow worker |
| Signal-quality feedback policy | shipped narrow auto-apply | `files/signal_quality_feedback.py` | `cooldown_harm`, replay-confirmed quality deltas | keep auto-apply limited to replay-confirmed cooldown |
| Peak-risk lifecycle telemetry | shipped shadow-only | `docs/specs/peak-risk-shadow.md` | event count, `peak_within_n_bars`, false-positive continuation rate | collect shadow rows before any exit change |
| Hypothesis queue | shipped diagnostic-only | `docs/specs/hypothesis-queue.md` | ranked hypotheses, linked evidence, replay status | no auto-apply |
| Unified portfolio replacement | planned | `SCOUT_OPTIMIZATION_SPEC.md` | opportunity cost, replacement uplift, capture under `10/10` | replay grid required |
| Why-no-signal traces | shipped first read-only version | `docs/specs/p0-observability-foundation.md` | blocker chain per missed top mover | connect to daily reports |
| P0 observability foundation | shipped first additive slice | `docs/specs/p0-observability-foundation.md` | `blocked_reason_harm`, evaluator coverage | expand blocker context fields |
| Canonical metrics map | shipped | `docs/specs/metrics-canonical.md` | objective decision consistency | use in roadmap reviews |
| Failure casebook | diagnostic-only complete | `docs/specs/failure-casebook.md` | ranked real failure cases, opportunity cost, replayable hypotheses | replay early-block-to-entry rescue first |
| Early-block-to-entry rescue replay | diagnostic proxy complete, advances to candle replay | `docs/specs/early-block-to-entry-rescue-replay.md` | rescued missed winners, proxy opportunity gain, non-positive cases | implement candle-level causal replay with false-positive gate |
| Early-block rescue event replay | diagnostic-only complete, rejected | `docs/specs/early-block-rescue-event-replay.md` | top15 precision, false-positive ratio, missed-winner coverage | keep diagnostic; needs extra causal discriminator |
| Post-block causal discriminator dataset | research-only complete | `docs/specs/post-block-causal-discriminator-dataset.md` | post-block feature coverage, useful-missed-winner class balance | run chronological discriminator audit |
| Post-block causal discriminator audit | research-only complete, rejected | `docs/specs/post-block-causal-discriminator-audit.md` | useful precision, top15 precision, candidate pressure | improve labels or add score/rank trajectory before next model |
| Post-block experiment suite | research-only complete, selected delayed confirmation target | `docs/specs/post-block-experiment-suite.md` | holdout precision lift, candidate pressure, target selection | replay `top15_and_tradable_120m` delayed confirmation candidate |
| Post-block delayed confirmation replay | research-only complete, rejected | `docs/specs/post-block-delayed-confirmation-replay.md` | post-entry return, MFE/MAE, chase risk | do not continue as entry policy; move decision point earlier |

## Governance

1. Follow `docs/specs/spec-first-workflow.md` for every non-trivial change.
2. Any new live decision path needs a feature spec, a rollback switch, and a replay acceptance rule.
3. Shadow-only instrumentation may ship before replay when it does not alter BUY/SELL behavior.
4. Evaluator findings may create hypotheses automatically, but production changes still require replay evidence.
5. ML/RL changes must report effect on the bot objective, not only on surrogate PnL metrics.
