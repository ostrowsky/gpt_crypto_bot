# Mission-aligned learning cycle

Registered2026-10-09, user authorizes sequential execution of all6 planning points.
TH01..12 and crypto-bot-truth-harness apply; mechanical full PASS at start.
No production trading relaxation until actual maximum-history and fresh forward
criteria pass. Supervised model training is not itself mission improvement.

## Ordered work

1. Restore forward-label collection: diagnose actual OS lock/process/stop marker,
   preserve incident evidence, repair bounded transient recovery, supervise only
   identified bot learning worker. Never delete an active byte-lock or kill
   unrelated processes. Verify real successful collection/label counts afterward.
   Entry acceptance/event timestamps remain separate; legacy clocks stay unknown.
2. Register one versioned mission contract for new experiments/forward artifacts:
   exchangeTop20 intersect immutable as-of watchlist, complete local00:00..24:00
   Europe/Budapest day, day-window quote volume>=1mUSDT. Existing22h goals and
   historical watchlistTop15 are legacy/diagnostic, not relabeled as new contract.
   Fetch maximum available raw history and label universe; receipt coverage,
   current-selected historical universe vs certified PIT explicitly separated.
   Online inference sees only past features; final ranks/returns are labels only.
3. Implement separated leader discovery/ranking and continuation/re-entry targets
   on ALL actual candidates/decision states, including blocked/unchosen cases.
   Exact available clocks, mature-only training, purged chronological splits,
   fixed initial parameters/thresholds before results, no retrospective upgrades
   of trade acceptance/fills. Teacher positives do not substitute for future targets.
4. Maximum-period full10-slot policy replay vs original identical control, including
   replacement, cooldown and protective exits. Raw same-contract early/coverage/
   unique precision plus matched immutable-state accompaniment and retained rise.
   Independent audit, candidate/source/data SHA snapshots; exposed history never
   claimed sealed. Preserve rejected rules and all failed intermediate artifacts.
5. Register fixed acceptance BEFORE experiment: primary early leader count,
   coverage/unique BUY precision noninferiority and companion exit-quality checks.
   Publish3-calendar-day paired block intervals and denominator counts. Fresh
   fixed-model forward cohort remains separate from model/threshold selection.
6. Deliver bounded forward shadow and release readiness using SAME objective,
   monitored data coverage/drift/label maturity, verified acceptance clocks,
   source-bound inference and immediate rollback. No canary/live BUY/SELL until
   fresh evidence and registered guardrails pass; absent future outcomes are
   pending, never manufactured. Do all presently possible steps autonomously.

## Safety and engineering

Costs/account DD are safety diagnostics; mission winner is early discovery,
coverage/selection and accompaniment. No account access or exchange orders.
Collector/control-plane recovery may start after source/tests verification;
trading gates stay intact. Tests for transient locks vs integrity failures,
objective contract/cutoffs/DST/coverage, future-prefix invariance, label maturity,
actual candidate/position identity, matched-policy accounting and release guards.
Complete relevant tests/diff review, staged Harness and commit/push intended
sources/spec/tests only. Never commit runtime evidence/prices/weights/state.
Every stage records actual completion, failures and externally pending evidence.

## Point1 registered recovery detail

Current OS byte-lock probe succeeds; do not unlink it. Collector process loaded
before subsequent recovery code and stayed disabled after Oct6 timeout; source
also checked only one exception cause. New helper walks bounded/cycle-safe cause
chains, retries only exact dataset-lock TimeoutError or Windows sharing errors32/33,
at most3 attempts separated300seconds. Malformed/ACL/unknown failures stay blocked.
Incident marker removed only after successful actual cycle. Status separates
enabled/running/healthy/retry_wait/blocked; failed cycle never becomes success.
Identity-checked workspace worker/wrapper restart preserves incident/status/lock
evidence, touches no trading process and never removes dataset byte-lock. Any
train-lock cleanup requires dead owner; preserve all models/datasets/history.

## Point2 raw collection and boundary repair

V1 fetched191days over699requested public/historically-mentioned symbols.43
files were rejected because last native day ends early on delisting, although
their prior complete days are valid. Preserve V1. V2 changes ONLY record-level
coverage handling: retain SHA-bound full native days, reject partial last row
on that day, disclose partial symbols in coverage. No model has been fitted or
threshold tuned. Current+historically-reported symbol union is observational,
not a retrospective PIT certification. Failed/partial histories never become
zero returns or clean proof of absence.

V3 labels reuse the identical V2 raw receipts, without refetching or model fitting:
valid below-volume assets are retained as known negative candidates; liquidity
filters only exchange ranking, never the training denominator. Partial native
days remain unknown. Intraday extension preserves SHA-bound archived prefixes,
fetches missing tails/pairs, records unavailable/gapped histories separately,
and never silently reduces the declared requested watchlist.

## Points3..6 fixed experiment protocol (before fitting)

All baseline replay candidates, including later rejected candidates, form entry
population. Two independent CatBoost classifiers predict (a) final global leader
membership and (b) early membership at the current candidate price. Closed17-bar
15m prefixes, timeframe, cyclic UTC clock, original causal score and signal mode
only; no symbol/day IDs or legacy future annotations. Parameters:300trees,
depth4, learning_rate0.03, l2=10, seed42, 2threads, no auto class weights.
Refit each30days after initial30days; inner validation starts at80% whole local
days of matured history. Train label availability<validation boundary; validation
availability<fit clock. Unknown labels/features do not become negatives.

One registered exploratory ranking arm adds at most4score points:
8*(early_probability-0.5), with missing/untrained scores unchanged. This can
affect ranking/admission only inside OFFLINE unchanged10-slot replay. No threshold
search or classifier accuracy claim authorizes deployment. Control candles,
candidate identities, modes, cooldowns, replacement and hard exits stay identical.
Held position ticks and blocked cooldown states are separately recorded from the
control replay. Continuation head labels next1h positive close return AND no
future close drawdown>2originATR; reentry head uses4h. These are explicit proxy
targets, not proof of exit improvement. Both stay action-neutral in this cycle;
the next policy relaxation requires its own registered full replay.

Entry acceptance gate: TEST early count gain>=1,3calendar-day paired95% lower
bound for early delta>0, coverage and unique BUY precision not worse, complete
companion exit metrics with no worse held-time/retention on matched entries.
Unknown/unmatched companion states block an overall PASS. Fresh forward gate:
fixed hashes, at least30 completed local days/100 eligible leader-day targets,
PIT universe and watchlist before predictions, feature receive<=issue clock,
verified actual acceptance clocks and complete data coverage. Retrospective
passes never bypass this gate. Drift/error/missing provenance => shadow only;
rollback is disablement of the optional model, never removing protective gates.

Collector persistence repair: each collector snapshot is buffered in memory
until the cycle's market requests finish, then NEW IDs are appended under one
shared dataset lock/uniqueness scan. Existing decisions/features/labels are
preserved, including higher-priority monitor BUY/block evidence; collector
snapshots never rewrite an existing decision. Mark IDs logged only after durable
append. Forward labels use existing strict batched update. Malformed history
aborts before any append; zero successful market pairs cannot mark recovery
healthy. Cycle stats publish requested/ok/failed and newly persisted IDs.
Recovery status write failures cannot swallow the integrity incident or retry.
Wrapper startup must verify actual PID/heartbeat and retain stderr; discover
identity-verified orphan learning workers before restart, never duplicates.

Verified wrapper failure: Set-Content heartbeat sharing IOException terminated
the supervisor but left Python alive. Heartbeat uses UTF8 temporary file and
atomic Replace/Move with bounded5s retry. Publication failure logs warning and
keeps supervising the child; it never claims a fresh heartbeat. No lock deletion.
AttachOnly may attach a repaired supervisor to exactly one verified orphan,
without restarting training/collector/report schedulers. An existing wrapper,
ambiguous workers or wrong executable/source path rejects attachment.

The legacy lock budget120s equals atomic replacement retry120s, leaving no
budget for parsing/serialization or queueing behind a writer. Default acquisition
budget becomes max(300s,2*replacement_budget+60s); lock owner/release semantics
unchanged. A bounded timeout still records failure, never unlocks another writer.

Windows snapshot readers share READ/WRITE/DELETE so atomic replacement can
publish a new path while the reader finishes its immutable original handle.
Retry ONLY actual Win32 sharing/lock errors32/33 for up to30s; access denial,
missing files and malformed records remain failures. No ACL changes, lock
deletion, forced handle closure, or bypass of dataset write locks.
On Windows publish uses ReplaceFileW (preserving destination ACLs), not MoveFileEx
overwrite, which failed with WinError5 in the real shared-reader test. Existing
bounded replacement retry remains; partial/native errors fail closed. Tests
verify old-reader bytes, new-path bytes and an unmodified missing-file failure.

## Fixed forward shadow implementation

Bounded35days; STOP file halts only the shadow. Latest mature discovery/early
models are mandatory; weights, inference/kernel sources and model hashes frozen,
no forward refitting. Feature_clock>model_fit_at, receipt>=feature_clock,
issued>=receipt; max5minute issue delay. Immutable per-cycle public raw klines,
exchange registry, watchlist and receipts. No keys/account endpoints/orders.
Training-only1%/99% feature bounds flag at least4outside dimensions as drift.
Collect true future daily outcomes from exact UTC hourly grids;23/25hour DST
days are valid, gaps are unknown. Only full local-day origin grids with no
missing/stale pairs qualify. Brier/base rates describe forecasts, not BUY effect.
Actual acceptance remains explicitly unverified; retrospective scores or these
proxy forecasts cannot unlock canary. Independent raw/fold/control/denominator
audit is required before any historical conclusion.

Before outcome replay/evaluation completes, register a calendar correction:
initial runner computes3surviving-row blocks. Partial native days can leave
calendar gaps, so that does not implement the promised3calendar-day blocks.
Preserve that initial artifact; separately recompute paired intervals with a
full calendar and NaN masks for absent days, normalized to observed-day count.
No new model/parameters/thresholds/trades or rule selection. Corrected evaluation
is authoritative; initial compressed-row intervals never approve release.

Collector freshness correction: a successful kline response from a BREAK/delisted
symbol is not fresh market coverage. Snapshot public exchangeInfo once per cycle,
preserve raw/hash/receive clock, exclude non-TRADING pairs explicitly before
requests. Reject future or stale closed features (>one interval+30s age).
Publish requested/watchlist/excluded/failed pair counts separately. Persist a
fetch-return upper-bound clock, not a fabricated network receipt or acceptance.
Existing records are never backfilled with these new provenance fields.
Known public API connection/timeout/429/5xx failures and zero successful tradable
pairs reuse the same bounded3retry recovery. Permanent4xx/schema/integrity/ACL
errors do not. Failure and stale coverage are never success, even if retried.

Prepared experiment V1 models/candidates are preserved. During the first control
replay, inherited research policy MagicMock exit/disabled-model wrappers retain
every call and array view (process reached7.56GB). Stop that experiment only;
resume V2 using identical SHA-bound prepared features/candidates/predictions and
models, with ordinary pure function overrides instead of call-recording mocks.
No parameters, scores, labels, admission rules or outcomes change. Full prior
control prefix parity is mandatory. Preserve V1 aborted-resource receipt; it
cannot be presented as a completed replay.
Forward target ranking uses native local-day bars and the union of asof cycle
registries plus a label-only end registry, so later listings can affect ranks
without leaking into features. DST days use exact hourly aggregation. Changed
intraday watchlist invalidates the fixed cohort; partial snapshots are excluded.
Additional audit reconstructs ALL known entry/continuation/reentry targets from
raw prices and probes actual BTC/ETH/AMP/SOL candidate generation after replacing
all future bars with neutral unclosed sentinels. Finite probes are not PIT proof.

Verified wrapper failure: Set-Content heartbeat sharing IOException terminated
the supervisor but left Python alive. Heartbeat uses UTF8 temporary file and
atomic Replace/Move with bounded5s retry. Publication failure logs warning and
keeps supervising the child; it never claims a fresh heartbeat. No lock deletion.
