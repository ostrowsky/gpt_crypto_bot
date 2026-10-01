# Independently evaluated ranker rollout

Date: 2026-10-01. Status: disabled-by-default adapter and evaluator; no candidate
approved and no production improvement claimed. Parent: continuous-signal-improvement.md.
TH-01 through TH-12 apply. Existing proxy readiness cannot approve this adapter.

## Independent evidence

The evaluator consumes immutable raw paired trades and closed 15m price paths,
not trainer verdicts or summarized PnL. It independently recomputes two unified
ten-slot accounts after fees/slippage and against BTC. Full history, sealed test,
shadow and canary are separate bundles. Historical evidence must cover the whole
available period; test starts at least 48h after the last model exposure. Full
history includes exposure and is diagnostic, while sealed-test evidence is OOS.
Prices must be finite, positive, unique and contiguous; trades stay in-window.
Both arms use identical population, costs and dates, with complete valuation.

Evidence also requires a coverage authority HMAC attestation binding every input
byte, exact model/champion hashes, live/replay parity, point-in-time population,
closed raw-data provenance and actual canary assignment. These facts are NOT
inferred from an arbitrary JSON boolean. No authority keys are provisioned here.
Separate service accounts/ACLs must prevent trainer access to authority/evaluator
keys and release state. HMAC is integrity under that deployment assumption, NOT
an OS security boundary or public-key signature. No unsigned evidence is usable.

Fixed initial budgets: fees >=7.5bps/side, slippage >=5bps/side, drawdown <=15%,
candidate drawdown no more than champion+1pp, candidate portfolio delta >0 and
candidate after-cost BTC alpha >=0. Sealed test/shadow/canary >=30 days;
each arm >=100 trades. Paired daily portfolio-return deltas require at least 30
days and a positive lower 95% interval from a fixed-seed 2000-sample seven-day
moving-block bootstrap (including canary). Historical bundle covers maximum available interval.
These conservative budgets are engineering guardrails, not market-validated
strategy parameters. Human authorization must explicitly approve the experiment;
intervals are conditional on certified inputs, not guarantees. Wider optimization
still needs preregistered regime tests. Full Harness TH-11 FAIL remains visible.

## Safe application

Latest forward cohort must end within 24h; sealed, shadow and canary windows
cannot overlap. Evaluator authorizations bind model, champion, stage, timestamps and evaluator
source hash. CANARY requires historical + sealed + shadow; PROMOTED additionally
requires canary. An operator-enabled switch is false by default. Adapter reads
authorization every decision, checks integrity/expiry/champion identity/source
and immutable candidate bytes. Canary uses fixed symbol-level 5% assignment,
not selected winners; promoted uses 100%. Change to champion bonus stays within +/-1 point;
no SELL, cooldown, risk or capacity relaxation. Missing/corrupt/expired evidence,
kill switch or model mismatch returns existing champion, without a restart.
Runtime fit is never a release authorization.

Atomic compare-and-swap pointer writes use a lock and expected prior digest;
rollback atomically clears the overlay. Last-known-good champion file is never
overwritten. This is an adapter, not a completed automated trainer/forward worker.
Runtime activation is blocked until externally certified evidence and operator
enablement exist. The legacy ranker scorer must also be explicitly enabled with
nonzero weight; this adapter never turns it on implicitly. Config/source changes
invalidate tickets and require renewed independent evaluation/certification.
Next: complete live-outcome
collector, scheduled controller integration and autonomous rollback on risk events.
The uncertainty estimator is delivered, but independent regime/repeated-testing
budgets and the event-driven forward collector remain follow-up work.

CLI: `independent_portfolio_gate.py --request REQUEST.json --output TICKET.json`
reads explicit candidate/champion file paths and phase-specific bundle/certification
paths. `validated_ranker_rollout.py --authorization TICKET.json --candidate MODEL
--champion CHAMPION [--expected POINTER_SHA]` installs the pointer, not config.
`--rollback --expected POINTER_SHA` clears it. Evaluator uses the two separate
RANKER_COVERAGE_AUTHORITY_KEY/RANKER_EVALUATOR_KEY environment secrets; runtime
needs only evaluator verification key. Never put keys in Git or command arguments.

## Verification

Focused tests cover forged/changed attestation, NaN, gaps, duplicate candles,
future/expired tickets, source/model/champion mismatch, CAS races, rollback,
canary assignment and bounded bonus. Synthetic data prove plumbing only.
Run existing release/evaluator/ranker tests and full/staged Truth Harness,
diff check, commit and push source/spec/tests only. Real maximum archive is
partial (93/105), rule-only baseline and negative; cannot authorize release.
