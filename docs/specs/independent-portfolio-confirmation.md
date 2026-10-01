# Independent portfolio confirmation intake

Date: 2026-10-01. Status: fail-closed evidence intake, not strategy approval.
Parent: scheduled-forward-evidence.md. TH-01 through TH-12 apply.

The scheduled evaluator must report portfolio confirmation separately from T+5
prediction quality. It reads an evaluator-owned `portfolio_inputs.json`, never
creates certificates from trainer verdicts, and independently recomputes every
available historical/sealed/shadow/canary bundle with independent_portfolio_gate.
The manifest contract is `independent-portfolio-inputs-v1`, with stage CANARY or
PROMOTED, candidate/champion descriptors, and evidence keyed by required phase.
Each descriptor contains a relative path and exact SHA256. Each evidence phase
contains bundle and certification descriptors. Paths must remain under the
manifest directory, including resolved symlinks. Raw bytes, not JSON reserialization,
are hashed. Certification still requires the separate coverage authority secret.

Every phase reports PASS/REJECTED/UNKNOWN, dates, sample size, paired interval,
after-cost portfolio alpha and drawdown from recomputation. Missing phases remain
UNKNOWN. No partial phase success, proxy label, legacy file, mutable model, unsigned
certificate or fresh training timestamp can create release authority. Full history
must span maximum available bounds certified by the coverage authority; sealed and
forward phases require separate complete >=30-day >=100-trade populations with
48-hour exposure embargo and existing risk/uncertainty budgets.

Only successful existing gate authorization, including full Truth Harness, allows
atomic publication of portfolio_request.json. Scheduled controller additionally
requires current confirmation READY, preventing reuse of an old request after a
missing/corrupt/rejected intake. Failure preserves champion and clears a prior
overlay through existing CAS rollback. No BUY/SELL/config gates are relaxed.

Rollback: revert intake integration; existing controller still requires signed
phase evidence and full Harness. Do not remove gate requirements to unblock release.

Verification: synthetic signed complete cohorts test plumbing only; test missing
phases, changed bytes/model, path escape, unsupported stage, losses, stale input,
Harness failure, atomic publication and rollback of a previously active overlay.
Existing maximum archived 181-day replay is partial/current-rule diagnostic, not
paired champion/challenger evidence. It cannot certify live-policy parity or an
unobserved forward period. Actual admission/fill log generation, authority-backed
market/universe provenance and future cohorts are still required external inputs;
this implementation does not invent them. Full Harness FAIL TH-11 remains visible.
Protected Windows deployment requires a pinned commit and restore snapshot; a
checkout change is not reported as deployed under the separate service accounts.

## Input example

All files are placed within the evaluator-protected manifest directory. SHA values
must be exact digests of raw files; the example placeholders do not pass validation.

```json
{"contract":"independent-portfolio-inputs-v1","stage":"CANARY",
 "candidate":{"path":"frozen_candidate.json","sha256":"SHA256"},
 "champion":{"path":"frozen_champion.json","sha256":"SHA256"},
 "evidence":{
  "historical":{"bundle":{"path":"historical.json","sha256":"SHA256"},"certification":{"path":"historical.cert.json","sha256":"SHA256"}},
  "sealed":{"bundle":{"path":"sealed.json","sha256":"SHA256"},"certification":{"path":"sealed.cert.json","sha256":"SHA256"}},
  "shadow":{"bundle":{"path":"shadow.json","sha256":"SHA256"},"certification":{"path":"shadow.cert.json","sha256":"SHA256"}}}}
```

CLI: `independent_portfolio_confirmation.py --manifest INPUT --request REQUEST
--report REPORT` under evaluator identity; secrets come from environment, never
command-line arguments. Nonzero exit means BLOCKED, not negative profitability.
