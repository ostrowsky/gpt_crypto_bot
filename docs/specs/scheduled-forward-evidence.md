# Scheduled forward collection and Windows role separation

Date: 2026-10-01. Parent: validated-ranker-rollout.md.
Status: infrastructure, not an approved trading-policy hypothesis. TH-01..TH-12.

Pre-implementation check: maximum archived baseline is 181 days, 93/105 symbols,
10538 trades and -95.716019% after costs; partial/current-rule diagnostic only.
Existing rollout regression tests pass. No BUY/SELL relaxation is supported.
Full Harness remains FAIL TH-11. Infrastructure hypothesis: restartable prospective
collection and scheduled fail-closed control preserve evidence without approving
stale, retrospective, incomplete or trainer-authored evidence. Historical backtests
cannot prove future cohort quality or Windows account isolation; test those directly.

Collector freezes candidate registry and a hash-chained append-only observation/
outcome journal. First observation must be within 120s of feature availability and
after frozen holdout start; decision provenance must be valid and before target.
Label/teacher fields never enter frozen features. No retrospective admission of
already labeled rows. Mature ret_5 labels require original provenance, exact label
time, finite value and unchanged causal record. Duplicate observations/outcomes are
idempotent; conflicting IDs and chain corruption block collection. Invalid/late
rows are not misses or successes. Outcomes remain T+5 proxies, not portfolio fills.
Collector exports an evaluator-owned snapshot and calls existing independent
evaluator; incomplete population is never certified for portfolio rollout.

Recovery contract (2026-10-02): collector and release-pointer locks are now
nonblocking OS-owned byte/file locks. Process death automatically releases them;
an orphan legacy sentinel cannot block collection permanently. Never remove a
lock file while writers can run: that permits locking distinct inodes. A live
owner still blocks a second writer; no age-based stealing, PID guessing or journal
reset. Existing journal hashes, frozen candidate and holdout dates are retained.
Test real subprocess contention, abrupt termination, legacy empty file and error
cleanup. Rollback uses the same crash-safe lock and original compare-and-swap.
Deployment under isolated accounts requires the new process_lock.py dependency;
checkout tests alone do not establish restored scheduled collection.

Evaluator-owned training export excludes all features/labels at or after frozen
holdout start. Trainer receives only this sanitized file and cannot read raw
critic data, evaluator registry, journal, coverage/evaluator secrets or active
release pointer. Trainers write candidates only, never active model files.

Windows provisioning requires administrator and distinct new non-admin local
users GptBotTrainer/GptBotEvaluator. Random account credentials are not logged or
committed. Credentials are DPAPI-protected in admin-only bootstrap storage. Exact
NTFS ACLs grant trainer read-only executable/source access, read training export,
modify candidate output; evaluator modify evidence, read inputs and candidates.
Explicit trainer denies protect raw dataset/evidence/release state. No ACL changes
remove existing operator/bot access to Telegram tokens or position files. Deployment source files are
frozen copies, not trainer-writable checkout code. Tasks use least-privilege S4U,
evaluator every minute, trainer hourly; no overlapping instances. Run checks use
actual Windows SID, not an environment role assertion. Collector and controller
share evaluator schedule but trainer has neither signing key. Keys must be supplied
separately; provisioning does not synthesize coverage certifications or enable BUY.
The frozen evaluator runs the full Harness against the administrator-configured
live project root, not an empty copied runtime. Existing legacy worker training is
not claimed to be migrated; its outputs cannot substitute for this isolated cohort.
The first candidate remains frozen throughout its holdout; new hourly candidates
do not silently replace it or gain access to holdout labels.
Provisioning may resume only with the exact previously observed account SIDs;
it refuses unrelated existing principals. ACL work is scoped to the raw source and
sealed evidence paths, not all runtime logs/locks. Partial installation is reported
as such; only completed task registration writes the public installation marker.
Cross-user S4U registration supplies the account's DPAPI-decrypted password only
in-process to the registration API (Microsoft Security Contexts for Tasks); never
as shell/process arguments or logs. Read back LogonType=S4U before starting a task.
The service accounts belong only to standard Users, not Administrators.
Use native RegisterTaskDefinition with explicit TASK_LOGON_S4U=2, because the
PowerShell User/Password overload replaces the principal logon type. Task DACLs
grant role accounts read/execute only; only SYSTEM/Administrators can modify them.
Embedded Python explicitly prioritizes frozen role source over its checkout ._pth.
Task updates during recovery require the same verified service principal; other
existing tasks are never overwritten. Registration success alone is not run health.
Triggers and demand starts are explicitly enabled; battery conditions must not
silently suspend collection. Before any frozen candidate exists, training intake
advances to evaluator's current time. After registration it is bounded by fixed
holdout_start. Never deadlock bootstrap on an insufficient installation-day sample.
Provision only SeBatchLogonRight for the two exact service SIDs using additive
LSA account-rights API. Do not grant administrator/debug/service privileges,
overwrite global policy, or clear deny-logon rights. Security event 4625 status
0xc000015b means scheduled logon has not been permitted; registration is not health.
If a Windows trainer read handle prevents atomic training-file replacement (error
32), preserve the complete old eligible snapshot and defer publication to the next
tick. Report export row count as unknown, not the unpublished count. Other IO
failures still block; never truncate an input used by the trainer.

The scheduled roles now publish immutable content-addressed snapshots instead of
replacing the reader's training.jsonl. Evaluator writes a complete fsynced file,
stores it as training_snapshots/SHA256.jsonl, and atomically publishes only a small
training.snapshot.json descriptor. Trainer pins and verifies digest/byte size/path
before fitting and digest again before publishing a candidate; corrupt/missing
pointers block training, with no legacy-file fallback. Rows remain pre-holdout and
existing ML quality gates remain mandatory. Old files are retained while readers
may use them; no automatic pruning or permission expansion. Publication failures
remain BLOCKED and do not advance the pointer. Snapshot row counts are not proof
of useful learning. Source/spec/tests are committed; input snapshots stay runtime.
Regression verification includes locked legacy file, repeated publication,
concurrent old reader, pointer failure, tamper, path escape and frozen cutoff.
Identical eligible row IDs are deduplicated; conflicting eligible IDs block
publication. Trainer status carries the exact snapshot digest, existing dataset
quality verdict and chronological train/validation/test counts, not an improvement
claim. No learning quality gate is weakened to make the task report success.

Controller runs on every evaluator tick. Unsigned/missing portfolio request,
certification, keys, stale cohort or failed Harness leaves BLOCKED. Previously
active overlay is cleared by atomic compare-and-swap on failed evaluation; rollback
conflict is BLOCKED and runtime expiry remains an independent safety fallback.
State/status contains reason, run time and next scheduled task identity. No automatic
promotion from proxy collector output. Automatic portfolio bundle generation from
real champion/challenger admission/fill logs is NOT certified by this stage.
The evaluator now reads protected portfolio_inputs.json, recomputes independent
phase evidence and publishes portfolio_confirmation_latest.json on every tick.
Current confirmation must be READY before the controller may use a request;
missing evidence cannot fall back to a previously successful request. See
independent-portfolio-confirmation.md. Protected deployment remains separately
versioned; checkout changes alone do not update Windows role processes.

Verify synthetic arrival->maturity->evaluation, future/stale/conflicting labels,
restart/idempotence, journal tampering, trainer SID mismatch, evaluation failure
rollback and scheduler ACL intent. Real collector bootstrap may be BLOCKED by old
candidate exposure provenance. Report that rather than fabricating a cohort.
Commit/push code/spec/tests only. After deployment restart Telegram bot using scoped
launcher and verify exactly one PID/startup health. Admin/UAC refusal is a deployment
blocker, not a claim of installed role isolation. Never restart arbitrary Python.
