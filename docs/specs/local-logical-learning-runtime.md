# Local logical learning runtime

2026-10-02. User explicitly accepts same-Lenovo logical role separation instead
of new Windows accounts/UAC. TH-01..TH-12 remain applicable.

Separate user-level processes run collector/evaluator every 60s, training exporter
and trainer hourly, portfolio confirmation/controller every 300s. Each role is
single-flight with crash-released locks; supervisor restarts failed workers.
Blocked bootstrap retries after at most 60s instead of waiting an hour for intake.
Lifecycle files distinguish an in-progress tick from its last completed result.
Expensive training/export/controller work cannot block the minute collector.
No administrator, new accounts, privilege grants, ACL bypass or cloud upload.
Existing protected deployment is untouched, with a separate local registry,
candidate/intake and release pointer to avoid cross-deployment writers.

Same SID is checked and explicitly accepted in local deployment; trainer skips
the NTFS-denial probe only in this opt-in mode. This is NOT OS isolation: Lenovo
can read all local files. Trainer code receives only immutable pre-holdout intake,
not raw holdout as its training argument. Frozen candidate registration, label
timing, deduplication, chronological splits and 48h embargo are unchanged.
Child environment is allowlisted: no inherited Telegram/exchange keys; only
controller may inherit an explicitly supplied evaluator key. No keys, certificates
or successful no-trainer-access assertions are fabricated by initialization.

Existing cryptographic provenance, full-period portfolio comparison, forward and
canary cohorts, Harness and rollback gates remain fail-closed. Local execution
does not approve a candidate or automatically connect its dedicated release pointer
to production. Coverage certification still requires genuine inputs; accepting
logical process separation does not prove untouched holdout or population parity.
Current maximum 181-day paired replay has zero uplift: no policy relaxation here.
Missing intake/candidate/evidence is BLOCKED, not learning success. Collection is
T+5 PROXY_ONLY; until live receipts and independent objective results exist the
loop remains NOT_CLOSED/UNKNOWN. Report isolation_mode and os_access_isolation=false.

Start: powershell -NoProfile -ExecutionPolicy Bypass -File start_learning_local.ps1.
Stop: same command with -Stop. Stop marker is checked between ticks; supervisor
terminates only its own children, preserving atomic snapshots/journal integrity.
Rollback: stop local supervisor; protected service deployment and trading config
were not modified. Logon autostart is not installed; processes require user session.

Verification: synthetic tests for explicit opt-in, SID mismatch, environment
filtering, immutable intake-only fit, controller isolation, singleton supervision
and truthful health. Regression tests cover collection, provenance, portfolio gate,
snapshot publication and rollback. Runtime artifacts/credentials never committed.
