# RFP-1 journal I/O recovery and verified continuation

Registered 2026-11-19 after v1 gzip write OSError errno22; disk has free space.
Root cause of the OS error is not certified. Parent RFP-1 remains INCOMPLETE;
never overwrite its source snapshots, status, partial journal or incident.
No economic/model/risk/cost/selection changes or actual trading operations.

JR01: write bounded <=1MiB JSONL blocks as deterministic concatenated gzip
members, with short-lived binary append handles and fsync. Exact original text
bytes/order, no skipped events. Atomic final rename remains parent's operation.
Fresh own-runtime .partial file only; reject an existing path. Keep per-journal
uncompressed hash/record/byte counts and storage/recovery evidence separately.

JR02: for OSError EINVAL/EIO only, at most two retries of the SAME member. Before
retry prove the existing committed prefix length/hash, truncate ONLY own new
partial file back to that durable offset, then append once. Detect a short write;
do not duplicate partially written members. Record every recovery. Changed or
missing committed prefix aborts; other/persistent errors propagate INCOMPLETE.
On economic exception preserve a bounded unflushed buffer as aborted evidence;
never publish a complete case. No parent/runtime cleanup or lost failure logs.

JR03: a separate continuation controller binds the ORIGINAL unchanged study,
financial engine, quote reader, audit, feature data, coefficients and parameters.
The original economic registration must be byte/hash equal in its fresh child
run; verify all original input hashes, parent source and completed manifests.
Only module-local I/O/simulation dependency bindings are injected in this own
offline process, restored in finally. No global standard-library monkeypatch,
current bot worker or network change. A supplemental preregistration pins new
controller/writer/spec, parent incident, parent registration and reuse vector.

JR04: reuse EVERY fully sealed parent case with valid manifest/result/journal/
audit/registration/policy binding, independent of its PnL. Re-emit exact original
events through the durable writer and independently audit them again. An invalid
sealed completed case fails preflight; incomplete cases are wholly recomputed,
never resumed in the middle of a trade/day or rebased to current balances.
Require identical signal-manifest and TRAIN microprice coefficient hashes.
Remaining cases run the exact original simulator from original per-case100USDT.
No model refit/search except the same deterministic original TRAIN-only baseline.

JR05: gzip read modes retain standard semantics. Verify decompressed journal
SHA/record count after execution for all61 cases and freeze continuation evidence.
No success before original source validation, all ledger audits and original
report finish. Failed parent is not renamed PASS. Full Truth Harness FAIL TH-11
and all financial limitations remain. Runtime/model/report data never staged.

Tests before code: real temporary file prefix/partial-write rollback and exact
decompressed content, persistent error, prefix tampering, existing path, mixed
read/write dispatch, registration/economic mismatch, sealed reuse corruption
and replay-vs-original economic event parity. Run relevant regressions and
staged Truth Harness, review own diff, commit and push. Rollback stops own
offline continuation, retaining all evidence and original trading settings.
