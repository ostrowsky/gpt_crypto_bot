# ML label writer CPU optimization

2026-10-14. Performance-only; no trading, polling, ranking or label-definition
change. TH-01..12 apply. The current full Harness FAIL about historical portfolio
source drift is retained, not repaired by rewriting evidence hashes.

## Contract

`_rewrite_records` may receive an explicit mutation scope. Callers supply exactly
the same decoded ID or symbol/timeframe equality guard as their mutator. Continue
decoding and validating every row; malformed/non-object records retain existing
cleanup/counting behavior. Copy a valid out-of-scope row directly without JSON
serialization, preserving a final newline. Serialize all in-scope rows, including
mutators returning false: provenance helpers may have side effects in that case.
Without a scope retain legacy behavior. Keep file/thread/process lock, atomic
replacement, exception cleanup and no-change no-rewrite semantics unchanged.

JSON whitespace of copied rows need not be normalized. Decoded records, labels,
provenance, counts and record order must match the legacy writer. No raw substring
predicate may decide membership: compact or escaped duplicate IDs must still be
updated when the rewrite's original preflight finds a change.

## Verification

Focused tests cover untouched serialization count, duplicate IDs with alternative
encoding, false-return provenance side effects, malformed cleanup, final newline,
failed replacement and atomic persistence. Existing ML/provenance suites remain
required. A deterministic temporary synthetic benchmark compares scoped writes
with the same function's legacy unscoped path, with identical semantic outputs;
report rows/bytes/repeats and CPU times. It is not a measurement of the whole bot
or evidence of trading improvement. No maximum strategy replay is required for
a serialization optimization that does not relax policy or change label values.

Deployment: user authorized a short main-bot restart after tests, preserving
positions and launch arguments. Do not restart the stopped autonomous trainer or
other projects. Measure post-restart CPU separately; short-window fluctuation is
not a guaranteed long-run reduction. Rollback: revert the scoped-write change and
restart the main bot; existing JSONL remains compatible with the legacy writer.
