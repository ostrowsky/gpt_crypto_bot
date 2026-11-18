# FAS-1: bounded streaming audit of completed public flow capture

2026-11-18. Corrective SPEC -> RED -> CODE. Independent data audit only.
TA12 in binance-demo-trade-aware-replay.md remains unchanged. Existing failed
PFD-AUDIT-20261118-v1 keeps its MemoryError and partial status, not a data FAIL.

FAS01: Never request more than64KiB per decompressed stream read. Iterate JSONL
records incrementally; no whole-chunk8MiB allocation or splitlines materialization.
Bound a pending record and total decompressed chunk by original8MiB contract.
Hash exactly the same bytes, including newlines; preserve final nonterminated
line semantics, counts, receipt/session checks and all source metadata checks.

FAS02: Split records across read boundaries without losing or duplicating bytes.
Oversized line/chunk, bad gzip, malformed JSON and source/hash/receipt errors
fail closed; no partial full-capture PASS. No weakening any quality assertion.
The producer validation profile and NOT_TRAINED/financial_certification=false
remain. Seven calendar days do not establish uninterrupted coverage or profit.

FAS03: Preserve failed audit source/log/incident; rerun into a new own output
folder with the same immutable capture. CLI writes INCOMPLETE_AUDIT status and
safe exception class on failure. No automatic extension/restart of collector,
training, keys or orders. Tests must reproduce the excessive read before code;
run full app regressions, diff/staged gates, intended-only commit and push.
