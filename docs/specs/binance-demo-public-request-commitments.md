# PRC-1: public research request commitments before redaction

2026-11-05. SPEC -> RED -> CODE. Research-only journal extension; TH-01..12.
Reason: COOKIEUSDT is a public market symbol, but generic credential redaction
correctly masks strings containing cookie. Literal parameter equality then
cannot certify three of the 613 IC-1 native request/cache links. Earlier
invocations retain their FAIL/INCOMPLETE; their absent hashes are not invented.

PR01: Do not weaken or bypass sanitize, reveal credentials, or change the live
adapter/journal. A PublicResearchJournal subclass adds public_descriptor_sha256
to GET REST_BEGIN/REST_END before existing redaction. Descriptor canonical UTF-8
JSON contains method, endpoint, private=false, parameters (null for terminal).
Only /api/v3/klines and the isolated official Spot DAILY1s archive path scope
are eligible. Reject private, non-GET or unknown endpoints before recording.
The digest is not a permission to request arbitrary endpoints or discover keys.

PR02: A separate IC-1 verifier requires the commitment on EVERY native attempt,
validates it against the immutable registered exact window query, and checks
logged fields equal sanitize(expected). Verify main-file chain, identity,
sequential/unique records, unique attempt IDs, paired begin/end/run ownership,
closed runs, receipt/status/IDs, exact public raw bytes, cache SHA and decoded
rows. A matching body with a wrong/missing query hash cannot pass. No tolerance
or favourable exception for masked symbols. Legacy literal verifier outcomes
stay unchanged and are separate from this new stronger evidence profile.

PR03: Archive verification uses the same commitment for its exact validated URL
and parameters, including a masked URL containing COOKIEUSDT. Every receipt is
bound to begin/end IDs and the stored public file bytes; malformed paths,
private events, orphan attempts, missing hashes, modified caches/files or open
WAL/runs reject. A public file hash never certifies source compatibility or PnL.

PR04: New invocations register this source/spec and current journal hashes.
Already-running and failed registrations stay immutable with source snapshots.
Tests precede code: COOKIEUSDT masking stays enabled, committed native and
archive links verify, altered queries/bodies/hashes and private requests reject.
Run focused/app regressions, diff/staged gates, commit and push intended source,
tests/specs only. No trading behaviour, keys, limits, workers or promotion.
