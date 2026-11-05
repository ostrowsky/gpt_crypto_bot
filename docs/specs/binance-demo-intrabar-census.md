# IC-1: complete ambiguity census and separately labelled archive refinement

Date 2026-11-05. SPEC -> RED -> CODE. Independent research, TH-01..12.
Parents RDE-1, RDE-2A/C and EOS-1 remain immutable. No trading-policy change.

## Frozen population and question

IC01: Enumerate EVERY ambiguity from ALL 168 sealed EOS-1 portfolios, including
TUNE/TEST, base/stress, STOP_FIRST/TARGET_FIRST and all original seeds. Verify
EOS registration, compressed report metadata/hashes and its frozen input hashes;
require COMPLETE and 168 financial passes/156 reference matches. Bind each case
to its single closed trade (symbol, entry<=origin<=exit), frozen entry price,
stop/target and chosen reason. Reject missing/duplicate/malformed bindings.
Do not select cases by sign, performance or whether the earlier six probes
resolved. Deduplicate GETs by symbol/origin, never drop differing trade levels.
Expected observed census: 613 windows, 223 symbols, 1457 portfolio-case bindings.
Those counts are checked against actual input, not manufactured as outcomes.
Bindings sharing a market window are dependent observations. Counts by case
are not an independent-trade sample or a profitability confidence interval.

IC02: This is an execution-order DATA DIAGNOSTIC, not a full portfolio replay.
Changing an exit changes subsequent admissions: never patch old trade PnL or
sum hypothetical wins. No new models/features/HPO/seeds/risk limits. All
outputs remain SIMULATION / RETROSPECTIVE_REUSED_HOLDOUT, promotion_allowed=false,
actual_execution_reconstructed=false, full_replay_complete=false.

## Stage M: complete same-source minute census

IC03: Register all windows/cases and original archive page/content hashes before
requests. Reuse frozen RDE-1 manifest/source-page binding and original_bar reads.
Pin EOS registration/report hashes, new source/spec hashes and current full
Truth Harness path/hash/status. Source archive stays read-only. Changed inputs,
registrations, cached responses or resumes reject, with failures preserved.

IC04: Reuse credential-free DemoReadClient and PacedReadTransport, exactly
GET https://demo-api.binance.com/api/v3/klines, interval1m, UTC, exact15m window.
Existing production adapter/host allowlists remain unchanged. Collect every
window even if another one fails. Reuse validate_window's exact Decimal OHLC/
trade-count and preregistered additive-volume rounding checks. No tolerance
relaxation, source replacement, inferred flat candles, seed/day omission.
Missing/error/mismatch responses cannot determine first touch.

IC05: Minute classification uses chronological OPEN before intraminute HIGH/LOW.
An open at/beyond one level resolves that first touch. If both levels are first
touched within the same minute with open between them, ORDER_UNRESOLVED.
Otherwise first low<=stop or high>=target resolves SL_FIRST/TP_FIRST. Use
Decimal(str(frozen float level)), matching RDE-2C; any numerical-boundary
contradiction/no touch remains unresolved. Do not infer a within-minute path.
Return first bucket origin/resolution, not an exact exchange fill timestamp.

IC06: Close journal before verifying hash chain, paired public requests and raw
response/cache links using existing verifier. Require native provenance VERIFIED
for accepted first-touch results. Numerical MATCH and provenance are distinct.
Record requested/matched/failed windows, every case and group denominator,
SL_FIRST/TP_FIRST/unresolved, plus contradictions of each EOS convention.
Neither stop-first nor target-first is asserted as historical truth by default.
Minute success remains MINUTES_ONLY / PARTIAL_ORDER, never the old RDE-2 gate PASS.
Use the separately tested PRC-1 pre-redaction GET commitments from
[public request commitments](binance-demo-public-request-commitments.md).
COOKIEUSDT stays masked in journal text; every exact query is verified by its
registered descriptor hash. Old missing-commitment runs keep their failures.

## Stage S: optional official archive seconds, an explicit new source

IC07: After M, register ALL still unresolved cases with accepted Demo1m matches,
parent registration/native audit/response SHAs and their exact original15m.
Also include the six RDE-2A original windows as fixed compatibility controls.
Download official public Spot DAILY1s files from data.binance.vision, together
with their .CHECKSUM. This is a SEPARATE archive-source compatibility experiment,
not a fallback relabelled as native Demo history. Never alter old gates/reports.
Reference: https://github.com/binance/binance-public-data/blob/master/README.md
(Spot timestamps since 2025 are microseconds; archives can be revised).

IC08: An isolated research transport permits only HTTPS data.binance.vision:443,
GET exact /data/spot/daily/klines/{symbol}/1s/{symbol}-1s-YYYY-MM-DD.zip[.CHECKSUM].
No query, redirect, proxy/environment endpoint discovery, arbitrary hosts,
private endpoints or credentials. Unicode symbols are URL-encoded. Validate
symbol/date/path before connecting; strict TLS and <=15s total per attempt.
Sequential >=.3s pacing; no IP-ban retry, honour Retry-After<=60s or halt UNKNOWN;
max3 attempts/logical GET, <=2000 attempts and <=3600s per invocation. Persist
HTTP/non-HTTP errors without credential/exception text. Never treat []/404 as
proof that all history is nonexistent. No automatic recurring worker.

IC09: Seal raw ZIP and checksum bytes plus public request/receipt metadata and
SHA before parsing. Journal every GET begin/end, referencing exact immutable
public files for binary bodies; post-close verify chain, pairing, endpoint,
receipt and file hashes. Require checksum SHA256 and exact ZIP basename.
ZIP must contain exactly the expected CSV filename, no directories/encryption/
extra members; reject traversal, corrupt CRC, compressed>16MiB, expanded>64MiB,
more than 86400 rows or duplicate/unordered/off-day timestamps. Stream CSV,
never extract files. Fixed schema, strings/integers, no silent header guessing.
Timestamp unit is determined by UTC DATE (<2025 milliseconds, >=2025
microseconds), with exact open/close quantum before conversion; never truncate
an arbitrary microsecond timestamp onto the grid. Require all900 seconds of
each evaluated15m, then validate_seconds against original15m AND EACH accepted
Demo1m. A whole-window price match alone is insufficient. All accepted source
files are immutable snapshots; later revisions require another registration.

IC10: Only after checksum/schema/provenance/nested aggregation MATCH classify
first touch at1s by the same chronological rule. Both inside one second stays
unresolved; no artificial curve, random order, hidden fallback or repricing.
Report archive-derived order separately from minute-derived and unresolved
cases. Matching market tape does NOT certify Demo fills, queue, liquidity,
latency, live PnL, source identity outside these windows or old native gate.

## Resources, evidence and delivery

IC11: Runtime only under own .runtime/intrabar-execution/IC-1-20261105*;
no keys/accounts/orders/training/production workers. <=2 CPU threads, one writer,
one portfolio/report in memory. Native journal<=256MiB; archive cache<=1GiB,
expanded member<=64MiB (streamed), audit<=128MiB; total new runtime<=2GiB.
Budget exhaustion leaves explicit pending/unresolved, never a smaller winning
sample. A bounded invocation may resume only its immutable registration and
successful files after integrity verification; failed attempts remain retained.
Missing past events cannot be repaired by future observations. Do not start
an indefinite forward collector; document exactly what forward evidence is needed.

IC12: Add genuine RED tests before source. Cover full case population/binding,
open priority, same-bucket ambiguity, minute/source rejection, immutable parent,
download path/TLS/budget/HTTP errors, checksum/ZIP/units/nested mismatch and
binary audit linkage. Relevant regressions, git diff --check, selective staged
Truth Harness, review, commit and push spec/tests/source only. Full Truth Harness
FAIL remains FAIL; artifact accounting/source checks cannot certify earnings.
Exercise the complete collector/report contract with its registration metadata,
including rounding_contract, before native requests. A failed invocation keeps
its exact source/spec snapshot; reruns use a new registration and own journal.
