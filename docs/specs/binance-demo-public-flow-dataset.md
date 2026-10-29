# Forward public Spot flow dataset (PFD-1)

Status: engineering/data collection only, 2026-10-29. Companion to FSA-1.
A clean new continuous dataset is required: existing Spot trades/L2 overlap is
119 seconds; long legacy archive cannot bridge snapshot/diff sequences. Old
aggregates contain unmarked trade gaps and are forbidden as training evidence.

Independent credential-free collector under own application package/runtime.
No adapter, .env, private stream, orders, account or existing worker mutation.
Fixed Binance Spot Demo public host wss://demo-stream.binance.com, combined
streams BTCUSDT/ETHUSDT/SOLUSDT aggTrade and depth20@100ms only. Verify official
Demo host at launch; never silently switch venue. No production stream changes.

Start immediately after tested implementation. Immutable schedule: first next
Europe/Budapest midnight plus seven complete calendar days; initial partial day
is warmup. Capture ends at this registered timestamp or explicit own stop file,
disk reserve<20GiB, compressed budget8GiB, protocol/persistence incident. These
stops are INCOMPLETE, not success. No outcome-driven stop. One bounded reconnecting
connection, exponential backoff <=60s, below-normal own process, bounded buffers.
Registration includes fixed streams, origin, planned end, quotas and source hash.

Store raw public payloads, stream, connection/session id, wall receipt ms and
monotonic receipt ns. Flush gzip chunks every <=60s or <=8MiB uncompressed;
close chunks atomically and hash compressed bytes in manifest. Durable status
contains messages per topic, first/last receipts, gaps/conflicts/reconnections,
actual interval and resource state. Persistence failure aborts, never drops silently.

Partial depth20 is a full standing top20 snapshot, not diff depth and not full
exchange book/cancel flow. Validate positive sorted noncrossed levels and monotone
lastUpdateId per symbol/session. No exchange timestamp is supplied: exchange-age
UNKNOWN; do not invent it. Identical repeated update is duplicate, conflicting
same update invalid; lower update invalidates session coverage. AggTrade validates
IDs, event/trade timestamps and maker flag; retain first/last trade IDs. Detect
aggregate/trade ID gaps, reconnect gaps, clock regressions and stale receipts.
Never turn a disconnected interval into zero trades. A new session resets
continuity; future features must require complete receipt windows and warmup.
All bad messages remain raw evidence with reason, not deleted or repaired by future
snapshots. Public matching data is not actual account execution or profitability.
Backward/overlapping underlying trade IDs at a new aggregate ID and backward
source event/trade timestamps are protocol incidents. Session-start receipts
also enforce nonregression, including connections that never deliver a frame.

No model fit, PnL superiority, adoption or seven-day financial proof from collector
health. Historical candle flow experiment remains separately named proxy. Later
raw-flow/L2 model experiment needs frozen features, label timing, costs and
chronological train/test protocol before fitting this dataset. Do not imply
seven days guarantees adequate power. Runtime, raw messages and reports untracked.

PFD01 malformed/unexpected venue/topic rejected; no credential/order dependency.
PFD02 raw receipts/session/sequence preserved; gaps never become zeros.
PFD03 standing partial-depth semantics, duplicate/conflict/crossed validation.
PFD04 immutable finite DST-aware schedule, caps/own stop produce explicit status.
PFD05 durable hashed gzip roundtrip, tampering/persistence failure rejected.
Write tests and establish genuine missing-API RED before implementation.
Rollback is own stop file only, no trading worker restart or policy mutation.

## Protocol references and executable contract

Host verified 2026-10-29 against official
[Demo endpoint table](https://github.com/binance/binance-spot-api-docs/blob/master/demo-mode/general-info.md).
Payload semantics use official [market streams](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md).
Demo observations remain Demo data; the official endpoint table cautions that
realistic Demo data does not establish live strategy performance.

`make_registration(started_at_ms)` fixes the first following Budapest midnight,
then adds seven local calendar days (DST-aware), fixed host/topics/quotas and
the collector source hash. `FlowValidator` accepts raw combined frames plus
wall/monotonic receipt and explicit session IDs; `ChunkStore` persists exact raw
strings in bounded gzip chunks, with immutable registration and a verified
manifest. `verify_dataset` rejects compressed/source tampering. Existing dataset
folders are not silently resumed or overwritten. `stop_reason` exposes finite
schedule, own stop, reserve and byte-cap outcomes. The CLI is explicit `--collect`.
An optional output path must resolve inside this application's `.runtime`.

Repeated identical IDs are duplicates, different content at the same ID and
backward IDs are protocol incidents. Forward aggregate/underlying trade ID gaps,
reconnects and receipt gaps over five seconds are retained as coverage gaps;
they cannot certify a zero-flow interval. Valid payloads after a gap remain raw
observations, with downstream warmup required. All bad frames are durably flushed
before protocol termination. Known `!serverShutdown` is a recorded reconnect
control. Receipt clock regressions terminate after preserving the offending frame.
No exact exchange-age claim is made from an uncalibrated local wall clock.
