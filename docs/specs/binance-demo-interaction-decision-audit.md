# Binance Demo: полный аудит взаимодействий и решений

Дата: 2026-10-11, обновлено 2026-10-27. Статус: **AUD-A/B1/B2/B3/C1/C2a implemented;
170 audit focused / 665 app GREEN; full interaction audit incomplete**.
User requirement: все взаимодействия с Binance и все принятые решения нужны
для последующего анализа эффективности. Scope: только `apps/binance_demo_bot`.
Parent: [программа](binance-demo-autonomous-trading-program.md),
[dataset/outcomes](binance-demo-phase5-outcomes-training-dataset.md).

## Required coverage

AUD-01: append-only audit в own `.runtime`, начиная до первого подключения,
до первой сделки и до включения learning. Все REST attempts (включая retries,
rate-limit/timeout/error/cancel), WSS connect/disconnect, subscribe/ACK/control/
heartbeat и входящие сообщения public/private регистрируются. Open candles,
отклонённые/невалидные сообщения и решения ничего не делать не исключаются.
Нельзя семплировать журнал взаимодействий или логировать только успешные BUY.
В текущем P2 пока существуют snapshots/incidents; они не объявляются полным
аудитом. Будущие OMS send/query/cancel/replace/TP/SL/reconcile также обязательны.

AUD-02: InteractionEnvelope содержит app/environment/account episode/scope,
event/schema/process/source-code IDs, interaction/request/correlation/attempt/
connection IDs, endpoint/method/direction, safe parameters, sequence, causal
parent IDs, actual send/receipt/monotonic clocks, response/status/error code,
rate-budget/latency/size, raw-content hash и normalized snapshot references.
Origin event time не заменяет receipt time. UNKNOWN/timeout не является ACK;
BEGIN без terminal response после crash остаётся незавершённым наблюдением.
Каждая повторная попытка получает свой attempt_id с общим request_id.

AUD-03: API secret, API key, HMAC signatures, Authorization/cookies и access
tokens/listen keys не сохраняются, в том числе в URL/query/headers/exception
text и вложенных объектах. Хранятся sanitized structured параметры и whitelist
headers; auth определяется opaque key/account identity, без секрета. Исходные
public market bytes/JSON/hash сохраняются; private payload после обязательной
redaction остаётся помеченным sanitized, не "exact raw". Балансы/fills/fees
относятся к own account namespace. Credentials/runtime audit не коммитятся.

AUD-04: DecisionEnvelope связывает data cutoff и доступные snapshot IDs с
policy/model/risk/schema versions, candidate population, BUY/SELL/HOLD/WAIT/
BLOCK/NO_ACTION, explicit reasons, scores/uncertainty, selected/rejected intent,
quantity/entry/TP/SL parameters, position/portfolio references. Operational
decisions о refresh, reconnect, backfill, quota/backpressure/readiness также
логируются со своим типом, не выдаются за торговые решения. Событие решения
не означает placement/ACK/fill. Learning changes сохраняют hypothesis/experiment
IDs, prior/new parameter versions и validation/promotion/rollback evidence.

AUD-05: send intent/decision durable до финансового действия; actual response
и fills append отдельные события. Для incoming market group raw audit и
derived snapshots имеют общую причинную связь и durable commit ACK. Ограниченная
очередь/диск, audit persistence failure/overflow и crash gaps явны. При отказе
durable intent запрещены новые ордера; продолжаются reconcile/protection по OMS.
При отказе incoming audit поток не READY, evidence retention/gap UNKNOWN
видны, нет silent drop или best-effort "полный журнал". Не вводить unbounded
очередь. Первичные timestamps/IDs не подделываются после задержки записи.

AUD-06: retention/rotation/compression/archive с immutable manifests, counts,
time range, checksums и проверяемой цепочкой. Не удалять данные active exposure,
нефинализированных дней или текущих исследований; policy/version эксплицитны.
Coverage denominator берётся из transport/decision counters, не из самих
сохранённых строк. Показывать expected/recorded/gap n/N отдельно для каждого
канала/attempt type; не переименовывать downtime/missing data в нулевой результат.

## Analysis and delivery gate

Связь decision -> intent -> submit -> ACK -> fill -> fees -> valuation -> daily
net equity PnL позволяет анализировать входы/отказы, сопровождение/TP/SL,
задержки, комиссии, ошибочные решения и качество обучения. Counterfactual/
backtest/proxy результаты отделены от actual account outcome. Export замораживает
IDs/hashes/cutoffs/versions; future labels доступны только после maturity.

Обязательный порядок: SPEC -> focused tests/RED -> code -> GREEN/regressions
-> completeness/redaction/crash/latency verification -> staged Truth Harness
-> commit/push. Проверить retries/error/WS raw/open/malformed/control, secret
redaction во всех путях, no-action population, correlation, ACK vs unknown,
disk/queue failure, restart/rotation/hash integrity и analytical reconstruction.
Реализация аудита имеет следующий приоритет после P2B2-12 writer и обязательна
до authenticated autonomous execution/learning. Эта спецификация не является
заявлением, что текущий бот уже логирует все перечисленные взаимодействия.

## AUD-A: durable REST attempts and decision journal (2026-10-26)

Implemented slice. Full AUD-01..06 remains a release
gate: public/private WSS, runtime-wide operational decisions, atomic incoming
snapshot links and archive retention are subsequent slices, not claimed here.
Rollback: omit explicit `audit=` on the read-only library client; this makes
coverage UNCONFIGURED, never complete, and cannot qualify autonomous execution.
No trade policy, credentials discovery, order sending or financial core changes.

- A01: explicit `AuditJournal(AppPaths, account_alias, episode_id, scope_id)`
  owns `audit.sqlite` with an OS writer lease, WAL/FULL, immutable identity and
  append-only rows. Each event has sequence, event/run/process IDs, source-code
  hash, actual wall/monotonic times, previous/content hashes. Verify all rows,
  identity and immutable triggers at reopen. A foreign account/episode is denied.
- A02: bounded synchronous writes (no unbounded async queue): at most 8 active
  or waiting submissions, at most 48 MiB serialized event and explicit total
  database+WAL disk budget (default 512 MiB, no automatic deletion). Overflow,
  disk/SQL failure and commit uncertainty latch FAILED; never retry an ambiguous
  commit or return its event ID as acknowledged. No new REST transport call after
  its audit BEGIN fails. Terminal audit failure does not become a successful read.
- A03: each actual REST attempt has durable BEGIN before transport and separate
  END with the same request/attempt/parent IDs. Retries share request ID and have
  distinct attempt IDs. BEGIN is intent, not proof bytes were sent; send and
  receipt clocks are measured outside the audit commit. END precedes parsing,
  deadline/status/auth validation, so malformed/429/5xx/redirect/error/timeout/
  cancellation observations survive. No exception message is persisted. Transport
  counts increment independently of saved rows. A crash between records remains
  unresolved, including after reopen; do not infer a response from absence.
  A BEGIN commit consuming the deadline produces a CANCELLED terminal with no
  send clock or transport attempt. Return both BEGIN/END IDs in successful read
  provenance. Binary encoding of safe public bytes must remain lossless even
  when that encoding coincidentally contains a redaction keyword/credential.
- A04: structured parameters, whitelisted numeric rate headers and bounded
  response payloads only; never store signed query or request auth headers.
  Remove sensitive keys recursively, known credential values, token-bearing
  embedded URLs/text (including percent encoding). Exact public response bytes
  and SHA256 are retained only when safe; otherwise mark SANITIZED/OMITTED.
  Private JSON is always SANITIZED and hashes its sanitized content; malformed
  private content is OMITTED. Never label private payload as exact raw.
- A05: explicit `record_decision` records OPERATIONAL or TRADING decisions,
  including WAIT/HOLD/BLOCK/NO_ACTION, reasons, cutoff, available snapshot IDs,
  candidate population, policy/model/risk versions and optional intent. Supplied
  source available_at must be <= cutoff. It returns an audit ID, not an order
  receipt. No synthetic automatic policy decisions are invented by this slice.
- A06: detached bounded-page reads and frozen manifest (identity, first/last seq,
  row count, chain tip, digest) reconstruct durable evidence. Health publishes
  expected/ACKed/gap by event kind from process-local submission counters, a
  separate persistent unresolved-BEGIN count, incomplete prior runs, and explicit
  scope REST_ATTEMPTS_AND_EXPLICIT_DECISIONS / full_audit_complete=False. Zero
  submissions are not 100% coverage. Run close cannot erase failures or unknowns.
  Counter basis is explicitly PROCESS_SUBMISSIONS with run/process IDs; previous
  process denominators cannot be reconstructed from successful rows. OS process
  death must release ownership without removing the durable unknown attempt.
- A07: public and authenticated read clients accept an explicit matching journal.
  Shared public wrapper can use it through the same DemoReadClient; no credential
  access in public code. Mismatched identity is denied before network activity.
  Existing unaudited read-only callers retain behavior but expose UNCONFIGURED.

Test-first acceptance: real SQLite/foreign readers and abort triggers, timeout
then retry, 5xx then success, malformed JSON and 429, cancellation, known/nested/
URL secrets, private malformed bodies, append mutation/identity/tamper/restart,
capacity/disk/commit failures, independent denominators, causal WAIT/NO_ACTION,
manifest stability and no implicit network/credential access. Green unit tests
prove only these contracts; full-market latency, signed-account certification,
complete WSS capture and daily PnL remain unproven.

### AUD-A verification

SPEC -> 18 intended missing-module RED tests -> 18 GREEN. Six additional
boundaries reproduced 1 failure/2 errors (deadline reason, missing terminal ID,
base64 collision); three preserved size/lease/identity contracts already GREEN.
Next counter-basis probe reproduced a missing field; real OS-exit recovery
already passed. Encoded sensitive-key probe then reproduced retained secret
content before repair. Final 27 focused / 522 app GREEN; no skips, xfail or
weakened assertions. Initial default-temp fixture permission errors were tooling,
not behavioral RED; fixtures now live in the own ignored runtime namespace.
Full regression needs junction creation rights for an existing isolation test;
an unprivileged failure is not reported as a passing run. Synthetic transports,
foreign readonly SQL, abort triggers and own subprocess death; no real keys or
network needed. Root spec tests14 and harness tests13 pass. Full Truth Harness
remains **FAIL TH-11**: canonical portfolio evidence references an old
`replay_backtest.py` hash; this unrelated blocker is not waived by unit success.
No full-market speedup, current private permission or financial improvement is
claimed. Next: AUD-B WSS/control/raw-invalid/open-candle capture and causal
snapshot linkage, operational-decision integration, then archive/export policy.

## AUD-B1: WSS application-boundary capture (2026-10-26)

Implemented after spec-first tests. Explicit journal on the shared REST client also
enables public market and private readers; no journal means UNCONFIGURED.
Identity must match the reader's snapshot/evidence store before connect. No
changes to financial kernel, trade policy, FULL/250ms guards or order authority.
Rollback: omit journal only for diagnostic read-only libraries; no autonomous
qualification. Full audit remains incomplete until protocol-level controls,
all runtime operational decisions and retained archives are implemented.

- B01: own `AuditedWsTransport` wraps default and injected application transports.
  Every connect/send/recv/ping/close attempt gets durable WS_BEGIN and WS_END,
  correlation/operation/connection IDs, operation sequence, actual clocks and
  fixed safe outcome. No third-party exception text. Failed connect, timeout,
  malformed/open/duplicate messages, rejected subscribe ACK and false Pong are
  retained; no sampling. Each reconnect gets a new connection ID. Health counts
  physical calls/results/errors/frames independently of durable journal ACKs.
- B02: raw public application-message bytes (<=1MiB) are exact when safe; private
  messages are sanitized/omitted. Signed subscribe auth is redacted, including
  its signature echoed in later text. Safe normalized subscription method is
  explicit; no credentials are loaded by public readers. Automatic wire-level
  ping/pong/close control frames consumed inside websockets are NOT observed by
  this boundary: `protocol_controls_observed=False`, never full coverage.
- B03: capture original receipt wall/mono immediately after physical recv and
  before journal serialization/commit. Reader normalization and latency guards
  use that original receipt, including startup buffered events. Disk delay may
  invalidate a session; timestamps must not be moved forward to conceal it.
- B04: no connect/send/recv/ping after audit intent failure. A terminal journal
  failure closes/invalidate reader and never returns successful normalized data.
  Physical close is always attempted even if audit is broken; repeated close is
  idempotent, cleanup errors are safe and health/gaps stay visible. WS_BEGIN
  without WS_END after crash is an unresolved WSS operation, not a successful
  connection. Disk/queue limits and unknown commit remain sticky.
- B05: audit processing records reference raw receipt IDs and confirmed market
  snapshot IDs after the prepared group's durable ACK, or the private evidence
  store namespace/database/record sequence after its ACK. Open candles may have
  no snapshots; raw evidence still exists. Public startup buffered frames also
  get linkage. Snapshot failure has no fabricated confirmed IDs. Distinct WAL
  databases are NOT one transaction: raw receipt survives a derived failure;
  missing processing marker after crash is UNKNOWN, not processed. Post-commit
  audit failure invalidates the session even when normalized rows already exist.
- B06: health exposes bounded detached counters, audit state and explicit scope;
  source-code hash includes the audit bridge and both readers. ACK evidence does
  not certify private permissions, exchange reconciliation, all-market latency,
  archive completeness or PnL. No production service starts in these tests.

Acceptance: real journal and store commits with synthetic public/private
transports; malformed/open/raw/ACK capture, secret echo and nested redaction,
original clocks under commit delay, connection lifecycle and failed heartbeat,
intent/terminal/processing abort triggers, foreign identity, independent n/N,
durable linkage/failed snapshot and unresolved WSS at restart. SPEC -> tests
RED -> implementation -> focused/full GREEN -> staged Truth Harness -> commit/push.

### AUD-B1 verification

18 new intended missing-interface RED cases preceded implementation. First
implementation run exposed private audit failure being replaced by
INVALID_INCIDENT; repair keeps precise audit reason in health and records a
STREAM_GAP in the established private-evidence vocabulary.18 then GREEN.
Six added boundaries: startup/closed-candle linkage, persistent WSS unknowns,
private receipt clocks and committed-fill survival already GREEN; heartbeat
audit reason was RED. Public timeout/heartbeat added a second RED reproduction.
Both now retain the original audit failure. Final25 new /547 app GREEN,
47.463s final full suite; root14 spec +13 harness tests PASS. No skip/xfail,
weakened assertions, real keys, exchange requests or production processes.
Actual SQLite aborts/foreign file reads/reopen verify durability. Distinct WAL
journals are not claimed atomic, and a durable fill can survive a failed
processing marker while the reader correctly stops. Mandatory full Harness
remains FAIL TH-11 (old replay source hash); no policy relaxation or financial
claim. No new full-market latency trial; previous FAIL remains authoritative.
Next AUD-B2: wire-level control observation, runtime-wide decisions and
completeness/processing-gap analysis; archive/retention integration remains open.

## AUD-B2: coordinator decisions and recorded-prefix processing coverage

Implemented after spec-first tests, 2026-10-26. Scope: current read-only market
coordinator and stream supervisor, plus public/private incoming-message
dispositions. Policy/OMS/learning do not yet exist and cannot claim coverage.
Protocol-level automatic controls and retained archives remain separate gates.
Rollback: explicit absence of journal keeps diagnostic libraries UNCONFIGURED;
it cannot qualify autonomous execution. No financial kernel/trading changes.

- B201: `OperationalAudit` records each covered branch before its network or
  scheduling side effect: START/STOP, metadata/clock/quote refresh, not-due waits,
  per-symbol history-ready/cooldown/backfill, bounded-work/quota blocks, shard
  reconnect/backoff/rotation, L2 readiness/buffering/snapshot and known failures.
  A NO_ACTION/WAIT is a real decision, not a synthetic trading signal. Each
  record has component/operation/reason, cutoff, exact considered population,
  explicit population state, versions and safe structured context. Independent
  helper expected/ACKed/gap counters identify their component/process scope.
- B202: universe/history source IDs resolve against the owned store; use the
  actual stored available_at, never a supplied fictitious timestamp. Any source
  after cutoff or foreign audit/store namespace is denied before side effects.
  Unknown initial population is NOT_DISCOVERED, not an empty successful market.
- B203: decision-write failure latches AUDIT_FAILED and aborts new work, stops
  owned readers/maintenance/writer where applicable, preserves the exact safe
  audit reason and reports cleanup failures. Stop is attempted even when its own
  decision cannot persist. Background handlers cannot overwrite an audit failure
  with a generic runtime error or keep a fresh-data readiness verdict.
  An audit abort inside the coordinator lock must release owned resources
  without waiting on its own coordinator thread blocked on that same lock;
  a subsequent external stop joins the stopped coordinator normally.
- B204: every completed application RECV receives one final disposition when
  known: CONTROL_ACK after validated subscribe response, PROCESSED after derived
  ACK (including open/no-write candle), or REJECTED after schema/normalization/
  subscription failure. Failure to persist a disposition remains UNKNOWN;
  never create a second marker to hide an ambiguous commit. Pre-ACK buffers keep
  original receipt IDs/clocks. Broken/missing normalization remains visible.
- B205: `AuditJournal.processing_report(through_seq=None)` is a read-only frozen
  prefix report. Categories received_N = processed_n + rejected_n + control_ack_n
  + unknown_n + conflict_n. Duplicate/conflicting/future/foreign references are
  failures, not extra successes. Return invalid_marker_n and fixed prefix seq/
  chain tip/digest. A bounded SQL scan streams results; no all-log Python list.
  Older prefix results remain identical after later markers arrive or restart.
- B206: denominator basis is DURABLE_COMPLETED_RECV, not all exchange traffic.
  Capture coverage stays UNKNOWN; independent transport/submission counts are
  reported separately, never reconstructed from saved rows. Zero received rows
  yield UNKNOWN. Even RESOLVED_RECORDED_PREFIX cannot set full_audit_complete or
  enable trading. Health exposes audit failure/counters with no payloads; health
  reads themselves do not generate decisions or mutate the journal.

Acceptance: SPEC -> RED tests for branch provenance/side-effect ordering,
causal history/source rejection, per-symbol WAIT/NO_ACTION, reconnect/quota/L2,
intent failure/owned cleanup and background reason preservation; actual reader
ACK/malformed/private rejection and ambiguity; frozen prefix/late marker/zero/
duplicates/foreign IDs and native SQL denominator tests. Then implementation,
full regression and staged Harness, selective commit/push. No live keys/orders
or profitability/market-latency claims. Full Harness TH-11 remains independently
FAIL until current-source historical evidence is regenerated.

### AUD-B2 verification

18 intended missing-module/interface errors RED ->18 GREEN; an initial full
565-test regression was GREEN. Three new boundaries plus a stronger original
failure-reason assertion then reproduced4 failures: leaked SQL error, lost
foreground REST audit reason, cleanup overwriting reason, and fresh-looking
health after an external journal failure. After repair21 focused GREEN.
Next five cases plus stronger malformed-marker checks reproduced4 failures:
unclassified pre-ACK event/invalid subscription ID, a missing connected-session
NO_ACTION, and invalid marker schema counted as processed. One fixture
inheritance mistake produced unrelated setup errors; these are not behavioral
RED and were corrected before reproducing the actual failures.26 then GREEN.
Finally two real blocked-coordinator join probes reproduced skipped cleanup in
runtime and supervisor; abort now closes resources without joining under its
own lock.28 focused GREEN; final575 app,14 root spec and13 harness tests GREEN.
No skips/xfail or weaker assertions. Own SQLite abort triggers, stored source
cutoffs, real threads, frozen-prefix/reopen SQL and synthetic REST/WSS establish
these mechanics; no real keys or network calls. Full Harness remains FAIL TH-11
for the historical replay hash. No full-market rerun or earnings claim. Next:
automatic protocol-control capture and retained archive/export lifecycle;
market latency and account/ownership qualification still precede execution.

## AUD-B3: parsed WebSocket control frames (2026-10-27)

Specified before tests/code; IMPLEMENTED with16 focused GREEN. Extend opt-in audit on owned default transports,
using the pinned websockets17.1 ClientConnection factory. No dependency-file
patches, endpoints/TLS/proxy changes, credential discovery or order capability.
Injected transports without the observer remain explicitly unobserved.

- B301: install per-connection hooks before the native reader thread starts.
  Observe parsed PING/PONG/CLOSE before protocol interpretation and automatic
  responses; preserve original wall/mono, connection ID, direction, fixed opcode,
  phase and causal parent. Public decoded payload <=125 bytes may be exact when
  safe, but is not a serialized/masked wire capture. Private payload is always
  sanitized/omitted. Close payload is always OMITTED (only byte count), even
  on public streams. No exception text, close reason, keys or signature leakage.
- B302: outbound control phases INTENT (durable before queue), QUEUED (after
  protocol serialization), WRITE_ATTEMPT and WRITE_RETURNED/WRITE_UNKNOWN are
  distinct. Local sendall return is not a remote ACK. Batch send failure leaves
  every pending control uncertain even if earlier bytes may have been sent.
  Max64 pending control intents; no unbounded queue. Data-message payloads are
  not duplicated by the control observer. Malformed frames rejected before the
  parsed-frame hook remain unobserved; full capture coverage stays UNKNOWN.
- B303: counters for observed inbound/outbound controls, queued controls,
  socket-write attempts/returns/unknowns and audit ACK/gaps come from observer
  actions independently of saved rows. Actual hook installation, not merely a
  transport capability or mocked factory, enables protocol_controls_observed.
  Reset connection identity on reconnect; installed callbacks freeze that ID.
- B304: journal failure/overflow closes the raw socket without recursively
  invoking ClientConnection.close while its protocol mutex is held. Suppress
  further protocol processing/queueing/writes after failure; native teardown
  must terminate. Preserve the first safe audit error through high-level
  transport/reader handling. No false ACK or silently successful fallback.
- B305: controls rejected during content interpretation get a separate REJECTED
  marker; parser/TCP/TLS gaps cannot become full coverage. Health has counters/
  scope/unknowns but no payloads. Audit absence retains the original factory and
  unaudited behavior. Hooks are supported only for the exact pinned backend;
  unsupported versions/duplicate ownership are rejected before use.

Test-first gate: real Sans-I/O protocol for automatic Pong/close echo, client
Ping/unsolicited Pong/content rejection, no data duplication and bounded pending
work; real native ClientConnection + own bounded loopback pair for masked responses/write
outcomes and journal abort under the protocol lock (no external network). Verify
default public/private wiring, absent/injected observer, source clocks, private
redaction and first-error preservation. Then full regressions/staged Harness,
commit/push. No real keys, Binance connection, trading or latency/PnL claim.
Archive/rotation/compression/export remains the following implementation gate;
active evidence must not be deleted to evade the existing finite disk budget.

### AUD-B3 verification

The15 new tests first failed on the intended missing `protocol_audit` module
(RED), then passed after implementation. The Windows socketpair fallback blocked
at accept in the sandbox; diagnostic traceback identified a fixture/tooling
problem, not a successful run. Replaced it with an equivalent own loopback pair
with two-second connect/accept/receive bounds; no external server or keys.
A public Close-code1000 case initially stayed GREEN because its invalid UTF-8
code bytes already caused omission. A valid application Close-code3072 case
then reproduced exact reason retention (RED); all Close payloads are now
explicitly omitted, including automatic echo. Final16 focused GREEN (3.068s);
complete app591 GREEN. Native automatic Pong
passes a real masked socket send, and journal abort under the protocol mutex
terminates the native receive thread. These tests certify the stated parsed
boundary only. Archive/retention/export and full-market latency remain open;
no trading or PnL improvement is claimed.
Root14 specification and13 Truth Harness tests pass; `git diff --check` passes.
Full Harness remains FAIL TH-11: canonical portfolio artifact has a stale
`replay_backtest.py` source hash. This read-only observability slice does not
repair or rely on that portfolio evidence; a current causal replay is required
before profitability claims or trading-policy relaxation.

## AUD-C1: frozen archive/export and preserve-all retention (2026-10-27)

Specified before tests/code; IMPLEMENTED with38 focused GREEN. Public API: `AuditArchive(AppPaths, *,
max_compressed_bytes=64MiB, max_uncompressed_bytes=512MiB,
max_total_bytes=512MiB, max_archives=64, fault_hook=None)` in `audit_archive.py`;
`export(journal, through_seq=None, archive_id=None)`,
`verify(archive_id, expected_identity=None, expected_manifest_sha256=None)`,
`inventory(protected_archive_ids=())`. All IDs are32 lowercase hex characters;
omitted export IDs are generated. APIs return detached metadata, never secrets.

- C101: export uses an independent read-only SQLite WAL snapshot. Pin source
  identity/cutoff under the short journal lock, release that lock before gzip,
  hashing, fsync or verification. Stream exactly seq1..cutoff in bounded memory;
  original immutable envelopes, IDs, source fingerprints, clocks and hashes are
  preserved. Validate contiguous sequence, row columns, identity and entire
  previous/content hash chain. Concurrent appends cannot alter this prefix.
  Empty prefix0 is explicit; invalid cutoff, foreign runtime and closed source
  are denied. A failed source may export its verified committed prefix for
  forensics, without calling it healthy or resolving unknown ACKs.
- C102: publish a directory under own `.runtime/audit-archives/<id>` containing
  deterministic `records.jsonl.gz` and `manifest.json`. Manifest freezes schema,
  identity, seq range/rows, kind counts, min/max receipt wall time, chain tip,
  compressed/uncompressed SHA256 and byte counts, exporter source fingerprint,
  source state/run at pin and `PRESERVE_ALL_V1`. Count basis is
  DURABLE_EXPORTED_PREFIX; capture remains UNKNOWN/full_audit_complete=False.
  No inferred transport denominator, mature labels or profitability claim.
- C103: immutable publication: no overwrite/reuse. Serialize exports with an
  app-owned writer lease. Write into unique `.pending-<id>`, flush/fsync both
  files, fully verify them, then rename into the final directory. No success
  until final verification. Faults before rename leave explicit incomplete
  evidence; faults after rename are PUBLICATION_UNKNOWN, resolved only by a
  separate verification/inventory. No directory-fsync/power-loss durability
  claim on Windows. Faults never delete records, change source health, append
  artificial source events or reset source counters.
- C104: verifier streams/decompresses with byte/record/manifest limits; rejects
  truncation, CRC/hash/count/chain/identity mismatch, extra files, malformed
  schema, path traversal, symlink/junction escapes and wrong external pin.
  Self-consistent hashes provide INTERNAL_VERIFIED integrity, not authenticity.
  Only a caller-provided matching manifest SHA yields PINNED_MANIFEST; a
  same-file checksum is not an external trust anchor. Errors expose fixed codes,
  no raw payload/path/exception text. Never load executable/pickle artifacts.
  Require the exact versioned manifest keys and typed counts/ranges (JSON bool
  is not a count); invalid deflate streams map to the same safe integrity error.
- C105: separate compressed/uncompressed/total storage and directory-count
  limits, including incomplete exports. Refuse new work explicitly on limits;
  never silently delete archives or truncate source evidence. Bounds apply to
  verification too, including a malicious compression bomb or long JSON line.
  Unexpected entries are surfaced rather than followed or removed.
- C106: inventory verifies all published archives, distinguishes valid/invalid/
  incomplete, returns byte/count denominators, and lists explicit protected IDs.
  Missing protected IDs stay UNKNOWN; corrupt copies FAIL. Retention policy is
  PRESERVE_ALL_V1, deletion_eligible_n=0. No delete API: active exposure,
  unfinalized days and current research are preserved by construction. This is
  export/retention inventory, not space reclamation or live segment rotation.
  Archive counts conserve N: valid+invalid+incomplete=archives_expected_N.
  Layout violations are separate; if unexpected layout prevents complete byte
  accounting, stored_bytes_n=null and observed_file_bytes_n is a lower bound.
  The writer lease path is checked before opening, including junction escapes.

Acceptance: real SQLite/gzip/filesystem fixtures; causal concurrent append,
prefix0, source failure/restart, immutable duplicate, secret-redacted roundtrip,
corrupt/truncated gzip and forged metadata/chain, foreign identity/pin/runtime,
bounded decompression/line/disk/count, publication faults and preserved source,
incomplete/protected inventory and path escape. SPEC -> missing-interface RED ->
code -> focused/full GREEN -> staged Harness -> commit/push. Rollback: do not
call the opt-in exporter; journal behavior and trading guards are unchanged.
Next AUD-C2: certified rollover/reclamation with durable restart anchors and
explicit legal holds; no sustained-storage readiness before that gate.

### AUD-C1 verification

28 new tests initially RED for the intended missing `audit_archive` module;
then28 GREEN. Additional boundary tests reproduced unsafe zlib exception,
incorrect nonobject-source error and missing strict manifest schema (three RED
before repair). Actual child-process deaths before/after directory rename,
actual post-commit source ACK uncertainty and forged inflation sizes passed.
Further RED reproduced archive-count double counting and unchecked writer-lease
junction before repair. Final38 focused GREEN (9.288s),629 app GREEN (68.545s),
14 root spec +13 Truth Harness GREEN. No test skips/credentials/external network.
Manual snapshot export leaves source rows/health/counters unchanged and allows
concurrent append. Integrity is separate from external authenticity and physical
traffic completeness. Full Harness remains FAIL TH-11 for the historical replay
source hash; no current profitability conclusion or live market qualification.
Rollback is omission of explicit exporter calls. Certified segment rollover,
reclamation, restart anchors and future policy/OMS audit remain unimplemented.

## AUD-C2a: sealed segment rollover and restart anchors (2026-10-27)

Specified before tests/code; IMPLEMENTED with36 focused GREEN. `AuditRollover(AppPaths, account_alias=...,
episode_id=..., scope_id=..., archives=None, max_segments=64, fault_hook=None)`;
`rotate(journal, quiesced=False)` returns the successor `AuditJournal`;
`recover(database='audit.sqlite')` validates the chain and opens its active head.
This is an explicit maintenance API. Callers stop/drain readers first and rebind
them to the returned journal; no live transport swapping or latency claim.

- C201: deny rotation without explicit quiescence, wrong runtime/identity,
  failed/closed source, outstanding REST/WSS operations, unresolved prior runs
  or unresolved/conflicting processing markers. Verify source integrity and
  require zero process submission/ACK gaps. Do not clear gaps to pass a gate.
  Serialize rollover/recovery with an app-owned OS lease; guard all paths and
  junctions before opening. No legacy imports/process changes/order capability.
- C202: while source callers are quiesced, export its exact current prefix via
  AUD-C1, verify the manifest pin and tip, then exclusively reserve/fsync the
  successor file in own `audit-segments/<32hex>.sqlite`. No overwritten paths.
  Seal the source in ONE FULL transaction: final RUN_END with archive/next links
  and immutable `meta.sealed` containing that event's seq/hash/ID. A successful
  seal fences/closes the old writer. Ordinary `AuditJournal` startup refuses
  sealed sources before appending; calls on the old handle cannot write again.
  Seal failure/unknown commit never yields a usable successor or fallback.
- C203: successor metadata `parent_segment` and first RUN_BEGIN freeze the
  parent database, final seq/hash/ID, archived prefix and manifest pin. Local
  sequences restart at1; `(database,event_id)` and ancestry remain explicit.
  No invented global sequence or cross-segment processing-coverage claim.
  Validate the parent seal/archive before opening a child, including ordinary
  journal reopen. Caller-supplied ancestry must match the frozen metadata.
- C204: reserve the child before seal, so a crash after seal can initialize the
  reserved empty file. After child initialization, persist immutable
  `meta.successor_opened` on the sealed parent with the child's first record
  hash before returning the child. Recovery follows bounded validated links,
  verifies source rows/seal/archive pins and child ancestry/first-record receipt.
  A missing child, or empty child after an activation receipt, is UNKNOWN/error;
  never recreate it as fresh zero history. Crash-created BEGIN records remain
  in history and incomplete-run evidence remains visible. No duplicate seals.
- C205: crash before seal leaves original source usable and any reserved file
  retained as unlinked evidence. Crash after seal resumes only from verified
  anchors. Missing/corrupt archives, metadata, identity, rows or receipt prevent
  startup. External file disappearance is not interpreted as an empty episode.
  Transition errors have fixed codes without paths/raw exceptions. Windows
  directory-fsync/power-loss durability remains unclaimed.
  Ordinary child startup also verifies activation's first-record pin, even if
  the child was maliciously repacked with internally consistent hashes. Chain
  depth limits include reserved children. Unexpected maintenance exceptions
  return AUDIT_ROLLOVER_TRANSITION_UNKNOWN; durable anchors remain recoverable.
  A terminal seal event without `meta.sealed`, or a child BEGIN without matching
  frozen parent metadata, is corruption; absence cannot downgrade sealed/child
  evidence into an ordinary root journal.
  Rotation checks the successor's full ancestry budget before exporting/sealing;
  it must not create a chain its configured recovery limit cannot open.
  Freeze `journal_limits` (max_bytes/max_inflight) in the seal and propagate
  them to the child and recovered head; maintenance cannot silently reset a
  tighter configured resource guard to defaults.
  Ordinary child reopen clamps resource settings to its frozen parent ceilings;
  explicit tighter settings remain possible, relaxation through defaults does not.
- C206: no application deletion, truncation, VACUUM or automatic reclamation in
  C2a. Retain original DB evidence, archives, reservations and anchors; normal
  SQLite checkpoint/close may merge and retire WAL files. Source accounting hashes
  and market250ms/FULL guards are unchanged. C2b must add certified hold/release
  evidence and full sealed-tail archival/restoration before any reclamation.
  Existing per-segment analysis remains scoped; sustained end-to-end readiness
  additionally requires maintenance/rebinding and cross-segment analytics gates.

Acceptance: SPEC -> focused RED -> implementation -> GREEN/full regression,
actual process death around reservation/seal/open/activation, unresolved work,
seal transaction rollback/unknown commit, immutable reopen/parent receipts,
multiple rotations/restart, corrupt/missing/empty successor and archive pins,
own-path/lease guards. Rollback is omission of the opt-in rollover calls;
sealed evidence is never unsealed as a rollback. Full Harness/staged review,
commit and push required. No current trading/PnL improvement claim.

### AUD-C2a verification

27 missing-module RED first,27 GREEN after implementation and a fixture repair
that isolates metadata corruption from the separate trigger-presence guard.
Three additional RED cases reproduced ordinary-open receipt bypass, reserved
depth bypass and exception misclassification; fixed to30 GREEN. Metadata-removal
and rotation-budget RED reproductions were fixed to33 GREEN /662 full GREEN.
Two resource-limit RED cases found default reset on rotation/ordinary reopen;
fixed to35 GREEN /664 full GREEN. Malformed parent JSON then reproduced an
uncaught parser error; fixed before final36 focused GREEN (20.898s),665 full
app GREEN (91.594s),14 root specification +13 Truth Harness GREEN.
Actual process death covers four transition boundaries, including committed
seal before child initialization and child BEGIN before activation receipt.
Original evidence, failure reasons and unknown prior runs remain visible.
Full Harness remains FAIL TH-11 for the historical replay source hash; no
market/financial qualification follows from these maintenance-only tests.
No network/keys/orders/legacy-worker changes or runtime artifacts in the commit.
Next: full sealed-tail backup/restoration and certified holds before C2b
reclamation; live maintenance/rebinding and cross-segment analytics remain gates.
