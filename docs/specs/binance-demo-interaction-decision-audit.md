# Binance Demo: полный аудит взаимодействий и решений

Дата: 2026-10-11, обновлено 2026-10-26. Статус: **AUD-A/B1/B2 implemented;
80 audit focused / 575 app GREEN; full interaction audit incomplete**.
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
