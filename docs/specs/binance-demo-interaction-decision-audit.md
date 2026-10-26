# Binance Demo: полный аудит взаимодействий и решений

Дата: 2026-10-11, обновлено 2026-10-26. Статус: **AUD-A implemented;
27 focused / 522 app GREEN; full interaction audit incomplete**.
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
