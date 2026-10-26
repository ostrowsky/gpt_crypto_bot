# Binance Demo: полный аудит взаимодействий и решений

Дата: 2026-10-11. Статус: **SPECIFIED — реализация и тесты ещё не завершены**.
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
