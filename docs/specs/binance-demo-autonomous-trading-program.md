# Binance Demo: программа автономной торговли по дневному PnL

Дата: 2026-10-27. Status: **P0-A/B + P1-A/B + P2-A/B2 + AUD-A/B1/B2/B3/C1 foundations / 629 GREEN; all-market latency FAIL; trading runtime pending**.
Owner: repository maintainer. Objective contract: `daily_net_equity_pnl_v1`.
Полный [аудит Binance и решений](binance-demo-interaction-decision-audit.md)
обязателен до автономного исполнения/learning. AUD-A: durable opt-in REST attempts
и explicit causal decisions implemented; AUD-B1 пишет WSS application calls/raw
messages и causal snapshot links. AUD-B2 добавляет coordinator/supervisor
decisions и frozen-prefix processing coverage; capture coverage остаётся UNKNOWN.
AUD-B3 фиксирует parsed PING/PONG/CLOSE, automatic responses и отдельные
локальные socket-write outcomes; это не remote ACK или полный wire capture.
AUD-C1 реализует verified immutable gzip exports зафиксированного префикса,
manifest pins и preserve-all inventory;38 новых /629 app GREEN. Экспорт не
удаляет source/archives; certified rollover/reclamation/restart anchors и future
policy/OMS decisions ещё не завершены. Полный audit не объявляется готовым.
Основание: [дорожная карта](../roadmaps/binance-demo-daily-pnl-roadmap.md)
и подтверждённая пользователем цель — чистый дневной PnL в USDT после комиссий.
TH-01..TH-12 из [Truth Harness](truth-harness.md) обязательны.
Binding user constraint: [самостоятельное приложение](binance-demo-application-isolation.md)
в `apps/binance_demo_bot`, со своим package/env/state/processes. Текущий бот
не расширяется и не используется как runtime dependency.

## Problem

Текущие main/agent BUY и SELL создают сигналы и виртуальные JSON-позиции.
В репозитории нет внедрённого демо-брокера с подтверждёнными fills, денежным
учётом и разрешённым обновлением политики по дневному финансовому результату.
Разовый успешный account GET проверяет авторизацию чтения, а не завершает
интеграцию, проверку TRADE-права или доказательство заработка.

## Objective fit

Новый демо-контур оптимизирует средний дневной net account PnL при неизменном
капитале и зарегистрированных ограничениях риска. Early capture, recall,
precision и удержание тренда объясняют результат, но не заменяют деньги.
Требование прибыли каждый календарный день не является обещанием системы.

Новый контракт применяется только к явно помеченной фазе Binance Spot Demo.
Существующие [Scout](../../SCOUT_OPTIMIZATION_SPEC.md),
[mission learning](mission-aligned-learning-cycle.md) и WATCH-релизы сохраняют
свою область и прежние результаты. Их `PASS`, proxy и виртуальные сделки
не разрешают новый финансовый релиз. Установка этих спецификаций не меняет
ни старую runtime-политику, ни её текущие метрики.

## Scope

Самостоятельный Spot Demo app, long-only, собственные средства, одна каноническая
книга выделенного счёта и один свой OMS. Main/agent здесь — только новые app-local
policy plugins, не текущие процессы. Публичные и приватные источники имеют явную среду. Все доступные
аккаунту активы наблюдаются; покупка ограничивается проверенной торговой
популяцией, бюджетом и качеством данных. Реальные средства, margin, futures,
withdraw/transfer, автоматическое повышение риска и произвольное изменение
кода learner в этот scope не входят.

Реализовано отдельное offline financial core с own package/env/isolation kernel,
P1-A/B read-only account evidence и P2-A REST / P2-B1 stream/L2 foundations. Публичные Demo
API проверены; authenticated integration, размещение ордеров, фоновый торговый
сервис, новые модели и rollout ещё не выполнены. Фактический статус этапа меняется вместе с кодом, тестами
и source-bound evidence, а не по наличию документа.

## Priority and dependency gates

| Приоритет | Каноническая спецификация | Зависимость перед enablement | Результат реализации |
|---|---|---|---|
| Boundary / prerequisite | [Изоляция приложения](binance-demo-application-isolation.md) | Нет | Own package/env/runtime; без изменений текущего приложения |
| P0 / фаза 0 | [Финансовый контракт](binance-demo-phase0-financial-contract.md) | Boundary | Тестируемая бухгалтерия, risk contract и формат evidence |
| P1 / фаза 1 | [Account adapter](binance-demo-phase1-account-adapter.md) | P0 идентификаторы/границы | Demo read/auth/capability, без execution enablement |
| P2 / фаза 2 | [Universe и market data](binance-demo-phase2-universe-market-data.md) | P1 права/снимок аккаунта | Полная мониторинговая популяция и causal snapshots |
| P3 / фаза 3 | [OMS и риск](binance-demo-phase3-order-management-risk.md) | P0–P2 и технические тесты | Контролируемый защищённый BUY/SELL и reconcile |
| P4 / фаза 4 | [Базовая политика](binance-demo-phase4-baseline-policy.md) | P3; historical/forward eligibility | Ограниченная автоматическая демо-политика |
| P5 / фаза 5 | [Outcomes и dataset](binance-demo-phase5-outcomes-training-dataset.md) | P0–P3 contracts | Проверенный feedback и дневная аналитика |
| P6 / фаза 6 | [Обучение и проверка гипотез](binance-demo-phase6-profit-hypothesis-evaluation.md) | P2/P5 + quality/power gates | Независимый financial verdict, без прямого deployment |
| P7 / фаза 7 | [Автономная promotion](binance-demo-phase7-autonomous-promotion.md) | P3/P6 + новые cohorts | Ограниченные обновления и проверяемый rollback |

P5 ledger/outcome capture строится уже с P0–P3: первая демо-сделка не может
предшествовать журналу исполнения и учёту. P1/P2 и сбор данных могут идти
параллельно после фиксации интерфейсов. P6 можно разрабатывать offline;
отсутствие forward evidence не разрешает P4/P7 торговое enablement.
Первый вертикальный milestone: snapshot → intent → actual fill → active
protection → actual SELL → reconcile → объективный PnL одной демо-сделки.

P0-A/B: [110 tests GREEN](../../apps/binance_demo_bot/tests/README.md) в собственной
среде после recorded RED. Financial assertions выполнены; monetary/valuation
core и начальные isolation/startup guards реализованы. P0-B WAL journal,
atomic outbox/replay/restart и immutable report revisions реализованы.
P1-A REST: 56 focused tests после RED; безопасные account/orders/fill reads,
clock/signing/rate guards и реальный public Demo ping/time реализованы.
P1-B: 69 new tests после RED; private reader, устойчивые события/cursors и
diagnostic comparison с financial journal реализованы. Authenticated private
smoke, exclusive ownership и certified as-of reconcile/anchors pending.
P2-A: 57 новых тестов после RED, полный registered-category каталог,
immutable receipt-as-of snapshots, закрытые 1m features и whole-inventory
midpoint marks. Публичный smoke успешен; полный ручной прогон 173.67s не
подтверждает непрерывный real-time SLA. P2-B1: 46 новых тестов после RED,
потоки/shards/reconnect/atomic L2; bounded sample BTC/ETH/SOL проверен, sustained
all-market readiness не установлена. P2-B2: 36 новых тестов после RED, refresh,
closed-candle streams, atomic store/FIFO, coordinator и честные health counters.
Всего 374 GREEN; полные сетевые прогоны дали FAIL по WS_PROCESSING_LAG, 0/6
подключённых shards при 1375 активных парах. Следующий приоритет — устранить
полномасштабные задержки P2-B2, затем P2-C certified valuation/authenticated integration;
order sending disabled. Готовность заработать не доказана.

P2-B2 contention repair: ещё 26 тестов, всего 400 GREEN; per-thread WAL reads,
prepared atomic writes, indexed candle guards и освобождение reader connections.
Полные прогоны всё ещё FAIL: последний 46.23s достигал 4/6 подключённых shards,
финал 0/6, закрытая история 14/1375. Измерены commit-паузы до 400.63ms.
Следующий приоритет — storage/checkpoint latency с FULL и прежним 250ms,
обновление SQLite перед конкурентным checkpoint, затем P2-C. Не обходить gate.

P2B2-07: закреплён APSW 3.53.4.0 / SQLite 3.53.4 только для market store,
добавлен отдельный PASSIVE checkpoint с восстановлением прежней политики,
WAL pressure и failure/busy diagnostics. 22 новых теста после RED, всего
422 GREEN. Financial journal/fingerprints и текущее приложение не изменены.
Непрерывная полномасштабная доступность остаётся FAIL; фазу P2 не закрывать.

P2B2-08: native cache/CPU attribution, bounded multi-value INSERT, unlocked
standalone result read и group-local dependency availability cache. 12 новых
тестов после RED, всего 434 GREEN. Исправлены pinned return и bootstrap трёх
закрытых свечей при наличии текущей открытой. Primary receipts/hashes, atomicity, FULL и
250ms сохранены. Сетевые диагностические прогоны сохраняют отдельный verdict;
full P2/account/financial qualification и торговля ещё не завершены.

P2B2-09: end-to-end steady-state timing до durable commit, строгий post-commit
guard250ms и сохранение отвергнутых пакетов. 17 новых тестов после RED,
451 GREEN. Отказ хранения виден, соединение закрывается; counts не дублируются.
Evidence/invalidation/incidents отказа теперь сохраняются одной атомарной
FULL-транзакцией; ошибка любой строки откатывает всю группу отказа.
Последний public trial1791737943969095300:52.9277s,0/6 connected,
quotes0/1375,history8/1375 — FAIL. Порог250ms сохраняется.
Startup latency не измерена этим slice. Торговля остаётся выключенной;
зелёные тесты не подтверждают full-market SLA или рост дневного PnL.

P2B2-10: одна readonly WAL-транзакция на подготовку группы вместо отдельных
BEGIN/COMMIT для каждого lookup; восемь записей с одной зависимостью9->1.
Срез освобождается до writer handoff; caller pins/overlapping contexts сохраняют
его до последнего read lease.10 новых тестов после RED,461 app GREEN.
FULL/250ms и causal envelope/guard сохраняются; full-market gate остаётся FAIL
до отдельной свежей сетевой проверки. Финансовые fingerprints не менялись.
Свежий public trial1791740709548236200:47.9992632s,1/6 connected,
quotes21/1375,history10/1375 — FAIL. Следующий gate — owned TLS context reuse
с проверкой TLS guards, затем повтор полного потока; ORDER_SEND выключен.

P2B2-11: owned TLS context reuse до stream startup и при reconnect вместо
повторной Windows trust-store загрузки. Unsafe/mutated contexts отклоняются;
TLS verification/hostname/minTLS1.2 и Demo endpoint сохранены.10 новых тестов
после RED,471 GREEN. Это hot-path correctness, а не full-market или PnL pass.
Public trial1791741563868852000:startup6/6, после tick0/6; финал48.0524598s,
0/6 streams,REST quotes1375/1375,history14/1375 — FAIL. Только1 market TLS
build; serialization/scheduler/writer остаются отдельным gate.

P2B2-12: bounded common market writer, coalesced FULL commit и durable ACK,
FIFO/causal receipts/guards, isolation fallback, explicit commit uncertainty,
queue pressure/evidence и owned shutdown.24 новых теста после RED,495 GREEN.
Финансовые fingerprints не менялись; полный аудит interactions/decisions
следующий и обязателен до execution/learning. Full-market gate отдельно.
Public trial1791744230720272600:47.555206s,0/6 streams,REST1375/1375,
history18/1375 — FAIL; writer1914 groups/1166 commit ACKs,zero writer failures.
Native/serialization/read/ACK latency остаётся gate; orders disabled.

## Shared interfaces and ownership

`BinanceDemoAdapter` реализован как P1-A read-only REST foundation.
`UniverseService` и `MarketDataService` реализованы в пределах P2-A;
`AccountReconciler` реализован для P1-B diagnostics, не certified readiness.
Остальные интерфейсы пока планируемые: `DecisionIntent`,
`RiskManager`, `OrderManager`, `DailyPerformanceService`,
`ProfitEvaluator`, `DemoReleaseController`.

Канонический `EventEnvelopeV1`:

- `schema_version=1`, `application_id=binance_demo_bot`, `environment=BINANCE_SPOT_DEMO`,
  `objective_contract_id=daily_net_equity_pnl_v1`;
- `event_id`, `event_type`, `causation_id`, `account_episode_id`,
  `portfolio_scope_id`; `arm_id` при разделении политик;
- `exchange_event_time_ms` nullable, `received_at_ms` фактический UTC clock,
  `received_monotonic_ns`, `process_instance_id`, `recorded_at_ms`;
- для решений/исполнений: `decision_id`, `policy_version`, `model_id`,
  `source_sha256`, `objective_contract_id`, `risk_contract_id`;
- IDs заявки/сделки, `state_version`, worker lease epoch и payload определяются
  фазами 0/1/3. Отсутствующий exchange clock не заполняется придуманным временем.

Схемы и денежные числа версионируются; Decimal сериализуется строками.
Внешний UID аккаунта, ключи, секреты, подписи запросов и заголовки не входят
в публичные IDs, source manifests, сообщения, fixtures или Git.
Raw приватные ответы и runtime ledger имеют локальный ограниченный доступ.
Trainer читает snapshot; evaluator не обучает; governor не меняет evidence;
OMS принимает только разрешённые intents. LLM не владеет ключом и не пишет OMS.

## Primary metrics

Главная: mean daily net equity PnL в USDT и одинаковый fixed-budget daily return.
Успех обучения требует финансового gate фазы 6/7, а не улучшения surrogate.
Обязательные ограничения: дневной убыток, drawdown, exposure/concentration,
незащищённая quantity/time, корректность reconcile и причинные data clocks.
Каждая доля содержит `n/N`; missing и нулевой знаменатель не превращаются в 0%.
Economic PnL отдельно вычитает AI/инфраструктуру; торговые комиссии повторно
из equity не вычитаются. Baselines: frozen rule policy и hold одинаковой initial
inventory; USDT/no-trade только для сверенного USDT-only бюджета, BTC hold
отдельно как market-risk diagnostic.

## Acceptance criteria

- Все восемь specs зарегистрированы, имеют зависимости, границы, тестовые
  сценарии, verification gate и безопасный rollback.
- Для каждой фазы отдельно видны `spec_status`, `implementation_status`,
  `technical_readiness`, `financial_evidence_status`, `execution_enabled`.
  Документ `PLANNED` не способен дать `execution_enabled=true`.
- Enablement требует точного совпадения environment/objective/episode/scope/
  risk/source/model и cohort manifest. Старый WATCH ticket отклоняется.
- Неизвестные поля/версия/права/evidence блокируют зависимый путь с причиной
  и планом восстановления; missing evidence не выдаёт прибыльный verdict.
- Новый runtime не открывает Binance exposure при записи виртуального BUY.
  Удаление JSON-позиции не является закрытием биржевого актива.

## Risk and trade-offs

Строгая сверка может временно запрещать входы; объём universe ограничивает
частоту глубокого анализа. Финансовые интервалы могут оставаться недостаточно
точными. Увеличение числа сделок ради выборки запрещено. Демо-fill evidence
не доказывает возможность получить те же fills или доходность на real exchange.
Control plane реализуется внутри нового package с собственным lifecycle.
Старые решения — read-only design references; runtime imports существующих
модулей/состояния и перенос старого approval запрещены. Source hashes,
dependencies, fixtures и frozen contracts принадлежат только новому app.

## Backtest / verification gate

Каждая торговая гипотеза: maximum available исторический портфельный replay,
идентичная причинная популяция, fees/fills/cash/valuation обеих политик,
независимый audit, затем новый зарегистрированный forward и canary.
Разработка account/данных/учёта не требует обещать торговый edge; её gate —
технические критерии соответствующей фазы. Controlled integration order test
не является разрешением автоматического торгового эксперимента.

Планируемые сквозные тесты: `PROGRAM-01` legacy ticket rejected;
`PROGRAM-02` phase missing disables successor; `PROGRAM-03` virtual BUY does
not submit; `PROGRAM-04` one OMS owns all intents; `PROGRAM-05` no secret in
exceptions/manifests; `PROGRAM-06` actual/shadow results never mix;
`PROGRAM-07` money and exposure survive restart/rollback.
Обязательный порядок пользователя: **spec → focused tests → RED → code →
GREEN → refactor/regressions → diff review → staged Truth Harness → commit → push**.
Тесты пишутся до реализации по acceptance criteria; RED должен подтвердить
отсутствующее/неверное поведение, а не сбой среды. Always-green mocks,
skip/xfail и ослабление assertions не заменяют этот этап. Для preserving refactor
сначала contract coverage и явное обоснование GREEN без искусственного RED.
Правило сохранено в [AGENTS.md нового приложения](../../apps/binance_demo_bot/AGENTS.md)
и [процессе разработки](spec-first-workflow.md). Эта документационная поставка проверяет
ссылки/регистрацию/согласованность и существующие spec/harness regressions;
она не заявляет выполненными будущие финансовые или брокерские тесты.

## Rollback switch

До реализации rollback — не включать новые компоненты; существующий бот
продолжает работать в прежней области. После появления actual exposure
rollback запрещает новые входы/updates, сохраняет ledger и exchange-side
protection, выполняет reconcile и только затем разрешённую migration/close.
Переключение версии никогда не стирает остатки активов и pending orders.
