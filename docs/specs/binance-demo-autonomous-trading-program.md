# Binance Demo: программа автономной торговли по дневному PnL

Дата: 2026-10-10. Status: **specs complete; P0-A tests written / expected RED; runtime not implemented**.
Owner: repository maintainer. Objective contract: `daily_net_equity_pnl_v1`.
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

Эта поставка — спецификации, реестр и порядок реализации. Вызовы API,
размещение ордеров, запуск фоновых служб, новые модели и rollout здесь
не выполняются. Фактический статус этапа меняется вместе с кодом, тестами
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

Первый test-only шаг P0-A: [64 offline unit-теста](../../apps/binance_demo_bot/tests/README.md)
написаны после фиксации financial API; verified RED (64 missing-interface
errors, exit code 1). Реализация финансового модуля отсутствует; assertions
ещё не проходили. P0 integration и P1–P7 tests остаются planned. Торговое
enablement не выполнялось.

## Shared interfaces and ownership

Планируемые интерфейсы, а не существующие реализованные классы:
`BinanceDemoAdapter`, `UniverseService`, `MarketDataService`, `DecisionIntent`,
`RiskManager`, `OrderManager`, `AccountReconciler`, `DailyPerformanceService`,
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
