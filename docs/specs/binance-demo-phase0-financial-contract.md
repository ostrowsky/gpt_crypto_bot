# Binance Demo — фаза 0: финансовый контракт

Дата: 2026-10-10. Статус: **P0-A/B LOCAL_CORE_AND_JOURNAL_IMPLEMENTED / 110 GREEN; exchange anchors pending**. Владелец: сопровождающий репозитория. Приоритет:
P0, до подключения торговли. **Реализовано offline financial core; полный контракт фазы ещё не завершён.** Документ не
подключает аккаунт, не размещает заявки и не меняет действующие настройки.

Программа: [автономная демо-торговля](binance-demo-autonomous-trading-program.md). Источник приоритетов:
[дорожная карта](../roadmaps/binance-demo-daily-pnl-roadmap.md).
Обязательная граница: [изоляция нового приложения](binance-demo-application-isolation.md).

## Problem

Виртуальные BUY/SELL, суммы продаж и сумма доходностей сделок не подтверждают финансовый результат счёта.
Необходимо учитывать открытые позиции, locked balance, фактические исполнения, комиссии, внешние потоки и
сбросы демо. Удержание убытка, перестановка денег и закрытие только выигрышей не должны увеличивать целевую
метрику без роста стоимости всего портфеля.

## Objective fit

Подтверждённая пользователем цель: **чистый дневной PnL в USDT после комиссий**. Идентификатор нового
контракта: `daily_net_equity_pnl_v1`. Оптимизируется средний дневной результат при одинаковом капитале и
фиксированном допустимом риске, а не обязательная прибыль каждого дня. WAIT — допустимое решение. Результат
Binance Demo виртуальный; он не является реальной выручкой.

Существующие [mission-aligned-learning-cycle](mission-aligned-learning-cycle.md) и
`SCOUT_OPTIMIZATION_SPEC.md` описывают прежний контур. Их early capture, coverage и precision сохраняются
как исторические/диагностические показатели; прошлые результаты не становятся доказательством прибыли по
новому контракту. Финансовый runtime создаётся как независимое приложение `apps/binance_demo_bot`,
с явно выбранным `objective_contract_id`; это не миграция действующего бота или его позиций.

## Scope

Первый runtime: эксклюзивный Spot Demo аккаунт нового приложения, без плеча/шортов, один учитываемый
портфель. Другой API key того же аккаунта не создаёт изоляцию. Эксклюзивность account identity закреплена
в собственном account manifest; неподтверждённое ownership блокирует торговое enablement. Мониторинг всех
активов не означает торговлю всеми. По аккаунту оцениваются **все** ненулевые free+locked assets, включая
dust, fee assets и неторгуемые seed assets. Seed inventory — opening valuation, не автоматическая торговая
позиция и не разрешение продать старые активы/сопровождать чужие orders.

Планируемые компоненты: `FinancialLedger`, `AccountReconciler`, `ValuationService`,
`DailyPerformanceService`; transactional SQLite WAL с одним владельцем записи, атомарным commit
события/проекции/outbox и immutable журналом. Binance balance/fills — источник подтверждения, ledger —
воспроизводимая локальная проекция только нового приложения. Собственные package, `.venv`, `.env` и
`.runtime` расположены под `apps/binance_demo_bot`; SQLite ledger, cursors, reports и outbox принадлежат
только этому namespace. Нет чтения/импорта старых `files/*`, позиций, моделей, runtime, `.env`, sharing loop
или импорта их fills. Ключи не копируются автоматически. Схема событий общая только для фаз нового приложения.

### Идентичность и типизированные события

`EventEnvelope` для каждого события:

```text
schema_version: integer = 1
application_id: binance_demo_bot
environment: BINANCE_SPOT_DEMO
objective_contract_id: daily_net_equity_pnl_v1
account_episode_id, portfolio_scope_id, event_id, event_type: string
causation_id: string | null
exchange_event_time_ms: UTC integer | null
received_at_ms: actual UTC integer
received_monotonic_ns: integer
process_instance_id: string
recorded_at_ms: UTC integer
decision_id, arm_id, model_id, policy_version, source_sha256,
risk_contract_id: string | null, according to event type
```

`received_at_ms` фиксируется при фактическом получении, не выводится из свечи или времени записи. Monotonic
clock измеряет задержку одного процесса; при рестарте меняется `process_instance_id`. Exchange timestamp —
время события, а не доказательство доступности модели. Финансовый replay может использовать поздно
полученный fill для исправления отчёта; decision replay этого не может. Все финансовые
amounts/quantity/prices передаются decimal strings с asset/unit; binary float и молчаливое преобразование
валют запрещены. Новый неизвестный `schema_version` отклоняется с сохранением quarantine evidence.

| Тип payload | Обязательные поля и смысл |
|---|---|
| `AccountEpisodeOpened` | account identity без секретов, episode start UTC, opening reconciliation/valuation IDs, seed lots, objective/risk IDs, starting budget, scope manifest/hash. |
| `AccountBalanceObserved` | snapshot ID, account/asset, free, locked, источник, interval/as-of certainty, reconciliation anchor; наблюдение не является прибылью или переводом. |
| `ExchangeFillRecorded` | symbol, exchange order/trade IDs, client order ID, side, gross base quantity, gross quote quantity, price, commission quantity/asset, exchange clock, arm, raw payload hash. |
| `ExternalFlowRecorded` | flow ID, deposit/withdrawal/internal allocation classification, effective UTC, asset/quantity, sign, contemporaneous USDT valuation ID, provenance. |
| `AssetFeeRecorded` | fill/fee source ID, asset/quantity, expense USDT valuation ID; одна комиссия учитывается ровно один раз, отдельно от gross fill. |
| `ValuationSnapshotRecorded` | boundary UTC, asset quantities, conversion routes/hash, leg event/receipt clocks, mark prices, freshness flags, equity and liquidation diagnostic. |
| `ReconciliationRecorded` | interval/cursors, fills/flows coverage, per-asset balance residuals, open orders/reservations consistency, unresolved causes, status. |
| `DailyPerformanceFinalized` | day boundaries, revision/source hash, snapshot/episode/scope IDs, financial metrics, completeness, reasons, policy versions. |
| `AccountEpisodeClosed` | reset/closure reason, final certified snapshot or explicit gap, new episode linkage; reset difference не классифицируется как прибыль. |

Unique fill key: `(application_id, environment, account_episode_id, symbol, exchange_trade_id)`. REST backfill и WebSocket
delivery с тем же ключом не создают второе исполнение. Изменённый payload по уже принятому ключу — конфликт
для reconciliation. Комиссия из fill порождает один `AssetFeeRecorded` по уникальному fill/commission key;
две deliveries не создают два fee debits. ACK, intent, trigger, cancel и order reservation не являются
fill/fee/PnL. Reservation переносит free в locked без изменения total quantity/equity.

`AccountEpisode` начинается с подтверждённого seed состояния; начальный баланс не прибыль. Reset, смена
аккаунта/среды или необъяснённый разрыв баланса закрывает эпизод и запрещает склейку equity через разрыв.
Факт reset требует evidence; необъяснённую потерю нельзя списать на reset. Итог прежнего эпизода
сохраняется.

`portfolio_scope_id` неизменно задаёт account identity, `arm_id`, asset/lot ownership и budget. Несколько
ключей одного аккаунта не создают разные счета. Первые arm сравнения отключены. Позднее isolated OMS
subportfolios требуют непересекающихся lots/reservations/orders, зарегистрированных allocation flows и
тождества суммы arms всему account; межполитическое неттирование запрещено. PnL arms не складывается с
account PnL повторно. Неизвестные/manual/other-app orders и fills отражаются в сверке всего аккаунта, но
остаются `ownership=UNATTRIBUTED`: не получают app order/position ownership и не отменяются/продаются
автоматически. Их появление нарушает эксклюзивность, блокирует entries/promotion до разрешения incident;
app-owned exchange protection сохраняется. Нельзя объявить внешнюю сделку своим финансовым улучшением.

### Inventory и комиссии

Активы учитываются один раз как `free + locked`; позиции/ордера не добавляют их стоимость повторно. FIFO
lots имеют lot/source-fill/scope IDs, remaining quantity, USDT carrying cost и acquisition UTC. Метод FIFO
фиксируется на эпизод и не выбирается после результата. Initial lots получают basis opening mark: прибыль до
начала эпизода не приписывается боту. Lot ownership различает `SEED`, `APPLICATION`, `UNATTRIBUTED`;
opening SEED не проходит автоматически в application-managed exit inventory.
FIFO cost attribution — бухгалтерская проекция, не источник разрешения OMS на продажу; execution ownership
и доступный остаток application fills проверяются отдельно даже для того же fungible base asset.

Gross fill сначала создаёт asset movements; продажа списывает FIFO lots, partial fill меняет только
исполненное количество. Для non-USDT quote сохраняются её USDT conversion и FIFO disposal attribution;
поступивший asset получает transaction-price USDT basis. Fee debit отдельно списывает fee asset: fee expense
равен contemporaneous USDT value, disposal PnL fee asset равен этому value минус FIFO carrying cost. Поэтому
fee в base/quote/BNB, включая подорожание BNB, корректно сверяется с equity. Фактическая комиссия в ledger
проверяется против payload; повторная нормированная комиссия не списывается.

Bridge проверяется независимо: `DailyPnL = gross realized FIFO PnL - fee expenses + delta(unrealized FIFO
PnL)` при согласованных flow/basis reclassifications. Неизвестный basis помечает attribution incomplete;
значение canonical equity PnL разрешено только при полных balance/flow/valuation данных. Fees и realized PnL
раскрываются отдельно. Negative inventory, oversell и ownership conflict — accounting ERROR. Dust не
удаляется из equity.

## Primary metrics

Для актива `a`: `Q_a = free_a + locked_a`, `M_USDT = 1`. `Equity_USDT(t) = sum_a(Q_a(t) * M_a_USDT(t))`.
Другие stablecoins не оцениваются как 1 без рыночной конверсии. Routes зарегистрированы заранее, используют
demo prices одного venue без циклов; каждый leg содержит источник/время/side. Основная оценка — fresh
bid/ask mid; liquidation value по bid/depth и ожидаемым выходным расходам — отдельный diagnostic. Нельзя
оценить отсутствующую цену в ноль, выбрать маршрут по выгодному PnL или интерполировать boundary будущими
ценами.

```text
NetExternalFlow_d = sum_inflows(value_USDT_at_flow) - sum_outflows(value_USDT_at_flow)
DailyNetPnL_d = Equity_end_d - Equity_start_d - NetExternalFlow_d
EconomicPnL_d = DailyNetPnL_d - non_account_AI_cost_d - non_account_infra_cost_d
DailyReturn_d = DailyNetPnL_d / Equity_start_d   # only flow-free, positive equity
MeanDailyNetPnL = sum(DailyNetPnL_d for complete days) / N_complete_days
ProfitableDays = count(DailyNetPnL_d > 0) / N_complete_days
```

Actual fees и фактическая цена исполнения уже влияют на equity; расчётный slippage/spread/fee второй раз из
DailyNetPnL не вычитается. Sales receipts и turnover показываются отдельными sums, не входят в reward.
AI/infra costs начисляются по заранее утверждённому usage/allocation правилу: расходы, уже списанные со
счёта, не вычитаются снова. Нет billing evidence — EconomicPnL pending с причиной, а не фиктивные нулевые
расходы.

При внутридневном flow сохраняются equity непосредственно до/после flow с одинаковыми marks. Публикуется
`TWR = product(E_before_next / E_after_previous)-1` по flow-separated segments; отсутствие таких snapshots
делает TWR pending. `DailyReturn` простым делением в таком дне не публикуется как сопоставимый return.
Portfolio comparisons используют одинаковый заранее выделенный budget и flow-free episodes; day PnL flows
остаётся видимым отдельно. Дополнительное финансирование проигрывающего arm не допускается в сравнительную
когорту. Flows считаются относительно scope: перевод между free/locked или arms не внешний flow всего
account; связанные меж-arm allocations дают сумму ноль. Free USDT не равно полной equity: прочие seed assets
также входят в счёт. Baseline начинает с того же inventory; USDT-only/no-trade возможен только для
подтверждённого USDT scope либо после отдельно согласованной нейтрализации.

Каждый ratio хранит numerator/denominator; `N=0` или start equity <=0 даёт null с причиной. No-trade дни
включаются: overnight inventory MTM и комиссии могут дать ненулевой PnL без новых BUY/SELL. Ноль корректен
только при полной сверке и неизменном USDT-only состоянии без flows/расходов.

### Сутки, boundary и completeness

День — `[local 00:00, next local 00:00)` в IANA `Europe/Budapest`; UTC boundaries хранятся явно, дни DST
имеют 23/25 часов. Не прибавлять фиксированные 24h UTC. Fill ровно на правой границе относится к следующим
суткам. Closing state предыдущего дня равен opening state следующего до таких fills.

Historical boundary реконструируется по подтверждённым fills/flows и complete reconciliation cursors;
сегодняшний balance/current price не заменяет midnight state. Для timestamp-capable источника: mark event
time <= boundary, age <=**5s** как proposed default; позднее получение возможно только для учёта. Для REST/
bookTicker без event time поле остаётся null. До оценки регистрируется route
`time_basis=EXCHANGE_EVENT|OBSERVED_RESPONSE`. OBSERVED_RESPONSE допустим только по сохранённому completed
response interval и реальному received_at <= boundary, с age <=5s по observation clock и проверенным
source-health; возраст биржевого tick остаётся неизвестным и явно раскрывается. Receipt не переименовывается
в exchange time; нынешний REST response не восстанавливает прошлую границу. Обе политики используют
одинаковый protocol; выбрать clock после PnL нельзя. Freshness/precision tolerances и finalization timeout
фиксируются в versioned registry до запуска; quantity tolerances выводятся из точности биржи.

`DailyPerformanceSnapshot` хранит episode/scope/day/schema IDs, UTC boundaries,
snapshot/valuation/reconciliation IDs, opening/closing equity, flows, fees, realized/unrealized/net/economic
PnL, TWR, available risk/equity extrema, `status`, `reason_codes`, version hashes, numerator/denominator и
revision. Статусы: `PENDING` (сутки/сверка/данные не завершены), `COMPLETE` (доказаны все компоненты),
`ERROR` (конфликт/неустранимый gap/нарушение тождества). Экономические затраты и attribution имеют
собственный completeness, не маскируются net PASS.

Net COMPLETE требует coverage всех ненулевых assets, fresh boundary marks, подтверждённых flow clocks и всех
fills/fees до границы, сходящихся quantity/balance и equity/flow bridge в зарегистрированном tolerance.
FIFO attribution получает COMPLETE только после независимой сверки realized/unrealized/fee bridge.
Первая/последняя неполная сутка и сутки reset не входят в primary full-day sample; причины раскрываются.
Late data создают immutable corrected revision с audit trail и переоценкой выводов; старый отчёт не
переписывается незаметно. После восстановления весь PnL периода downtime учитывается, включая убытки.

Отчёт показывает `N_calendar`, `N_complete`, `N_pending`, `N_error`, partial boundaries, reset episodes,
coverage и список пропущенных суток. Paired comparison использует одинаковые полные даты/контракт/budget с
predeclared gap rules; исключение плохих дней по результату запрещено. Незакрытый gap не разрешает financial
PASS. Primary/actual/demo/shadow/simulation labels не смешиваются.

## Acceptance criteria

### P0-B: durable journal / outbox / corrected reports

Контракт фиксируется до тестов/кода (2026-10-10). Scope — собственная SQLite
проекция fills и immutable daily report inputs, не account adapter/OMS и не
подтверждение фактической прибыли. Public module: `binance_demo_bot.journal`.

Verified: 29 durable integration tests GREEN после missing-module RED,
outbox/provenance/corruption RED и report-retry RED; общая партия 110 GREEN.
Before/after-commit deaths и report after-commit death выполнены реальными
subprocesses. SQLite fixture connections закрываются явно; assertions сохранены.

`JournalStore(paths, *, episode_id, scope_id, opening_lots,
database="ledger.sqlite", fault_hook=None)` использует только
`AppPaths.runtime_path(database)` и соседний app-local writer lease. Один
writer по OS file lock; после process death lock освобождается ОС, чужие PID
не завершаются, чужие lock files не удаляются. Второй owner → `WriterBusy`.
SQLite `journal_mode=WAL`, `synchronous=FULL`, `foreign_keys=ON`, explicit
transactions. Namespace/schema/episode/scope/opening inventory frozen в meta;
reopen с иной identity/неизвестной version → `JournalIntegrityError`, без migration.
Все domain exceptions наследуют ValueError, accounting ошибки — AccountingError.

| Interface | Observable contract |
|---|---|
| `append_fill(envelope, fill, marks)` | APPLIED/DUPLICATE. Envelope: schema_version=1, application_id, environment, objective_contract_id, account_episode_id, portfolio_scope_id, event_id, event_type=ExchangeFillRecorded, exchange_event_time_ms, received_at_ms, received_monotonic_ns, process_instance_id. Clocks integers; exchange clock совпадает с fill; process ID не смешивается при monotonic анализе. recorded_at_ms назначается store при insert. |
| `projection(marks, *, as_of_received_ms=None)` | Decimal balances/managed_quantities/attribution, source_sha256, evidence_kind=LOCAL_PROJECTION, exchange_reconciled=false. Финансовый replay сортирует unique fills по `(exchange_event_time_ms, insertion sequence)`; receipt-cutoff replay сначала исключает ещё не полученные/future-clock fills. |
| `events()` / `receipts()` | Immutable accepted economic events и отдельные deliveries с сохранёнными clocks. Same financial key / equal normalized payload и marks → одна financial event/fee, несколько distinct receipts; exact event_id retry не добавляет receipt. |
| `sqlite_settings()` | Actual connection values `{journal_mode:wal, synchronous:2, foreign_keys:1}` для проверки storage protocol; не declaration из config. |
| `pending_outbox(limit=100)` / `ack_outbox(id)` | Durable local notifications only; no ORDER_SEND. Неacknowledged сообщения после рестарта возвращаются с тем же ID и payload. ACK идемпотентен; unknown ID отвергается. Delivery at-least-once, consumer обязан dedup по ID. |
| `record_daily_report(day, *, opening, closing, flows, reconciliation_status, economic_costs_usdt)` | Вызывает financial daily_performance; inputs immutable, full Budapest day boundaries/episode/scope проверены, future/unended day не финализируется. Добавляет revision, supersedes, source_sha256, content_sha256 и actual recorded UTC. |
| `reports(day)` | Все revisions по порядку, не только выгодные/COMPLETE. `stale` вычисляется относительно текущей source generation. Новое economic event консервативно invalidates старые reports; новый report append, старый не overwritten. Returned provenance остаётся LOCAL_PROJECTION, не exchange certification. |
| `close()` / context manager | Освобождает только свой SQLite handle/file lease, idempotent close. Нет worker/network/process-control side effects. |

Optional envelope fields causation_id/decision_id/arm_id/model_id/policy_version/
source_sha256/risk_contract_id сохраняются как string/null, не отбрасываются.
В frozen meta входит hash собственного accounting code (financial/journal/isolation);
reopen с другим code artifact отвергается до явной versioned migration.

Money записываются только canonical decimal strings (equivalent decimal
formatting не создаёт конфликт). Финансовый identity включает все обязательные
fill поля P0-A и frozen contemporaneous marks; delivery metadata не меняют его.
Пересмотр mark/basis принятого fill требует будущей versioned correction event,
а не тихого overwrite. Duplicate event_id с иной delivery, trade payload или
marks → LedgerConflict, без mutation accepted events/projection/outbox.
Invalid namespace/schema/types/amounts и oversell/replay failure отвергаются
атомарно. Caller обязан сохранить rejected raw evidence/quarantine incident;
полный incident lifecycle/reconciliation block реализуется с adapter/OMS.

Economic event + receipt + materialized balances/managed projection + outbox
notification committed одной transaction. Before-commit exception/process
death оставляет всё прежним; after-commit caller uncertainty безопасна при
retry того же event_id. Report revision + report outbox тоже атомарны.
`fault_hook(stage)` — явный engineering injection `before_commit`/`after_commit`,
по умолчанию None; интеграционные тесты используют реальные subprocess deaths.
Ни hooks, ни fixtures не запускают/останавливают существующий бот.

Exact report retry по canonical inputs + day + текущей source generation
возвращает прежний ID/revision без нового outbox. Money поля snapshots/flows/
economic costs нормализуются; after-commit process death безопасен при retry.
Изменённые inputs или новая financial generation append новую revision.

Events, receipts, meta и reports защищены от UPDATE/DELETE SQL triggers.
На reopen проверяются canonical content hashes, receipt/event associations и
materialized projection против replay. Несоответствие → JournalIntegrityError;
не пытаться обнулить историю/переписать anchors или автоматически исправлять
повреждённый journal. Outbox ACK — отдельное mutable delivery state, payload
не меняется.
Outbox ID/topic/payload/hash также immutable, DELETE запрещён; допустим только
ACK. На reopen hash и association с соответствующим event/report проверяются.
Это локальная integrity check, не защита от администратора, переписавшего все
данные и hashes; external signed anchors ещё не реализованы.

P0-B scenarios (все сначала RED, затем implementation/GREEN):
`DUR-01` persistence/reopen and WAL/FULL;
`DUR-02` REST/WS + exact retry dedup/Decimal equivalence;
`DUR-03` conflicting payload/event ID/marks, atomicity;
`DUR-04` exception and real process death before commit;
`DUR-05` real process death after commit/retry;
`DUR-06` deterministic late-fill FIFO rebuild;
`DUR-07` receipt-cutoff excludes future knowledge across restart;
`DUR-08` outbox repeat delivery/idempotent ACK/reopen;
`DUR-09` immutable corrected losing report/stale favorable revision;
`DUR-10` report atomic failure/day/episode/scope/future validation;
`DUR-11` schema/identity/cache tamper/SQL immutability;
`DUR-12` second writer refused, crashed writer lease recoverable, app path boundary.

Remaining full-P0 gaps: actual account balance/fill/flow cursors and anchors,
episode reset evidence, conversion coverage/precision/flow boundary protocol,
cross-arm budgets, quarantine resolution and independent exchange recompute.
Durable local fills/report storage alone не завершает FIN-01..12 или P0.
Receipt-cutoff ограничивает только доступность journal fills. Историческая
доступность opening seed anchor и переданных marks этим API не доказана:
P1/P2 обязаны поставить PIT-certified anchors/valuation snapshots. Source hash
проекции не заменяет hash текущей valuation или model feature snapshot.

### P0-A: публичный контракт первой партии unit-тестов

Контракт зафиксирован до написания тестов (2026-10-10). Будущий импорт:
`binance_demo_bot.financial` из собственного `apps/binance_demo_bot/src`.
Это offline financial core; SQLite/outbox проверены в P0-B. Account adapter,
OMS и полная изоляция runtime требуют следующих партий. Ни наличие
unit-тестов, ни их будущий GREEN не завершают FIN-01..12 целиком.

API использует обычные mappings и Decimal results, чтобы не связывать тесты
с внутренней структурой классов. Денежные входы — Decimal или decimal strings;
Арифметика использует локальный Decimal Context(prec=50, ROUND_HALF_EVEN),
независимый от настроек/ловушек вызывающего процесса; глобальный context не меняется.
float, bool, NaN/Infinity отвергаются `ValueError`. На входах с quantities,
fees, prices дополнительно проверяется допустимый знак. Error types
`AccountingError` и `LedgerConflict` наследуют `ValueError`.

| Интерфейс | Вход и наблюдаемый результат |
|---|---|
| `decimal_amount(value)` | Конечный Decimal без промежуточного float; signed amounts допустимы. |
| `day_bounds(day, timezone)` | ISO local date и IANA zone → tuple `(start_ms, end_ms)` UTC; DST учитывается. |
| `day_for_timestamp(timestamp_ms, timezone)` | ISO local day события; правая граница принадлежит следующим суткам. |
| `value_portfolio(balances, quotes, routes, *, boundary_ms, time_basis, max_age_ms)` | `balances[asset]={free,locked}`; `quotes[symbol]={base,quote,bid,ask,venue,event_time_ms,received_at_ms,response_started_at_ms,source_healthy}`; `routes[asset]=[(symbol,inverse),...]`. Только `BINANCE_SPOT_DEMO`, детерминированный путь до USDT. Результат `{status,equity_usdt,marks,reason_codes}`. USDT=1; inverse использует reciprocal(mid), не mid reciprocals. Все nonzero assets обязательны; zero assets не требуют quotes. |
| `FinancialLedger(*, episode_id, opening_lots)` | Opening lot `{asset,quantity,basis_usdt,ownership}` с total carrying cost, ownership `SEED`/`APPLICATION`/`UNATTRIBUTED`. Pure in-memory проекция; durable wrapper — P0-B JournalStore. |
| `ledger.apply_fill(fill, marks)` | Fill содержит `application_id`, `environment`, `episode_id`, `symbol`, `trade_id`, `order_id`, `client_order_id`, `base_asset`, `quote_asset`, `side`, `quantity`, `quote_quantity`, `price`, `commission_asset`, `commission_quantity`, `exchange_event_time_ms`, `ownership`. `marks[asset]` — contemporaneous USDT mark. Возвращает `APPLIED`/`DUPLICATE`; меняется только исполненная quantity. |
| `ledger.balances()` / `ledger.managed_quantity(asset)` | Все total balances как `dict[asset,Decimal]`; отдельно net application-owned inventory для разрешения exit. SEED inventory не даёт permission на SELL. |
| `ledger.attribution(marks)` | `{gross_realized_usdt,fee_expenses_usdt,unrealized_usdt}`. Gross realized включает disposal fee asset; fees не списываются дважды. FIFO seed basis — opening mark; financial FIFO не заменяет operational ownership. |
| `daily_performance(opening, closing, flows, *, reconciliation_status, economic_costs_usdt)` | Snapshots `{boundary_ms,equity_usdt,status,episode_id,scope_id}`. Flows `{effective_ms,signed_usdt,status,classification}` с уже подтверждённой contemporaneous valuation; classification `DEPOSIT`, `WITHDRAWAL`, `INTERNAL_ACCOUNT_TRANSFER`. Internal account movement не external flow. Результат `{status,net_pnl_usdt,net_external_flow_usdt,daily_return,economic_pnl_usdt,economic_status,reason_codes}`. |
| `time_weighted_return(segments)` | Хронологические flow-separated `{start_equity_usdt,end_equity_usdt}`; результат `{status,value}`. Missing или nonpositive start → `PENDING`, value null; пустой список тоже PENDING. |
| `reconcile_pnl(*, net_pnl_usdt, gross_realized_usdt, fee_expenses_usdt, unrealized_delta_usdt, tolerance_usdt)` | `{status,residual_usdt}`; residual = net − (realized − fees + unrealized delta). Только abs(residual) <= заранее заданной nonnegative tolerance даёт COMPLETE. |
| `aggregate_latest_daily(calendar_days, reports)` | ISO calendar manifest и reports `{day,revision,status,net_pnl_usdt}`. Последняя revision на каждую дату; результат `{N_calendar,N_complete,N_pending,N_error,positive_days,mean_daily_net_pnl_usdt,positive_day_fraction}`. Fraction `{n,N,value}`, N=0 → value null. Нет report → PENDING; неполные/error дни не становятся нулевыми returns. |

Clock для valuation передаётся явно и не выбирается по результату. `EXCHANGE_EVENT`:
event_time <= boundary и age <= max_age; поздняя receipt допустима только для
бухгалтерии. `OBSERVED_RESPONSE`: completed interval
response_started_at <= received_at <= boundary, observation age <= max_age,
source healthy; неизвестный exchange clock остаётся null. Future/stale/missing
leg даёт PENDING и null equity всего портфеля. Crossed/nonpositive quote,
negative inventory, неверный venue/циклический маршрут дают ERROR и null equity.

Дополнительная FIN-07 граница: новые fills P0-A применяются в неубывающем
exchange event time; одинаковый timestamp сохраняет порядок принятия.
Новый fill с более ранним clock отвергается атомарно и не резервирует dedup key.
Повтор уже принятого fill допустим независимо от последующих clocks.
Полный durable журнал в следующей партии обязан сохранить late event и
пересобрать хронологическую проекцию/corrected revision; in-memory core
сам не выдаёт arrival-order correction за честный FIFO replay.

Ledger rejects чужой application/environment/episode, неверное fill arithmetic
`quantity * price != quote_quantity`, oversell и недостаток fee inventory.
Validation failure атомарна: баланс, FIFO и accepted trade key не меняются.
Dedup key — episode/symbol/trade, не order ID; transport metadata `source` и
`received_at_ms` не меняют economic payload. Другой economic payload под тем
же ключом → LedgerConflict, без mutation. Частичные fills одного order с
разными trade IDs независимы. `UNATTRIBUTED` не получает managed quantity;
полный reconcile/manual-activity gate остаётся интеграционной задачей.

Daily core не повторяет оценку market data: принимает сертифицированные
snapshots/flows и explicit reconciliation status. ERROR component → ERROR;
PENDING component/null equity → PENDING; только COMPLETE позволяет net reward.
Episode/scope mismatch → ERROR; right boundary <= left → ValueError. Flows
учитываются в `[opening.boundary_ms, closing.boundary_ms)`; наличие реального
external flow убирает simple daily_return даже при суммарном net flow=0.
Economic cost null означает economic PENDING независимо от net COMPLETE;
заданный cost — только extra-account AI/infra, finite nonnegative Decimal.
Fees уже содержатся в equity и не вычитаются повторно. Нулевой opening equity
не мешает абсолютному net PnL, но запрещает daily_return.

Aggregator отвергает duplicate `(day,revision)`, даты вне manifest и COMPLETE
report с null PnL. Последняя revision может ухудшить COMPLETE до PENDING/ERROR;
нельзя выбирать последнюю выгодную или только complete revision. Calendar
count включает downtime. Mean/positive fraction используют только COMPLETE;
числитель, знаменатель и все gap counts сохраняются. Старые revisions не
удаляются входным сервисом; этот pure core не реализует хранение/approval.

P0-A milestone: public financial core реализован; original 64 financial tests
и новые edge/isolation/startup tests проходят в own venv (81 GREEN суммарно).
RED до implementation и отдельный RED для позднего fill записаны в test evidence.
P0-B journal/replay/outbox/revisions реализованы; cross-arm sum, reset evidence,
actual account ownership/anchors и full process isolation остаются PLANNED.

| ID | Планируемый автоматический сценарий и ожидаемый результат |
|---|---|
| FIN-01 | BUY partial → OCO reserve → partial SELL: free+locked identity, exact fills, FIFO и equity bridge сходятся. |
| FIN-02 | Base, quote, BNB fee, duplicate REST/WS fill: одна комиссия; third-asset disposal учтён, прибыль не удвоена. |
| FIN-03 | Deposit/withdrawal non-USDT, virtual reset и unexplained loss: flow-adjusted PnL корректен; reset episodes изолированы; loss не списан как reset. |
| FIN-04 | No-trade overnight loss; zero-trade USDT day; unmatched/manual order: MTM loss виден, нулевой denominator null, атрибуция нарушена. |
| FIN-05 | Budapest DST 2026-03-29/2026-10-25 и boundary fill: 23/25h; смежные balances совпадают, событие включено ровно один раз. |
| FIN-06 | Stale/missing conversion leg, dust/no market, next-day current mark: COMPLETE отвергается, unknown asset не обнулён. |
| FIN-07 | Restart/outbox retry/late fill/conflicting trade payload: replay deterministic, duplicates безопасны, revisions объяснимы. |
| FIN-08 | Fixed-budget vs injected-capital arms, unavailable flow boundary, zero equity: comparisons не подменяют метрику; TWR undefined раскрыт. |
| FIN-09 | Gross receipts, fee/slippage attribution, external AI cost: sales не reward; costs не вычтены дважды. |
| FIN-10 | Arm lot/reservation sum к account, negative stock, balance residual: ownership инварианты либо ERROR. |
| FIN-11 | Missing downtime day → recovered losing day: gap видим, убыток возвращён, old favorable verdict пересмотрен. |
| FIN-12 | Старые files/runtime/models/.env недоступны, новый package/venv/SQLite/loop независимы; same-account/different-key не изоляция, SEED/unknown orders не становятся app positions. |

P0-A/B: **110 GREEN** в собственном Python 3.14 `.venv`; финансовые
assertions реально выполнены, включая исходные 64 теста. Предварительно проверены
missing-interface RED, missing-entrypoint RED и reproduction late-fill failure.
Реализация: `apps/binance_demo_bot/src/binance_demo_bot/financial.py` и isolation
kernel и P0-B JournalStore; не exchange-certified account reporting. Покрытие и команда запуска:
[tests/README](../../apps/binance_demo_bot/tests/README.md). Остальные части
FIN-01..12 остаются **planned integration scenarios**. Фаза завершена только после реализации,
focused tests и независимого recompute одинакового input log; численный net PnL должен сверяться с account
anchors, а completeness и denominator — с календарным manifest. Секреты не нужны в unit fixtures.

## Risk/tradeoffs

2026-10-27: пользователь явно подтвердил dedicated account для нового бота.
[P0-C1 ownership/opening manifest](binance-demo-account-opening-manifest.md)
добавляет durable declaration и provisional source-backed opening inventory,
все ненулевые returned balances и Demo midpoint conversion без peg assumption.
Это ещё не AccountEpisodeOpened, независимая финансовая сверка или разрешение
продавать SEED. Равенство двух REST balances не доказывает отсутствие операций
между чтениями; certified cut, fills/flows и reset evidence остаются отдельными
требованиями. Начальный капитал не считается дневной прибылью.

Mid equity отличается от доступной цены ликвидации; публикуем обе оценки. Fee asset valuation, latency и
недостаток historic marks ограничивают coverage; точность отчёта важнее искусственного полного sample.
Provisional pilot limits из roadmap (0.1% stop-risk/trade, 5% position, 25% gross notional, сначала одна
позиция, day loss 1%, drawdown 3%) **не внедрены и не утверждены как оптимальные**. Они
фиксируются/тестируются в фазе 3 до торговли; learner их не ослабляет. Общий loss stop использует
complete/reconciled equity, а missing/stale accounting требует отдельного entry block, не «нулевого убытка».
Gap/slippage может превысить плановый stop loss; контракт не обещает гарантированной дневной прибыли.

## Backtest/verification gate

TH-01–TH-12 применяются по [truth-harness](truth-harness.md). Spec shipping и механический Harness PASS не
являются финансовым PASS. Фаза 0 проверяет учёт, не подтверждает trading alpha. Перед любой новой торговой
политикой требуется полный portfolio replay на maximum available historical period с фиксированным
контрактом/капиталом/costs и disclosure исторических PIT/receipt gaps; свежий forward проверяется отдельно.
Старые virtual fills не actual demo evidence.

## Rollback

Контракт, app-local source artifacts/scope и risk IDs закреплены в собственном manifest; reader отвергает
неизвестную версию или чужой application namespace. При нарушении сверки новые entries блокируются будущим
RiskManager, app-owned exchange protection сохраняется; чужие orders не меняются. Возврат
software/policy версии не откатывает fills и баланс: journal остаётся, проекции пересчитываются, отчёт
переходит в PENDING/ERROR до восстановления. Смена accounting rule создаёт новый contract version и
параллельный пересчёт; выбирать выгодную версию запрещено. Rollback действует только в новом приложении;
старый бот, его процессы, ledger, конфигурация и позиции не затрагиваются.

## Dependencies and handoff

Фаза 0 задаёт интерфейсы для [фазы 1: account adapter](binance-demo-phase1-account-adapter.md), [фазы 2:
universe/data](binance-demo-phase2-universe-market-data.md), [фазы 3:
OMS/risk](binance-demo-phase3-order-management-risk.md), [фазы 4:
baseline](binance-demo-phase4-baseline-policy.md), [фазы 5:
dataset](binance-demo-phase5-outcomes-training-dataset.md), [фазы 6:
evaluation](binance-demo-phase6-profit-hypothesis-evaluation.md) и [фазы 7:
promotion](binance-demo-phase7-autonomous-promotion.md).
