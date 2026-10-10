# Binance Demo — фаза 0: финансовый контракт

Дата: 2026-10-10. Статус: **PLANNED / specification only**. Владелец: сопровождающий репозитория. Приоритет:
P0, до подключения торговли. **Исполняемая реализация этого контракта ещё не внедрена.** Документ не
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

Это **planned test scenarios**, не выполненные тесты реализации. Фаза завершена только после реализации,
focused tests и независимого recompute одинакового input log; численный net PnL должен сверяться с account
anchors, а completeness и denominator — с календарным manifest. Секреты не нужны в unit fixtures.

## Risk/tradeoffs

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
