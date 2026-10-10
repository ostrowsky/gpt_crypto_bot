# Binance Demo phase 3: order management, protection and risk

Registered: 2026-10-10. Status: **PLANNED — specification only**.
Objective contract: `daily_net_equity_pnl_v1`. Environment: `BINANCE_SPOT_DEMO`.
Priority: P3; execution/accounting prerequisite for automatic trading, roadmap phase 3.
This document enables no exchange requests, orders, policy changes or learning.

## Problem

Сигнал BUY/SELL и изменение виртуального JSON не подтверждают покупку/продажу
на Binance. Несколько производителей сигналов, неизвестный результат отправки,
частичное исполнение и гонка SELL со стопом могут создать повторную покупку,
продажу сверх остатка или незащищённую экспозицию. Без фактических fills и
сверенного учёта дневной PnL и обучающие метки недостоверны.

## Objective fit and dependencies

Фаза создаёт единый путь от immutable решения до биржевого исполнения,
комиссии и остатка. Финансовый результат считает контракт фазы 0; эта фаза
доказывает техническую готовность, а не прибыльность или финансовый uplift.
Старые виртуальные сделки остаются отдельной популяцией.

Зависимости: [program](binance-demo-autonomous-trading-program.md),
[financial contract](binance-demo-phase0-financial-contract.md),
[account adapter](binance-demo-phase1-account-adapter.md),
[universe/data](binance-demo-phase2-universe-market-data.md),
[roadmap](../roadmaps/binance-demo-daily-pnl-roadmap.md),
[Truth Harness](truth-harness.md). TH-01..TH-12 обязательны.

## Scope

Spot Demo, long-only, покупки за выделенные средства; pilot execution — USDT
пары, хотя мониторинг охватывает весь разрешённый universe. Один OrderManager
(OMS), RiskManager и фактический ledger обслуживают main/agent/model intents.
Включены BUY, защита, HOLD, частичный/полный SELL, cancel/reconcile, восстановление.
Исключены production hosts, плечо, shorts, выводы, переводы, свободный RL,
торговля по Telegram-сообщениям и автоматическое увеличение бюджета.

Планируемые компоненты: `OrderManager`, единственный credential-owning
`OrderExecutor`, `RiskManager`, `AccountReconciler`; transactional SQLite WAL
с durable outbox и журналом событий. Имена — design targets, не наличие кода.

## Contracts and durable identity

Общий EventEnvelope: `schema_version=1`, `environment`, `account_episode_id`,
`portfolio_scope_id`, `event_id`, `event_type`, `causation_id`,
`exchange_event_time_ms` (nullable), фактические `received_at_ms` UTC,
`received_monotonic_ns`, `process_instance_id`, `recorded_at_ms`; при наличии `decision_id`, `arm_id`,
`model_id`, `policy_version`, `source_sha256`, `risk_contract_id`.
OMS payload добавляет `objective_contract_id`, `position_id`, `client_order_id`,
`exchange_order_id`, `exchange_order_list_id`, `exchange_trade_id`,
`state_version`, `lease_epoch`, `request_hash`, quantities/prices/fee asset.
Исходный ответ и normalized event связываются hash; секреты не сохраняются.

DecisionIntent содержит producer, immutable arm/scope, действие, symbol, sizing,
TP/SL, reason, данные/feature cutoff, market/account snapshot IDs, срок годности,
policy/model/source/risk versions. Уникальный ключ логического действия:
`(account_episode_id, portfolio_scope_id, arm_id, decision_id, action_seq)`.
Повтор того же ключа возвращает записанное состояние; иной payload с тем же
ключом — integrity incident. Завершённые intent tombstones сохраняются.

Резерв, approved intent и outbox создаются одной транзакцией. До подтверждения
результата reservation не освобождается только из-за timeout. Client order ID
генерируется до отправки, соответствует требованиям API и неизменно связан с
payload. Он не заменяет журнал: биржа допускает некоторые повторные IDs после
завершения старой заявки. Не обещаем exactly-once transport через сеть.

Каждый arm получает заранее фиксированный реально выделенный бюджет и
commission reserve. Ownership intent → order/list → fill → fee → позиция
неизменен. Нет неттинга ордеров arms, перераспределения fill после результата
или использования средств другого arm. Сумма arm ledgers/reservations плюс
зарегистрированный unallocated scope сходится с полным счётом. Несколько API
ключей одного счёта не создают независимые бюджеты. Manual/unowned order или
fee — incident и запрет новых входов, не автоматически присвоенная прибыль.

## Order state machine and concurrency

`RECEIVED → VALIDATED/BLOCKED → RESERVED → OUTBOX_READY → SEND_STARTED →`
`ACKNOWLEDGED/QUERY_REQUIRED/REJECTED → PARTIALLY_FILLED/FILLED/CANCELED/EXPIRED`.
Позиция отдельно: `NONE → OPEN_UNPROTECTED → OPEN_PROTECTED → EXIT_PENDING →`
`CLOSED` либо `RECOVERY_REQUIRED/DUST`. Order/list и protection state отдельны:
ACK, pending leg, stop trigger, order cancel и fill не взаимозаменяемы.

`SEND_STARTED` записывается до сети. Crash/timeout/5xx с неясным результатом
ведёт в QUERY_REQUIRED: запросить ордер/list по сохранённым IDs, fills и баланс,
совместить private events; не слать новый BUY/SELL с новым ID. Not-found в одном
ответе при отставании источника не доказывает отсутствия ордера. Повтор допустим
только после зарегистрированного протокола доказанного non-acceptance либо
однозначного pre-engine rejection; unresolved статус блокирует конфликтующие
действия. Query retries ограничены по времени/числу и rate budget.

Account-scoped единственный подписывающий executor держит OS exclusive lock
на весь срок владения, включая отправку. Producers signing access не имеют.
Durable lease epoch и state-version CAS предотвращают stale/out-of-order jobs;
проверка epoch происходит в executor непосредственно перед network send.
Lease timeout сам по себе не гарантирует fencing на Binance. Новый владелец
не начинает отправку, пока старый процесс не исключён как sender и его in-flight
requests не сверены. При сомнении takeover блокируется; split-brain не лечится
вторым executor. Backups не запускаются одновременно с живым writer.

Непосредственно перед `SEND_STARTED` executor проверяет актуальные release
generation, ticket/revocation, intent expiry, risk state и lease. Откат атомарно
отзывает queued-but-unsent entry intents; accepted/possibly-sent requests
сначала сверяются. Protective SELL/recovery имеют отдельную authority и не
отзываются как новые входы при смене entry policy.

Дубли `executionReport`/REST fills не создают второй fill/fee. Fill uniqueness:
`(environment, account_episode_id, symbol, exchange_trade_id)`; order/list
идентичность проверяется отдельно. Противоречащий payload с тем же ключом —
incident. Cumulative quantity согласуется с unique fills; поздний NEW не
откатывает FILLED. Arrival order/event timestamps не заменяют reconciliation.
Статус, reserve update, ledger effect и consumed-event marker пишутся атомарно.

## Pre-send risk and accounting

Все деньги/quantity/price — Decimal; float не участвует в проверке лимитов или
подписи суммы. Свежие permissions, symbol status, order types, применимые
tick/step/notional/percent-price/order-count filters проверяются до отправки и
после округления. Новый неизвестный filter — BLOCK. Post-only/limit rejection
не превращается без разрешённого правила в MARKET. Pending BUY и резервы всех
producers входят в aggregate exposure; free balance не включает locked средства.

Reservation учитывает worst permitted fill price, fees и price movement budget;
BUY никогда не занимает средства. SELL quantity ограничена реально полученным
и доступным остатковым base соответствующего arm, минус активные SELL/reserves.
Возможная SELL commission в base также резервируется: quantity плюс эта fee
не превосходят доступный base; неожиданная fee требует reconcile/entry stop.
OCO две альтернативные legs не считаются двумя независимыми владениями base.
Base fee уменьшает доступную quantity; quote/BNB fee учитывается в своём asset
ledger один раз. Фактическая commission имеет приоритет над предварительной
оценкой. Native OPO fee из free funds также должна иметь предварительный arm
reserve; она не разрешает скрытое финансирование соседним arm.

Pilot limits из roadmap **provisional, не включены**: stop-risk ≤0.1% equity
на вход, позиция ≤5%, суммарная exposure ≤25%, сначала одна позиция, daily loss
stop 1%, drawdown stop 3%. Фаза 0 регистрирует окончательный risk contract;
другие пределы требуют новой версии/валидации. Learner их не ослабляет.
Stop-risk estimate не гарантирует максимальный убыток при gap/stop-limit.

До первой BUY обязательны конечные положительные preregistered
`max_unprotected_notional_usdt`, `max_unprotected_duration_ms`, recovery attempt
budget и deadline, измеренные transport/stream freshness limits, dust cap и
emergency-close slippage/quantity limits. Пока значения отсутствуют — вход
BLOCK. Значения выбираются из pilot budget/проверенной latency до результатов,
не увеличиваются при неудачном тесте. Показываются текущие/максимальные observed
values, breaches и n/N. Нет свежего account/valuation или stream gap — новых
входов нет; старые защиты продолжают жить на бирже.

## Exchange-side protection and partial entries

Базовый путь — verified BUY fills → SELL OCO на доступную net quantity.
Для linked LIMIT/LIMIT_MAKER входа испытать OTOCO; OPOCO — отдельная capability
проверка. Типы legs, правильность TP/SL, filters и поддержка конкретного demo
symbol/аккаунта подтверждаются phase 1 evidence, не общей документацией.

OTOCO/OPOCO pending exits включаются только после **полного** working BUY;
partial fill уже является экспозицией. OPOCO quantity определяется полученным
asset с правилами commission/rounding; activation filters могут отклонить exits.
PENDING_NEW не активная защита. Проверяется статус обеих legs/list, net covered
quantity и freshness; неожиданное reject/cancel/expiry снимает PROTECTED.
[Binance order lists](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#order-lists),
[OPO rules](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/opo.md).

Partial BUY: экспозиция учитывается с первого unique fill. OMS по заранее
выбранному bounded протоколу защищает пригодный остаток либо отменяет remainder
working entry, сверяет новые fills/native exits и освобождение locked base,
затем размещает защиту/разрешённый recovery SELL. Cancel может гоняться с full
fill и активацией linked exits; сначала проверить list, а не добавить второй
SELL. Если partial quantity меньше min-notional/step — остановить remainder,
сверить, закрыть только допустимую quantity при разрешённых filters. Остаток
dust не исчезает: asset ledger/valuation/risk/причина/план восстановления.
Автоматическая дополнительная BUY ради достижения min-notional запрещена.

Защита отклонена/потеряна: entry kill, incident, bounded query/cancel/reprotect
или emergency exit. Превышенный time/notional/dust предел запрещает дальнейшие
входы и расширение canary даже после последующего успешного закрытия. Если
биржа недоступна, record `RECOVERY_REQUIRED` и сохранять exposure; не обещать
мгновенный выход и не объявлять protected-cycle PASS.

Stop trigger и stop-limit placement не равны продаже. LIMIT stop может
остаться без fill; MARKET stop имеет slippage. Recovery использует доступные
типы в зарегистрированных пределах; таймер наблюдает outstanding stop и fills.
Amend/replacement разрешён только при проверенной capability и race протоколе.
Неподдерживаемое обновление сохраняет действующую защиту и отмечает degraded
management; запрещён blind cancel единственного стопа ради нового параметра.

## SELL, kill switches and restart

Signal SELL — intent к OMS. Для выхода сверить linked protection/fills, отменить
конфликтующую защиту, подтвердить результат отмены, повторно вычислить остаток
после возможного TP/SL fill, затем продать допустимый remainder. Partial SELL
уменьшает exposure только по fill; оставшаяся quantity требует защиты/recovery.
Timeout cancel/SELL — QUERY_REQUIRED; zero remainder не создаёт ещё одну продажу.

Entry kill switch останавливает BUY/outbox entries, сохраняет существующие
exchange stops и разрешённое сопровождение/SELL. Emergency-close — отдельный
scope/version с планом отмены, сверки, безопасного закрытия и bounds; не удаление
позиций из JSON. Нет pruning реальной exposure из-за превышения capacity.
Delist/CANCEL_ONLY/maintenance не гарантирует возможность SELL: действия только
по подтверждённым capability, остаток остаётся виден с incident.

Startup/reconnect: exclusive ownership → запись gap → account/open orders/lists
snapshot → backfill trades/orders с durable cursor, overlap и pagination →
dedup/private event merge → reconciliation quantities/fees/free/locked/reserves →
проверка защиты → resume разрешённых действий. Не предполагается replay всех
пропущенных WebSocket events. Старые terminal orders/fills сохраняются даже если
API историю больше не отдаёт; неразрешимый gap блокирует новые входы и полную
атрибуцию. Balance mismatch, manual trade, reset начинают incident/episode
протокол фазы 0; старые незакрытые экспозиции не забываются при перезапуске.

## Primary metrics and acceptance criteria

Business metric: общий daily net equity PnL, не intent count/proceeds/сумма
выборочных удачных сделок. Фаза 3 публикует readiness отдельно от profitability.
Технически: protected net quantity/total open net quantity по каждому asset;
разные assets агрегируются только по зарегистрированной USDT value, не суммой
несопоставимых quantities. Также protected-cycle
count/attempt count; fill completeness; reconciliation asset deltas; duplicate
send incidents; unprotected notional/duration/max/breaches; pending unknown
count/age; rejected/canceled/partial/fill counts; latency p50/p95/max и N.
Нет denominator — null; unknown/recovery не успешная защита. Dust показывается
отдельно, а не убирается из знаменателя риска. Все попытки/потери сохраняются.

Acceptance: (1) complete planned scenarios below pass; (2) ledger независимо
сходится с account/fills/fees в Decimal tolerances, зарегистрированных по asset;
(3) один bounded demo BUY → verified protection → actual SELL lifecycle доказан
exchange IDs и балансом; (4) restart/failure drills не дают лишний BUY/oversell;
(5) нулевые unresolved critical incidents для technical readiness и сохранённая
история breaches; (6) tested entry stop/emergency recovery и backup restore;
(7) source/spec/tests review, full/staged Harness и чистый intended diff.
Integration pilot только по отдельному разрешённому demo execution scope после
phases 0/1/2; этот spec-only commit такого разрешения не создаёт.

## Backtest verification, risk and rollback

Симулятор использует тот же intent/risk/reservation/partial/protection протокол;
latency, commissions, locked funds и неоднозначный TP/SL внутри свечи задаются
заранее. Finer data либо неблагоприятный заранее заданный порядок; future prices
не определяют действие. Прежде чем ослабить trading policy — максимальная
доступная история всей candidate population, целые отдельные cash ledgers,
walk-forward, untouched cohort, одинаковые risk/budget и named baselines.
Историческая свеча не доказывает фактический demo fill/capability; отсутствие
receipt/order-flow depth отмечается, не заменяется точным исполнением.

Rollback отключает новые entries/release pointer, удерживает ledger/cursors/
ownership и сопровождение existing exposure, запускает reconciliation.
Возврат к старому JSON engine не может забыть или повторно купить активы.
Schema/artifact upgrade имеет проверенный backward read/restore план; снимок
перед миграцией сохраняется. Production environment остаётся запрещённой.

## Planned focused tests and verification scenarios

Это обязательные будущие тесты; сейчас test implementation/results отсутствуют.

| ID | Проверяемый сценарий и ожидаемый результат |
|---|---|
| D3-01 | Main/agent duplicate intent, conflicting payload: один reserve/outbox; конфликт BLOCK. |
| D3-02 | Crash до/после send, ACK потерян, delayed not-found: query, без нового BUY ID. |
| D3-03 | Terminal client ID reuse: journal tombstone запрещает повтор завершённого решения. |
| D3-04 | Duplicate/out-of-order WS/REST fills и fee: один денежный effect, без regress state. |
| D3-05 | Two workers, stale epoch, takeover with in-flight: только один sender; uncertainty BLOCK. |
| D3-06 | Decimal tick/step/notional/percent/limits, stale filters, free vs locked: safe reject. |
| D3-07 | Concurrent budgets/pending BUY, base/quote/BNB fees: нет overspend/cross-financing. |
| D3-08 | BUY full/partial, OCO/OTOCO/OPOCO activation/reject: pending != protection. |
| D3-09 | Partial ниже protective min-notional, cancel/full-fill race: dust учтён, лишнего SELL нет. |
| D3-10 | Protection reject, timeout, bounded recovery, недоступная биржа: entry kill/incident. |
| D3-11 | Breach unprotected notional/time/dust: attempt FAIL, не стирается после recovery. |
| D3-12 | TP/SL trigger, partial fill, signal SELL и cancel race: продаётся только net remainder. |
| D3-13 | Unsupported amend: старый стоп сохранён; explicit replacement race проверен отдельно. |
| D3-14 | Stop-limit не fill, slippage/gap, exhausted rate budget: bounded recovery, видимый риск. |
| D3-15 | Kill switch vs emergency close: entries off; защита/ledger/exits не исчезают. |
| D3-16 | Restart/backfill/pagination/gap, external order/reset: reconcile до resume, unknown не PASS. |
| D3-17 | Capacity exceeded/CANCEL_ONLY/delist: exposure сохранена, никакого виртуального закрытия. |
| D3-18 | Actual arms equal budget/immutable ownership, partial fees: сумма к account, no posthoc split. |
| D3-19 | Full lifecycle demo evidence + independent money recomputation: integration, не profit claim. |
| D3-20 | Ledger backup/restore/migration rollback: незакрытые orders и tombstones восстановлены. |

## Primary API references

Reviewed 2026-10-10; deployment повторно проверяет актуальные symbol/account
capabilities. Использованы [REST unknown execution/status rules](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md),
[private execution/account events](https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md),
[filters](https://github.com/binance/binance-spot-api-docs/blob/master/filters.md),
[amend capability](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/order_amend_keep_priority.md).
API docs описывают возможности; они не доказательство их поддержки этим demo
аккаунтом и не evidence успешного выполнения запланированных сценариев.
