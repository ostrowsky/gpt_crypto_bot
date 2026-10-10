# Binance Demo phase 7: автономные обновления политики и rollback

Дата: 2026-10-10. Status: **PLANNED — specification only; not implemented**.
Owner: repository maintainer. Contract: `daily_net_equity_pnl_v1`.
Parent: [программа](binance-demo-autonomous-trading-program.md).
Boundary: [отдельное приложение](binance-demo-application-isolation.md):
own policies/model pointer/keys/ledger/services в `apps/binance_demo_bot`.
Depends on: [P0](binance-demo-phase0-financial-contract.md),
[P3 OMS](binance-demo-phase3-order-management-risk.md),
[P4 baseline](binance-demo-phase4-baseline-policy.md),
[P5 feedback](binance-demo-phase5-outcomes-training-dataset.md),
[P6 independent evaluation](binance-demo-phase6-profit-hypothesis-evaluation.md).

## Problem

Trainer, который ежедневно переписывает active model после хорошей training
метрики, способен ухудшать деньги и ломать уже открытые позиции. Старый
WATCH/ranker rollout ticket не описывает private exchange OMS, риск,
actual fills и новую финансовую цель. Нужен работающий путь evidence→adoption,
с ограниченной автономией, а не бесконечный shadow или самовольная мутация.

## Objective fit

Разрешать новые параметры тогда, когда подтверждено улучшение дневного net
PnL на том же бюджете и риске; наблюдать результат после релиза и откатывать
ошибку. Автономность означает выполнение зарегистрированного протокола без
ежедневного ручного выбора thresholds, не обещание ежедневной прибыли.

## Scope and allowed changes

`DemoReleaseController` — единственный владелец compatible active policy
pointer **нового приложения**; его OMS владеет только своими ордерами,
его RiskManager — собственными неизменными hard constraints. Старые learning
workers, release controller, config, policy pointers и services не вызываются.
Trainer/LLM/evaluator не пишут active pointer и не используют ключи.
Новая capability `BINANCE_SPOT_DEMO_DAILY_PNL_V1` не принимает legacy WATCH
или score-overlay ticket, даже если подпись его файла валидна.

До первого canary регистрируется finite parameter allowlist, bounds,
evaluation protocol и risk contract. Допускаются проверенные entry/ranking
thresholds, model version и TP/SL/trailing/holding параметры. Любой непроверенный
parameter domain или новая policy family возвращается к P6 и registration.
Капитал, environment/keys, risk limits, metric, universe semantics, labels,
evaluator, code и model permissions learner изменять не может.
Наличие расписания не обязует менять champion при REJECTED/INCONCLUSIVE.

## Release states

```text
REGISTERED → HISTORICAL_EVALUATED → FORWARD_SHADOW
           → READY_FOR_CANARY → CANARY → CANARY_EVALUATED → PROMOTED
any eligible stage → PAUSED_FOR_RISK / INVALIDATED / REJECTED / ROLLED_BACK
```

Stage отдельно от result/status/reason. READY_FOR_CANARY — техническая
готовность P0–P3 и пройденные maximum-history + новые sealed forward checks P6.
Он разрешает bounded demo experiment; не обещает финансовую состоятельность.
PROMOTED — отдельный новый, заранее зарегистрированный actual canary cohort,
financial gate P6, integrity и runtime safety. Forward shadow, readiness и
actual canary cohort disjoint; данные, по которым меняли candidate, не новый test.
CONTINUING/WAITING_FOR_DATA/UNDERPOWERED имеют next action/review deadline,
не маскируются под DONE или автоматическое включение.

## Ticket and activation contract

`DemoReleaseTicketV1`: schema/capability/environment/objective IDs,
candidate+champion model/policy/source hashes, account_episode_id,
portfolio_scope_id/arm design, risk_contract_id и hash, parameter bounds,
historical/forward/canary manifest IDs, independent verifier receipt/hash,
allowed stage, activated budget/capacity, issued/expiry clocks, nonce,
prior pointer hash и revocation reference. Подпись не заменяет проверки
scope/source/freshness/cohort; evaluator signing key не Binance API key.

Activation — atomic compare-and-swap pointer только при текущем matching
champion/lease и полном ticket. Read failure, stale/expired/replayed ticket,
source/model change, missing receipt, reset episode или risk mismatch
отклоняют activation. Permanent journal хранит до/после версии и причину.
Не переиздавать успешный ticket на другой candidate/budget ради обхода gate.

Canary имеет fixed start/end/budget, quantitative stop rules, источник данных,
arm IDs, scope и observer. Нельзя выбрать удобные торговые дни post-hoc.
Изолированные comparable actual budgets и order/fill ownership определены
P0/P3/P6; одинаковые ключи одного аккаунта не делают два независимых счёта.
Переход на полный капитал/новые активы требует отдельной scale verification:
маленький arm не доказывает те же fills на большом notional.

## Open positions and policy transitions

Каждый actual lot/позиция сохраняет immutable entry policy/model/risk/arm
identity. Новая entry policy применяется только к новым intents. Existing
positions продолжают управляться compatible pinned exit policy, пока separate
position-migration record не докажет quantity/protection/order compatibility.
Миграция не ослабляет зарегистрированный hard risk и не отменяет stop заранее.
Нет молчаливой смены TP/SL/holding limit всех старых сделок при model reload.

Rollback атомарно возвращает compatible prior entry pointer и отзывает только
queued-but-unsent entry intents. Executor непосредственно перед SEND_STARTED
сверяет active release generation/ticket/revocation, expiry, risk и lease.
Accepted/possibly-sent entry requests переходят в reconciliation/query-required,
а не удаляются. Protective SELL/recovery сохраняют отдельную authority.
Для реально принятой
заявки controller ждёт reconciliation и знает её остаток. RiskManager/OMS/
account observations и защитные ордера остаются. Prior version всегда
принадлежит только новому приложению. Если prior model/source/scope
несовместим, freeze new entries безопаснее незаверенного fallback.

## Monitoring and automatic response

Проверять account reconcile, protection quantity/time, pending order certainty,
freshness/gaps, authorization, exposure/day loss/DD, label lag/calibration,
cost/latency/rejects, ticket/source/lease validity и actual financial cohort.
Risk stop незамедлительно запрещает новые входы и сохраняет protective exits;
отдельный emergency close регулируется P3, без обещания fill при outage.
Loss/breach/failure наблюдения остаются в financial result попытки.

Data/model drift сам по себе вызывает diagnosis/quality block/retraining
candidate, не автоматическое снижение confidence threshold. Повторно включать
входы после risk/integrity incident можно лишь по зарегистрированному recovery
gate с актуальным reconcile и исправлением причины. Controller не повышает
loss budget ради достижения дневной цели.

## Primary metrics

Daily net PnL/return against comparable champion/no-trade, lower CI/SESOI/power
по P6 и риск по P0. Technical: permitted/rejected activations n/N, time-to-
rollback, unmatched intents/orders, protection и reconciliation violations,
complete/requested cohort days. Число models/promotions не является успехом.
Readiness/profitability/actual vs simulated и epoch/source versions публикуются
отдельно. CostAI/infrastructure ограничены экономическим budget.

## Acceptance criteria

- Только authenticated exact-scope ticket проходит CAS; legacy/stale/replayed/
  wrong episode/risk/source ticket rejected с evidence.
- CANARY стартует лишь после readiness и preregistration; PROMOTED требует
  нового actual cohort, independent money audit и earning/uplift gate P6.
- Неисправный/underpowered candidate не включается; loop продолжает сбор
  данных или другую зарегистрированную гипотезу с конечным budget.
- Current champion остаётся frozen в сравниваемом периоде; изменение параметра
  завершает version/candidate cohort, не бесшовно переписывает прошлое.
- Restart/worker lease/CAS/risk breach не теряют orders, quantities, fees,
  open-position exit identity или arm budgets.
- Bounded rollout и обратный путь проверены до автоматических обновлений.
  Autonomous daily parameter mutation без passing ticket невозможна.

## Risk / trade-offs

Качество evidence ограничивает частоту обновлений; положительный исторический
результат может не повториться. Политики с общим счётом/ресурсами зависят друг
от друга: stop treatment и interference раскрываются до эксперимента.
Новый параметр выхода может повредить существующие позиции — поэтому exit
pinning/migration обязательны. Демо-approval не является real-money approval.

## Backtest / verification gate

P6 maximum-period causal portfolio comparison и separate fresh forward/canary;
P3 failure injection на actual order lifecycle до release-controller enablement.
Проверить known champion/source compatibility, complete risk contract, latest
full Truth Harness, exact hashes и независимые receipts. Повторная promotion
по той же раскрытой выборке не разрешена; sequential monitoring protocol
financial quality задан заранее, технические risk stops действуют всегда.

Планируемые tests: `REL-01` legacy capability; `REL-02` wrong/expired/replayed
ticket; `REL-03` model/data/source tamper; `REL-04` CAS/lease race;
`REL-05` readiness!=promotion/disjoint cohorts; `REL-06` risk allowlist
immutable; `REL-07` open-position exit pinning; `REL-08` in-flight rollback;
`REL-09` restart keeps budgets/orders; `REL-10` rejected/underpowered no update;
`REL-11` reset invalidation; `REL-12` trigger/timeout never fake fill;
`REL-13` scale change needs new verification; `REL-14` bounded recovery and
no forced trade/daily profit; `REL-15` secret-free report/signature separation.
Планируемые тесты не считаются выполненной runtime-валидацией сейчас.

## Rollback switch

Disable promotions + `entry_enabled=false`; preserve ledger/protection,
reconcile accepted orders и atomically restore compatible prior pointer.
Новая phase-7 authority отключается отдельно от account observer/OMS exits.
Отзыв ticket/source сохраняется append-only; historical losses/failed canary
не удаляются. До реализации execution и автономная promotion не включены.
