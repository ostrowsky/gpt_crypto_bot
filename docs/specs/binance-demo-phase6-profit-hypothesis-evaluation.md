# Binance Demo phase 6: обучение и проверка финансовых гипотез

Дата: 2026-10-10. Status: **PLANNED — specification only; not implemented**.
Owner: repository maintainer. Contract: `daily_net_equity_pnl_v1`.
Parent: [программа](binance-demo-autonomous-trading-program.md).
Depends on: [финансовый контракт](binance-demo-phase0-financial-contract.md),
[causal universe/data](binance-demo-phase2-universe-market-data.md),
[baseline](binance-demo-phase4-baseline-policy.md),
[dataset](binance-demo-phase5-outcomes-training-dataset.md).

## Problem

Улучшение direction accuracy, teacher score или precision не доказывает
заработок счёта. Многократный поиск на уже просмотренной истории переобучает
исследователя и правила отбора. Legacy replay с virtual fills/30-day cap,
WATCH capability или ограниченным ranker overlay не оценивает новый OMS/PnL.

## Objective fit

Замкнуть воспроизводимый исследовательский цикл по среднему дневному net PnL
при одинаковом капитале и неизменном risk budget. Candidate должен не только
терять меньше baseline, но и пройти отдельную проверку положительного результата.
Здесь создаётся evidence, а не право trainer размещать ордера.

## Scope

Первый learner: supervised CatBoost/простые expectancy и ranking/exit models,
train-only calibration и ограниченный, заранее заданный поиск параметров.
LLM может предложить typed hypothesis после проверки своей добавочной пользы;
не выбирает скрытые labels, не меняет evaluator/цель/риск и не пишет production
config. Unrestricted RL, live exploration на ключе и автоматическое
переписывание кода вне scope. Неподтверждённый candidate сохраняет статус
research и не становится champion по расписанию обучения.

Reuse `mission_learning`, calendar evaluator и durable experiment loop только
через новые contract/capability adapters. `portfolio_alpha`/book simulator
расширить для actual candidate population, arbitrary maximum range, partial
exits, lifecycle/timing и continuous MTM. Старый `MAX_REPLAY_DAYS=30` не
обосновывает отсутствие остальных доступных месяцев.

## Experiment and model contracts

`ProfitHypothesisV1`: hypothesis/attempt IDs, question, `reopen_basis`, scope,
objective/risk/venue IDs, exact population, primary metric, SESOI (минимальный
практически значимый эффект), risk noninferiority margins, parameter domain,
baseline/source/data hashes, splits/purge, horizon/label definition,
execution/fee/latency assumptions, alpha/family, resampling/power protocol,
evaluation end rule, budgets, maximum range manifest и rollback.
Все обязательные поля фиксируются до наблюдения результата.

`TrainingSnapshotV1` — frozen phase-5 manifest с fit cutoff, mature-only labels,
изолированными train/tune/calibration/evaluation intervals. `ModelArtifactV1`
содержит feature order/schema, target, exact parameters/seed, fit cutoff,
framework versions, native model hash, candidate policy hash и receipts.
`FinancialEvidenceV1` содержит raw decision/order/cash/equity traces,
source-bound independent audit, coverage, denominators, intervals и verdict;
это вход release controller, не изменение active pointer.

Гипотезы проверяются последовательно: entry expectancy/weak-entry abstention,
bounded sizing/concentration, exit/trailing/time-stop, replacement и затем
order-flow execution. Каждая имеет отдельный attempt и scope. Ранее отвергнутые
entry/order-flow идеи сохраняются; повтор только при документированном новом
target/data/venue/evidence основании, не при переименовании или retune старого TEST.

## Causal learning and evaluation

Признаки доступны до decision origin; labels полностью созрели и были доступны
до fit. Нормализация, imputation, feature selection, calibration и thresholds
fit только на past training/tuning; никого не обучать на evaluation interval.
Purge/embargo по максимальному label/holding overlap; общий future interval,
position и родственные rows не пересекают train/validation/test как независимые
примеры. Regime/subgroup границы определяются из train, не post-hoc winners.

Walk-forward по календарю; selection только inner folds. All-candidate entry
targets отдельно от actual chosen-trade rewards. Exit targets conditioned on
каузально достижимом состоянии позиции; replay каждой политики строит свою
траекторию, не приклеивает новые exit predictions к выигрышным trades baseline.
Future-prefix mutation проверяет все features/intents, не только итоговую метрику.

Maximum available history: immutable PIT universe/candidate snapshots,
все допустимые даты/режимы/листинги/делистинги и честные exclusions. Недостающие
данные backfill с provenance, иначе missing periods раскрываются. Текущий
список торговых монет не является историческим universe. Просмотренная история
остаётся exposed research, даже после нового random seed/model version.

Full portfolio replay включает cash/free/locked/reserves, actual order timing
или явно simulated fills, partials/dust, fees в разных активах, TP/SL, cap,
holding/replacement/cooldown и MTM всех активов. В coarse bar с TP и SL нет
удобного oracle: finer data либо preregistered adverse execution. Не удваивать
стоимость: реальный fill уже содержит shortfall, exchange fees уже в equity.

## Controls and evidence types

Текущая frozen baseline policy — control. Все arms имеют одинаковую исходную
полную inventory/equity, risk contract и причинные данные. Для оставленных
seeded BTC/BNB/других активов обязательный control — hold initial inventory
без торговли: его MTM не выдаётся за талант модели. USDT/no-trade корректен
после отдельно сверенного стартового USDT-only budget/normalization; стартовая
конверсия и costs одинаковы. BTC hold — отдельный market-risk diagnostic.

Historical и paired forward shadow оба simulated; actual demo отдельно.
Actual-fill сравнение требует двух реально торгуемых arms с одинаковыми
бюджетами: отдельные доступные demo accounts либо P0/P3 isolated subledgers
с immutable `arm_id`, отдельными orders/fills/fees и account reconciliation.
Нет post-hoc netting, cross-financing или переназначения выгодных fills.
Глобальные risk/resource ограничения воздействуют симметрично; interference
и капитал испытания раскрываются. Actual candidate + simulated champion не
сертифицируют actual uplift, даже если обе кривые нарисованы одинаково.

## Primary metrics and statistical gate

Primary: paired mean daily net PnL/return candidate–champion на одном бюджете;
отдельный earning test against соответствующий no-trade control. Report:
absolute/normalized delta, full account return, costs, DD/loss/exposure,
positive days n/N, complete/requested days, known/unknown orders/outcomes,
per-regime/asset/day contributions и sensitivity к costs/latency/gaps.
Все financial metrics из P0, без замены reward turnover/sales receipts.

До запуска: одна primary гипотеза, фиксированный evaluation end/complete-size
rule и maximum calendar budget, 95% уровень, SESOI, power/MDE расчёт и block
length из training dependence. Для family comparisons заранее Holm или
явно зарегистрированный equivalent protocol. Calendar-block bootstrap
сохраняет missing календарные дни и временную зависимость, не превращает
множество trades одного дня в независимые дни. Bootstrap сам не исправляет
multiple testing. Повторный ежедневный significance peek/early stop запрещён
без preregistered sequential procedure; risk stop всегда действует.

SUPPORTED требует validated integrity, lower confidence bound paired
candidate–champion delta **выше заранее зарегистрированного SESOI** в единицах
primary metric, **положительный lower bound абсолютного среднего DailyNetPnL**,
и положительный paired uplift относительно соответствующего no-trade control.
Эти bounds/tests учитывают заранее выбранную family correction; для simultaneous
CI допустим preregistered Bonferroni, для test p-values — Holm. Выбирать вариант
после результата запрещено. Риск соблюдён. Более слабое point+CI правило
требует отдельного явного protocol version до запуска. REJECTED —
отсутствие эффекта/нарушение финансовых constraints по заранее заданному rule.
INCONCLUSIVE/UNDERPOWERED — не хватает точности; INVALID — данные/исполнение
не проходят integrity. Inconclusive не означает «эффекта нет» или approval.
Каждый terminal результат сохраняет source/period/population/n/N, CI и reason.

Forward minimum для financial promotion: 30 полных новых дней и 100 закрытых
сделок **на каждый actual trading arm**, плюс достаточная power, не автоматический
PASS от двух счётчиков. Нулевой-trade control не обязан создавать 100 сделок.
Low-turnover policy продлевает study только по preregistered end/extension
rule, независимому от наблюдаемого эффекта, или регистрирует новую попытку
с новым sealed cohort. Она не торгует ради квоты. Disclosure/адаптация на test
требует нового cohort.

## Acceptance criteria

- Trainer не читает sealed labels и не авторизует deployment. Independent
  verifier пересчитывает account traces из frozen inputs, не импортируя
  producer aggregation как единственное доказательство.
- Финансовый winner использует всю actual eligibility population и maximum
  archive; старые proxy/harness PASS не проходят capability check.
- Раздельны actual/simulated/legacy, exposed/sealed и readiness/profitability.
  Подтверждённая стратегия должна пройти новый forward, а не один удачный месяц.
- Bounded research job достигает terminal result либо operational failure;
  retries имеют новый attempt/retry_of, budget и неизменные frozen inputs.
- Недостаток power получает конкретный план data recovery/extension и срок
  review, не бесконечный поиск thresholds на одной раскрытой истории.

## Risk / trade-offs

Положительный эффект может не существовать; модель может улучшать forecast,
но терять деньги. Большой universe и узкий effect повышают объём проверки.
Пропущенные дни и непроверенные receipt/universe histories ограничивают
сертификацию; они не стираются ради хорошего результата. Модели на другую
venue/частоту нельзя переносить как verified execution edge.

## Backtest / verification gate

Maximum historical + новый frozen forward перед READY_FOR_CANARY;
новый actual cohort перед [phase 7 promotion](binance-demo-phase7-autonomous-promotion.md).
Stress по spread/fees/latency/gaps/partial orders и worst-day/asset contributions.
Model comparison не заканчивается regression loss; cash identities и baseline
parity независимо проверяются. Full Truth Harness обязателен перед verdict.

Планируемые tests: `EVAL-01` mature/purged calendar split; `EVAL-02` future-prefix;
`EVAL-03` train-only preprocessing/calibration; `EVAL-04` maximum archive and
PIT omissions; `EVAL-05` independent money recompute including partial exits;
`EVAL-06` initial-inventory/no-trade control; `EVAL-07` intrabar adverse path;
`EVAL-08` calendar missing/dependence/power; `EVAL-09` fixed-end/multiple tests;
`EVAL-10` actual/simulated provenance; `EVAL-11` rejected/retry retention;
`EVAL-12` losing-less candidate != earning PASS; `EVAL-13` evaluator has no
write capability; `EVAL-14` immutable result/dataset/source tamper rejection.
Это будущие implementation checks, не уже выполненные эксперименты.

## Rollback switch

Отменить bounded training/evaluation job, сохранить attempt и inputs,
mark evidence invalid/withdrawn, revoke зависимые release tickets.
Остановить promotions и вернуть compatible champion только через P7,
не меняя существующие позиции/ledger/exchange protection. Ни compiler failure,
ни rejected hypothesis не запускают более рискованный fallback.
