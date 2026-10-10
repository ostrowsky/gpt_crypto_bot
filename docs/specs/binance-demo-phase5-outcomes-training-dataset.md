# Binance Demo phase 5: дневной результат и данные обучения

Дата: 2026-10-10. Status: **PLANNED — specification only; not implemented**.
Owner: repository maintainer. Contract: `daily_net_equity_pnl_v1`.
Parent: [программа](binance-demo-autonomous-trading-program.md).
Boundary: [отдельное приложение](binance-demo-application-isolation.md),
own `apps/binance_demo_bot/.runtime` и dataset namespace; без старого state.
Depends on: [P0 money](binance-demo-phase0-financial-contract.md),
[P2 data](binance-demo-phase2-universe-market-data.md),
[P3 OMS](binance-demo-phase3-order-management-risk.md).
Ledger capture внедряется до первой демо-сделки, хотя полный отчёт — фаза 5.

## Problem

Обучение по удачным закрытым сделкам, teacher labels или виртуальным BUY
теряет отказы, стоимость исполнения и открытые убытки. Без причинной цепочки
decision→fill→money нельзя объяснить дневной PnL или безопасно учить политику.

## Objective fit

Поставить learner проверенные будущие outcomes и состояние всего счёта;
сохранить primary daily PnL, не подменяя его долей угаданных направлений.
Report должен отличать наблюдение, гипотезу, intent, ACK, trigger и actual fill.

## Scope

Append-only decision/outcome ledger, immutable dataset revisions, daily
financial report и обучение readiness. Никакого нового ranking/order
enablement, настройки thresholds или automatic promotion в этой фазе.
Legacy dataset не читается новым runtime. Если отдельно разрешён offline import
immutable exported archive, он остаётся `legacy_virtual/unknown_provenance`;
не импортируется в actual demo reward с выдуманными clocks/fills. Свои raw,
snapshots, models, reports, cursors и backups находятся в own `.runtime`.

## Data contracts

| Запись | Обязательные поля и происхождение |
|---|---|
| `DecisionObservationV1` | EventEnvelope, immutable snapshot IDs, feature cutoff и actual receipt clocks, BUY/SELL/HOLD/WAIT/BLOCK, reasons, eligibility/candidate population, portfolio state, available/reserved cash, model/policy/risk versions |
| `ExecutionOutcomeV1` | decision/client/exchange IDs, immutable arm ownership, submit/ACK/fill clocks, executed quantities/prices, fees/assets, partial/cancel/TP/SL evidence, query/reconcile state |
| `MarketTargetV1` | origin, named horizon/definition, future observation IDs, label_end_ms, label_available_at_ms, maturity state, unknown/gap reasons; без выдуманного actual fill |
| `TradeAttributionV1` | FIFO lot identity/remaining qty, causal exit reason, gross_realized_fifo_pnl, fee_expenses, delta_unrealized_fifo_pnl и actual fee movements по P0; net attribution derived без второго fee debit, отдельно от account reward |
| `DailyOutcomeV1` | AccountEpisode/scope, local day boundaries, start/end valuations/flows, revision/hash, PnL/return/DD, coverage and finalization/reconcile state |
| `DatasetManifestV1` | Exact immutable IDs/hashes, feature/target schema versions, fit cutoff, episode/scope/venue, source/policy/risk versions, exclusions, sample/day/class/regime counts |

Schema_version общей оболочки = 1. Raw timestamps, price/quantity Decimal
и registry revisions никогда не заменяются округлённой строкой отчёта.
Corrections append новую revision с `supersedes`, старые outputs остаются.
Training snapshot frozen; поздняя correction не меняет уже выпущенный model
artifact и старый результат без отдельной invalidation/re-evaluation.

Все допустимые кандидаты и исключения попадают в decision population:
не только выбранные BUY и не только финальные winners. Для blocked/WAIT
наблюдаются будущие market targets, а не доходность несуществующего ордера.
Counterfactual replay хранится отдельно с execution assumptions.
Actual daily reward принадлежит whole policy trajectory, не каждой строке
одновременно; сумма придуманной «награды всех кандидатов» запрещена.

TP/SL event требует биржевого order/list identity и фактического состояния.
Trigger без fill не закрывает label позиции. Late fills/fees включаются
в accounting revision, reconciled fills не дублируются с WS cumulative qty.
HOLD targets могут перекрываться: группировка/split/purge учитывают одну
позицию, episode и доступность outcome до fit.

## Timing and reporting

`label_available_at_ms <= fit_cutoff_ms` и закрытый observation interval
обязательны для train/calibration. Наличие будущего файла не означает,
что label был доступен на historical fit. Feature transforms используют
только train; labels/financial report join не включаются в inference features.
Нет реального receipt — provenance unknown; такой ряд не сертифицируется
простым сравнением exchange timestamp с origin.

Daily report строится после local-day boundary по P0; canonical status
`PENDING | COMPLETE | ERROR`, incomplete — reason/component completeness,
не четвёртый status. PENDING/ERROR получает reason и bounded recovery/backfill deadline. Readiness ежедневна,
но не требует ежедневного переобучения или обновления политики.
Показывать absolute/reconciled PnL, realized и unrealized, costs/fees,
balances/exposure/DD, actions и executions, protected quantity/time,
policy/model version и каждую пропущенную часть календаря.
Известный loss при outage не удаляется из отчёта как missing winning day.

## Primary metrics

Daily net PnL/return и money reconciliation по P0. Data quality:
verified decision chains / all emitted intents, mature valid labels / eligible
origins по каждому target, complete days / all requested days, stale/gap/
unknown counts, feature/label lag и изменения schemas. Каждый ratio имеет
n/N; zero N → null, не 100%. Model calibration/accuracy — отдельно от прибыли.

## Acceptance criteria

- Independent пересчёт daily money совпадает с fills/account snapshots
  в зарегистрированной Decimal tolerance; differences блокируют finalization.
- Каждое действие связано с immutable source snapshot и policy version;
  disconnect/restart не удваивает rows, fills, fees или labels.
- Dataset включает всю зарегистрированную популяцию и причины неизвестного
  результата; selected-only training не утверждает selection improvement.
- Mature-only training, no future transforms, exact namespace для actual/
  simulation/legacy; wrong venue/episode/arm/history capability rejected.
- Feedback available для profit evaluator без права писать execution config.
  Nightly report не заявляет обучение успешным по числу новых моделей.

## Risk / trade-offs

Future outcome collection требует времени и может быть неполной; label
failure не является нулевой доходностью. Drift/changing universe меняет
base rates, но не разрешает переопределить старые labels. Аккаунт с ручными
сделками, внешними flows или reset требует P0 episode/attribution treatment.
Приватные raw records защищаются локально; source/spec/tests содержат только
синтетические fixtures без реальных API credentials/UID/балансов.

## Backtest / verification gate

На maximum available decision/market archive независимо проверить population,
timing, feature prefix и все денежные identities. Historical receipt gaps
раскрываются; свежий demo corpus не смешивается с legacy provenance.
Заявление model/policy uplift проходит [phase 6](binance-demo-phase6-profit-hypothesis-evaluation.md),
релиз — [phase 7](binance-demo-phase7-autonomous-promotion.md).

Планируемые tests: `DATASET-01` blocked market target != actual PnL;
`DATASET-02` duplicate WS/reconcile; `DATASET-03` delayed fee revision;
`DATASET-04` label maturity and availability; `DATASET-05` label fields stripped
from features; `DATASET-06` prefix future mutation; `DATASET-07` selected/all
denominators; `DATASET-08` UTC/DST daily join; `DATASET-09` missing/recovered
loss day; `DATASET-10` frozen manifest immutable after correction;
`DATASET-11` reset/manual activity invalidates relevant attribution;
`DATASET-12` no secrets in reports/fixtures.
Это тест-план, не подтверждение реализации.

## Rollback switch

Остановить dataset export/retraining; сохранить raw/ledger/cursors и
финансовое наблюдение. Повреждённую revision пометить invalid, append исправление;
не удалить неудобные сделки. До исправления quality gate запрещает dependent
training/release; существующая exchange protection и account reconciliation
остаются активными. Откат отчёта не меняет фактический денежный результат.
