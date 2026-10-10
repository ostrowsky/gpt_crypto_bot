# Binance Demo phase 4: автоматическая базовая торговая политика

Дата: 2026-10-10. Status: **PLANNED — specification only; not implemented**.
Owner: repository maintainer. Contract: `daily_net_equity_pnl_v1`.
Parent: [программа](binance-demo-autonomous-trading-program.md).
Depends on: [финансовый контракт](binance-demo-phase0-financial-contract.md),
[market data](binance-demo-phase2-universe-market-data.md),
[OMS/risk](binance-demo-phase3-order-management-risk.md).

## Problem

Существующий сигнал по цене свечи не подтверждает покупку на бирже.
Раздельные main/agent JSON-позиции и capacity-pruning нельзя напрямую
превратить в реальные ордера. Для измерения новой цели нужна одна frozen
политика, воспроизводимая в симуляции и демо, с реальным денежным состоянием.

## Objective fit

Создать исходный champion для дневного net PnL, с которым сравниваются
последующие модели. Технически исправное исполнение не доказывает, что эта
политика прибыльна. Диагностика лидеров объясняет входы, не является reward.

## Scope

Существующие правила извлекаются в чистый decision layer или adapter без
скрытых JSON/Telegram/order side effects. Baseline manifest замораживает
entry/exit modes, thresholds, cooldown, replacement, holding limits, TP/SL,
universe eligibility, price clocks, sizing, portfolio cap и source hashes.
Прежняя политика не переносится как «точный baseline», если эти поля изменены:
изменение записывается отдельным baseline version с причиной.

Decision layer принимает immutable causal snapshot и подтверждённый ledger,
возвращает `BUY | HOLD | SELL | WAIT | BLOCK` intent с reason codes.
Main/agent — источники кандидатов одного scheduler; один risk/OMS решает
admission и резерв средств. Виртуальный log не подтверждает fill.
Модель обучения, новый entry edge, order-flow delay и real trading здесь
не внедряются. Каждая новая стратегия проходит [phase 6](binance-demo-phase6-profit-hypothesis-evaluation.md).

## Decision contract

`DecisionIntent`: envelope программы + `decision_id`, `action`, `symbol`,
`requested_quantity/quote_budget`, `reference_price`, `feature_cutoff_ms`,
`universe_snapshot_id`, `account_snapshot_id`, `reason_codes`, `expires_at_ms`,
`policy_version`, `risk_contract_id`, план защитных уровней/holding limit.
Один intent относится к одному episode/scope/arm; имеет стабильный hash.

Каждый origin обрабатывает только данные, реально доступные до decision.
Новые свечи не достраивают старый snapshot, unclosed candle не участвует
в closed-bar сигнале. Price forecast, early-rank или trader score не отменяют
проверку fees/liquidity/funds/risk. Полностью определённое tie-breaking
ранжирует одновременных кандидатов; losers сохраняют BLOCK reason.

После принятия решения OMS ещё раз проверяет свежесть, цену, свободный бюджет
и exposure. Истёкший intent не переисполняется как новый без нового snapshot.
SELL указывает подтверждённую позицию/остаток и причину; защитный exit
не откладывается ради favourable order-flow forecast.
Отсутствие positive edge допускает WAIT, без требования дневной квоты сделок.

## Risk parameters

Начальные ограничения pilot — предложения roadmap: риск по стопу до 0.1%
equity, notional одной позиции до 5%, общий notional до 25%, одна открытая
позиция на первом запуске, daily loss stop 1%, DD stop 3%.
Это не включённые параметры и не доказательство оптимальности. Финальный
`risk_contract_id` регистрирует denominator, precision, slippage reserve,
действие на breach, finite max holding и protection timing до первого BUY.
Больший капитал/число позиций — отдельная проверка масштаба, не право learner.

Если planned quantity не проходит min-notional/step или превышает риск,
выдать BLOCK; нельзя округлить её вверх через риск-лимит.
Capacity освобождается только по подтверждённому exit, не по intent/cancel
или удалению строки. Replacement учитывает два fills, время и стоимость
перехода; не создаёт дополнительного кредита под ожидаемую продажу.

## Primary metrics

Daily net PnL/return, max DD, day loss, exposure, turnover/fees и equity
reconciliation. Атрибуция использует поля P0: `gross_realized_fifo_pnl`,
`fee_expenses`, `delta_unrealized_fifo_pnl`; net realized = gross realized
минус fee expenses. Canonical equity PnL не вычитает комиссию повторно.
Technical: filled/submitted `n/N`, protected net quantity/current open net
quantity по каждому asset; aggregate protection только по USDT value,
не суммой BTC/ETH/SOL quantities. Dust остаётся в exposure. Latency и counts query-required/
rejected/partial/dust. Отдельно early/coverage/precision с определённой
популяцией и denominator. Дневной PnL без fills ещё может изменяться из-за MTM.

## Acceptance criteria

- Все источники BUY/SELL идут через один account-aware OMS; никакая ветка
  monitor/agent не может отдельно «купить» актив только в JSON.
- Frozen snapshot повторно воспроизводит action/reason/quantity; prefix future
  mutation не меняет уже выпущенные intents.
- Одновременные BUY не превышают free/reserved budget, cap или correlated
  exposure; активная защита и actual position сохраняются после restart.
- Replay и demo policy используют одинаковые entry/exit/risk definitions.
  Отсутствие подтверждённой parity раскрыто как blocker adoption, не PASS.
- Controlled vertical slice доказывает integration readiness. Автоматический
  demo experiment включается только отдельным admission record с лимитами,
  owner, периодом и rollback; прибыльность остаётся отдельным verdict.

## Risk / trade-offs

Прежние правила могут быть убыточными при честных fills и полном учёте.
Обновлённый all-universe отличается от старого watchlist: не переносить старые
результаты как evidence на новую популяцию. Более редкие входы могут снижать
охват лидеров; увеличение капитала может ухудшить fills. Strict stale gates
создают явные WAIT/BLOCK, а не ретроспективные успешные сделки.

## Backtest / verification gate

Maximum available history с point-in-time universe, causal snapshots и полным
портфелем обеих политик. Проверить partial fills/exits, timing/fees, TP+SL
intrabar ambiguity и holding limits. Если только coarse bars доступны,
использовать зарегистрированный adverse path; execution parity не объявляется
сертифицированной. Старое `MAX_REPLAY_DAYS=30` не является максимальной историей.
Нужны новый forward по frozen policy и техническая готовность P0–P3 перед
автоматическим canary; переход в прибыльный champion — [phase 7](binance-demo-phase7-autonomous-promotion.md).

Планируемые focused tests: `POLICY-01` closed-bar/receipt cutoff;
`POLICY-02` main+agent capacity race; `POLICY-03` expired intent;
`POLICY-04` entry fee/stop distance size; `POLICY-05` replacement reservation;
`POLICY-06` actual exit before capacity release; `POLICY-07` snapshot/prefix
parity; `POLICY-08` no-trade/held-loss daily MTM; `POLICY-09` kill switch keeps
exits; `POLICY-10` old watchlist result cannot approve expanded universe.
Tests указаны для реализации; сейчас не выполнялись как runtime checks.

## Rollback switch

`entry_enabled=false` останавливает admission, не protective SELL.
Отключение policy adapter не удаляет actual exposure. Ledger/reconcile/OMS
остаются до разрешения всех orders и остатков; model pointer возвращается
к проверенному compatible champion без пересчёта исторических решений.
При несовместимом risk/episode/scope откат блокирует новые входы вместо
включения legacy JSON execution. До реализации новые действия выключены.
