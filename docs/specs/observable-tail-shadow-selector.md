# Observable Tail Shadow Selector

Date: 2026-08-03
Status: shadow-only

## Objective

Collect independent forward evidence for retaining a 50% protected tail after
selected exits, without changing production SELL behavior.

## Selector

`exclude_ema_and_false_cleanup` registers a shadow candidate only when all of
the following are observable at the exit decision:

- exit reason is not an EMA-break cleanup;
- realized PnL is greater than `-0.5%`;
- prior maximum favorable excursion is at least `1.0%`.

The shadow watch records counterfactual returns after 2, 5, and 10 bars and the
50% partial-tail delta. `OBSERVABLE_TAIL_SHADOW_ENABLED` is the rollback switch.
No position size, exit timing, Telegram signal, or live order changes.

## Replay evidence

Maximum available period: 2026-05-05 through 2026-08-02, 76 report days and
1,010 labeled rows. Chronological holdout contains 303 rows.

- holdout average uplift across all cases: `+0.2581%`;
- selected-case average uplift: `+1.0291%`;
- selected-case median uplift: `+0.5233%`;
- selected-case worse rate: `21.05%`;
- selected rate: `25.08%`;
- false-positive selected rate: `0.0%`.

Recent 14-day stability window: 227 rows / 68 test rows. The same selector is
ranked first, with `+0.1439%` average test uplift, `+0.5152%` selected average,
`+0.4098%` selected median, `21.05%` selected worse rate, and `0.0%`
false-positive selected rate.

## Promotion gate

The replay authorizes shadow collection only. Production SELL changes require
an independent mature forward cohort that preserves positive median partial-
tail delta without degrading realized PnL, drawdown, or the bot's early-capture
north-star metric.
