# Early Block Rescue Event Replay

Status: diagnostic-only  
Last updated: 2026-05-19

## Purpose

Validate the early-block rescue hypothesis on causal event logs, not only on final
top-gainer critic rows.

Question:

```text
If the bot created a rescue candidate after repeated early blocks, how many of
those candidates later became final watchlist top15 winners, and how many were
false positives?
```

## Inputs

- `files/bot_events.jsonl`;
- `files/agent_events.jsonl`;
- `.runtime/reports/top_gainer_critic_*_final.json` for post-factum labels.

## Candidate Definition

A rescue candidate is a `(local_day, symbol)` pair with:

- at least `N` blocked events before/equal the configured local hour;
- blocker reason in an allowed rescue set;
- no dependency on final-day outcome at candidate time.

The first qualifying event is the causal candidate timestamp.

## Metrics

Report:

- candidate count;
- final-top15 precision;
- false-positive count;
- false-positive ratio;
- missed-winner coverage;
- bought-before-rescue count;
- top examples with post-factum opportunity where available.

## Promotion Gate

This diagnostic can only advance to a behavioral replay if a variant:

- covers a meaningful number of missed winners;
- has acceptable top15 precision relative to candidate volume;
- does not explode false-positive candidates;
- defines the exact runtime gate fields needed for implementation.

No production behavior or Telegram signal is authorized by this audit alone.


## First Run Result

Latest event-level audit loaded:

- labeled days: `43`;
- labeled final top-gainer day/symbol rows: `645`;
- blocked events loaded: `669,641`;
- entries loaded: `1,273`.

Result: **rejected at event proxy gate**.

The broad high-opportunity variant:

- reason set: `agent_plus_score`;
- max first-block hour: `12`;
- min blocked count: `3`;
- candidates: `1,937`;
- top15 candidates: `341`;
- false-positive candidates: `1,596`;
- top15 precision: `17.60%`;
- false-positive ratio: `82.40%`;
- missed top15 candidates: `165`;
- proxy top opportunity: `287.272%`.

A stricter score-only variant reduced volume but still failed the stricter gate:

- reason set: `score_only`;
- max first-block hour: `6`;
- min blocked count: `50`;
- candidates: `195`;
- top15 candidates: `45`;
- false-positive candidates: `150`;
- top15 precision: `23.08%`;
- false-positive ratio: `76.92%`;
- missed top15 candidates: `13`;
- proxy top opportunity: `8.975%`.

## Interpretation

The previous winner-only proxy overstated the opportunity. Real event logs show
that repeated early blocks are common among both future winners and non-winners.
Using them directly as a rescue admission signal would likely create too much
candidate pressure and false-positive risk.

The casebook insight remains valid ? some huge winners were blocked early ? but
`early repeated block` alone is not a sufficient causal selector.

## Decision

Do not advance early-block rescue to behavior replay or Telegram.

The next useful step is not another broad rescue replay. A future version would
need an additional causal discriminator, for example:

- market-relative acceleration after the early block;
- watchlist rank improvement after the block;
- volume/range expansion after the block;
- portfolio opportunity-cost filter.

Until then, keep early-block rescue as diagnostic evidence only.
