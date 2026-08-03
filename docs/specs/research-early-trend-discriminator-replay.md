# Research Early-Trend Discriminator Replay

Date: 2026-08-03
Status: replay passed for independent shadow telemetry; production and Telegram unchanged

## Problem

The broad research-universe collector now has mature causal labels, but the
existing V1 structural signals capture only a small fraction of persistent
T+5/T+10 upside episodes. Simple `alignment` / `trend` expansion and the
previous `regime_start/base_recovery_v1` profile both failed their evidence
gates. A materially different discriminator must be tested without weakening
the production score, blocker, portfolio, or BUY gates.

## Hypothesis

A nonlinear classifier trained only on decision-time 15-minute features can
identify an additive early-trend shadow signal. The candidate is evaluated as
an adjunct to the first existing V1 structural signal per symbol/local-day,
never as a replacement for V1.

Allowed features are the current research row's RSI, ADX, slope, relative
volume, normalized MACD histogram, ATR, candle body/wicks, 24-hour rank/change,
log quote volume, and current V1 rule-signal one-hot values. Forward labels,
final top-mover membership, later rows, and portfolio outcomes are forbidden
features.

## Causal Replay

- Use every mature row in `research_universe_shadow.jsonl` over the maximum
  available period.
- Group decisions by symbol and Europe/Budapest local day; at most the first
  signal from each policy is counted.
- Primary label: `ret_5 >= 0.5%` and `ret_10 >= 1.0%`.
- Strict guard label: `ret_5 >= 1.0%` and `ret_10 >= 2.0%`.
- Split ordered local days into 60% train, 20% validation, and 20% holdout.
  Purge one complete day on both sides of each boundary so T+10 outcomes do not
  leak across the decision split.
- Fit the classifier on train only. Select its probability threshold on
  validation only. Read holdout once after threshold selection.
- Join final critic artifacts only after threshold selection. The canonical
  north-star denominator is `exchange_top_gainers` filtered by
  `in_watchlist=true`; `watchlist_top_gainers` must not replace it.

## Validation Gate

Compared with the existing V1 structural baseline, validation must show:

- primary recall improvement of at least 2 percentage points;
- precision no more than 2 points worse;
- no lower average T+10 return;
- no more than 1.5x selected symbol/day pressure;
- strict recall no lower and strict precision no more than 2 points worse.

The untouched holdout must independently show at least +2 points primary
recall, no precision loss, no lower average T+10, no more than 1.5x pressure,
and no strict-label precision/recall loss.

## Promotion Boundary

Passing this replay can advance the profile only to independent shadow
telemetry. It cannot enable BUY or realtime Telegram because forward-return
labels are a proxy for the canonical same-day top-mover objective. At least
five holdout critic days and five canonical top movers are required even for a
north-star directional check. The candidate must not reduce top-mover recall;
limited critic coverage is reported explicitly rather than treated as proof.

Any production proposal requires a frozen model/config artifact, independent
forward shadow labels, fee/slippage and ten-slot portfolio replay, and a
rollback switch.

When every replay gate passes, the CLI may export the frozen model and metadata
under `.runtime/models` with `--export-shadow-model`. The research collector
then annotates only new 15-minute, in-watchlist, `rule_signal=none` rows with
the model score and threshold. This field is telemetry inside the separate
research dataset: it sends no Telegram message, opens no position, and changes
no V1/V2 gate. Model/schema absence fails closed and is exposed in collector
status.

The runtime scorer verifies both the exact feature schema and the exported
model SHA-256 before loading. A missing hash, mismatched artifact, or schema
change fails closed.

The existing research-universe scorecard reports a separate forward cohort,
deduplicated to the first candidate per symbol/Europe-Budapest local day. It
tracks mature T+5/T+10 rows, primary/strict precision, and average/median
returns. Fewer than 30 mature first candidates is always `collect_forward_labels`;
the scorecard never marks the cohort production-eligible.

## Maximum-Period Result — 2026-08-03

The official replay loaded `130,686` mature 15-minute rows and used 30 train,
8 validation, and 10 untouched holdout local days plus four embargo days. Ten
of 16 validation thresholds passed. Validation selected probability threshold
`0.61248582875`.

On holdout, adding the discriminator to the existing V1 structural baseline:

- primary precision improved `7.7551% -> 9.2527%` (`+1.4976pp`);
- primary recall improved `10.1604% -> 13.9037%` (`+3.7433pp`);
- average T+10 improved `0.0762% -> 0.1224%` (`+0.0462pp`);
- strict precision improved `4.4898% -> 4.9822%`;
- strict recall improved `14.4737% -> 18.4211%`.

The final-critic directional check covered 9 holdout days and 26 canonical
watchlist top movers. Recall stayed `69.2308%`; precision changed
`7.5314% -> 6.5934%`, within the frozen 2-point guardrail. The candidate did
not add a previously uncaptured top mover, but it moved three captured top
movers earlier by 560 minutes on average.

Decision: enable independent dataset-only shadow annotation with the frozen
model. Do not enable BUY or Telegram. Forward promotion requires a new cohort;
the replay holdout cannot be reused as independent live evidence.
