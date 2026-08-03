# BTC Early Trend WATCH

Status: retired after failed live-forward gate; broad V2 shadow remains enabled
Last updated: 2026-08-03

## Problem

On 2026-07-17 the live V2 observer detected a closed-candle BTC transition to
`emerging_move` at 17:46:47 UTC, but no operator signal was sent. General V2
realtime Telegram is disabled because its broad upside precision is too low.
The main V1 path had independently produced a strong raw candidate score, but
its same-day Top-Gainer quality score correctly kept the BUY blocked.

## Behavior

Send a separate yellow operator WATCH only when all conditions hold:

- symbol is exactly `BTCUSDT` and timeframe is exactly `15m`;
- a non-bootstrap V2 material transition emits `emerging_move`,
  `elevate_priority`, and `early positive structure` from a closed candle;
- within the preceding 30 minutes, V1 logged a `top_gainer_score_gate` block
  for BTC 15m with raw `candidate_score >= 60`;
- the exact V2 event has not already been sent.

The alert must explicitly say WATCH and not BUY. It must not open a position,
weaken a gate, change ranking, enable broad V2 realtime Telegram, or affect
portfolio state. On worker restart, an eligible event up to 90 minutes old may
be caught up once; persistent event-key deduplication prevents duplicates.

## Maximum-Available Replay

The conjunction was evaluated across the complete available BTC V2 event
journal from 2026-05-17 through 2026-07-17. Each V2 early-positive 15m event
was paired only with a preceding V1 raw candidate of at least 60 within 30
minutes, then labeled causally from 15m candles. There were 47 paired events.
The chronological split used the first 30 observations as train and the final
17 as holdout.

Six-hour forward return (`24 x 15m`):

- all: average `+0.1152%`, median `+0.0503%`, positive `57.45%`;
- train: average `+0.1140%`, median `+0.0303%`, positive `56.67%`;
- holdout: average `+0.1174%`, median `+0.1352%`, positive `58.82%`.

Twelve-hour forward return (`48 x 15m`):

- all: average `+0.1351%`, median `+0.2844%`, positive `57.45%`;
- train: average `+0.0207%`, median `+0.2153%`, positive `56.67%`;
- holdout: average `+0.3372%`, median `+0.6685%`, positive `58.82%`.

The wider V2 stream and stricter raw-score thresholds were not promoted: the
general stream retains poor precision, while thresholds 70/80 were less stable
across the chronological split. The raw-60 conjunction is promoted only as an
operator WATCH because both train and holdout have positive average/median
trend-horizon return and above-50% positive rate.

## Verification and Rollback

Focused tests cover the exact V2 profile, causal V1 lookback, score threshold,
fresh catch-up, persistent deduplication, and WATCH/not-BUY wording.

Set `V2_BTC_TREND_WATCH_TELEGRAM_ENABLED=False` to stop these alerts without
disabling V2 shadow collection or changing trading behavior.

## Live-Forward Decision — 2026-08-03

The post-deployment cohort contained 11 independent episodes after a 12-hour
deduplication window. Six-hour return averaged `-0.1788%`, median was
`-0.2082%`, and only `36.36%` were positive. Twelve-hour average was only
`+0.0068%` (10 mature cases), despite a positive median, so the original replay
edge did not survive independent forward observation. The dedicated Telegram
WATCH is disabled by default. This retirement does not disable the broad V2
observer and does not change BUY admission.
