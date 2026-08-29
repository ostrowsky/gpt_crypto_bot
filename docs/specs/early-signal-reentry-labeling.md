# Early Signal And Re-entry Labeling

Status: production notification correctness repair
Date: 2026-08-29

## Problem

On 2026-08-29 `ICPUSDT` was detected before the reported start of its move:

- `01:48` Europe/Budapest: closed-bar discovery routed a 15m breakout;
- `01:50`: the production score gate rejected the BUY (`31.40 < 34.00`), and
  Telegram confirmed delivery of the existing strong blocked-signal alert at
  price `2.392`;
- `06:05`: the first accepted BUY was delivered at `2.424`;
- `09:16`, `10:16`, and `11:49`: later entries followed same-day exits.

The 11:49 message was formatted as a generic BUY, so an operator could
reasonably read it as the first discovery of the trend. The blocked-signal
message was also English and did not prominently say that it was the early
growth alert while automatic entry remained prohibited.

## Required behavior

- A high-quality candidate blocked only by the top-gainer score gate keeps the
  existing admission thresholds and remains non-trading.
- Its Telegram notification is headed `EARLY GROWTH SIGNAL` in Russian and
  explicitly says that the bot detected the move but did not open a position.
- The first accepted entry for a symbol/local day is labeled `BUY SIGNAL`.
- A second or later accepted entry for the same symbol/local day is labeled
  `RE-ENTRY`, includes its ordinal number, and references the first accepted
  entry time.
- Entry-alert history is bounded to the current local day and has no effect on
  BUY/SELL, scoring, cooldown, portfolio capacity, or exits.
- Restarting the process may lose notification-only entry history; it must not
  reconstruct a trading decision from mutable logs.

## Replay decision

The tempting trading hypothesis was frozen as a 15m breakout rescue with
`candidate_score >= 130`, live top-gainer score `[31, 34)`, and `vol_x >= 4`.
The maximum available mature event/candle cohort was 2026-06-28..2026-08-28:

| Segment | Eligible | Top movers | Precision | T+10 avg | T+10 median |
|---|---:|---:|---:|---:|---:|
| all mature | 41 | 1 | 2.44% | -0.43% | -0.62% |
| chronological holdout | 18 | 1 | 5.56% | -0.68% | -0.86% |
| recent stability | 4 | 0 | 0.00% | -1.05% | -1.12% |

Decision: reject the BUY rescue and retain the production score floor `34`.
The incident supports clearer early WATCH/re-entry semantics, not a trading
relaxation.

## Historical confirmation audit

The maximum available completed critic window for delivered early alerts is
2026-06-27..2026-08-28. The audit uses the first delivered alert per local
day/symbol as the opportunity unit, so Telegram retries and repeated bars do
not inflate the denominator. Of 3,825 delivery-`ok` rows, 3,614 could be
causally joined to a blocked-score label (94.5% coverage), yielding 1,916
unique opportunities across 62 local days.

Three meanings of confirmation must remain separate:

- operational confirmation: a normal production BUY followed the early alert;
- objective confirmation: the symbol finished in the final watchlist top set;
- forward market confirmation: T+10 net return after 20 bps costs was positive.

Results:

| Class | Opportunities | Mature T+10 | T+10 positive | T+10 avg | T+10 median |
|---|---:|---:|---:|---:|---:|
| later BUY + final top | 63 | 58 | 55.2% | +0.83% | +0.37% |
| later BUY + not final top | 759 | 710 | 49.6% | +0.13% | -0.03% |
| no later BUY + final top | 12 | 8 | 50.0% | +0.66% | -0.37% |
| no later BUY + not final top | 1,082 | 979 | 23.3% | -0.62% | -0.62% |

Therefore 822/1,916 (42.9%) were later accepted by the normal BUY path, but
only 63/1,916 (3.29%) were both operationally and objectively confirmed.
Among the 1,755 mature opportunities, only 32 (1.82%) met all three tests.
The operational confirmation has information value, but it is not permission
to relabel the original alert as a proven trade.

### Replay-only hypothesis

An early alert may move from `EARLY` to `CONFIRMED BY CURRENT BUY POLICY` only
when the unchanged production BUY path subsequently accepts the same
symbol/local day. This is a message-state transition at the later timestamp,
not a backdated claim and not an alternative entry. It must retain a separate
`final_top=unknown`/`T+10=pending` status until those labels mature.

The narrower causal slice `trend|strong_trend|retest` with live score `[32,34)`
is eligible for a prospective shadow test, not production promotion. Its later
BUY conversion was 163/343 (47.5%) overall and 47/90 (52.2%) on the
chronological holdout, but holdout final-top precision was only 3/90 (3.3%)
and mature T+10 median remained -0.20%. Promotion therefore requires a new
one-bar persistence discriminator to improve final-top precision and T+10
without increasing alert rate; the existing features alone fail that gate.

## Acceptance criteria

- focused tests prove first-entry vs same-day re-entry classification;
- a new local day resets the ordinal to one;
- the early blocked-signal alert remains one-shot per bar and explicitly says
  that no position was opened;
- no trading-policy/config threshold changes;
- Truth Harness staged-change review and `git diff --check` complete.

## Rollback

Revert the notification helper and message wording. Trading behavior is
unchanged, so rollback cannot alter positions.

## Canary verification

After restart, inspect the first eligible blocked-score alert and the first
same-day re-entry in `bot_events.jsonl`: Telegram delivery must be `ok`, the
early alert must remain non-trading, and the re-entry label must agree with the
accepted entry sequence. If wording or ordinals are wrong, use the rollback
above; do not change score or admission thresholds as a workaround.
