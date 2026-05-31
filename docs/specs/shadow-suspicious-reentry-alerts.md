# Shadow Suspicious Re-entry Alerts

Date: 2026-05-31
Status: shadow-only live telemetry; no production BUY/SELL changes

## Problem

Exit monetization is the current bottleneck. Broad replay showed that the isolated `baseline_suspicious_reentry` policy improves 14d and 30d full-watchlist replay, but it must not be promoted directly to live trading.

## Goal

Emit Telegram shadow alerts when the live bot would have bypassed cooldown after a suspicious exit and a normal continuation candidate appears.

## Non-goals

- Do not open real positions.
- Do not bypass cooldown in production.
- Do not change SELL logic.
- Do not change portfolio accounting.

## Live Shadow Flow

1. A real SELL happens.
2. The monitor scores the exit as potentially suspicious using exit reason, realized PnL, MFE proxy, mode, and bars held.
3. If score passes threshold, the symbol gets a short shadow re-entry watch window.
4. During cooldown, if a normal entry candidate appears and passes stricter confirmation gates, Telegram receives a `SHADOW RE-ENTRY` alert.
5. The alert is logged to `bot_events.jsonl` for post-factum scoring.

## Promotion Gate

After several days, compare shadow alerts against actual forward returns and daily top-mover outcomes. Only if shadow alerts remain positive under replay + live shadow evaluation may a production re-entry spec be proposed.
