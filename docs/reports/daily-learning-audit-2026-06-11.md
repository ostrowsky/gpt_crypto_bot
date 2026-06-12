# Daily Learning Report Audit — 2026-06-11

Date: 2026-06-12
Status: research-only audit; no production BUY/SELL/rotation changes

## Objective Contract

The bot's primary objective remains early capture of same-day watchlist top movers
under the portfolio cap. Decisions must improve or protect:

- watchlist-filtered top-mover capture;
- early capture / remaining daily move after entry;
- false-positive BUY control;
- exit monetization / MFE retention;
- portfolio replacement quality.

## Report Validity

The 2026-06-11 daily learning report is internally consistent, but correctly low
confidence:

- `watchlist_top_count=1`, `bought=1`, `early=0` means daily capture was 100%,
  but daily early capture was 0% on a denominator of one.
- rolling early capture fell from 36.19% to 12.5%, but the last 7-day window has
  only 4 days with positive watchlist-top denominator.
- candle coverage is partial (`186/206`), but triage marks missing series as
  inactive `BREAK` symbols, so this is a warning rather than a full blocker.
- broad trend-start `miss_rate=82.66%` is not the same denominator as watchlist
  top-mover capture and should be read as a separate funnel-health warning.

## Historical Checks Run

### Portfolio replacement

Command:

```powershell
pyembed\python.exe files\report_portfolio_replacement_shadow_reward.py --json --no-save
```

Coverage:

- events loaded: 1,138,000;
- replacement events: 274;
- closed incoming outcomes: 268.

Result:

- average replacement delta: -0.0879%;
- median replacement delta: -0.3835%;
- positive delta rate: 37.69%;
- replaced non-losing positions are especially harmful: avg delta -0.3159%.

Decision:

- keep live replacement behavior unchanged unless explicitly gated;
- keep `AGENT_REPLACEMENT_BLOCK_NON_LOSING_ENABLED=False`;
- keep shadow monitoring active;
- do not add new replacement relaxations before rotation outcome analysis.

### Chase guard

Existing behavior replay report reviewed:

- `docs/reports/chase-guard-behavior-replay-2026-06-03.md`.

Replay result over 14d / full watchlist rejected all chase-guard relaxations:

- `chase_guard_off`: ΔPnL -27.8493 vs baseline;
- `chase_guard_rsi_off`: ΔPnL -36.1160;
- `chase_guard_rsi_82`: ΔPnL -35.8748.

The blocker reward table for 2026-06-11 shows `chase_guard` with only
`net_harm=0.75%` after protection credit. This is a weak blocker candidate, not a
production relaxation signal.

Decision:

- do not relax chase guard;
- only targeted behavior replay is allowed;
- daily report wording must not call weak net-harm blockers simply "harmful".

## Code Change

`files/learning_progress_report.py` now distinguishes weak vs strong blocker
candidates in next actions:

- weak if `net_harm_pct < 5.0`;
- strong otherwise.

This is observability-only and does not affect trading logic.

## Verification

```powershell
cd D:\Projects\gpt_crypto_bot\files
..\pyembed\python.exe test_learning_progress_report.py
```

```powershell
pyembed\python.exe files\test_market_signal_agent_replacement_policy.py
pyembed\python.exe files\test_replay_chase_guard_variants.py
pyembed\python.exe files\test_report_portfolio_replacement_shadow_reward.py
pyembed\python.exe files\test_report_entry_admission_shadow_reward.py
pyembed\python.exe files\test_replay_observable_tail_selector.py
```

All tests passed.

## Next Priority

1. Keep portfolio replacement in shadow and analyze skipped/kept rotation
   outcomes before enabling any policy.
2. Continue exit monetization search; current tail selector does not pass gate.
3. Treat early-capture degradation as a low-confidence warning until more
   positive-denominator watchlist-top days accumulate.
4. Resolve runtime discipline: live processes currently run from
   `D:\Projects\gpt_crypto_bot`, while `D:\Projects\gpt_crypto_bot_release`
   reports are stale.
