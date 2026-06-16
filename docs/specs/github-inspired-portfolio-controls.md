# GitHub-Inspired Portfolio Controls

Status: research-only replay harness

Last updated: 2026-06-16

## Problem

Recent 14d progress reporting shows the bot still captures final top movers, but
precision and exit quality are weak. Similar open-source trading bots emphasize
separate risk/protection layers, portfolio exposure controls, backtestable
protections, and robustness checks. This spec tests transferable portfolio-control
ideas without changing live BUY, SELL, alert, or replacement behavior.

## Objective Fit

- Better selection under the unified 10-position cap.
- Better preservation of good existing positions when a new candidate appears.
- Safer operation by keeping risk controls replay-gated before live use.

## Scope

Now:

- Add a replay-only variant, `score_replace_cluster_non_losing`, combining:
  - existing score replacement;
  - existing signal-cluster exposure caps;
  - a non-losing replacement block.
- Fix `bot-progress-report` handling of `null` rate fields so progress reports
  can be generated for incomplete quality windows.
- Remove the Render blueprint because Render migration is cancelled.

Not now:

- No production BUY, SELL, cooldown, replacement, alert, or Telegram behavior.
- No automatic parameter optimization or live adoption.
- No migration to Render or cloud runtime.

## Primary Metrics

- `pnl_total`
- `trade_precision`
- `symbol_precision`
- `capture_rate`
- `capture_ratio_at_entry`
- `exit_efficiency`
- `giveback_pct`
- `cooldown_harm_pct`
- `replacements_total`
- `replacement_policy_skipped`
- `skipped_cluster_cap`

## Backtest Evidence

All runs use:

```text
.\pyembed\python.exe .\files\replay_backtest.py --variant <variant> --top-gainer-score-min 34 --objective-top-n 15 --no-baseline --json
```

### 7d

| Variant | Trades | PnL total | Capture | Trade precision | Median capture | Median exit eff | Median giveback |
|---|---:|---:|---:|---:|---:|---:|---:|
| `score_replace` | 384 | -80.99% | 100.00% | 26.30% | 0.0751 | -0.3137 | 1.0625 |
| `protected_trailing_exit` | 356 | -57.03% | 100.00% | 26.69% | 0.0297 | -0.3333 | 1.3731 |
| `partial_profit_take` | 384 | -79.55% | 100.00% | 26.30% | 0.0751 | -0.3137 | 1.1061 |
| `score_replace_cluster` | 391 | -70.42% | 100.00% | 26.85% | 0.0021 | -0.4167 | 1.0433 |
| `replacement_block_non_losing` | 376 | -67.27% | 100.00% | 26.60% | 0.0680 | -0.3875 | 1.1001 |
| `chase_guard_rsi_82` | 384 | -80.30% | 100.00% | 27.60% | 0.0645 | -0.2857 | 1.0625 |
| `protected_weak_only` | 351 | -57.12% | 100.00% | 27.07% | 0.0297 | -0.3520 | 1.4290 |
| `score_replace_cluster_non_losing` | 384 | -64.88% | 100.00% | 27.08% | 0.0239 | -0.2941 | 1.0461 |

### 14d

| Variant | Trades | PnL total | Capture | Trade precision | Median capture | Median exit eff | Median giveback |
|---|---:|---:|---:|---:|---:|---:|---:|
| `score_replace` | 666 | -257.76% | 100.00% | 28.98% | 0.0000 | -0.4545 | 1.2883 |
| `protected_trailing_exit` | 593 | -263.27% | 100.00% | 29.34% | 0.0000 | -0.6333 | 1.6760 |
| `score_replace_cluster` | 666 | -246.97% | 100.00% | 29.28% | 0.0000 | -0.4821 | 1.2064 |
| `replacement_block_non_losing` | 673 | -238.28% | 100.00% | 29.12% | 0.0000 | -0.5000 | 1.2702 |
| `score_replace_cluster_non_losing` | 671 | -237.56% | 100.00% | 28.61% | 0.0000 | -0.5385 | 1.1789 |

## Decision

- Do not promote any tested idea to production.
- `protected_trailing_exit` is rejected for now: 7d looked better, but 14d
  worsened PnL and exit metrics.
- `partial_profit_take` is rejected for now: negligible improvement and worse
  giveback.
- `chase_guard_rsi_82` is rejected for now: slightly better precision but no
  meaningful PnL improvement and worse cooldown harm.
- Portfolio controls are worth further research:
  - `replacement_block_non_losing` improved 14d PnL from -257.76% to -238.28%.
  - `score_replace_cluster_non_losing` improved 14d PnL to -237.56%, but with
    weaker trade precision and exit efficiency.

## Acceptance Criteria

This research slice is complete when:

- replay-only variant is available through CLI;
- 7d and 14d baseline-vs-variant results are recorded;
- Render blueprint is removed;
- no live behavior changes are made;
- compile checks pass.

## Risks / Trade-Offs

- Portfolio controls can preserve mediocre positions and block better leaders.
- Cluster caps can reduce correlated exposure but may also block real sector-wide
  top-mover days.
- Non-losing replacement blocks improve churn control but can lower opportunity
  capture during fast rotations.

## Next Gate

Before any live adoption:

1. Run a 30d replay and split results by market regime.
2. Add a combined variant with cluster cap plus non-losing replacement plus a
   leader-delta override grid.
3. Require no loss of capture, better 30d PnL, and no degradation in trade
   precision or exit efficiency.

## Rollback

The replay-only variant is disabled by default. Remove
`score_replace_cluster_non_losing` from `files/replay_backtest.py` if the harness
is no longer useful.
