# V2 V1 Market-Structure Feature Audit

Status: research-only  
Last updated: 2026-05-19

## Purpose

Before building a new market-environment model from scratch, audit whether v1
already contains reusable market-structure observations and accumulated data.

The target is not to copy v1 BUY logic into v2. The target is to reuse v1's
validated *measurements* where they improve v2's ability to infer which policy
family should be active.

## Hypothesis

v1 has accumulated useful market-structure signals in two places:

1. the watchlist top-gainer model feature schema / weights;
2. `ml_dataset.jsonl` rows containing per-symbol structure, BTC context,
   sequence trends, and signal-mode flags.

If those features are useful, a selected subset should improve causal horizon
policy-advantage classification versus the current broad feature set.

## Candidate V1 Feature Families

- trend structure:
  - EMA stack / close-vs-EMA;
  - slope;
  - ADX;
  - RSI;
  - MACD histogram;
  - ATR / daily range.
- participation / activity:
  - `vol_x`;
  - body / wick proportions;
  - sequence trend / tail features.
- market context:
  - `btc_vs_ema50`;
  - `btc_momentum_4h`;
  - `market_vol_24h`.
- v1 decision context:
  - signal mode flags;
  - projected candidate / leader scores already ported into v2 admission rows.

## Diagnostic Protocol

1. Inspect `files/watchlist_top_gainer_model.json` for feature names and largest
   absolute model weights.
2. Inspect `files/ml_dataset.jsonl` for row count, feature coverage, and time
   span.
3. Evaluate causal horizon classification using selected feature groups from the
   current v2 market-breadth samples:
   - v1 projected structure;
   - v1 projected trend deltas;
   - v1 row-breadth aggregates;
   - OHLCV market breadth;
   - BTC / ETH risk context;
   - simple greedy feature selection.

## Acceptance Criteria

This audit does not authorize switched policy or RL.

A v1-derived feature subset is promising only if it beats the majority baseline
by at least `3pp` on `1h` or `2h` horizon classification with meaningful coverage.

If no subset passes, the result still constrains the next work: v1 features can
remain useful primitives, but v2 needs better target/modeling or more causal data.


## Audit Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_v1_market_structure_features.py --json
```

### V1 reusable signal inventory

The v1 watchlist top-gainer model contains `60` features, of which `49` are
market-structure related by name/schema. The strongest structure weights include:

| Feature | Weight |
|---|---:|
| `btc_vs_ema50` | `-0.184128` |
| `daily_range` | `0.138813` |
| `close_vs_ema200` | `0.133966` |
| `close_vs_ema20` | `0.123564` |
| `close_vs_ema50` | `0.119290` |
| `signal_alignment` | `-0.105937` |
| `slope` | `0.099746` |
| `seq_last_slope` | `0.099746` |

`files/ml_dataset.jsonl` is present and contains accumulated v1 data:

- rows: `6279`;
- time span: `2024-07-22T01:00:00Z -> 2026-05-18T13:45:00Z`;
- timeframes: `15m=3619`, `1h=2660`;
- structure features with near/full coverage: `49`.

This confirms that v1 has reusable measurement history. It should be used as a
feature source, but not as an unexamined policy.

### Causal feature-selection result

Broad groups still do not pass the gate, but small selected subsets do:

| Horizon | Selected features | Accuracy | Majority | Edge | Verdict |
|---|---|---:|---:|---:|---|
| `1h` | `market_ret4_positive_share`, `prefix_projected_leader_score_trend` | `0.649718` | `0.608939` | `+0.040779` | pass |
| `2h` | `market_btc_ret4_pct`, `market_volume_gt_mean20_share` | `0.680000` | `0.636872` | `+0.043128` | pass |

Important caution: this is still feature selection on the same causal sample, not
a final policy validation. The result authorizes a switched-replay gate, not live
usage.

## Interpretation

The result supports the user's thesis: v2 should not ignore v1. The useful
features are not the full v1 model wholesale; they are compact structural and
market-context measurements:

- market participation over the last hour;
- v1 projected leader score as a symbol-structure aggregate;
- BTC short-horizon context;
- market volume expansion breadth.

The broad feature sets underperform because they mix useful state with too much
noise. The v2 belief model should therefore prefer selected / regularized
observation sets over naive ?add every feature? expansion.

## Decision

Promote to the next research gate only:

```text
selected-feature market-environment switched replay
```

No BUY / SELL / Telegram behavior changes are authorized.
