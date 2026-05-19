# V2 Action-Level Exit Advantage Dataset

Status: research-only  
Last updated: 2026-05-19

## Purpose

The position-aware exit selector showed that manual threshold overrides are not
enough. The next step is to build an action-level ground truth for exit decisions.

Instead of asking:

```text
which full policy is better over a horizon?
```

this dataset asks:

```text
while already in a position, is SELL now better than continuing with the current
candidate hold/exit path?
```

## Target

For each in-position frame under the fixed temporal candidate policy:

```text
sell_now_reward = reward if the bot exits at this frame
continuation_reward = realized reward from candidate policy from this frame until exit
sell_advantage = sell_now_reward - continuation_reward
```

Labels:

- `sell_advantage_positive`: `sell_advantage > 0`
- `sell_advantage_strong`: `sell_advantage >= 1.0`
- `hold_advantage_strong`: `sell_advantage <= -1.0`

## Features

Use only causal features available at the decision frame:

- symbol belief masses;
- v1 projected structure;
- temporal deltas;
- position context:
  - bars held;
  - unrealized PnL;
  - MFE;
  - giveback;
  - current lifecycle state for offline analysis only.

## Protocol

1. Build offline episodes from the v2 admission dataset.
2. Roll out the fixed temporal candidate policy.
3. Track the open position state for every frame.
4. For every in-position frame before/at exit, compute sell-now reward and
   continuation-to-exit reward.
5. Save JSONL rows and an audit summary.

## Acceptance Criteria

This package is diagnostic/data only. It passes if:

- rows are produced for in-position frames;
- class balance is reported;
- top positive / negative examples are inspectable;
- feature coverage is reported.

No live behavior or Telegram signal is authorized by this dataset alone.


## Audit Result

Latest corrected build produced:

- rows: `99,935` in-position decision frames;
- output: `.runtime/reports/v2_action_level_exit_advantage_15m.jsonl`;
- audit: `.runtime/reports/v2_action_level_exit_advantage_audit_15m.json`;
- `sell_positive`: `50,002` rows (`50.03%`);
- `sell_strong`: `33,309` rows (`33.33%`);
- `hold_strong`: `31,598` rows (`31.62%`);
- feature coverage: `100%` for the selected causal feature set.

The corrected target includes the candidate policy's realized per-step rewards
from the current in-position frame through the eventual exit. This matters: an
exit label that only compares against the final SELL reward would miss the real
path-dependent benefit or damage of continuing to hold.

## Interpretation

The dataset is now useful as a supervised/offline-learning target for exit logic:

- it is action-level rather than whole-policy-level;
- it directly compares `SELL now` against the realized `HOLD / candidate-exit`
  continuation path;
- it contains both strong sell and strong hold examples, instead of a collapsed
  one-sided label;
- examples show the intended contrast:
  - early `emerging_move` positions where holding was much better;
  - mature / overextended positions where selling immediately avoided large
    subsequent loss.

This does **not** authorize live behavior, Telegram alerts, or production SELL
changes. It only creates the ground truth needed for the next offline gate.

## Decision

Advance to a chronological baseline model:

1. train a simple transparent predictor for `sell_advantage` / `sell_strong`;
2. evaluate it on a time-forward holdout split;
3. replay a position-aware exit policy driven by the prediction;
4. compare against the fixed temporal candidate and current base policy.

Promotion is blocked until the learned exit policy beats the fixed candidate in
full offline replay, not merely on classification accuracy.
