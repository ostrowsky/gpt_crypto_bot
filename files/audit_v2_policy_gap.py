from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v2_belief_action_calibration import _oracle_policy, _threshold_policy
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout
from v2.state import Action


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_policy_gap_audit_15m.json"


def build(history_root: Path, labels_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    oracle_rollouts = [rollout(env, _oracle_policy) for env in episodes]
    threshold_rollouts = [rollout(env, _threshold_policy(0.70, 0.70)) for env in episodes]
    payload = {
        "episodes": len(episodes),
        "policies": {
            "lifecycle_oracle": _summarize_trades(oracle_rollouts),
            "threshold_o0.70_s0.70": _summarize_trades(threshold_rollouts),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _summarize_trades(rollouts) -> dict:
    entries = Counter()
    exits = Counter()
    trades = []
    for steps in rollouts:
        active = None
        for idx, step in enumerate(steps):
            if step.action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
                entries[step.frame.label.state.value] += 1
                active = {
                    "entry_index": idx,
                    "reward_total": step.reward.total,
                }
            elif active is not None:
                active["reward_total"] += step.reward.total
                if step.action == Action.SELL:
                    exits[step.frame.label.state.value] += 1
                    trades.append(
                        {
                            "bars_held": idx - active["entry_index"] + 1,
                            "realized_pnl_reward": step.reward.realized_pnl_reward,
                            "giveback_penalty": step.reward.giveback_penalty,
                            "total_reward": active["reward_total"],
                        }
                    )
                    active = None
    count = len(trades) or 1
    return {
        "trade_count": len(trades),
        "entries_by_true_state": dict(sorted(entries.items())),
        "exits_by_true_state": dict(sorted(exits.items())),
        "avg_realized_pnl_reward": round(sum(t["realized_pnl_reward"] for t in trades) / count, 6),
        "avg_giveback_penalty": round(sum(t["giveback_penalty"] for t in trades) / count, 6),
        "avg_total_reward_per_trade": round(sum(t["total_reward"] for t in trades) / count, 6),
        "avg_bars_held": round(sum(t["bars_held"] for t in trades) / count, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["policies"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
