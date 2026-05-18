from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows, _threshold_policy
from audit_v2_policy_gap import _summarize_trades
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import lifecycle_oracle_policy, rollout, summarize_policy
from v2.state import Action


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_residual_gap_decomposition_15m.json"
PRODUCTIVE_STATES = {"emerging_move", "confirmed_trend"}
LATE_STATES = {"mature_trend", "exhaustion", "reversal"}


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    oracle_rollouts = [rollout(env, lifecycle_oracle_policy) for env in episodes]
    improved_rollouts = [rollout(env, _threshold_policy(rows, admission="combined")) for env in episodes]
    oracle_summary = summarize_policy("lifecycle_oracle", oracle_rollouts)
    improved_summary = summarize_policy("belief_plus_projected_v1_admission_policy", improved_rollouts)
    payload = {
        "episodes": len(episodes),
        "policies": {
            "lifecycle_oracle": {
                "summary": oracle_summary,
                "admission": _admission_mix(oracle_rollouts),
                "exit": _summarize_trades(oracle_rollouts),
            },
            "belief_plus_projected_v1_admission_policy": {
                "summary": improved_summary,
                "admission": _admission_mix(improved_rollouts),
                "exit": _summarize_trades(improved_rollouts),
            },
        },
        "reward_gap_vs_oracle": _reward_gap(oracle_summary, improved_summary),
    }
    payload["dominant_next_bottleneck"] = _dominant_bottleneck(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _admission_mix(rollouts_) -> dict:
    counts = Counter()
    for steps in rollouts_:
        for step in steps:
            if step.action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
                counts[step.frame.label.state.value] += 1
    total = sum(counts.values()) or 1
    productive = sum(counts[state] for state in PRODUCTIVE_STATES)
    late = sum(counts[state] for state in LATE_STATES)
    noise = counts["noise"]
    return {
        "entries_by_true_state": dict(sorted(counts.items())),
        "productive_entry_share": round(productive / total, 6),
        "noise_entry_share": round(noise / total, 6),
        "late_entry_share": round(late / total, 6),
    }


def _reward_gap(oracle: dict, improved: dict) -> dict:
    out = {}
    for key, oracle_value in oracle["reward_components"].items():
        out[key] = round(improved["reward_components"][key] - oracle_value, 6)
    out["total_reward"] = round(improved["total_reward"] - oracle["total_reward"], 6)
    return out


def _dominant_bottleneck(payload: dict) -> dict:
    improved = payload["policies"]["belief_plus_projected_v1_admission_policy"]
    oracle = payload["policies"]["lifecycle_oracle"]
    reward_gap = payload["reward_gap_vs_oracle"]
    admission_noise_gap = improved["admission"]["noise_entry_share"] - oracle["admission"]["noise_entry_share"]
    giveback_gap = reward_gap["giveback_penalty"]
    false_buy_gap = reward_gap["false_buy_penalty"]
    realized_gap = reward_gap["realized_pnl_reward"]
    if abs(giveback_gap) > abs(false_buy_gap) and abs(giveback_gap) > abs(realized_gap):
        label = "exit_monetization"
    else:
        label = "residual_admission"
    return {
        "label": label,
        "noise_entry_share_gap": round(admission_noise_gap, 6),
        "giveback_penalty_gap": giveback_gap,
        "false_buy_penalty_gap": false_buy_gap,
        "realized_pnl_reward_gap": realized_gap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["dominant_next_bottleneck"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
