from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_temporal_exit_window_stability_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    ordered = sorted(episodes, key=lambda env: env.current_frame.label.local_day)
    windows = _split_windows(ordered, parts=4)

    aggregate = _summarize_window("aggregate", ordered, rows, temporal_rows)
    window_payloads = [
        _summarize_window(f"window_{idx + 1}", window, rows, temporal_rows)
        for idx, window in enumerate(windows)
    ]
    wins = [item for item in window_payloads if item["delta_vs_base_reward"] > 0]
    payload = {
        "candidate": {
            "late_mass_floor": 0.50,
            "mature_delta_max": -0.20,
            "late_mass_delta_min": 0.10,
        },
        "aggregate": aggregate,
        "windows": window_payloads,
        "stability": {
            "winning_windows": len(wins),
            "total_windows": len(window_payloads),
            "winning_share": round(len(wins) / len(window_payloads), 6) if window_payloads else 0.0,
            "passes_majority_gate": len(wins) >= 3 and aggregate["delta_vs_base_reward"] > 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _summarize_window(name: str, episodes, rows, temporal_rows) -> dict:
    base_runs = [rollout(env, _base_policy(rows)) for env in episodes]
    candidate_runs = [rollout(env, _grid_policy(rows, temporal_rows, -0.20, 0.10)) for env in episodes]
    base = summarize_policy("base_sell_0_70", base_runs)
    candidate = summarize_policy("mature_decay_0_20_late_rise_0_10", candidate_runs)
    days = sorted({env.current_frame.label.local_day for env in episodes})
    return {
        "name": name,
        "episodes": len(episodes),
        "start_day": days[0] if days else None,
        "end_day": days[-1] if days else None,
        "base_reward": base["total_reward"],
        "candidate_reward": candidate["total_reward"],
        "delta_vs_base_reward": round(candidate["total_reward"] - base["total_reward"], 6),
        "base_trade_count": base["trade_count"],
        "candidate_trade_count": candidate["trade_count"],
    }


def _split_windows(items: list, *, parts: int) -> list[list]:
    n = len(items)
    base, rem = divmod(n, parts)
    windows = []
    start = 0
    for idx in range(parts):
        size = base + (1 if idx < rem else 0)
        windows.append(items[start : start + size])
        start += size
    return windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["stability"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
