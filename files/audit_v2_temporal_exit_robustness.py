from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_policy_gap import _summarize_trades
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy
from v2.state import Action, SymbolState


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_temporal_exit_robustness_15m.json"
MATURE_THRESHOLDS = (-0.10, -0.15, -0.20)
LATE_RISE_THRESHOLDS = (0.05, 0.10, 0.15)


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    out = {}

    base_runs = [rollout(env, _base_policy(rows)) for env in episodes]
    out["base_sell_0_70"] = {
        "summary": summarize_policy("base_sell_0_70", base_runs),
        "exit": _summarize_trades(base_runs),
    }
    base_reward = out["base_sell_0_70"]["summary"]["total_reward"]

    grid = []
    for mature_threshold in MATURE_THRESHOLDS:
        for late_rise_threshold in LATE_RISE_THRESHOLDS:
            name = _profile_name(mature_threshold, late_rise_threshold)
            runs = [
                rollout(env, _grid_policy(rows, temporal_rows, mature_threshold, late_rise_threshold))
                for env in episodes
            ]
            summary = summarize_policy(name, runs)
            payload = {
                "summary": summary,
                "exit": _summarize_trades(runs),
                "thresholds": {
                    "late_mass_floor": 0.50,
                    "mature_delta_max": mature_threshold,
                    "late_mass_delta_min": late_rise_threshold,
                },
                "delta_vs_base_reward": round(summary["total_reward"] - base_reward, 6),
            }
            out[name] = payload
            grid.append(
                {
                    "policy": name,
                    "thresholds": payload["thresholds"],
                    "total_reward": summary["total_reward"],
                    "delta_vs_base_reward": payload["delta_vs_base_reward"],
                }
            )

    grid.sort(key=lambda item: (item["thresholds"]["mature_delta_max"], item["thresholds"]["late_mass_delta_min"]))
    positive = [item for item in grid if item["delta_vs_base_reward"] > 0]
    center = next(
        item
        for item in grid
        if item["thresholds"]["mature_delta_max"] == -0.15
        and item["thresholds"]["late_mass_delta_min"] == 0.10
    )
    best = max(grid, key=lambda item: item["delta_vs_base_reward"])
    worst = min(grid, key=lambda item: item["delta_vs_base_reward"])
    payload = {
        "episodes": len(episodes),
        "base_reward": base_reward,
        "policies": out,
        "grid": grid,
        "robustness": {
            "positive_cells": len(positive),
            "total_cells": len(grid),
            "positive_share": round(len(positive) / len(grid), 6),
            "center": center,
            "best": best,
            "worst": worst,
            "locally_robust": len(positive) >= 5 and center["delta_vs_base_reward"] > 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _base_policy(admission_rows):
    def policy(env):
        frame = env.current_frame
        probs = frame.belief.probabilities
        open_mass = probs[SymbolState.EMERGING_MOVE] + probs[SymbolState.CONFIRMED_TREND]
        if env.position is None:
            if open_mass < 0.70:
                return Action.IGNORE
            row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
            if not row:
                return Action.IGNORE
            projected = row.get("v1_projected_structural") or {}
            admission_mass = float(row["belief"].get("emerging_move", 0.0)) + float(
                row["belief"].get("confirmed_trend", 0.0)
            )
            leader = float(projected.get("projected_leader_score_trend", 0.0))
            return Action.OPEN_FULL if admission_mass >= 0.50 and leader >= 3.0 else Action.IGNORE
        late_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
        return Action.SELL if late_mass >= 0.70 else Action.HOLD

    return policy


def _grid_policy(admission_rows, temporal_rows, mature_threshold: float, late_rise_threshold: float):
    def policy(env):
        frame = env.current_frame
        probs = frame.belief.probabilities
        open_mass = probs[SymbolState.EMERGING_MOVE] + probs[SymbolState.CONFIRMED_TREND]
        if env.position is None:
            if open_mass < 0.70:
                return Action.IGNORE
            row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
            if not row:
                return Action.IGNORE
            projected = row.get("v1_projected_structural") or {}
            admission_mass = float(row["belief"].get("emerging_move", 0.0)) + float(
                row["belief"].get("confirmed_trend", 0.0)
            )
            leader = float(projected.get("projected_leader_score_trend", 0.0))
            return Action.OPEN_FULL if admission_mass >= 0.50 and leader >= 3.0 else Action.IGNORE
        late_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
        temporal = temporal_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
        if not temporal.get("has_3bar_history"):
            return Action.HOLD
        sell = (
            late_mass >= 0.50
            and float(temporal["mature_delta_3"]) <= mature_threshold
            and float(temporal["late_mass_delta_3"]) >= late_rise_threshold
        )
        return Action.SELL if sell else Action.HOLD

    return policy


def _profile_name(mature_threshold: float, late_rise_threshold: float) -> str:
    mature = f"{abs(mature_threshold):.2f}".replace(".", "_")
    late = f"{late_rise_threshold:.2f}".replace(".", "_")
    return f"mature_decay_{mature}_late_rise_{late}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["robustness"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
