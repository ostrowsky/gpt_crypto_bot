from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_policy_gap import _summarize_trades
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy
from v2.state import Action, SymbolState


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_exit_quality_baselines_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    policies = {
        "base_sell_0_70": _policy(rows, profile="base_sell_0_70"),
        "early_sell_0_60": _policy(rows, profile="early_sell_0_60"),
        "exhaustion_sensitive": _policy(rows, profile="exhaustion_sensitive"),
        "reversal_sensitive": _policy(rows, profile="reversal_sensitive"),
        "hybrid_peak_guard": _policy(rows, profile="hybrid_peak_guard"),
    }
    out = {}
    for name, policy in policies.items():
        runs = [rollout(env, policy) for env in episodes]
        out[name] = {
            "summary": summarize_policy(name, runs),
            "exit": _summarize_trades(runs),
        }
    base_reward = out["base_sell_0_70"]["summary"]["total_reward"]
    for value in out.values():
        value["delta_vs_base_reward"] = round(value["summary"]["total_reward"] - base_reward, 6)
    best = max(out.items(), key=lambda item: item[1]["summary"]["total_reward"])
    payload = {
        "episodes": len(episodes),
        "policies": out,
        "selection": {"best_total_reward": {"policy": best[0], **best[1]}},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _policy(admission_rows: dict[tuple[str, int], dict], *, profile: str):
    def policy(env):
        frame = env.current_frame
        probs = frame.belief.probabilities
        open_mass = probs[SymbolState.EMERGING_MOVE] + probs[SymbolState.CONFIRMED_TREND]
        late_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
        if env.position is None:
            if open_mass < 0.70:
                return Action.IGNORE
            row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
            if not row:
                return Action.IGNORE
            projected = row.get("v1_projected_structural") or {}
            combined_open_mass = float(row["belief"].get("emerging_move", 0.0)) + float(
                row["belief"].get("confirmed_trend", 0.0)
            )
            leader = float(projected.get("projected_leader_score_trend", 0.0))
            return Action.OPEN_FULL if combined_open_mass >= 0.50 and leader >= 3.0 else Action.IGNORE
        if profile == "base_sell_0_70":
            sell = late_mass >= 0.70
        elif profile == "early_sell_0_60":
            sell = late_mass >= 0.60
        elif profile == "exhaustion_sensitive":
            sell = probs[SymbolState.EXHAUSTION] >= 0.35 or late_mass >= 0.60
        elif profile == "reversal_sensitive":
            sell = probs[SymbolState.REVERSAL] >= 0.30 or late_mass >= 0.60
        elif profile == "hybrid_peak_guard":
            favorable_move = 0.0
            if env.position is not None:
                favorable_move = ((env.position.peak_price / env.position.entry_price) - 1.0) * 100.0
            sell = late_mass >= 0.60 or (favorable_move >= 2.0 and late_mass >= 0.45)
        else:
            raise ValueError(f"unknown exit profile: {profile}")
        return Action.SELL if sell else Action.HOLD

    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
