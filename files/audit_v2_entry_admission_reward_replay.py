from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy
from v2.state import Action, SymbolState


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_reward_replay_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    admission_rows = _load_admission_rows(dataset_path)
    policies = {
        "threshold_policy_base": _threshold_policy(admission_rows, admission="none"),
        "belief_admission_policy": _threshold_policy(admission_rows, admission="belief"),
        "belief_plus_projected_v1_admission_policy": _threshold_policy(admission_rows, admission="combined"),
    }
    rollouts = {name: [rollout(env, policy) for env in episodes] for name, policy in policies.items()}
    summaries = {name: summarize_policy(name, runs) for name, runs in rollouts.items()}
    entry_state_mix = {name: _entry_state_mix(runs) for name, runs in rollouts.items()}
    base_reward = summaries["threshold_policy_base"]["total_reward"]
    payload = {
        "episodes": len(episodes),
        "policies": {
            name: {
                "summary": summaries[name],
                "entry_state_mix": entry_state_mix[name],
                "delta_vs_base_reward": round(summaries[name]["total_reward"] - base_reward, 6),
            }
            for name in policies
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_admission_rows(path: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        out[(str(row["symbol"]), int(row["ts_ms"]))] = row
    return out


def _threshold_policy(admission_rows: dict[tuple[str, int], dict], *, admission: str):
    def policy(env):
        frame = env.current_frame
        probs = frame.belief.probabilities
        open_mass = probs[SymbolState.EMERGING_MOVE] + probs[SymbolState.CONFIRMED_TREND]
        sell_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
        if env.position is None:
            if open_mass < 0.70:
                return Action.IGNORE
            if admission == "none":
                return Action.OPEN_FULL
            row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
            if not row:
                return Action.IGNORE
            if admission == "belief":
                return Action.OPEN_FULL
            projected = row.get("v1_projected_structural") or {}
            combined_open_mass = float(row["belief"].get("emerging_move", 0.0)) + float(
                row["belief"].get("confirmed_trend", 0.0)
            )
            leader = float(projected.get("projected_leader_score_trend", 0.0))
            return Action.OPEN_FULL if combined_open_mass >= 0.50 and leader >= 3.0 else Action.IGNORE
        return Action.SELL if sell_mass >= 0.70 else Action.HOLD

    return policy


def _entry_state_mix(rollouts_) -> dict:
    counts = Counter()
    for steps in rollouts_:
        for step in steps:
            if step.action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
                counts[step.frame.label.state.value] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["policies"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
