from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy
from v2.state import Action, SymbolState


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_belief_action_calibration_15m.json"


def build(history_root: Path, labels_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    naive = summarize_policy("belief_policy_v1", [rollout(env, _naive_policy) for env in episodes])
    oracle = summarize_policy("lifecycle_oracle", [rollout(env, _oracle_policy) for env in episodes])
    variants = []
    for open_threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        for sell_threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
            policy = _threshold_policy(open_threshold, sell_threshold)
            summary = summarize_policy(
                f"threshold_o{open_threshold:.2f}_s{sell_threshold:.2f}",
                [rollout(env, policy) for env in episodes],
            )
            variants.append(
                {
                    "open_threshold": open_threshold,
                    "sell_threshold": sell_threshold,
                    "summary": summary,
                    "delta_vs_naive": round(summary["total_reward"] - naive["total_reward"], 6),
                    "delta_vs_oracle": round(summary["total_reward"] - oracle["total_reward"], 6),
                }
            )
    best_reward = max(variants, key=lambda item: item["summary"]["total_reward"])
    payload = {
        "episodes": len(episodes),
        "baselines": {"belief_policy_v1": naive, "lifecycle_oracle": oracle},
        "variants": variants,
        "selection": {"best_reward": _key(best_reward)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _naive_policy(env):
    state = env.current_frame.prediction
    if env.position is None:
        return Action.OPEN_FULL if state in {SymbolState.EMERGING_MOVE, SymbolState.CONFIRMED_TREND} else Action.IGNORE
    return Action.SELL if state in {SymbolState.EXHAUSTION, SymbolState.REVERSAL} else Action.HOLD


def _oracle_policy(env):
    state = env.current_frame.label.state
    if env.position is None:
        return Action.OPEN_FULL if state in {SymbolState.EMERGING_MOVE, SymbolState.CONFIRMED_TREND} else Action.IGNORE
    return Action.SELL if state in {SymbolState.EXHAUSTION, SymbolState.REVERSAL} else Action.HOLD


def _threshold_policy(open_threshold: float, sell_threshold: float):
    def policy(env):
        probs = env.current_frame.belief.probabilities
        open_mass = probs[SymbolState.EMERGING_MOVE] + probs[SymbolState.CONFIRMED_TREND]
        sell_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
        if env.position is None:
            return Action.OPEN_FULL if open_mass >= open_threshold else Action.IGNORE
        return Action.SELL if sell_mass >= sell_threshold else Action.HOLD

    return policy


def _key(item: dict) -> dict:
    return {
        "open_threshold": item["open_threshold"],
        "sell_threshold": item["sell_threshold"],
        "summary": item["summary"],
        "delta_vs_naive": item["delta_vs_naive"],
        "delta_vs_oracle": item["delta_vs_oracle"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
