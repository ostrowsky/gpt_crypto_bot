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
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_exhaustion_aware_exit_baselines_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    names = (
        "base_sell_0_70",
        "late_mass_rsi_weak",
        "late_mass_ema_loss",
        "exhaustion_belief_combo",
        "consensus_exhaustion",
    )
    policies = {name: _policy(rows, profile=name) for name in names}
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

        row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
        projected = row.get("v1_projected_structural") or {}
        late_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
        rsi = float(projected.get("rsi", 100.0))
        price_vs_ema20 = float(projected.get("price_vs_ema20_pct", 999.0))
        mature = probs[SymbolState.MATURE_TREND]
        exhaustion = probs[SymbolState.EXHAUSTION]

        if profile == "base_sell_0_70":
            sell = late_mass >= 0.70
        elif profile == "late_mass_rsi_weak":
            sell = late_mass >= 0.55 and rsi <= 56.0
        elif profile == "late_mass_ema_loss":
            sell = late_mass >= 0.55 and price_vs_ema20 <= 0.35
        elif profile == "exhaustion_belief_combo":
            sell = exhaustion >= 0.35 and mature <= 0.25
        elif profile == "consensus_exhaustion":
            sell = late_mass >= 0.50 and rsi <= 56.0 and price_vs_ema20 <= 0.35
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
