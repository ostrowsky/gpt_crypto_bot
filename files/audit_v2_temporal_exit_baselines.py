from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_temporal_exit_baselines_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    names = (
        "base_sell_0_70",
        "late_mass_acceleration",
        "mature_decay_late_rise",
        "rsi_ema_decay",
        "consensus_temporal",
    )
    policies = {name: _policy(rows, temporal_rows, profile=name) for name in names}
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


def _build_temporal_rows(rows: dict[tuple[str, int], dict]) -> dict[tuple[str, int], dict]:
    grouped = defaultdict(list)
    for (symbol, ts_ms), row in rows.items():
        grouped[symbol].append((ts_ms, row))
    out = {}
    for symbol, items in grouped.items():
        items.sort(key=lambda item: item[0])
        for idx, (ts_ms, row) in enumerate(items):
            if idx < 3:
                out[(symbol, ts_ms)] = {"has_3bar_history": False}
                continue
            prev = items[idx - 3][1]
            current_belief = row.get("belief") or {}
            previous_belief = prev.get("belief") or {}
            current_projected = row.get("v1_projected_structural") or {}
            previous_projected = prev.get("v1_projected_structural") or {}
            out[(symbol, ts_ms)] = {
                "has_3bar_history": True,
                "late_mass_delta_3": _late_mass(current_belief) - _late_mass(previous_belief),
                "mature_delta_3": float(current_belief.get("mature_trend", 0.0))
                - float(previous_belief.get("mature_trend", 0.0)),
                "rsi_delta_3": float(current_projected.get("rsi", 0.0))
                - float(previous_projected.get("rsi", 0.0)),
                "price_vs_ema20_delta_3": float(current_projected.get("price_vs_ema20_pct", 0.0))
                - float(previous_projected.get("price_vs_ema20_pct", 0.0)),
            }
    return out


def _policy(admission_rows: dict[tuple[str, int], dict], temporal_rows: dict[tuple[str, int], dict], *, profile: str):
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
        if profile == "base_sell_0_70":
            sell = late_mass >= 0.70
        else:
            temporal = temporal_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
            if not temporal.get("has_3bar_history"):
                return Action.HOLD
            late_delta = float(temporal["late_mass_delta_3"])
            mature_delta = float(temporal["mature_delta_3"])
            rsi_delta = float(temporal["rsi_delta_3"])
            ema_delta = float(temporal["price_vs_ema20_delta_3"])
            if profile == "late_mass_acceleration":
                sell = late_mass >= 0.55 and late_delta >= 0.15
            elif profile == "mature_decay_late_rise":
                sell = late_mass >= 0.50 and mature_delta <= -0.15 and late_delta >= 0.10
            elif profile == "rsi_ema_decay":
                sell = late_mass >= 0.45 and rsi_delta <= -4.0 and ema_delta <= -0.40
            elif profile == "consensus_temporal":
                sell = late_mass >= 0.50 and late_delta >= 0.10 and rsi_delta <= -3.0 and ema_delta <= -0.25
            else:
                raise ValueError(f"unknown temporal exit profile: {profile}")
        return Action.SELL if sell else Action.HOLD

    return policy


def _late_mass(belief: dict) -> float:
    return float(belief.get("exhaustion", 0.0)) + float(belief.get("reversal", 0.0))


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
