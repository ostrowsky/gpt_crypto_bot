from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_policy_gap import _summarize_trades
from audit_v2_reward_weighted_market_selector_offline_replay import _build_episodes_from_admission
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from v2.policy_baselines import rollout, summarize_policy
from v2.state import Action, SymbolState


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_position_aware_exit_selector_15m.json"

PROFILES = {
    "base_override_late_mass": {
        "late_mass_min": 0.70,
        "late_delta_min": None,
        "mature_delta_max": None,
        "price_vs_ema20_max": None,
        "rsi_max": None,
    },
    "base_override_late_accel": {
        "late_mass_min": 0.60,
        "late_delta_min": 0.10,
        "mature_delta_max": None,
        "price_vs_ema20_max": None,
        "rsi_max": None,
    },
    "base_override_decay": {
        "late_mass_min": 0.55,
        "late_delta_min": 0.05,
        "mature_delta_max": -0.10,
        "price_vs_ema20_max": None,
        "rsi_max": None,
    },
    "base_override_ema_break": {
        "late_mass_min": 0.50,
        "late_delta_min": 0.05,
        "mature_delta_max": -0.05,
        "price_vs_ema20_max": -0.25,
        "rsi_max": 55.0,
    },
    "base_override_strict_break": {
        "late_mass_min": 0.60,
        "late_delta_min": 0.10,
        "mature_delta_max": -0.10,
        "price_vs_ema20_max": -0.40,
        "rsi_max": 50.0,
    },
}


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    rows = _load_admission_rows(dataset_path)
    episodes = _build_episodes_from_admission(history_root, labels_path, rows)
    temporal_rows = _build_temporal_rows(rows)
    base_policy = _base_policy(rows)
    candidate_policy = _grid_policy(rows, temporal_rows, -0.20, 0.10)
    policies = {
        "fixed_base": base_policy,
        "fixed_candidate": candidate_policy,
    }
    for name, profile in PROFILES.items():
        policies[name] = _position_aware_policy(rows, temporal_rows, profile)

    rollouts = {name: [rollout(env, policy) for env in episodes] for name, policy in policies.items()}
    summaries = {name: summarize_policy(name, runs) for name, runs in rollouts.items()}
    exits = {name: _summarize_trades(runs) for name, runs in rollouts.items()}
    base_reward = summaries["fixed_base"]["total_reward"]
    candidate_reward = summaries["fixed_candidate"]["total_reward"]
    out = {}
    for name in policies:
        out[name] = {
            "summary": summaries[name],
            "exit": exits[name],
            "delta_vs_fixed_base": round(summaries[name]["total_reward"] - base_reward, 6),
            "delta_vs_fixed_candidate": round(summaries[name]["total_reward"] - candidate_reward, 6),
            "override_summary": _override_summary(rollouts[name]) if name not in {"fixed_base", "fixed_candidate"} else None,
            "profile": PROFILES.get(name),
        }
    ranked = sorted(
        [
            {"policy": name, "total_reward": value["summary"]["total_reward"], "delta_vs_fixed_candidate": value["delta_vs_fixed_candidate"]}
            for name, value in out.items()
            if name not in {"fixed_base", "fixed_candidate"}
        ],
        key=lambda item: item["total_reward"],
        reverse=True,
    )
    payload = {
        "episodes": len(episodes),
        "policies": out,
        "selection": {
            "best": ranked[0] if ranked else None,
            "ranked": ranked,
            "fixed_base_reward": base_reward,
            "fixed_candidate_reward": candidate_reward,
        },
        "decision": _decision(out, ranked),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _position_aware_policy(admission_rows, temporal_rows, profile: dict):
    base = _base_policy(admission_rows)
    candidate = _grid_policy(admission_rows, temporal_rows, -0.20, 0.10)

    def policy(env):
        candidate_action = candidate(env)
        if env.position is None:
            return candidate_action
        base_action = base(env)
        if base_action == Action.SELL and candidate_action != Action.SELL:
            if _risk_gate(env, admission_rows, temporal_rows, profile):
                return Action.SELL
        return candidate_action

    return policy


def _risk_gate(env, admission_rows, temporal_rows, profile: dict) -> bool:
    frame = env.current_frame
    probs = frame.belief.probabilities
    late_mass = probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]
    if late_mass < float(profile["late_mass_min"]):
        return False
    row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
    projected = row.get("v1_projected_structural") or {}
    temporal = temporal_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
    checks = []
    if profile.get("late_delta_min") is not None:
        checks.append(float(temporal.get("late_mass_delta_3", 0.0)) >= float(profile["late_delta_min"]))
    if profile.get("mature_delta_max") is not None:
        checks.append(float(temporal.get("mature_delta_3", 0.0)) <= float(profile["mature_delta_max"]))
    if profile.get("price_vs_ema20_max") is not None:
        checks.append(float(projected.get("price_vs_ema20_pct", 0.0)) <= float(profile["price_vs_ema20_max"]))
    if profile.get("rsi_max") is not None:
        checks.append(float(projected.get("rsi", 0.0)) <= float(profile["rsi_max"]))
    return all(checks) if checks else True


def _override_summary(steps_by_episode) -> dict:
    # We infer override-like sells as SELL actions emitted before candidate's temporal profile would normally sell.
    # This is a lightweight diagnostic; exact pairwise action decomposition lives in selector-failure audit.
    sells_by_state = Counter()
    total_sells = 0
    for steps in steps_by_episode:
        for step in steps:
            if step.action == Action.SELL:
                total_sells += 1
                sells_by_state[step.frame.label.state.value] += 1
    return {"sell_count": total_sells, "sells_by_true_state": dict(sorted(sells_by_state.items()))}


def _decision(out: dict, ranked: list[dict]) -> dict:
    if not ranked:
        return {"promotion_gate_passed": False, "recommendation": "no candidate profiles"}
    best = ranked[0]
    fixed_candidate = out["fixed_candidate"]
    best_payload = out[best["policy"]]
    trade_ratio = best_payload["summary"]["trade_count"] / max(fixed_candidate["summary"]["trade_count"], 1)
    giveback_delta = best_payload["exit"]["avg_giveback_penalty"] - fixed_candidate["exit"]["avg_giveback_penalty"]
    passed = best["delta_vs_fixed_candidate"] > 0 and trade_ratio <= 1.10 and giveback_delta >= -0.10
    return {
        "best_policy": best,
        "trade_count_ratio_vs_candidate": round(trade_ratio, 6),
        "avg_giveback_penalty_delta_vs_candidate": round(giveback_delta, 6),
        "promotion_gate_passed": bool(passed),
        "recommendation": "advance to robustness/window stability" if passed else "reject first position-aware profiles; inspect action-level labels",
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
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
