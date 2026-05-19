from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_reward_weighted_market_selector_offline_replay import _build_episodes_from_admission
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _grid_policy
from v2.reward import RewardInputs, compute_reward
from v2.state import Action, SymbolState


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_action_level_exit_advantage_15m.jsonl"
DEFAULT_AUDIT = ROOT.parent / ".runtime" / "reports" / "v2_action_level_exit_advantage_audit_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path, audit_output: Path) -> dict:
    admission_rows = _load_admission_rows(dataset_path)
    episodes = _build_episodes_from_admission(history_root, labels_path, admission_rows)
    temporal_rows = _build_temporal_rows(admission_rows)
    policy = _grid_policy(admission_rows, temporal_rows, -0.20, 0.10)
    rows = []
    for env in episodes:
        rows.extend(_episode_rows(env, policy, admission_rows, temporal_rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    audit = _audit(rows, output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _episode_rows(env, policy, admission_rows: dict, temporal_rows: dict) -> list[dict]:
    env.reset()
    active = None
    current_trade_rows = []
    completed = []
    while True:
        frame = env.current_frame
        if active is not None:
            active["peak_price"] = max(active["peak_price"], frame.bar.high)
        action = Action.SELL if env.done and env.position is not None else policy(env)
        if active is not None:
            sell_now = _sell_now_reward(frame, active)
            current_trade_rows.append(
                {
                    "symbol": frame.bar.symbol,
                    "local_day": frame.label.local_day,
                    "ts_ms": frame.bar.open_ts_ms,
                    "candidate_action": action.value,
                    "true_state": frame.label.state.value,
                    "bars_held": env.index - active["entry_index"] + 1,
                    "entry_price": round(active["entry_price"], 12),
                    "current_price": round(frame.bar.close, 12),
                    "peak_price": round(active["peak_price"], 12),
                    "unrealized_pnl_pct": round(_pnl_pct(frame.bar.close, active["entry_price"]), 6),
                    "mfe_pct": round(_pnl_pct(active["peak_price"], active["entry_price"]), 6),
                    "giveback_pct": round(max(0.0, _pnl_pct(active["peak_price"], active["entry_price"]) - _pnl_pct(frame.bar.close, active["entry_price"])), 6),
                    "sell_now_reward": round(sell_now, 6),
                    "features": _features(frame, admission_rows, temporal_rows),
                }
            )
        step = env.step(action)
        if active is not None and current_trade_rows:
            current_trade_rows[-1]["candidate_step_reward"] = round(step.reward.total, 6)
        if action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
            active = {
                "entry_index": env.index - 1 if not step.done else env.index,
                "entry_price": frame.bar.close,
                "peak_price": frame.bar.high,
            }
            current_trade_rows = []
        elif action == Action.SELL and active is not None:
            realized_rows = _finalize_trade_rows(current_trade_rows)
            completed.extend(realized_rows)
            active = None
            current_trade_rows = []
        if step.done:
            return completed


def _finalize_trade_rows(rows: list[dict]) -> list[dict]:
    suffix = []
    running = 0.0
    for row in reversed(rows):
        running += float(row.get("candidate_step_reward", 0.0))
        suffix.append(running)
    suffix = list(reversed(suffix))
    out = []
    for row, continuation in zip(rows, suffix):
        sell_advantage = row["sell_now_reward"] - continuation
        out.append(
            {
                **row,
                "continuation_reward": round(continuation, 6),
                "sell_advantage": round(sell_advantage, 6),
                "sell_advantage_positive": sell_advantage > 0.0,
                "sell_advantage_strong": sell_advantage >= 1.0,
                "hold_advantage_strong": sell_advantage <= -1.0,
            }
        )
    return out


def _sell_now_reward(frame, active: dict) -> float:
    realized = _pnl_pct(frame.bar.close, active["entry_price"])
    mfe = _pnl_pct(active["peak_price"], active["entry_price"])
    exit_eff = realized / mfe if mfe > 0 else 0.0
    giveback = max(0.0, mfe - realized)
    return compute_reward(
        RewardInputs(
            realized_pnl_pct=realized,
            exit_efficiency=exit_eff,
            giveback_pct=giveback,
        )
    ).total


def _features(frame, admission_rows: dict, temporal_rows: dict) -> dict:
    probs = frame.belief.probabilities
    row = admission_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
    projected = row.get("v1_projected_structural") or {}
    temporal = temporal_rows.get((frame.bar.symbol, frame.bar.open_ts_ms)) or {}
    return {
        "belief_noise": round(float(probs[SymbolState.NOISE]), 6),
        "belief_open_mass": round(float(probs[SymbolState.EMERGING_MOVE] + probs[SymbolState.CONFIRMED_TREND]), 6),
        "belief_mature": round(float(probs[SymbolState.MATURE_TREND]), 6),
        "belief_late_mass": round(float(probs[SymbolState.EXHAUSTION] + probs[SymbolState.REVERSAL]), 6),
        "projected_leader_score_trend": round(float(projected.get("projected_leader_score_trend", 0.0)), 6),
        "price_vs_ema20_pct": round(float(projected.get("price_vs_ema20_pct", 0.0)), 6),
        "rsi": round(float(projected.get("rsi", 0.0)), 6),
        "adx": round(float(projected.get("adx", 0.0)), 6),
        "daily_range_pct": round(float(projected.get("daily_range_pct", 0.0)), 6),
        "late_mass_delta_3": round(float(temporal.get("late_mass_delta_3", 0.0)), 6),
        "mature_delta_3": round(float(temporal.get("mature_delta_3", 0.0)), 6),
        "rsi_delta_3": round(float(temporal.get("rsi_delta_3", 0.0)), 6),
        "price_vs_ema20_delta_3": round(float(temporal.get("price_vs_ema20_delta_3", 0.0)), 6),
    }


def _audit(rows: list[dict], output: Path) -> dict:
    labels = Counter()
    states = Counter()
    actions = Counter()
    feature_coverage = Counter()
    for row in rows:
        labels["sell_positive"] += int(row["sell_advantage_positive"])
        labels["sell_strong"] += int(row["sell_advantage_strong"])
        labels["hold_strong"] += int(row["hold_advantage_strong"])
        states[row["true_state"]] += 1
        actions[row["candidate_action"]] += 1
        for name, value in row["features"].items():
            if value is not None:
                feature_coverage[name] += 1
    n = len(rows) or 1
    sorted_rows = sorted(rows, key=lambda row: row["sell_advantage"])
    return {
        "output": str(output),
        "rows": len(rows),
        "label_counts": dict(labels),
        "label_shares": {key: round(value / n, 6) for key, value in labels.items()},
        "states": dict(sorted(states.items())),
        "candidate_actions": dict(sorted(actions.items())),
        "feature_coverage": {key: round(value / n, 6) for key, value in sorted(feature_coverage.items())},
        "sell_advantage_avg": round(sum(row["sell_advantage"] for row in rows) / n, 6),
        "sell_advantage_min": round(sorted_rows[0]["sell_advantage"], 6) if rows else None,
        "sell_advantage_max": round(sorted_rows[-1]["sell_advantage"], 6) if rows else None,
        "top_hold_advantage_examples": sorted_rows[:10],
        "top_sell_advantage_examples": list(reversed(sorted_rows[-10:])),
    }


def _pnl_pct(current: float, entry: float) -> float:
    if entry == 0:
        return 0.0
    return (current / entry - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output, args.audit_output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
