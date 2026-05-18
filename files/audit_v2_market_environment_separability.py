from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_environment_separability_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    by_day = defaultdict(list)
    for env in episodes:
        by_day[env.current_frame.label.local_day].append(env)
    days = []
    for day, day_episodes in sorted(by_day.items()):
        base = summarize_policy("base", [rollout(env, _base_policy(rows)) for env in day_episodes])
        candidate = summarize_policy(
            "candidate",
            [rollout(env, _grid_policy(rows, temporal_rows, -0.20, 0.10)) for env in day_episodes],
        )
        delta = round(candidate["total_reward"] - base["total_reward"], 6)
        day_rows = _rows_for_episodes(day_episodes, rows)
        days.append(
            {
                "day": day,
                "episodes": len(day_episodes),
                "label": "candidate_favorable" if delta > 0 else "candidate_unfavorable",
                "reward_delta": delta,
                "features": _market_day_features(day_rows),
            }
        )
    grouped = defaultdict(list)
    for item in days:
        grouped[item["label"]].append(item)
    means = {label: _feature_means(items) for label, items in sorted(grouped.items())}
    ranked = _rank_feature_deltas(means)
    payload = {
        "days": days,
        "class_counts": {label: len(items) for label, items in sorted(grouped.items())},
        "feature_means": means,
        "ranked_feature_deltas": ranked,
        "verdict": _verdict(grouped, ranked),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _rows_for_episodes(episodes, rows) -> list[dict]:
    out = []
    for env in episodes:
        for frame in env._frames:
            row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
            if row:
                out.append(row)
    return out


def _market_day_features(rows: list[dict]) -> dict:
    total = len(rows) or 1
    states = defaultdict(int)
    values = defaultdict(list)
    for row in rows:
        states[row["true_state"]] += 1
        belief = row["belief"]
        projected = row.get("v1_projected_structural") or {}
        values["belief_late_mass"].append(float(belief.get("exhaustion", 0.0)) + float(belief.get("reversal", 0.0)))
        for name in (
            "projected_forecast_proxy_pct",
            "projected_leader_score_trend",
            "adx",
            "rsi",
            "vol_x",
            "daily_range_pct",
            "price_vs_ema20_pct",
        ):
            values[name].append(float(projected.get(name, 0.0)))
    features = {
        "noise_share": states["noise"] / total,
        "emerging_share": states["emerging_move"] / total,
        "confirmed_share": states["confirmed_trend"] / total,
        "mature_share": states["mature_trend"] / total,
        "exhaustion_share": states["exhaustion"] / total,
        "reversal_share": states["reversal"] / total,
    }
    features.update({name: sum(vals) / len(vals) for name, vals in values.items() if vals})
    return {name: round(value, 6) for name, value in sorted(features.items())}


def _feature_means(items: list[dict]) -> dict:
    names = items[0]["features"].keys() if items else []
    return {name: round(sum(item["features"][name] for item in items) / len(items), 6) for name in names}


def _rank_feature_deltas(means: dict) -> list[dict]:
    favorable = means.get("candidate_favorable", {})
    unfavorable = means.get("candidate_unfavorable", {})
    ranked = []
    for name in sorted(set(favorable) & set(unfavorable)):
        delta = favorable[name] - unfavorable[name]
        ranked.append(
            {
                "feature": name,
                "favorable_mean": favorable[name],
                "unfavorable_mean": unfavorable[name],
                "favorable_minus_unfavorable": round(delta, 6),
                "abs_delta": round(abs(delta), 6),
            }
        )
    ranked.sort(key=lambda item: item["abs_delta"], reverse=True)
    return ranked


def _verdict(grouped: dict, ranked: list[dict]) -> dict:
    favorable_n = len(grouped.get("candidate_favorable", []))
    unfavorable_n = len(grouped.get("candidate_unfavorable", []))
    if favorable_n < 3 or unfavorable_n < 3:
        return {"label": "inconclusive", "reason": "too_few_days_per_class"}
    top = ranked[:5]
    meaningful = [item for item in top if item["abs_delta"] >= 0.05]
    if len(meaningful) >= 3:
        return {"label": "separable_candidate", "top_features": [item["feature"] for item in top]}
    return {"label": "weak_signal", "top_features": [item["feature"] for item in top]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
