from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_policy_gap import _summarize_trades
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from audit_v2_temporal_exit_window_stability import _split_windows
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_temporal_exit_failure_slice_15m.json"


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    ordered = sorted(episodes, key=lambda env: env.current_frame.label.local_day)
    windows = _split_windows(ordered, parts=4)
    payloads = [
        _summarize_window(f"window_{idx + 1}", window, rows, temporal_rows)
        for idx, window in enumerate(windows)
    ]
    winning = [item for item in payloads if item["delta_vs_base_reward"] > 0]
    losing = [item for item in payloads if item["delta_vs_base_reward"] <= 0]
    comparison = _compare_groups(winning, losing)
    payload = {
        "candidate": {
            "late_mass_floor": 0.50,
            "mature_delta_max": -0.20,
            "late_mass_delta_min": 0.10,
        },
        "windows": payloads,
        "comparison": comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _summarize_window(name: str, episodes, rows, temporal_rows) -> dict:
    base_runs = [rollout(env, _base_policy(rows)) for env in episodes]
    candidate_runs = [rollout(env, _grid_policy(rows, temporal_rows, -0.20, 0.10)) for env in episodes]
    base_summary = summarize_policy("base_sell_0_70", base_runs)
    candidate_summary = summarize_policy("mature_decay_0_20_late_rise_0_10", candidate_runs)
    days = sorted({env.current_frame.label.local_day for env in episodes})
    episode_rows = _rows_for_episodes(episodes, rows)
    return {
        "name": name,
        "episodes": len(episodes),
        "start_day": days[0] if days else None,
        "end_day": days[-1] if days else None,
        "delta_vs_base_reward": round(candidate_summary["total_reward"] - base_summary["total_reward"], 6),
        "state_mix": _state_mix(episode_rows),
        "feature_means": _feature_means(episode_rows),
        "base": {
            "summary": base_summary,
            "trade": _summarize_trades(base_runs),
        },
        "candidate": {
            "summary": candidate_summary,
            "trade": _summarize_trades(candidate_runs),
        },
        "reward_component_delta": _component_delta(base_summary["reward_components"], candidate_summary["reward_components"]),
    }


def _rows_for_episodes(episodes, rows) -> list[dict]:
    out = []
    for env in episodes:
        for frame in env._frames:  # research audit: inspect immutable episode frames
            row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
            if row:
                out.append(row)
    return out


def _state_mix(rows: list[dict]) -> dict:
    counts = Counter(row["true_state"] for row in rows)
    total = sum(counts.values()) or 1
    return {
        state: {"count": count, "share": round(count / total, 6)}
        for state, count in sorted(counts.items())
    }


def _feature_means(rows: list[dict]) -> dict:
    if not rows:
        return {}
    features = {
        "belief_emerging": [],
        "belief_confirmed": [],
        "belief_mature": [],
        "belief_exhaustion": [],
        "belief_reversal": [],
        "belief_late_mass": [],
        "projected_forecast_proxy_pct": [],
        "projected_leader_score_trend": [],
        "adx": [],
        "rsi": [],
        "vol_x": [],
        "daily_range_pct": [],
        "price_vs_ema20_pct": [],
    }
    for row in rows:
        belief = row["belief"]
        projected = row.get("v1_projected_structural") or {}
        features["belief_emerging"].append(float(belief.get("emerging_move", 0.0)))
        features["belief_confirmed"].append(float(belief.get("confirmed_trend", 0.0)))
        features["belief_mature"].append(float(belief.get("mature_trend", 0.0)))
        features["belief_exhaustion"].append(float(belief.get("exhaustion", 0.0)))
        features["belief_reversal"].append(float(belief.get("reversal", 0.0)))
        features["belief_late_mass"].append(
            float(belief.get("exhaustion", 0.0)) + float(belief.get("reversal", 0.0))
        )
        for name in (
            "projected_forecast_proxy_pct",
            "projected_leader_score_trend",
            "adx",
            "rsi",
            "vol_x",
            "daily_range_pct",
            "price_vs_ema20_pct",
        ):
            features[name].append(float(projected.get(name, 0.0)))
    return {name: round(sum(values) / len(values), 6) for name, values in sorted(features.items())}


def _component_delta(base: dict, candidate: dict) -> dict:
    keys = sorted(set(base) | set(candidate))
    return {key: round(candidate.get(key, 0.0) - base.get(key, 0.0), 6) for key in keys}


def _compare_groups(winning: list[dict], losing: list[dict]) -> dict:
    winning_profile = _average_profiles(winning)
    losing_profile = _average_profiles(losing)
    ranked_feature_deltas = []
    for feature, losing_value in losing_profile["feature_means"].items():
        winning_value = winning_profile["feature_means"].get(feature, 0.0)
        ranked_feature_deltas.append(
            {
                "feature": feature,
                "winning_mean": winning_value,
                "losing_mean": losing_value,
                "losing_minus_winning": round(losing_value - winning_value, 6),
                "abs_delta": round(abs(losing_value - winning_value), 6),
            }
        )
    ranked_feature_deltas.sort(key=lambda item: item["abs_delta"], reverse=True)
    return {
        "winning_windows": [item["name"] for item in winning],
        "losing_windows": [item["name"] for item in losing],
        "winning_average": winning_profile,
        "losing_average": losing_profile,
        "ranked_feature_deltas": ranked_feature_deltas,
    }


def _average_profiles(windows: list[dict]) -> dict:
    if not windows:
        return {"feature_means": {}, "state_shares": {}, "reward_component_delta": {}}
    feature_names = windows[0]["feature_means"].keys()
    states = {state for window in windows for state in window["state_mix"]}
    reward_names = windows[0]["reward_component_delta"].keys()
    return {
        "feature_means": {
            name: round(sum(window["feature_means"][name] for window in windows) / len(windows), 6)
            for name in feature_names
        },
        "state_shares": {
            state: round(
                sum(window["state_mix"].get(state, {}).get("share", 0.0) for window in windows) / len(windows),
                6,
            )
            for state in sorted(states)
        },
        "reward_component_delta": {
            name: round(sum(window["reward_component_delta"][name] for window in windows) / len(windows), 6)
            for name in reward_names
        },
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
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["comparison"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
