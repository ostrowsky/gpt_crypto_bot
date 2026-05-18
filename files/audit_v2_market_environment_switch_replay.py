from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_market_environment_separability import _market_day_features
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_environment_switch_replay_15m.json"
PREFIX_BARS = 16
FEATURES = (
    "adx",
    "projected_leader_score_trend",
    "daily_range_pct",
    "projected_forecast_proxy_pct",
    "price_vs_ema20_pct",
)


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    by_day = defaultdict(list)
    for env in episodes:
        by_day[env.current_frame.label.local_day].append(env)

    day_info = _build_day_info(by_day, rows, temporal_rows)
    oracle_choice = {day: info["oracle_policy"] for day, info in day_info.items()}
    causal_choice = _causal_choices(day_info)

    policies = {
        "fixed_base": _base_policy(rows),
        "fixed_candidate": _grid_policy(rows, temporal_rows, -0.20, 0.10),
        "oracle_switched": _switched_policy(rows, temporal_rows, oracle_choice, cutoff_by_day=None),
        "causal_prefix_switched": _switched_policy(
            rows,
            temporal_rows,
            {day: item["policy"] for day, item in causal_choice.items()},
            cutoff_by_day={day: item["cutoff_ts_ms"] for day, item in causal_choice.items()},
        ),
    }
    summaries = {
        name: summarize_policy(name, [rollout(env, policy) for env in episodes])
        for name, policy in policies.items()
    }
    base_reward = summaries["fixed_base"]["total_reward"]
    candidate_reward = summaries["fixed_candidate"]["total_reward"]
    out = {
        name: {
            "summary": summary,
            "delta_vs_fixed_base": round(summary["total_reward"] - base_reward, 6),
            "delta_vs_fixed_candidate": round(summary["total_reward"] - candidate_reward, 6),
        }
        for name, summary in summaries.items()
    }
    payload = {
        "parameters": {"prefix_bars": PREFIX_BARS, "features": list(FEATURES)},
        "policies": out,
        "oracle_day_choices": oracle_choice,
        "causal_day_choices": causal_choice,
        "days": day_info,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _build_day_info(by_day, rows, temporal_rows) -> dict:
    out = {}
    for day, episodes in sorted(by_day.items()):
        base = summarize_policy("base", [rollout(env, _base_policy(rows)) for env in episodes])
        candidate = summarize_policy(
            "candidate",
            [rollout(env, _grid_policy(rows, temporal_rows, -0.20, 0.10)) for env in episodes],
        )
        delta = round(candidate["total_reward"] - base["total_reward"], 6)
        prefix_rows, cutoff_ts_ms = _prefix_rows(episodes, rows)
        out[day] = {
            "episodes": len(episodes),
            "reward_delta_candidate_vs_base": delta,
            "oracle_policy": "candidate" if delta > 0 else "base",
            "prefix_features": _market_day_features(prefix_rows),
            "cutoff_ts_ms": cutoff_ts_ms,
        }
    return out


def _prefix_rows(episodes, rows) -> tuple[list[dict], int]:
    ts_values = sorted({frame.bar.open_ts_ms for env in episodes for frame in env._frames})
    prefix_ts = set(ts_values[:PREFIX_BARS])
    cutoff = ts_values[PREFIX_BARS - 1]
    out = []
    for env in episodes:
        for frame in env._frames:
            if frame.bar.open_ts_ms in prefix_ts:
                row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
                if row:
                    out.append(row)
    return out, cutoff


def _causal_choices(day_info: dict) -> dict:
    out = {}
    history = []
    for day, info in sorted(day_info.items()):
        train = [item for item in history if item["label"] in {"candidate_favorable", "candidate_unfavorable"}]
        labels = {item["label"] for item in train}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            predicted = _nearest_centroid_label(train, info["prefix_features"])
            policy = "candidate" if predicted == "candidate_favorable" else "base"
            reason = "nearest_centroid"
        else:
            predicted = None
            policy = "base"
            reason = "insufficient_prior_classes"
        out[day] = {
            "policy": policy,
            "predicted_label": predicted,
            "reason": reason,
            "cutoff_ts_ms": info["cutoff_ts_ms"],
        }
        history.append(
            {
                "label": "candidate_favorable"
                if info["reward_delta_candidate_vs_base"] > 0
                else "candidate_unfavorable",
                "features": info["prefix_features"],
            }
        )
    return out


def _nearest_centroid_label(train: list[dict], features: dict) -> str:
    grouped = defaultdict(list)
    for item in train:
        grouped[item["label"]].append(item["features"])
    centroids = {
        label: {name: sum(row[name] for row in rows) / len(rows) for name in FEATURES}
        for label, rows in grouped.items()
    }
    return min(centroids, key=lambda label: _distance(features, centroids[label]))


def _distance(left: dict, right: dict) -> float:
    return sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in FEATURES))


def _switched_policy(admission_rows, temporal_rows, choices: dict, cutoff_by_day: dict | None):
    base = _base_policy(admission_rows)
    candidate = _grid_policy(admission_rows, temporal_rows, -0.20, 0.10)

    def policy(env):
        day = env.current_frame.label.local_day
        choice = choices.get(day, "base")
        if cutoff_by_day is not None:
            cutoff = cutoff_by_day.get(day)
            if cutoff is None or env.current_frame.bar.open_ts_ms <= cutoff:
                return base(env)
        return candidate(env) if choice == "candidate" else base(env)

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
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["policies"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
