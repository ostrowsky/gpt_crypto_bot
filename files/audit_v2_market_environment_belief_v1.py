from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_market_environment_separability import _market_day_features
from audit_v2_market_environment_switch_replay import (
    FEATURES,
    PREFIX_BARS,
    _build_day_info,
    _causal_choices,
    _nearest_centroid_label,
    _switched_policy,
)
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout, summarize_policy


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_environment_belief_v1_15m.json"
UPDATE_STEP_BARS = 8
PRIOR_WEIGHT = 0.70
OBS_WEIGHT = 0.30
DECISION_THRESHOLD = 0.65


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    by_day = defaultdict(list)
    for env in episodes:
        by_day[env.current_frame.label.local_day].append(env)

    day_info = _build_day_info(by_day, rows, temporal_rows)
    causal_choice = _causal_choices(day_info)
    belief_choice = _belief_choices(by_day, rows, day_info)

    fixed_base = _base_policy(rows)
    fixed_candidate = _grid_policy(rows, temporal_rows, -0.20, 0.10)
    policies = {
        "fixed_base": fixed_base,
        "fixed_candidate": fixed_candidate,
        "oracle_switched": _switched_policy(
            rows,
            temporal_rows,
            {day: info["oracle_policy"] for day, info in day_info.items()},
            cutoff_by_day=None,
        ),
        "causal_prefix_switched": _switched_policy(
            rows,
            temporal_rows,
            {day: item["policy"] for day, item in causal_choice.items()},
            cutoff_by_day={day: item["cutoff_ts_ms"] for day, item in causal_choice.items()},
        ),
        "belief_switched": _belief_switched_policy(rows, temporal_rows, belief_choice),
    }
    summaries = {
        name: summarize_policy(name, [rollout(env, policy) for env in episodes])
        for name, policy in policies.items()
    }
    base_reward = summaries["fixed_base"]["total_reward"]
    candidate_reward = summaries["fixed_candidate"]["total_reward"]
    payload = {
        "parameters": {
            "prefix_bars": PREFIX_BARS,
            "update_step_bars": UPDATE_STEP_BARS,
            "prior_weight": PRIOR_WEIGHT,
            "observation_weight": OBS_WEIGHT,
            "decision_threshold": DECISION_THRESHOLD,
            "features": list(FEATURES),
        },
        "policies": {
            name: {
                "summary": summary,
                "delta_vs_fixed_base": round(summary["total_reward"] - base_reward, 6),
                "delta_vs_fixed_candidate": round(summary["total_reward"] - candidate_reward, 6),
            }
            for name, summary in summaries.items()
        },
        "belief_day_paths": belief_choice,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _belief_choices(by_day, rows, day_info) -> dict:
    out = {}
    history = []
    for day, episodes in sorted(by_day.items()):
        labels = {item["label"] for item in history}
        prior = {"candidate_favorable": 0.5, "candidate_unfavorable": 0.5}
        current_policy = "base"
        path = []
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            ts_values = sorted({frame.bar.open_ts_ms for env in episodes for frame in env._frames})
            for end in range(PREFIX_BARS, len(ts_values) + 1, UPDATE_STEP_BARS):
                prefix_ts = set(ts_values[:end])
                prefix_rows = []
                for env in episodes:
                    for frame in env._frames:
                        if frame.bar.open_ts_ms in prefix_ts:
                            row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
                            if row:
                                prefix_rows.append(row)
                features = _market_day_features(prefix_rows)
                observation = _observation_probs(history, features)
                posterior = {
                    label: round(PRIOR_WEIGHT * prior[label] + OBS_WEIGHT * observation[label], 6)
                    for label in prior
                }
                total = sum(posterior.values()) or 1.0
                posterior = {label: value / total for label, value in posterior.items()}
                if posterior["candidate_favorable"] >= DECISION_THRESHOLD:
                    current_policy = "candidate"
                    decision = "candidate"
                elif posterior["candidate_unfavorable"] >= DECISION_THRESHOLD:
                    current_policy = "base"
                    decision = "base"
                else:
                    decision = "abstain"
                path.append(
                    {
                        "cutoff_ts_ms": ts_values[end - 1],
                        "bars_seen": end,
                        "observation": observation,
                        "posterior": posterior,
                        "decision": decision,
                        "policy_after_update": current_policy,
                    }
                )
                prior = posterior
        out[day] = {
            "oracle_policy": day_info[day]["oracle_policy"],
            "path": path,
            "fallback_policy": "base",
        }
        history.append(
            {
                "label": "candidate_favorable"
                if day_info[day]["reward_delta_candidate_vs_base"] > 0
                else "candidate_unfavorable",
                "features": day_info[day]["prefix_features"],
            }
        )
    return out


def _observation_probs(history: list[dict], features: dict) -> dict:
    favorable_train = [item for item in history if item["label"] == "candidate_favorable"]
    unfavorable_train = [item for item in history if item["label"] == "candidate_unfavorable"]
    predicted = _nearest_centroid_label(history, features)
    grouped = {
        "candidate_favorable": favorable_train,
        "candidate_unfavorable": unfavorable_train,
    }
    distances = {}
    for label, items in grouped.items():
        centroid = {name: sum(item["features"][name] for item in items) / len(items) for name in FEATURES}
        distances[label] = _distance(features, centroid)
    inv = {label: 1.0 / max(distance, 1e-9) for label, distance in distances.items()}
    total = sum(inv.values()) or 1.0
    probs = {label: inv[label] / total for label in inv}
    if predicted not in probs:
        return {"candidate_favorable": 0.5, "candidate_unfavorable": 0.5}
    return probs


def _distance(left: dict, right: dict) -> float:
    return sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in FEATURES))


def _belief_switched_policy(admission_rows, temporal_rows, belief_choice):
    base = _base_policy(admission_rows)
    candidate = _grid_policy(admission_rows, temporal_rows, -0.20, 0.10)

    def policy(env):
        day = env.current_frame.label.local_day
        info = belief_choice.get(day, {})
        selected = info.get("fallback_policy", "base")
        for update in info.get("path", []):
            if env.current_frame.bar.open_ts_ms > update["cutoff_ts_ms"]:
                selected = update["policy_after_update"]
        return candidate(env) if selected == "candidate" else base(env)

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
