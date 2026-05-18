from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_market_environment_switch_replay import _build_day_info
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from run_v2_policy_baselines import _build_episodes
from v2.policy_baselines import rollout
from v2.state import Action


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_environment_target_design_15m.json"
HORIZONS = {
    "1h": 4,
    "2h": 8,
    "rest_of_day": None,
}
EVAL_START_BAR = 16
EVAL_STEP_BARS = 8


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    by_day = defaultdict(list)
    for env in episodes:
        by_day[env.current_frame.label.local_day].append(env)
    day_info = _build_day_info(by_day, rows, temporal_rows)

    samples = []
    for day, day_episodes in sorted(by_day.items()):
        ts_values = sorted({frame.bar.open_ts_ms for env in day_episodes for frame in env._frames})
        for start in range(EVAL_START_BAR, len(ts_values), EVAL_STEP_BARS):
            anchor_ts = ts_values[start - 1]
            for horizon_name, horizon_bars in HORIZONS.items():
                end = len(ts_values) if horizon_bars is None else min(len(ts_values), start + horizon_bars)
                horizon_ts = set(ts_values[start:end])
                base_reward, base_states = _reward_on_horizon(day_episodes, rows, temporal_rows, horizon_ts, policy="base")
                candidate_reward, candidate_states = _reward_on_horizon(
                    day_episodes, rows, temporal_rows, horizon_ts, policy="candidate"
                )
                delta = round(candidate_reward - base_reward, 6)
                samples.append(
                    {
                        "day": day,
                        "anchor_ts_ms": anchor_ts,
                        "horizon": horizon_name,
                        "day_label": "candidate_favorable"
                        if day_info[day]["reward_delta_candidate_vs_base"] > 0
                        else "candidate_unfavorable",
                        "horizon_label": "candidate_favorable" if delta > 0 else "candidate_unfavorable",
                        "reward_delta": delta,
                        "future_state_mix": _state_mix(candidate_states or base_states),
                    }
                )
    payload = {
        "parameters": {
            "eval_start_bar": EVAL_START_BAR,
            "eval_step_bars": EVAL_STEP_BARS,
            "horizons": HORIZONS,
        },
        "horizons": {
            name: _summarize_horizon([sample for sample in samples if sample["horizon"] == name])
            for name in HORIZONS
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _reward_on_horizon(episodes, rows, temporal_rows, horizon_ts: set[int], *, policy: str) -> tuple[float, list[str]]:
    total = 0.0
    states = []
    chooser = _base_policy(rows) if policy == "base" else _grid_policy(rows, temporal_rows, -0.20, 0.10)
    for env in episodes:
        for step in rollout(env, chooser):
            if step.frame.bar.open_ts_ms in horizon_ts:
                total += step.reward.total
                states.append(step.frame.label.state.value)
    return total, states


def _state_mix(states: list[str]) -> dict:
    counts = Counter(states)
    total = sum(counts.values()) or 1
    return {state: round(count / total, 6) for state, count in sorted(counts.items())}


def _summarize_horizon(samples: list[dict]) -> dict:
    labels = Counter(sample["horizon_label"] for sample in samples)
    disagreements = [sample for sample in samples if sample["day_label"] != sample["horizon_label"]]
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["horizon_label"]].append(sample)
    return {
        "samples": len(samples),
        "class_counts": dict(sorted(labels.items())),
        "day_label_disagreement_count": len(disagreements),
        "day_label_disagreement_share": round(len(disagreements) / len(samples), 6) if samples else 0.0,
        "avg_reward_delta": round(sum(sample["reward_delta"] for sample in samples) / len(samples), 6)
        if samples
        else 0.0,
        "future_state_mix_by_label": {
            label: _average_mix(items)
            for label, items in sorted(grouped.items())
        },
    }


def _average_mix(items: list[dict]) -> dict:
    states = {state for item in items for state in item["future_state_mix"]}
    return {
        state: round(sum(item["future_state_mix"].get(state, 0.0) for item in items) / len(items), 6)
        for state in sorted(states)
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
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["horizons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
