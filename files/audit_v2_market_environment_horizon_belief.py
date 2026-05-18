from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_market_environment_separability import _market_day_features
from audit_v2_market_environment_target_design import EVAL_START_BAR, EVAL_STEP_BARS, HORIZONS, _reward_on_horizon
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from run_v2_policy_baselines import _build_episodes


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_environment_horizon_belief_15m.json"
FEATURES = (
    "adx",
    "projected_leader_score_trend",
    "daily_range_pct",
    "projected_forecast_proxy_pct",
    "price_vs_ema20_pct",
    "noise_share",
    "emerging_share",
    "mature_share",
    "belief_late_mass",
)


def build(history_root: Path, labels_path: Path, dataset_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    rows = _load_admission_rows(dataset_path)
    temporal_rows = _build_temporal_rows(rows)
    by_day = defaultdict(list)
    for env in episodes:
        by_day[env.current_frame.label.local_day].append(env)

    samples = []
    for day, day_episodes in sorted(by_day.items()):
        ts_values = sorted({frame.bar.open_ts_ms for env in day_episodes for frame in env._frames})
        for start in range(EVAL_START_BAR, len(ts_values), EVAL_STEP_BARS):
            anchor_ts = ts_values[start - 1]
            feature_rows = _prefix_rows(day_episodes, rows, set(ts_values[:start]))
            features = _select_features(_market_day_features(feature_rows))
            for horizon_name in ("1h", "2h"):
                horizon_bars = HORIZONS[horizon_name]
                end = min(len(ts_values), start + horizon_bars)
                horizon_ts = set(ts_values[start:end])
                base_reward, _ = _reward_on_horizon(day_episodes, rows, temporal_rows, horizon_ts, policy="base")
                candidate_reward, _ = _reward_on_horizon(
                    day_episodes, rows, temporal_rows, horizon_ts, policy="candidate"
                )
                delta = round(candidate_reward - base_reward, 6)
                samples.append(
                    {
                        "day": day,
                        "anchor_ts_ms": anchor_ts,
                        "horizon": horizon_name,
                        "label": "candidate_favorable" if delta > 0 else "candidate_unfavorable",
                        "reward_delta": delta,
                        "features": features,
                    }
                )

    payload = {
        "parameters": {
            "features": list(FEATURES),
            "eval_start_bar": EVAL_START_BAR,
            "eval_step_bars": EVAL_STEP_BARS,
        },
        "horizons": {
            horizon: _evaluate_horizon([s for s in samples if s["horizon"] == horizon])
            for horizon in ("1h", "2h")
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _prefix_rows(episodes, rows, prefix_ts: set[int]) -> list[dict]:
    out = []
    for env in episodes:
        for frame in env._frames:
            if frame.bar.open_ts_ms in prefix_ts:
                row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
                if row:
                    out.append(row)
    return out


def _select_features(features: dict) -> dict:
    return {name: float(features.get(name, 0.0)) for name in FEATURES}


def _evaluate_horizon(samples: list[dict]) -> dict:
    history = []
    predictions = []
    for sample in sorted(samples, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            pred, confidence = _nearest_centroid_prediction(history, sample["features"])
            predictions.append(
                {
                    "day": sample["day"],
                    "anchor_ts_ms": sample["anchor_ts_ms"],
                    "actual": sample["label"],
                    "predicted": pred,
                    "confidence": confidence,
                    "correct": pred == sample["label"],
                    "reward_delta": sample["reward_delta"],
                }
            )
        history.append(sample)
    counts = Counter(sample["label"] for sample in samples)
    pred_counts = Counter(p["predicted"] for p in predictions)
    correct = sum(1 for p in predictions if p["correct"])
    majority = counts.most_common(1)[0][1] / sum(counts.values()) if counts else 0.0
    favorable_predictions = [p for p in predictions if p["predicted"] == "candidate_favorable"]
    actual_favorable = [p for p in predictions if p["actual"] == "candidate_favorable"]
    true_favorable = [p for p in favorable_predictions if p["actual"] == "candidate_favorable"]
    wrong_confident = [p for p in predictions if not p["correct"] and p["confidence"] >= 0.60]
    return {
        "samples": len(samples),
        "class_counts": dict(sorted(counts.items())),
        "prediction_coverage": len(predictions),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "accuracy": round(correct / len(predictions), 6) if predictions else 0.0,
        "majority_baseline_accuracy": round(majority, 6),
        "favorable_precision": round(len(true_favorable) / len(favorable_predictions), 6)
        if favorable_predictions
        else None,
        "favorable_recall": round(len(true_favorable) / len(actual_favorable), 6)
        if actual_favorable
        else None,
        "wrong_confident_count": len(wrong_confident),
        "wrong_confident_share": round(len(wrong_confident) / len(predictions), 6) if predictions else 0.0,
        "verdict": _verdict(correct / len(predictions) if predictions else 0.0, majority, predictions),
        "predictions": predictions,
    }


def _nearest_centroid_prediction(history: list[dict], features: dict) -> tuple[str, float]:
    grouped = defaultdict(list)
    for item in history:
        grouped[item["label"]].append(item["features"])
    centroids = {
        label: {name: sum(row[name] for row in rows) / len(rows) for name in FEATURES}
        for label, rows in grouped.items()
    }
    distances = {label: _distance(features, centroid) for label, centroid in centroids.items()}
    predicted = min(distances, key=distances.get)
    inv = {label: 1.0 / max(distance, 1e-9) for label, distance in distances.items()}
    total = sum(inv.values()) or 1.0
    confidence = inv[predicted] / total
    return predicted, round(confidence, 6)


def _distance(left: dict, right: dict) -> float:
    return sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in FEATURES))


def _verdict(accuracy: float, majority: float, predictions: list[dict]) -> str:
    if not predictions:
        return "inconclusive_no_predictions"
    if accuracy > majority + 0.03:
        return "beats_majority_candidate"
    if accuracy >= majority - 0.02:
        return "near_majority"
    return "below_majority"


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
