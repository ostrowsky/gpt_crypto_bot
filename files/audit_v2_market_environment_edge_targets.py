from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_observation_features_15m.json"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_environment_edge_targets_15m.json"
THRESHOLDS = (0.5, 1.0, 2.0, 5.0, 10.0)


def build(input_path: Path, output: Path) -> dict:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    samples = source["samples"]
    payload = {
        "source": str(input_path),
        "thresholds": list(THRESHOLDS),
        "horizons": {
            horizon: {
                str(threshold): _evaluate_threshold(
                    [sample for sample in samples if sample["horizon"] == horizon],
                    threshold,
                )
                for threshold in THRESHOLDS
            }
            for horizon in ("1h", "2h")
        },
    }
    payload["selection"] = _select(payload["horizons"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _evaluate_threshold(samples: list[dict], threshold: float) -> dict:
    labeled = []
    for sample in samples:
        label = _edge_label(float(sample["reward_delta"]), threshold)
        labeled.append({**sample, "edge_label": label})
    actionable = [sample for sample in labeled if sample["edge_label"] != "no_edge"]
    history = []
    predictions = []
    for sample in sorted(actionable, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["edge_label"] for item in history}
        if labels == {"candidate_edge", "base_edge"}:
            pred, confidence = _nearest_centroid_prediction(history, sample["features"])
            predictions.append(
                {
                    "actual": sample["edge_label"],
                    "predicted": pred,
                    "confidence": confidence,
                    "correct": pred == sample["edge_label"],
                    "reward_delta": sample["reward_delta"],
                }
            )
        history.append(sample)
    counts = Counter(sample["edge_label"] for sample in actionable)
    all_counts = Counter(sample["edge_label"] for sample in labeled)
    correct = sum(1 for item in predictions if item["correct"])
    majority = counts.most_common(1)[0][1] / sum(counts.values()) if counts else 0.0
    candidate_predictions = [p for p in predictions if p["predicted"] == "candidate_edge"]
    actual_candidate = [p for p in predictions if p["actual"] == "candidate_edge"]
    true_candidate = [p for p in candidate_predictions if p["actual"] == "candidate_edge"]
    accuracy = correct / len(predictions) if predictions else 0.0
    verdict = _verdict(accuracy, majority, len(actionable), len(samples))
    return {
        "threshold": threshold,
        "samples": len(samples),
        "actionable_samples": len(actionable),
        "actionable_share": round(len(actionable) / len(samples), 6) if samples else 0.0,
        "class_counts": dict(sorted(counts.items())),
        "all_label_counts": dict(sorted(all_counts.items())),
        "prediction_coverage": len(predictions),
        "accuracy": round(accuracy, 6),
        "majority_baseline_accuracy": round(majority, 6),
        "candidate_edge_precision": round(len(true_candidate) / len(candidate_predictions), 6)
        if candidate_predictions
        else None,
        "candidate_edge_recall": round(len(true_candidate) / len(actual_candidate), 6)
        if actual_candidate
        else None,
        "verdict": verdict,
    }


def _edge_label(delta: float, threshold: float) -> str:
    if delta >= threshold:
        return "candidate_edge"
    if delta <= -threshold:
        return "base_edge"
    return "no_edge"


def _nearest_centroid_prediction(history: list[dict], features: dict) -> tuple[str, float]:
    names = list(features)
    grouped = defaultdict(list)
    for item in history:
        grouped[item["edge_label"]].append(item["features"])
    centroids = {
        label: {name: sum(row[name] for row in rows) / len(rows) for name in names}
        for label, rows in grouped.items()
    }
    distances = {label: _distance(features, centroid, names) for label, centroid in centroids.items()}
    predicted = min(distances, key=distances.get)
    inv = {label: 1.0 / max(distance, 1e-9) for label, distance in distances.items()}
    total = sum(inv.values()) or 1.0
    return predicted, round(inv[predicted] / total, 6)


def _distance(left: dict, right: dict, names: list[str]) -> float:
    return sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in names))


def _verdict(accuracy: float, majority: float, actionable: int, total: int) -> str:
    if actionable < max(30, int(total * 0.25)):
        return "too_low_coverage"
    if accuracy > majority + 0.03:
        return "beats_majority_candidate"
    if accuracy >= majority - 0.02:
        return "near_majority"
    return "below_majority"


def _select(horizons: dict) -> dict:
    candidates = []
    for horizon, values in horizons.items():
        for threshold, result in values.items():
            candidates.append(
                {
                    "horizon": horizon,
                    "threshold": float(threshold),
                    "accuracy": result["accuracy"],
                    "majority": result["majority_baseline_accuracy"],
                    "edge": round(result["accuracy"] - result["majority_baseline_accuracy"], 6),
                    "actionable_share": result["actionable_share"],
                    "verdict": result["verdict"],
                }
            )
    candidates.sort(key=lambda item: (item["edge"], item["actionable_share"]), reverse=True)
    return {"best_by_accuracy_edge": candidates[0] if candidates else None, "ranked": candidates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.input, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
