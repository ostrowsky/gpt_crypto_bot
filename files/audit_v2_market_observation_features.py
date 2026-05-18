from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_market_environment_target_design import EVAL_START_BAR, EVAL_STEP_BARS, HORIZONS, _reward_on_horizon
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from run_v2_policy_baselines import _build_episodes


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_observation_features_15m.json"
RECENT_BARS = 8
BASE_NAMES = (
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
            prefix_ts = set(ts_values[:start])
            recent_ts = set(ts_values[max(0, start - RECENT_BARS) : start])
            prior_ts = set(ts_values[: max(0, start - RECENT_BARS)])
            features = _observation_features(day_episodes, rows, prefix_ts, recent_ts, prior_ts)
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
        "parameters": {"recent_bars": RECENT_BARS, "eval_start_bar": EVAL_START_BAR, "eval_step_bars": EVAL_STEP_BARS},
        "feature_count": len(samples[0]["features"]) if samples else 0,
        "horizons": {
            horizon: _evaluate_models([s for s in samples if s["horizon"] == horizon])
            for horizon in ("1h", "2h")
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _observation_features(episodes, rows, prefix_ts: set[int], recent_ts: set[int], prior_ts: set[int]) -> dict:
    prefix = _aggregate_rows(_rows_for_ts(episodes, rows, prefix_ts))
    recent = _aggregate_rows(_rows_for_ts(episodes, rows, recent_ts))
    prior = _aggregate_rows(_rows_for_ts(episodes, rows, prior_ts))
    latest = _latest_rows(episodes, rows, prefix_ts)
    breadth = _breadth_features(latest)
    out = {}
    for name in BASE_NAMES:
        out[f"prefix_{name}"] = prefix.get(name, 0.0)
        out[f"recent_{name}"] = recent.get(name, 0.0)
        out[f"delta_{name}"] = recent.get(name, 0.0) - prior.get(name, 0.0)
    out.update(breadth)
    return {name: round(float(value), 6) for name, value in sorted(out.items())}


def _rows_for_ts(episodes, rows, ts_set: set[int]) -> list[dict]:
    out = []
    for env in episodes:
        for frame in env._frames:
            if frame.bar.open_ts_ms in ts_set:
                row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
                if row:
                    out.append(row)
    return out


def _latest_rows(episodes, rows, ts_set: set[int]) -> list[dict]:
    by_symbol = {}
    for env in episodes:
        for frame in env._frames:
            if frame.bar.open_ts_ms in ts_set:
                row = rows.get((frame.bar.symbol, frame.bar.open_ts_ms))
                if row:
                    by_symbol[frame.bar.symbol] = row
    return list(by_symbol.values())


def _aggregate_rows(items: list[dict]) -> dict:
    total = len(items) or 1
    states = Counter(row["true_state"] for row in items)
    vals = defaultdict(list)
    for row in items:
        belief = row["belief"]
        projected = row.get("v1_projected_structural") or {}
        vals["belief_late_mass"].append(float(belief.get("exhaustion", 0.0)) + float(belief.get("reversal", 0.0)))
        for name in (
            "adx",
            "projected_leader_score_trend",
            "daily_range_pct",
            "projected_forecast_proxy_pct",
            "price_vs_ema20_pct",
        ):
            vals[name].append(float(projected.get(name, 0.0)))
    out = {
        "noise_share": states["noise"] / total,
        "emerging_share": states["emerging_move"] / total,
        "mature_share": states["mature_trend"] / total,
    }
    out.update({name: _mean(values) for name, values in vals.items()})
    return out


def _breadth_features(items: list[dict]) -> dict:
    if not items:
        return {}
    forecast = []
    leader = []
    ema_positive = 0
    rsi_positive = 0
    forecast_gt1 = 0
    for row in items:
        projected = row.get("v1_projected_structural") or {}
        f = float(projected.get("projected_forecast_proxy_pct", 0.0))
        l = float(projected.get("projected_leader_score_trend", 0.0))
        forecast.append(f)
        leader.append(l)
        ema_positive += float(projected.get("price_vs_ema20_pct", 0.0)) > 0.0
        rsi_positive += float(projected.get("rsi", 0.0)) >= 50.0
        forecast_gt1 += f > 1.0
    n = len(items)
    return {
        "breadth_ema20_positive_share": ema_positive / n,
        "breadth_rsi50_share": rsi_positive / n,
        "breadth_forecast_gt1_share": forecast_gt1 / n,
        "breadth_forecast_mean": _mean(forecast),
        "breadth_forecast_std": _std(forecast),
        "breadth_leader_mean": _mean(leader),
        "breadth_leader_std": _std(leader),
    }


def _evaluate_models(samples: list[dict]) -> dict:
    centroid = _evaluate_centroid(samples)
    nb = _evaluate_naive_bayes(samples)
    return {"nearest_centroid": _compact(centroid), "gaussian_nb": nb}


def _evaluate_centroid(samples: list[dict]) -> dict:
    history = []
    predictions = []
    for sample in sorted(samples, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            pred, conf = _centroid_predict(history, sample["features"])
            predictions.append(
                {
                    "actual": sample["label"],
                    "predicted": pred,
                    "confidence": conf,
                    "correct": pred == sample["label"],
                }
            )
        history.append(sample)
    return _classification_summary(samples, predictions)


def _evaluate_naive_bayes(samples: list[dict]) -> dict:
    history = []
    predictions = []
    for sample in sorted(samples, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            pred, conf = _nb_predict(history, sample["features"])
            predictions.append(
                {
                    "actual": sample["label"],
                    "predicted": pred,
                    "confidence": conf,
                    "correct": pred == sample["label"],
                }
            )
        history.append(sample)
    return _classification_summary(samples, predictions)


def _classification_summary(samples: list[dict], predictions: list[dict]) -> dict:
    counts = Counter(s["label"] for s in samples)
    correct = sum(1 for p in predictions if p["correct"])
    majority = counts.most_common(1)[0][1] / sum(counts.values()) if counts else 0.0
    favorable_predictions = [p for p in predictions if p["predicted"] == "candidate_favorable"]
    actual_favorable = [p for p in predictions if p["actual"] == "candidate_favorable"]
    true_favorable = [p for p in favorable_predictions if p["actual"] == "candidate_favorable"]
    wrong_confident = [p for p in predictions if not p["correct"] and p["confidence"] >= 0.60]
    accuracy = correct / len(predictions) if predictions else 0.0
    return {
        "samples": len(samples),
        "class_counts": dict(sorted(counts.items())),
        "prediction_coverage": len(predictions),
        "accuracy": round(accuracy, 6),
        "majority_baseline_accuracy": round(majority, 6),
        "favorable_precision": round(len(true_favorable) / len(favorable_predictions), 6)
        if favorable_predictions
        else None,
        "favorable_recall": round(len(true_favorable) / len(actual_favorable), 6)
        if actual_favorable
        else None,
        "wrong_confident_share": round(len(wrong_confident) / len(predictions), 6) if predictions else 0.0,
        "verdict": _verdict(accuracy, majority, predictions),
    }


def _centroid_predict(history: list[dict], features: dict) -> tuple[str, float]:
    names = list(features)
    grouped = defaultdict(list)
    for item in history:
        grouped[item["label"]].append(item["features"])
    centroids = {
        label: {name: sum(row[name] for row in rows) / len(rows) for name in names}
        for label, rows in grouped.items()
    }
    distances = {label: _distance(features, centroid, names) for label, centroid in centroids.items()}
    pred = min(distances, key=distances.get)
    inv = {label: 1.0 / max(distance, 1e-9) for label, distance in distances.items()}
    total = sum(inv.values()) or 1.0
    return pred, round(inv[pred] / total, 6)


def _nb_predict(history: list[dict], features: dict) -> tuple[str, float]:
    names = list(features)
    grouped = defaultdict(list)
    for item in history:
        grouped[item["label"]].append(item["features"])
    logps = {}
    total = len(history)
    for label, rows in grouped.items():
        logp = math.log(len(rows) / total)
        for name in names:
            values = [row[name] for row in rows]
            mean = _mean(values)
            var = max(_variance(values, mean), 1e-6)
            x = features[name]
            logp += -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
        logps[label] = logp
    mx = max(logps.values())
    probs = {label: math.exp(value - mx) for label, value in logps.items()}
    total_prob = sum(probs.values()) or 1.0
    probs = {label: value / total_prob for label, value in probs.items()}
    pred = max(probs, key=probs.get)
    return pred, round(probs[pred], 6)


def _distance(left: dict, right: dict, names: list[str]) -> float:
    return math.sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in names))


def _compact(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "predictions"}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float], mean: float) -> float:
    return sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values, _mean(values))) if values else 0.0


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
