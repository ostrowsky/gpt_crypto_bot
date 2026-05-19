from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = ROOT.parent / ".runtime" / "reports" / "v2_market_breadth_observation_store_15m.json"
DEFAULT_MODEL = ROOT / "watchlist_top_gainer_model.json"
DEFAULT_ML_DATASET = ROOT / "ml_dataset.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_v1_market_structure_feature_audit_15m.json"

V1_STRUCTURE_TOKENS = (
    "ema",
    "slope",
    "adx",
    "rsi",
    "macd",
    "atr",
    "range",
    "vol_x",
    "wick",
    "body",
    "btc_",
    "market_vol",
    "signal_",
    "leader",
    "forecast",
    "candidate_score",
)

GROUPS = {
    "v1_projected_level": (
        "prefix_adx",
        "prefix_daily_range_pct",
        "prefix_price_vs_ema20_pct",
        "prefix_projected_forecast_proxy_pct",
        "prefix_projected_leader_score_trend",
        "recent_adx",
        "recent_daily_range_pct",
        "recent_price_vs_ema20_pct",
        "recent_projected_forecast_proxy_pct",
        "recent_projected_leader_score_trend",
    ),
    "v1_projected_delta": (
        "delta_adx",
        "delta_daily_range_pct",
        "delta_price_vs_ema20_pct",
        "delta_projected_forecast_proxy_pct",
        "delta_projected_leader_score_trend",
    ),
    "v1_lifecycle_mix": (
        "prefix_noise_share",
        "prefix_emerging_share",
        "prefix_mature_share",
        "prefix_belief_late_mass",
        "recent_noise_share",
        "recent_emerging_share",
        "recent_mature_share",
        "recent_belief_late_mass",
        "delta_noise_share",
        "delta_emerging_share",
        "delta_mature_share",
        "delta_belief_late_mass",
    ),
    "v1_row_breadth": (
        "breadth_ema20_positive_share",
        "breadth_rsi50_share",
        "breadth_forecast_gt1_share",
        "breadth_forecast_mean",
        "breadth_forecast_std",
        "breadth_leader_mean",
        "breadth_leader_std",
    ),
    "market_ohlcv_breadth": (
        "market_ret1_positive_share",
        "market_ret4_positive_share",
        "market_ret8_positive_share",
        "market_ret_day_positive_share",
        "market_above_ema20_share",
        "market_above_ema50_share",
        "market_volume_gt_mean20_share",
    ),
    "market_dispersion": (
        "market_ret4_mean",
        "market_ret4_std",
        "market_ret8_mean",
        "market_ret8_std",
        "market_ret_day_mean",
        "market_ret_day_std",
        "market_ret_day_top_decile_minus_median",
    ),
    "btc_eth_risk": (
        "market_btc_ret4_pct",
        "market_btc_ret_day_pct",
        "market_btc_price_vs_ema20_pct",
        "market_eth_ret4_pct",
        "market_eth_ret_day_pct",
        "market_eth_price_vs_ema20_pct",
    ),
}


def build(samples_path: Path, model_path: Path, ml_dataset_path: Path, output: Path) -> dict:
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = source["samples"]
    model_audit = _audit_watchlist_model(model_path)
    dataset_audit = _audit_ml_dataset(ml_dataset_path, model_audit.get("market_structure_features", []))
    selection = _feature_selection(samples)
    payload = {
        "source_samples": str(samples_path),
        "watchlist_model": model_audit,
        "ml_dataset": dataset_audit,
        "feature_selection": selection,
        "decision": _decision(selection),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _audit_watchlist_model(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = list(payload.get("feature_names") or [])
    weights = list((payload.get("model") or {}).get("weights") or [])
    structure = [name for name in names if _is_v1_structure_feature(name)]
    weighted = []
    for name, weight in zip(names, weights):
        if _is_v1_structure_feature(name):
            weighted.append({"feature": name, "weight": round(float(weight), 6), "abs_weight": round(abs(float(weight)), 6)})
    weighted.sort(key=lambda item: item["abs_weight"], reverse=True)
    return {
        "exists": True,
        "feature_count": len(names),
        "market_structure_feature_count": len(structure),
        "market_structure_features": structure,
        "top_abs_structure_weights": weighted[:20],
    }


def _audit_ml_dataset(path: Path, structure_features: list[str]) -> dict:
    if not path.exists():
        return {"exists": False}
    rows = list(_iter_jsonl(path))
    if not rows:
        return {"exists": True, "rows": 0}
    ts_values = []
    coverage = {name: 0 for name in structure_features}
    signal_counts = Counter()
    tfs = Counter()
    for row in rows:
        ts = _row_ts_ms(row)
        if ts is not None and _is_reasonable_ts_ms(ts):
            ts_values.append(ts)
        tfs[str(row.get("tf") or "")] += 1
        signal_counts[str(row.get("signal_type") or "")] += 1
        features = _model_feature_values(row, structure_features)
        for name, value in features.items():
            if value is not None:
                coverage[name] += 1
    n = len(rows)
    coverage_rows = [
        {"feature": name, "coverage": round(count / n, 6), "non_null_rows": count}
        for name, count in coverage.items()
    ]
    coverage_rows.sort(key=lambda item: (item["coverage"], item["feature"]), reverse=True)
    return {
        "exists": True,
        "rows": n,
        "start_utc": _fmt_ts(min(ts_values)) if ts_values else None,
        "end_utc": _fmt_ts(max(ts_values)) if ts_values else None,
        "timeframes": dict(sorted(tfs.items())),
        "signal_type_counts": dict(sorted(signal_counts.items())),
        "structure_feature_coverage_top": coverage_rows[:30],
        "structure_features_with_full_coverage": sum(1 for row in coverage_rows if row["coverage"] >= 0.99),
    }


def _feature_selection(samples: list[dict]) -> dict:
    available = sorted(set().union(*(set(sample.get("combined_features", {})) for sample in samples))) if samples else []
    group_results = {}
    for group, names in GROUPS.items():
        selected = [name for name in names if name in available]
        group_results[group] = {
            horizon: _evaluate_horizon(_select_samples(samples, horizon, selected))
            for horizon in ("1h", "2h")
        }
    all_v1_names = sorted(
        name
        for name in available
        if not name.startswith("market_") and (_is_v1_structure_feature(name) or name.startswith("breadth_"))
    )
    all_market_names = sorted(name for name in available if name.startswith("market_"))
    group_results["all_v1_structure"] = {
        horizon: _evaluate_horizon(_select_samples(samples, horizon, all_v1_names))
        for horizon in ("1h", "2h")
    }
    group_results["all_market"] = {
        horizon: _evaluate_horizon(_select_samples(samples, horizon, all_market_names))
        for horizon in ("1h", "2h")
    }
    group_results["all_combined"] = {
        horizon: _evaluate_horizon(_select_samples(samples, horizon, available))
        for horizon in ("1h", "2h")
    }
    greedy = {horizon: _greedy_forward(samples, horizon, available, max_features=8) for horizon in ("1h", "2h")}
    return {
        "available_feature_count": len(available),
        "group_results": group_results,
        "greedy_forward": greedy,
    }


def _select_samples(samples: list[dict], horizon: str, names: list[str]) -> list[dict]:
    out = []
    for sample in samples:
        if sample["horizon"] != horizon:
            continue
        features = sample.get("combined_features", {})
        out.append(
            {
                "day": sample["day"],
                "anchor_ts_ms": int(sample["anchor_ts_ms"]),
                "label": sample["label"],
                "features": {name: float(features.get(name, 0.0)) for name in names},
            }
        )
    return out


def _greedy_forward(samples: list[dict], horizon: str, candidates: list[str], max_features: int) -> dict:
    selected: list[str] = []
    steps = []
    remaining = list(candidates)
    best_edge = -999.0
    for _ in range(max_features):
        best = None
        for name in remaining:
            result = _evaluate_horizon(_select_samples(samples, horizon, selected + [name]))
            score = (result["accuracy_edge"], -result["wrong_confident_share"], result["accuracy"])
            if best is None or score > best[0]:
                best = (score, name, result)
        if best is None:
            break
        _, name, result = best
        selected.append(name)
        remaining.remove(name)
        best_edge = result["accuracy_edge"]
        steps.append({"feature": name, "selected": list(selected), "result": result})
        if result["accuracy_edge"] > 0.03:
            break
    final = steps[-1]["result"] if steps else _evaluate_horizon(_select_samples(samples, horizon, []))
    return {"selected_features": selected, "final": final, "steps": steps, "best_edge": best_edge}


def _evaluate_horizon(samples: list[dict]) -> dict:
    history = []
    predictions = []
    for sample in sorted(samples, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"} and sample["features"]:
            pred, confidence = _nearest_centroid_prediction(history, sample["features"])
            predictions.append({"actual": sample["label"], "predicted": pred, "confidence": confidence, "correct": pred == sample["label"]})
        history.append(sample)
    counts = Counter(sample["label"] for sample in samples)
    correct = sum(1 for item in predictions if item["correct"])
    majority = counts.most_common(1)[0][1] / sum(counts.values()) if counts else 0.0
    accuracy = correct / len(predictions) if predictions else 0.0
    wrong_confident = [p for p in predictions if not p["correct"] and p["confidence"] >= 0.60]
    return {
        "samples": len(samples),
        "features": len(samples[0]["features"]) if samples else 0,
        "prediction_coverage": len(predictions),
        "class_counts": dict(sorted(counts.items())),
        "accuracy": round(accuracy, 6),
        "majority_baseline_accuracy": round(majority, 6),
        "accuracy_edge": round(accuracy - majority, 6),
        "wrong_confident_share": round(len(wrong_confident) / len(predictions), 6) if predictions else 0.0,
        "verdict": _verdict(accuracy, majority, predictions),
    }


def _nearest_centroid_prediction(history: list[dict], features: dict) -> tuple[str, float]:
    names = list(features)
    stats = _history_stats(history, names)
    grouped = defaultdict(list)
    for item in history:
        grouped[item["label"]].append(_standardize(item["features"], names, stats))
    current = _standardize(features, names, stats)
    centroids = {label: {name: _mean([row[name] for row in rows]) for name in names} for label, rows in grouped.items()}
    distances = {label: _distance(current, centroid, names) for label, centroid in centroids.items()}
    predicted = min(distances, key=distances.get)
    inv = {label: 1.0 / max(distance, 1e-9) for label, distance in distances.items()}
    total = sum(inv.values()) or 1.0
    return predicted, round(inv[predicted] / total, 6)


def _history_stats(history: list[dict], names: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for name in names:
        values = [float(item["features"].get(name, 0.0)) for item in history]
        stats[name] = (_mean(values), max(_std(values), 1e-9))
    return stats


def _standardize(features: dict, names: list[str], stats: dict[str, tuple[float, float]]) -> dict:
    return {name: (float(features.get(name, 0.0)) - stats[name][0]) / stats[name][1] for name in names}


def _distance(left: dict, right: dict, names: list[str]) -> float:
    return math.sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in names))


def _decision(selection: dict) -> dict:
    best = []
    for name, horizons in selection.get("group_results", {}).items():
        for horizon, result in horizons.items():
            best.append({"type": "group", "name": name, "horizon": horizon, **result})
    for horizon, result in selection.get("greedy_forward", {}).items():
        best.append({"type": "greedy", "name": "greedy_forward", "horizon": horizon, **result["final"]})
    best.sort(key=lambda item: item["accuracy_edge"], reverse=True)
    winner = best[0] if best else None
    return {
        "best": winner,
        "promotion_gate_passed": bool(winner and winner["accuracy_edge"] > 0.03),
        "recommendation": "run switched replay from selected features" if winner and winner["accuracy_edge"] > 0.03 else "keep as research primitives; do not run switched replay",
    }


def _is_v1_structure_feature(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in V1_STRUCTURE_TOKENS)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def _model_feature_values(row: dict, names: list[str]) -> dict[str, float | None]:
    direct = row.get("f") or {}
    seq_names = row.get("seq_feature_names") or []
    seq = row.get("seq") or []
    values = {}
    for name in names:
        if name in direct:
            values[name] = _safe_float(direct.get(name))
        elif name.startswith("signal_"):
            values[name] = 1.0 if str(row.get("signal_type") or "") == name.removeprefix("signal_") else 0.0
        elif name.startswith("tf_"):
            values[name] = 1.0 if str(row.get("tf") or "") == name.removeprefix("tf_") else 0.0
        elif name.startswith("seq_"):
            values[name] = _seq_feature_value(name, seq, seq_names)
        else:
            values[name] = None
    return values


def _seq_feature_value(name: str, seq: list, seq_names: list[str]) -> float | None:
    parts = name.split("_", 2)
    if len(parts) != 3 or not seq or not seq_names:
        return None
    _, op, base = parts
    try:
        idx = seq_names.index(base)
    except ValueError:
        return None
    values = [_safe_float(row[idx]) for row in seq if isinstance(row, list) and len(row) > idx]
    if not values:
        return None
    if op == "last":
        return values[-1]
    if op == "mean":
        return _mean(values)
    if op == "tail":
        return _mean(values[-5:])
    if op == "trend":
        return values[-1] - values[0]
    return None


def _row_ts_ms(row: dict) -> int | None:
    if row.get("bar_ts") is not None:
        try:
            return int(row["bar_ts"])
        except Exception:
            return None
    text = row.get("ts_signal")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _is_reasonable_ts_ms(ts_ms: int) -> bool:
    # Keep the dataset audit robust against malformed runtime rows.
    return 946684800000 <= int(ts_ms) <= 4102444800000


def _fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out or math.isinf(out) else out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


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
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ml-dataset", type=Path, default=DEFAULT_ML_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.samples, args.model, args.ml_dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
