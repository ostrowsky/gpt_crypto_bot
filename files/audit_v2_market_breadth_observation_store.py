from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_SAMPLES = ROOT.parent / ".runtime" / "reports" / "v2_market_observation_features_15m.json"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_market_breadth_observation_store_15m.json"
LOCAL_TZ = ZoneInfo("Europe/Budapest")


def build(history_root: Path, samples_path: Path, output: Path) -> dict:
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = source["samples"]
    anchors = sorted({int(sample["anchor_ts_ms"]) for sample in samples})
    metrics_by_anchor, coverage = build_market_breadth_snapshots(history_root, anchors)

    enriched = []
    for sample in samples:
        anchor = int(sample["anchor_ts_ms"])
        breadth = metrics_by_anchor.get(anchor, {})
        enriched.append(
            {
                "day": sample["day"],
                "anchor_ts_ms": anchor,
                "horizon": sample["horizon"],
                "label": sample["label"],
                "reward_delta": sample["reward_delta"],
                "breadth_features": breadth,
                "combined_features": {**sample["features"], **{f"market_{k}": v for k, v in breadth.items()}},
            }
        )

    payload = {
        "source_samples": str(samples_path),
        "history_root": str(history_root),
        "coverage": coverage,
        "feature_counts": {
            "breadth": len(next(iter(metrics_by_anchor.values()), {})),
            "combined": len(enriched[0]["combined_features"]) if enriched else 0,
        },
        "horizons": {
            horizon: {
                "breadth_only": _evaluate_horizon(
                    [_as_feature_sample(s, "breadth_features") for s in enriched if s["horizon"] == horizon]
                ),
                "existing_plus_breadth": _evaluate_horizon(
                    [_as_feature_sample(s, "combined_features") for s in enriched if s["horizon"] == horizon]
                ),
            }
            for horizon in ("1h", "2h")
        },
        "samples": enriched,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_market_breadth_snapshots(history_root: Path, anchors: list[int]) -> tuple[dict[int, dict], dict]:
    symbol_metrics = {}
    tracked_symbols = 0
    for data_path in sorted(history_root.glob("*/15m.jsonl")):
        symbol = data_path.parent.name
        bars = _read_bars(symbol, data_path)
        if not bars:
            continue
        tracked_symbols += 1
        symbol_metrics[symbol] = _metrics_by_ts(symbol, bars)

    snapshots = {}
    for anchor in anchors:
        rows = []
        for symbol, by_ts in symbol_metrics.items():
            metrics = by_ts.get(anchor)
            if metrics:
                rows.append(metrics)
        snapshots[anchor] = _aggregate_market_breadth(rows, tracked_symbols)

    non_empty = sum(1 for features in snapshots.values() if features.get("available_symbols", 0.0) > 0)
    return snapshots, {
        "tracked_symbols": tracked_symbols,
        "anchors": len(anchors),
        "anchors_with_data": non_empty,
        "anchor_data_share": round(non_empty / len(anchors), 6) if anchors else 0.0,
    }


def _read_bars(symbol: str, path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            {
                "symbol": symbol,
                "open_ts_ms": int(payload["open_ts_ms"]),
                "open": float(payload["open"]),
                "high": float(payload["high"]),
                "low": float(payload["low"]),
                "close": float(payload["close"]),
                "volume": float(payload["volume"]),
            }
        )
    return sorted(rows, key=lambda row: row["open_ts_ms"])


def _metrics_by_ts(symbol: str, bars: list[dict]) -> dict[int, dict]:
    out = {}
    ema20 = None
    ema50 = None
    alpha20 = 2 / 21
    alpha50 = 2 / 51
    day_open = None
    day_key = None
    closes = []
    volumes = []
    for bar in bars:
        close = bar["close"]
        volume = bar["volume"]
        current_day = _local_day(bar["open_ts_ms"])
        if current_day != day_key:
            day_key = current_day
            day_open = close
        ema20 = close if ema20 is None else (close * alpha20) + (ema20 * (1 - alpha20))
        ema50 = close if ema50 is None else (close * alpha50) + (ema50 * (1 - alpha50))
        closes.append(close)
        volumes.append(volume)
        vol_window = volumes[-20:]
        mean_vol = _mean(vol_window)
        out[bar["open_ts_ms"]] = {
            "symbol": symbol,
            "ret_1bar_pct": _ret(close, closes[-2]) if len(closes) >= 2 else 0.0,
            "ret_4bar_pct": _ret(close, closes[-5]) if len(closes) >= 5 else 0.0,
            "ret_8bar_pct": _ret(close, closes[-9]) if len(closes) >= 9 else 0.0,
            "ret_day_pct": _ret(close, day_open) if day_open else 0.0,
            "above_ema20": 1.0 if close >= ema20 else 0.0,
            "above_ema50": 1.0 if close >= ema50 else 0.0,
            "price_vs_ema20_pct": _ret(close, ema20) if ema20 else 0.0,
            "volume_gt_mean20": 1.0 if mean_vol and volume > mean_vol else 0.0,
        }
    return out


def _aggregate_market_breadth(rows: list[dict], tracked_symbols: int) -> dict:
    available = len(rows)
    if not rows:
        return {
            "tracked_symbols": float(tracked_symbols),
            "available_symbols": 0.0,
            "available_share": 0.0,
        }
    ret1 = [row["ret_1bar_pct"] for row in rows]
    ret4 = [row["ret_4bar_pct"] for row in rows]
    ret8 = [row["ret_8bar_pct"] for row in rows]
    retd = [row["ret_day_pct"] for row in rows]
    by_symbol = {row["symbol"]: row for row in rows}
    btc = by_symbol.get("BTCUSDT", {})
    eth = by_symbol.get("ETHUSDT", {})
    return _round_features(
        {
            "tracked_symbols": float(tracked_symbols),
            "available_symbols": float(available),
            "available_share": available / tracked_symbols if tracked_symbols else 0.0,
            "ret1_positive_share": _positive_share(ret1),
            "ret4_positive_share": _positive_share(ret4),
            "ret8_positive_share": _positive_share(ret8),
            "ret_day_positive_share": _positive_share(retd),
            "above_ema20_share": _mean([row["above_ema20"] for row in rows]),
            "above_ema50_share": _mean([row["above_ema50"] for row in rows]),
            "volume_gt_mean20_share": _mean([row["volume_gt_mean20"] for row in rows]),
            "ret4_mean": _mean(ret4),
            "ret4_std": _std(ret4),
            "ret8_mean": _mean(ret8),
            "ret8_std": _std(ret8),
            "ret_day_mean": _mean(retd),
            "ret_day_std": _std(retd),
            "ret_day_top_decile_minus_median": _top_decile_minus_median(retd),
            "btc_ret4_pct": float(btc.get("ret_4bar_pct", 0.0)),
            "btc_ret_day_pct": float(btc.get("ret_day_pct", 0.0)),
            "btc_price_vs_ema20_pct": float(btc.get("price_vs_ema20_pct", 0.0)),
            "eth_ret4_pct": float(eth.get("ret_4bar_pct", 0.0)),
            "eth_ret_day_pct": float(eth.get("ret_day_pct", 0.0)),
            "eth_price_vs_ema20_pct": float(eth.get("price_vs_ema20_pct", 0.0)),
        }
    )


def _as_feature_sample(sample: dict, key: str) -> dict:
    return {
        "day": sample["day"],
        "anchor_ts_ms": sample["anchor_ts_ms"],
        "label": sample["label"],
        "features": sample[key],
    }


def _evaluate_horizon(samples: list[dict]) -> dict:
    history = []
    predictions = []
    for sample in sorted(samples, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"} and sample["features"]:
            pred, confidence = _nearest_centroid_prediction(history, sample["features"])
            predictions.append(
                {
                    "actual": sample["label"],
                    "predicted": pred,
                    "confidence": confidence,
                    "correct": pred == sample["label"],
                }
            )
        history.append(sample)
    counts = Counter(sample["label"] for sample in samples)
    correct = sum(1 for item in predictions if item["correct"])
    majority = counts.most_common(1)[0][1] / sum(counts.values()) if counts else 0.0
    wrong_confident = [p for p in predictions if not p["correct"] and p["confidence"] >= 0.60]
    accuracy = correct / len(predictions) if predictions else 0.0
    return {
        "samples": len(samples),
        "class_counts": dict(sorted(counts.items())),
        "prediction_coverage": len(predictions),
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
        if item["features"]:
            grouped[item["label"]].append(_standardize(item["features"], names, stats))
    current = _standardize(features, names, stats)
    centroids = {
        label: {name: _mean([row[name] for row in rows]) for name in names}
        for label, rows in grouped.items()
    }
    distances = {label: _distance(current, centroid, names) for label, centroid in centroids.items()}
    predicted = min(distances, key=distances.get)
    inv = {label: 1.0 / max(distance, 1e-9) for label, distance in distances.items()}
    total = sum(inv.values()) or 1.0
    return predicted, round(inv[predicted] / total, 6)


def _history_stats(history: list[dict], names: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for name in names:
        values = [float(item["features"].get(name, 0.0)) for item in history if item["features"]]
        mean = _mean(values)
        std = max(_std(values), 1e-9)
        stats[name] = (mean, std)
    return stats


def _standardize(features: dict, names: list[str], stats: dict[str, tuple[float, float]]) -> dict:
    return {name: (float(features.get(name, 0.0)) - stats[name][0]) / stats[name][1] for name in names}


def _distance(left: dict, right: dict, names: list[str]) -> float:
    return math.sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in names))


def _local_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=LOCAL_TZ).date().isoformat()


def _ret(current: float, previous: float | None) -> float:
    if previous is None or previous == 0:
        return 0.0
    return (current / previous - 1.0) * 100.0


def _positive_share(values: list[float]) -> float:
    return sum(1 for value in values if value > 0.0) / len(values) if values else 0.0


def _top_decile_minus_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    top_start = max(0, int(len(ordered) * 0.9) - 1)
    return _mean(ordered[top_start:]) - float(median(ordered))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float], mean: float) -> float:
    return sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values, _mean(values))) if values else 0.0


def _round_features(features: dict) -> dict:
    return {name: round(float(value), 6) for name, value in sorted(features.items())}


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
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.samples, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["horizons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
