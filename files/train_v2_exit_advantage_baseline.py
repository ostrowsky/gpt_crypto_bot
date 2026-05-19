from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_action_level_exit_advantage_15m.jsonl"
DEFAULT_MODEL = ROOT.parent / ".runtime" / "reports" / "v2_exit_advantage_baseline_model_15m.json"
DEFAULT_REPORT = ROOT.parent / ".runtime" / "reports" / "v2_exit_advantage_baseline_report_15m.json"

FEATURE_NAMES = [
    "belief_noise",
    "belief_open_mass",
    "belief_mature",
    "belief_late_mass",
    "projected_leader_score_trend",
    "price_vs_ema20_pct",
    "rsi",
    "adx",
    "daily_range_pct",
    "late_mass_delta_3",
    "mature_delta_3",
    "rsi_delta_3",
    "price_vs_ema20_delta_3",
]

DEFAULT_THRESHOLDS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]


def build(dataset_path: Path, model_output: Path, report_output: Path, train_fraction: float = 0.70, ridge_alpha: float = 1.0) -> dict:
    rows = _load_rows(dataset_path)
    train_rows, holdout_rows = _chronological_split(rows, train_fraction=train_fraction)
    x_train, y_train = _matrix(train_rows)
    x_holdout, y_holdout = _matrix(holdout_rows)
    scaler = _fit_scaler(x_train)
    x_train_s = _apply_scaler(x_train, scaler)
    x_holdout_s = _apply_scaler(x_holdout, scaler)
    model = _fit_ridge(x_train_s, y_train, ridge_alpha)
    pred_train = _predict(x_train_s, model)
    pred_holdout = _predict(x_holdout_s, model)
    thresholds = [_evaluate_threshold(holdout_rows, pred_holdout, threshold) for threshold in DEFAULT_THRESHOLDS]
    selected = _select_threshold(thresholds)
    payload_model = {
        "model_type": "ridge_linear_regression",
        "target": "sell_advantage",
        "feature_names": FEATURE_NAMES,
        "ridge_alpha": ridge_alpha,
        "train_fraction": train_fraction,
        "intercept": round(float(model[0]), 10),
        "coefficients": {name: round(float(value), 10) for name, value in zip(FEATURE_NAMES, model[1:])},
        "scaler": {
            "mean": {name: round(float(value), 10) for name, value in zip(FEATURE_NAMES, scaler["mean"])},
            "std": {name: round(float(value), 10) for name, value in zip(FEATURE_NAMES, scaler["std"])},
        },
        "selected_threshold": selected["threshold"],
    }
    report = {
        "dataset": str(dataset_path),
        "model_output": str(model_output),
        "rows": len(rows),
        "train": _split_summary(train_rows, pred_train, y_train),
        "holdout": _split_summary(holdout_rows, pred_holdout, y_holdout),
        "thresholds": thresholds,
        "holdout_advantage_baselines": _advantage_baselines(holdout_rows),
        "selected_threshold": selected,
        "top_coefficients_by_abs": _top_coefficients(payload_model["coefficients"]),
        "decision": _decision(selected, _advantage_baselines(holdout_rows)),
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(payload_model, ensure_ascii=False, indent=2), encoding="utf-8")
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "sell_advantage" in row and "features" in row and "ts_ms" in row:
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row["ts_ms"]), str(row.get("symbol", ""))))


def _chronological_split(rows: list[dict], train_fraction: float = 0.70) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    return rows[:split], rows[split:]


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[_feature(row, name) for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.array([float(row["sell_advantage"]) for row in rows], dtype=float)
    return x, y


def _feature(row: dict, name: str) -> float:
    value = (row.get("features") or {}).get(name, 0.0)
    try:
        value = float(value)
    except Exception:
        value = 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _fit_scaler(x: np.ndarray) -> dict:
    mean = x.mean(axis=0) if len(x) else np.zeros(len(FEATURE_NAMES))
    std = x.std(axis=0) if len(x) else np.ones(len(FEATURE_NAMES))
    std = np.where(std < 1e-9, 1.0, std)
    return {"mean": mean, "std": std}


def _apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    return (x - scaler["mean"]) / scaler["std"]


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def _predict(x: np.ndarray, model: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ model


def _split_summary(rows: list[dict], pred: np.ndarray, y: np.ndarray) -> dict:
    if len(rows) == 0:
        return {}
    errors = pred - y
    positive = y > 0.0
    pred_positive = pred > 0.0
    strong = y >= 1.0
    return {
        "rows": len(rows),
        "ts_start": int(rows[0]["ts_ms"]),
        "ts_end": int(rows[-1]["ts_ms"]),
        "sell_advantage_avg": round(float(y.mean()), 6),
        "predicted_advantage_avg": round(float(pred.mean()), 6),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 6),
        "directional_accuracy": round(float(np.mean(positive == pred_positive)), 6),
        "positive_rate": round(float(np.mean(positive)), 6),
        "predicted_positive_rate": round(float(np.mean(pred_positive)), 6),
        "strong_rate": round(float(np.mean(strong)), 6),
    }


def _evaluate_threshold(rows: list[dict], pred: np.ndarray, threshold: float) -> dict:
    y = np.array([float(row["sell_advantage"]) for row in rows], dtype=float)
    sell = pred >= threshold
    actual_positive = y > 0.0
    actual_strong = y >= 1.0
    sell_count = int(np.sum(sell))
    true_positive = int(np.sum(sell & actual_positive))
    true_strong = int(np.sum(sell & actual_strong))
    missed_strong = int(np.sum((~sell) & actual_strong))
    bad_sell = int(np.sum(sell & (y <= -1.0)))
    captured_advantage = float(np.sum(np.where(sell, y, 0.0)))
    return {
        "threshold": threshold,
        "sell_count": sell_count,
        "sell_rate": round(sell_count / len(rows), 6) if rows else 0.0,
        "positive_precision": round(true_positive / sell_count, 6) if sell_count else 0.0,
        "positive_recall": round(true_positive / int(np.sum(actual_positive)), 6) if np.sum(actual_positive) else 0.0,
        "strong_precision": round(true_strong / sell_count, 6) if sell_count else 0.0,
        "strong_recall": round(true_strong / int(np.sum(actual_strong)), 6) if np.sum(actual_strong) else 0.0,
        "missed_strong_sell": missed_strong,
        "bad_sell_count": bad_sell,
        "captured_advantage_sum": round(captured_advantage, 6),
        "avg_actual_advantage_when_sell": round(captured_advantage / sell_count, 6) if sell_count else 0.0,
    }


def _advantage_baselines(rows: list[dict]) -> dict:
    y = np.array([float(row["sell_advantage"]) for row in rows], dtype=float)
    if len(y) == 0:
        return {"never_sell": 0.0, "always_sell": 0.0, "oracle_positive_sell": 0.0}
    return {
        "never_sell": 0.0,
        "always_sell": round(float(np.sum(y)), 6),
        "oracle_positive_sell": round(float(np.sum(np.where(y > 0.0, y, 0.0))), 6),
    }


def _select_threshold(items: list[dict]) -> dict:
    return max(
        items,
        key=lambda item: (
            item["captured_advantage_sum"],
            item["strong_precision"],
            -item["bad_sell_count"],
            -item["sell_count"],
        ),
    )


def _top_coefficients(coefficients: dict[str, float], n: int = 8) -> list[dict]:
    return [
        {"feature": name, "coefficient": value}
        for name, value in sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    ]


def _decision(selected: dict, baselines: dict) -> str:
    if selected["captured_advantage_sum"] <= 0:
        return "research_only_rejected_negative_holdout_proxy"
    if selected["captured_advantage_sum"] < baselines.get("always_sell", 0.0):
        return "research_only_trained_but_underperforms_always_sell_proxy"
    if selected["strong_precision"] < 0.50:
        return "research_only_trained_but_precision_weak"
    return "research_only_advance_to_full_offline_replay"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.model_output, args.report_output, args.train_fraction, args.ridge_alpha)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print({"holdout": payload["holdout"], "selected_threshold": payload["selected_threshold"], "decision": payload["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
