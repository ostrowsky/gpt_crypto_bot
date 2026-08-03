from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from train_v2_exit_advantage_baseline import (
    DEFAULT_DATASET,
    _advantage_baselines,
    _evaluate_threshold,
    _feature,
    _load_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_exit_advantage_model_family_comparison_15m.json"
DEFAULT_THRESHOLDS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
STRUCTURAL_FEATURES = [
    "adx",
    "projected_leader_score_trend",
    "price_vs_ema20_pct",
    "daily_range_pct",
    "belief_late_mass",
    "belief_open_mass",
    "late_mass_delta_3",
    "rsi_delta_3",
]
POSITION_FEATURES = [
    "bars_held",
    "unrealized_pnl_pct",
    "mfe_pct",
    "giveback_pct",
    "candidate_action_sell",
]
DEFAULT_FEATURES = STRUCTURAL_FEATURES + POSITION_FEATURES


def build(
    dataset_path: Path,
    output: Path,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    purge_days: int = 1,
    bins: int = 6,
    min_support: int = 100,
) -> dict[str, Any]:
    rows = _load_rows(dataset_path)
    train_rows, validation_rows, holdout_rows, split = _chronological_day_split(
        rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        purge_days=purge_days,
    )
    validation_baselines = _advantage_baselines(validation_rows)
    holdout_baselines = _advantage_baselines(holdout_rows)
    candidates = []
    for feature in DEFAULT_FEATURES:
        candidates.append(
            _single_feature_candidate(
                train_rows,
                validation_rows,
                holdout_rows,
                feature,
                bins,
                min_support,
            )
        )
    for left, right in combinations(DEFAULT_FEATURES, 2):
        candidates.append(
            _pair_feature_candidate(
                train_rows,
                validation_rows,
                holdout_rows,
                left,
                right,
                bins,
                min_support,
            )
        )
    candidates = sorted(candidates, key=_validation_rank_key, reverse=True)
    best = candidates[0] if candidates else None
    payload = {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "holdout_rows": len(holdout_rows),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "purge_days": purge_days,
        "split": split,
        "bins": bins,
        "min_support": min_support,
        "structural_features": STRUCTURAL_FEATURES,
        "position_features": POSITION_FEATURES,
        "validation_advantage_baselines": validation_baselines,
        "holdout_advantage_baselines": holdout_baselines,
        "best_candidate": best,
        "top_candidates_validation_ranked": candidates[:25],
        "decision": _decision(best, holdout_baselines),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _chronological_day_split(
    rows: list[dict],
    *,
    train_fraction: float,
    validation_fraction: float,
    purge_days: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("validation_fraction must leave a positive holdout")
    purge_days = max(0, int(purge_days))
    days = sorted({_local_day(row) for row in rows if _local_day(row)})
    if len(days) < 8 + purge_days * 4:
        raise ValueError("not enough complete local days for purged three-way split")
    train_end = max(2, min(len(days) - 4, int(len(days) * train_fraction)))
    validation_end = max(train_end + 2, min(len(days) - 2, int(len(days) * (train_fraction + validation_fraction))))
    train_days = days[: max(1, train_end - purge_days)]
    validation_days = days[min(len(days), train_end + purge_days) : max(train_end + purge_days + 1, validation_end - purge_days)]
    holdout_days = days[min(len(days), validation_end + purge_days) :]
    if not train_days or not validation_days or not holdout_days:
        raise ValueError("purged split produced an empty partition")
    train_set = set(train_days)
    validation_set = set(validation_days)
    holdout_set = set(holdout_days)
    train_rows = [row for row in rows if _local_day(row) in train_set]
    validation_rows = [row for row in rows if _local_day(row) in validation_set]
    holdout_rows = [row for row in rows if _local_day(row) in holdout_set]
    split = {
        "days_total": len(days),
        "train_days": train_days,
        "validation_days": validation_days,
        "holdout_days": holdout_days,
        "purged_days": [day for day in days if day not in train_set | validation_set | holdout_set],
    }
    return train_rows, validation_rows, holdout_rows, split


def _local_day(row: dict) -> str:
    return str(row.get("local_day") or "")


def _single_feature_candidate(
    train_rows: list[dict],
    validation_rows: list[dict],
    holdout_rows: list[dict],
    feature: str,
    bins: int,
    min_support: int,
) -> dict[str, Any]:
    edges = _quantile_edges([_feature_value(row, feature) for row in train_rows], bins)
    train_stats: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for row in train_rows:
        train_stats[(_bin(_feature_value(row, feature), edges),)].append(float(row["sell_advantage"]))
    table = _mean_table(train_stats, min_support)
    fallback = _safe_mean(float(row["sell_advantage"]) for row in train_rows)
    validation_pred = np.array(
        [table.get((_bin(_feature_value(row, feature), edges),), fallback) for row in validation_rows],
        dtype=float,
    )
    holdout_pred = np.array(
        [table.get((_bin(_feature_value(row, feature), edges),), fallback) for row in holdout_rows],
        dtype=float,
    )
    return _candidate(
        "single_bin",
        [feature],
        edges,
        table,
        fallback,
        validation_rows,
        validation_pred,
        holdout_rows,
        holdout_pred,
    )


def _pair_feature_candidate(
    train_rows: list[dict],
    validation_rows: list[dict],
    holdout_rows: list[dict],
    left: str,
    right: str,
    bins: int,
    min_support: int,
) -> dict[str, Any]:
    left_edges = _quantile_edges([_feature_value(row, left) for row in train_rows], bins)
    right_edges = _quantile_edges([_feature_value(row, right) for row in train_rows], bins)
    train_stats: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for row in train_rows:
        key = (_bin(_feature_value(row, left), left_edges), _bin(_feature_value(row, right), right_edges))
        train_stats[key].append(float(row["sell_advantage"]))
    table = _mean_table(train_stats, min_support)
    fallback = _safe_mean(float(row["sell_advantage"]) for row in train_rows)
    validation_pred = _pair_predictions(validation_rows, left, right, left_edges, right_edges, table, fallback)
    holdout_pred = _pair_predictions(holdout_rows, left, right, left_edges, right_edges, table, fallback)
    return _candidate(
        "pair_bin",
        [left, right],
        {left: left_edges, right: right_edges},
        table,
        fallback,
        validation_rows,
        validation_pred,
        holdout_rows,
        holdout_pred,
    )


def _pair_predictions(
    rows: list[dict],
    left: str,
    right: str,
    left_edges: list[float],
    right_edges: list[float],
    table: dict[tuple[int, ...], float],
    fallback: float,
) -> np.ndarray:
    return np.array(
        [
            table.get(
                (
                    _bin(_feature_value(row, left), left_edges),
                    _bin(_feature_value(row, right), right_edges),
                ),
                fallback,
            )
            for row in rows
        ],
        dtype=float,
    )


def _candidate(
    family: str,
    features: list[str],
    edges: Any,
    table: dict[tuple[int, ...], float],
    fallback: float,
    validation_rows: list[dict],
    validation_pred: np.ndarray,
    holdout_rows: list[dict],
    holdout_pred: np.ndarray,
) -> dict[str, Any]:
    validation_thresholds = [
        _evaluate_threshold(validation_rows, validation_pred, threshold)
        for threshold in DEFAULT_THRESHOLDS
    ]
    selected = max(validation_thresholds, key=_threshold_rank_key)
    selected_threshold = float(selected["threshold"])
    holdout_result = _evaluate_threshold(holdout_rows, holdout_pred, selected_threshold)
    return {
        "family": family,
        "features": features,
        "table_cells": len(table),
        "fallback_advantage": round(float(fallback), 6),
        "validation_predicted_advantage_avg": round(float(np.mean(validation_pred)), 6) if len(validation_pred) else 0.0,
        "holdout_predicted_advantage_avg": round(float(np.mean(holdout_pred)), 6) if len(holdout_pred) else 0.0,
        "validation_thresholds": validation_thresholds,
        "selected_threshold_on_validation": selected,
        "holdout_result": holdout_result,
        "sample_table": _sample_table(table),
        "edges": edges,
    }


def _threshold_rank_key(item: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(item.get("captured_advantage_sum") or 0.0),
        float(item.get("strong_precision") or 0.0),
        -int(item.get("bad_sell_count") or 0),
        -int(item.get("sell_count") or 0),
    )


def _validation_rank_key(item: dict[str, Any]) -> tuple[float, float, int, int]:
    return _threshold_rank_key(item.get("selected_threshold_on_validation") or {})


def _feature_value(row: dict, name: str) -> float:
    if name == "candidate_action_sell":
        return 1.0 if str(row.get("candidate_action") or "").lower() == "sell" else 0.0
    if name in {"bars_held", "unrealized_pnl_pct", "mfe_pct", "giveback_pct"}:
        return _finite_float(row.get(name))
    return _feature(row, name)


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) or math.isinf(number) else number


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    clean = np.array([value for value in values if not math.isnan(value) and not math.isinf(value)], dtype=float)
    if len(clean) == 0:
        return []
    quantiles = [i / bins for i in range(1, bins)]
    edges = np.quantile(clean, quantiles)
    unique = []
    for value in edges:
        value = float(value)
        if not unique or abs(value - unique[-1]) > 1e-12:
            unique.append(value)
    return unique


def _bin(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.array(edges, dtype=float), float(value), side="right"))


def _mean_table(stats: dict[tuple[int, ...], list[float]], min_support: int) -> dict[tuple[int, ...], float]:
    return {
        key: float(np.mean(values))
        for key, values in stats.items()
        if len(values) >= min_support
    }


def _safe_mean(values: Any) -> float:
    data = list(values)
    return float(np.mean(data)) if data else 0.0


def _sample_table(table: dict[tuple[int, ...], float], n: int = 8) -> list[dict[str, Any]]:
    return [
        {"bin": list(key), "mean_advantage": round(float(value), 6)}
        for key, value in sorted(table.items(), key=lambda kv: kv[1], reverse=True)[:n]
    ]


def _decision(best: dict[str, Any] | None, baselines: dict[str, Any]) -> str:
    if not best:
        return "research_only_no_candidate"
    evaluation = best.get("holdout_result") or best.get("best_threshold") or {}
    captured = float(evaluation.get("captured_advantage_sum") or 0.0)
    always = float(baselines.get("always_sell") or 0.0)
    if captured <= 0:
        return "research_only_rejected_negative_proxy"
    if captured <= always:
        return "research_only_rejected_underperforms_always_sell_proxy"
    if float(evaluation.get("sell_rate") or 0.0) > 0.95:
        return "research_only_rejected_near_always_sell"
    return "research_only_advance_to_full_offline_replay"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(
        args.dataset,
        args.output,
        args.train_fraction,
        args.validation_fraction,
        args.purge_days,
        args.bins,
        args.min_support,
    )
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            {
                "best_candidate": payload["best_candidate"],
                "decision": payload["decision"],
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
