from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from train_v2_exit_advantage_baseline import (
    DEFAULT_DATASET,
    FEATURE_NAMES,
    _advantage_baselines,
    _chronological_split,
    _evaluate_threshold,
    _feature,
    _load_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_exit_advantage_model_family_comparison_15m.json"
DEFAULT_THRESHOLDS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
DEFAULT_FEATURES = [
    "adx",
    "projected_leader_score_trend",
    "price_vs_ema20_pct",
    "daily_range_pct",
    "belief_late_mass",
    "belief_open_mass",
    "late_mass_delta_3",
    "rsi_delta_3",
]


def build(
    dataset_path: Path,
    output: Path,
    train_fraction: float = 0.70,
    bins: int = 6,
    min_support: int = 100,
) -> dict:
    rows = _load_rows(dataset_path)
    train_rows, holdout_rows = _chronological_split(rows, train_fraction=train_fraction)
    baselines = _advantage_baselines(holdout_rows)
    candidates = []
    for feature in DEFAULT_FEATURES:
        candidates.extend(_single_feature_candidates(train_rows, holdout_rows, feature, bins, min_support))
    for left, right in combinations(DEFAULT_FEATURES, 2):
        candidates.extend(_pair_feature_candidates(train_rows, holdout_rows, left, right, bins, min_support))
    candidates = sorted(
        candidates,
        key=lambda item: (
            item["best_threshold"]["captured_advantage_sum"],
            item["best_threshold"]["strong_precision"],
            -item["best_threshold"]["bad_sell_count"],
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    payload = {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "train_fraction": train_fraction,
        "bins": bins,
        "min_support": min_support,
        "holdout_advantage_baselines": baselines,
        "best_candidate": best,
        "top_candidates": candidates[:25],
        "decision": _decision(best, baselines),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _single_feature_candidates(train_rows: list[dict], holdout_rows: list[dict], feature: str, bins: int, min_support: int) -> list[dict]:
    edges = _quantile_edges([_feature(row, feature) for row in train_rows], bins)
    train_stats = defaultdict(list)
    for row in train_rows:
        train_stats[(_bin(_feature(row, feature), edges),)].append(float(row["sell_advantage"]))
    table = _mean_table(train_stats, min_support)
    fallback = _safe_mean(float(row["sell_advantage"]) for row in train_rows)
    pred = np.array([table.get((_bin(_feature(row, feature), edges),), fallback) for row in holdout_rows], dtype=float)
    return [_candidate("single_bin", [feature], edges, table, fallback, holdout_rows, pred)]


def _pair_feature_candidates(
    train_rows: list[dict],
    holdout_rows: list[dict],
    left: str,
    right: str,
    bins: int,
    min_support: int,
) -> list[dict]:
    left_edges = _quantile_edges([_feature(row, left) for row in train_rows], bins)
    right_edges = _quantile_edges([_feature(row, right) for row in train_rows], bins)
    train_stats = defaultdict(list)
    for row in train_rows:
        key = (_bin(_feature(row, left), left_edges), _bin(_feature(row, right), right_edges))
        train_stats[key].append(float(row["sell_advantage"]))
    table = _mean_table(train_stats, min_support)
    fallback = _safe_mean(float(row["sell_advantage"]) for row in train_rows)
    pred = np.array(
        [
            table.get(
                (_bin(_feature(row, left), left_edges), _bin(_feature(row, right), right_edges)),
                fallback,
            )
            for row in holdout_rows
        ],
        dtype=float,
    )
    return [_candidate("pair_bin", [left, right], {left: left_edges, right: right_edges}, table, fallback, holdout_rows, pred)]


def _candidate(family: str, features: list[str], edges, table: dict, fallback: float, holdout_rows: list[dict], pred: np.ndarray) -> dict:
    thresholds = [_evaluate_threshold(holdout_rows, pred, threshold) for threshold in DEFAULT_THRESHOLDS]
    best = max(
        thresholds,
        key=lambda item: (
            item["captured_advantage_sum"],
            item["strong_precision"],
            -item["bad_sell_count"],
            -item["sell_count"],
        ),
    )
    return {
        "family": family,
        "features": features,
        "table_cells": len(table),
        "fallback_advantage": round(float(fallback), 6),
        "predicted_advantage_avg": round(float(np.mean(pred)), 6) if len(pred) else 0.0,
        "thresholds": thresholds,
        "best_threshold": best,
        "sample_table": _sample_table(table),
    }


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


def _safe_mean(values: object) -> float:
    data = list(values)
    return float(np.mean(data)) if data else 0.0


def _sample_table(table: dict[tuple[int, ...], float], n: int = 8) -> list[dict]:
    return [
        {"bin": list(key), "mean_advantage": round(float(value), 6)}
        for key, value in sorted(table.items(), key=lambda kv: kv[1], reverse=True)[:n]
    ]


def _decision(best: dict | None, baselines: dict) -> str:
    if not best:
        return "research_only_no_candidate"
    captured = best["best_threshold"]["captured_advantage_sum"]
    always = baselines.get("always_sell", 0.0)
    if captured <= 0:
        return "research_only_rejected_negative_proxy"
    if captured <= always:
        return "research_only_rejected_underperforms_always_sell_proxy"
    if best["best_threshold"]["sell_rate"] > 0.95:
        return "research_only_rejected_near_always_sell"
    return "research_only_advance_to_full_offline_replay"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.output, args.train_fraction, args.bins, args.min_support)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print({"best_candidate": payload["best_candidate"], "decision": payload["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
