from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from indicators import compute_features

from .history import CanonicalBar
from .lifecycle_labeling import LifecycleLabel
from .shadow_observer import FeatureSnapshot, estimate_shadow_state
from .state import SymbolState
from .teacher_confidence import TeacherConfidence


FEATURE_NAMES = (
    "ret_1",
    "ret_4",
    "price_vs_ema20",
    "slope",
    "adx",
    "rsi",
    "vol_x",
    "macd_hist_rel",
    "daily_range_pct",
)


@dataclass(frozen=True)
class ReconstructionRow:
    symbol: str
    local_day: str
    ts_ms: int
    features: tuple[float, ...]
    label: SymbolState
    confidence: float


def build_rows(
    bars: Iterable[CanonicalBar],
    labels_by_ts: dict[int, LifecycleLabel],
    confidence_by_ts: dict[int, TeacherConfidence],
) -> list[ReconstructionRow]:
    ordered = list(sorted(bars, key=lambda bar: bar.open_ts_ms))
    if len(ordered) < 40:
        return []
    o = np.array([bar.open for bar in ordered], dtype=float)
    h = np.array([bar.high for bar in ordered], dtype=float)
    l = np.array([bar.low for bar in ordered], dtype=float)
    c = np.array([bar.close for bar in ordered], dtype=float)
    v = np.array([bar.volume for bar in ordered], dtype=float)
    feat = compute_features(o, h, l, c, v)
    out = []
    for i, bar in enumerate(ordered):
        if i < 30 or bar.open_ts_ms not in labels_by_ts or bar.open_ts_ms not in confidence_by_ts:
            continue
        values = (
            _pct_change(c, i, 1),
            _pct_change(c, i, 4),
            _safe_pct(c[i], feat["ema_fast"][i]),
            float(feat["slope"][i]),
            float(feat["adx"][i]),
            float(feat["rsi"][i]),
            float(feat["vol_x"][i]),
            _safe_ratio(float(feat["macd_hist"][i]), c[i]) * 100.0,
            float(feat["daily_range_pct"][i]),
        )
        if not all(math.isfinite(value) for value in values):
            continue
        label = labels_by_ts[bar.open_ts_ms]
        conf = confidence_by_ts[bar.open_ts_ms]
        out.append(
            ReconstructionRow(
                symbol=bar.symbol,
                local_day=label.local_day,
                ts_ms=bar.open_ts_ms,
                features=tuple(float(value) for value in values),
                label=label.state,
                confidence=conf.value,
            )
        )
    return out


def chronological_split(rows: list[ReconstructionRow], train_ratio: float = 0.70):
    days = sorted({row.local_day for row in rows})
    cut = max(1, min(len(days) - 1, int(len(days) * train_ratio)))
    train_days = set(days[:cut])
    return [row for row in rows if row.local_day in train_days], [row for row in rows if row.local_day not in train_days]


def fit_majority(rows: list[ReconstructionRow]) -> SymbolState:
    return Counter(row.label for row in rows).most_common(1)[0][0]


def fit_centroids(rows: list[ReconstructionRow]) -> dict[SymbolState, tuple[float, ...]]:
    sums = defaultdict(lambda: np.zeros(len(FEATURE_NAMES), dtype=float))
    weights = defaultdict(float)
    for row in rows:
        weight = max(row.confidence, 1e-6)
        sums[row.label] += np.array(row.features) * weight
        weights[row.label] += weight
    return {label: tuple((sums[label] / weights[label]).tolist()) for label in sums}


def fit_scaler(rows: list[ReconstructionRow]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = np.array([row.features for row in rows], dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    return tuple(means.tolist()), tuple(stds.tolist())


def scale_features(
    features: tuple[float, ...],
    means: tuple[float, ...],
    stds: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(((np.array(features) - np.array(means)) / np.array(stds)).tolist())


def predict_centroid(features: tuple[float, ...], centroids: dict[SymbolState, tuple[float, ...]]) -> SymbolState:
    x = np.array(features)
    return min(centroids, key=lambda label: float(np.linalg.norm(x - np.array(centroids[label]))))


def predict_shadow_rule(row: ReconstructionRow) -> SymbolState:
    values = dict(zip(FEATURE_NAMES, row.features))
    decision = estimate_shadow_state(
        FeatureSnapshot(
            price=100.0 + values["price_vs_ema20"],
            ema20=100.0,
            slope=values["slope"],
            adx=values["adx"],
            rsi=values["rsi"],
            vol_x=values["vol_x"],
            daily_range=values["daily_range_pct"],
            macd_hist=values["macd_hist_rel"],
        )
    )
    return decision.state


def evaluate(rows: list[ReconstructionRow], predictions: list[SymbolState]) -> dict:
    labels = [row.label for row in rows]
    weights = [row.confidence for row in rows]
    states = list(SymbolState)
    weighted_correct = sum(w for y, p, w in zip(labels, predictions, weights) if y == p)
    weighted_total = sum(weights) or 1.0
    recalls = {}
    f1s = []
    for state in states:
        tp = sum(1 for y, p in zip(labels, predictions) if y == state and p == state)
        fp = sum(1 for y, p in zip(labels, predictions) if y != state and p == state)
        fn = sum(1 for y, p in zip(labels, predictions) if y == state and p != state)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recalls[state.value] = round(recall, 6)
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return {
        "rows": len(rows),
        "weighted_accuracy": round(weighted_correct / weighted_total, 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "recall_by_state": recalls,
    }


def _pct_change(values: np.ndarray, i: int, lookback: int) -> float:
    return _safe_pct(values[i], values[i - lookback]) if i >= lookback else float("nan")


def _safe_pct(value: float, base: float) -> float:
    return ((float(value) / float(base)) - 1.0) * 100.0 if base else float("nan")


def _safe_ratio(value: float, base: float) -> float:
    return float(value) / float(base) if base else float("nan")
