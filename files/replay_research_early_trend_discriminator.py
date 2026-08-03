from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
DEFAULT_DATASET = ROOT / "research_universe_shadow.jsonl"
DEFAULT_REPORTS_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = DEFAULT_REPORTS_DIR / "research_early_trend_discriminator_latest.json"
DEFAULT_SHADOW_MODEL = WORKSPACE_ROOT / ".runtime" / "models" / "research_early_trend_discriminator.cbm"
TIMEZONE_NAME = "Europe/Budapest"
RULE_SIGNALS = (
    "alignment",
    "retest",
    "trend",
    "impulse_speed",
    "breakout",
    "strong_trend",
    "impulse",
)
FEATURE_NAMES = (
    "rank_24h",
    "price_change_pct_24h",
    "log10_quote_volume_24h",
    "rsi",
    "adx",
    "slope",
    "vol_x",
    "macd_hist_norm",
    "atr_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
) + tuple(f"rule_{name}" for name in RULE_SIGNALS)
VALIDATION_QUANTILES = (
    0.80,
    0.85,
    0.88,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.992,
    0.995,
    0.997,
    0.999,
)


@dataclass(frozen=True)
class ResearchRow:
    symbol: str
    bar_ts: int
    local_day: str
    rule_signal: str
    in_watchlist: bool
    features: tuple[float, ...]
    ret_3: float
    ret_5: float
    ret_10: float


def load_rows(
    path: Path = DEFAULT_DATASET,
    *,
    timezone_name: str = TIMEZONE_NAME,
) -> tuple[list[ResearchRow], dict[str, int]]:
    rows: list[ResearchRow] = []
    stats = {"lines": 0, "malformed": 0, "non_15m": 0, "immature": 0, "invalid": 0}
    if not path.exists():
        return rows, stats
    zone = ZoneInfo(timezone_name)
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            stats["lines"] += 1
            try:
                payload = json.loads(raw_line)
            except Exception:
                stats["malformed"] += 1
                continue
            if not isinstance(payload, dict):
                stats["invalid"] += 1
                continue
            if str(payload.get("tf") or "") != "15m":
                stats["non_15m"] += 1
                continue
            labels = payload.get("labels") or {}
            if any(labels.get(f"ret_{horizon}") is None for horizon in (3, 5, 10)):
                stats["immature"] += 1
                continue
            try:
                bar_ts = int(payload["bar_ts"])
                symbol = str(payload["sym"])
                if not symbol or bar_ts <= 0:
                    raise ValueError("missing key")
                rule_signal = str(payload.get("rule_signal") or "none")
                rows.append(
                    ResearchRow(
                        symbol=symbol,
                        bar_ts=bar_ts,
                        local_day=datetime.fromtimestamp(bar_ts / 1000.0, timezone.utc)
                        .astimezone(zone)
                        .date()
                        .isoformat(),
                        rule_signal=rule_signal,
                        in_watchlist=bool(payload.get("in_trade_watchlist")),
                        features=feature_vector(payload),
                        ret_3=_number(labels["ret_3"]),
                        ret_5=_number(labels["ret_5"]),
                        ret_10=_number(labels["ret_10"]),
                    )
                )
            except Exception:
                stats["invalid"] += 1
    rows.sort(key=lambda row: (row.symbol, row.bar_ts))
    return rows, stats


def feature_vector(payload: dict[str, Any]) -> tuple[float, ...]:
    features = payload.get("f") or {}
    rule_signal = str(payload.get("rule_signal") or "none")
    numeric = (
        _number(payload.get("rank_24h"), 999.0),
        _number(payload.get("price_change_pct_24h")),
        math.log10(max(1.0, _number(payload.get("quote_volume_24h"), 1.0))),
        *(
            _number(features.get(name))
            for name in (
                "rsi",
                "adx",
                "slope",
                "vol_x",
                "macd_hist_norm",
                "atr_pct",
                "body_pct",
                "upper_wick_pct",
                "lower_wick_pct",
            )
        ),
        *(1.0 if rule_signal == name else 0.0 for name in RULE_SIGNALS),
    )
    return tuple(float(value) for value in numeric)


def chronological_day_splits(days: Sequence[str]) -> dict[str, tuple[str, ...]]:
    ordered = tuple(sorted(set(days)))
    if len(ordered) < 15:
        raise ValueError("at least 15 local days are required")
    train_count = int(len(ordered) * 0.60)
    validation_count = int(len(ordered) * 0.20)
    if train_count < 3 or validation_count < 3 or train_count + validation_count + 1 >= len(ordered):
        raise ValueError("insufficient days for purged split")
    train = ordered[: train_count - 1]
    validation = ordered[train_count + 1 : train_count + validation_count - 1]
    holdout = ordered[train_count + validation_count + 1 :]
    if not train or not validation or not holdout:
        raise ValueError("purged split produced an empty segment")
    embargo = (
        ordered[train_count - 1],
        ordered[train_count],
        ordered[train_count + validation_count - 1],
        ordered[train_count + validation_count],
    )
    return {
        "train": train,
        "validation": validation,
        "holdout": holdout,
        "embargo": embargo,
    }


def _arrays(rows: Sequence[ResearchRow]) -> dict[str, Any]:
    symbols = np.asarray([row.symbol for row in rows], dtype=str)
    local_days = np.asarray([row.local_day for row in rows], dtype=str)
    group_keys = np.char.add(np.char.add(symbols, "|"), local_days)
    _, group_index = np.unique(group_keys, return_inverse=True)
    group_count = int(group_index.max()) + 1 if len(group_index) else 0
    group_day = np.empty(group_count, dtype=object)
    group_symbol = np.empty(group_count, dtype=object)
    if group_count:
        group_day[group_index] = local_days
        group_symbol[group_index] = symbols
    return {
        "symbol": symbols,
        "bar_ts": np.asarray([row.bar_ts for row in rows], dtype=np.int64),
        "day": local_days,
        "rule": np.asarray([row.rule_signal for row in rows], dtype=str),
        "watch": np.asarray([row.in_watchlist for row in rows], dtype=bool),
        "features": np.asarray([row.features for row in rows], dtype=np.float32),
        "ret_5": np.asarray([row.ret_5 for row in rows], dtype=float),
        "ret_10": np.asarray([row.ret_10 for row in rows], dtype=float),
        "group_index": group_index,
        "group_count": group_count,
        "group_day": group_day,
        "group_symbol": group_symbol,
    }


def first_signal_indices(mask: np.ndarray, group_index: np.ndarray, group_count: int) -> np.ndarray:
    missing = len(mask)
    first = np.full(group_count, missing, dtype=np.int64)
    selected = np.flatnonzero(mask)
    if not len(selected):
        return first
    selected_groups = group_index[selected]
    keep = np.r_[True, selected_groups[1:] != selected_groups[:-1]]
    first[selected_groups[keep]] = selected[keep]
    return first


def _opportunity_groups(target: np.ndarray, watch: np.ndarray, arrays: dict[str, Any]) -> np.ndarray:
    result = np.zeros(arrays["group_count"], dtype=bool)
    np.logical_or.at(result, arrays["group_index"], target & watch)
    return result


def policy_metrics(
    first: np.ndarray,
    days: Iterable[str],
    target: np.ndarray,
    opportunities: np.ndarray,
    arrays: dict[str, Any],
) -> dict[str, Any]:
    day_mask = np.isin(arrays["group_day"], tuple(days))
    groups = np.flatnonzero(day_mask & (first < len(arrays["day"])))
    indices = first[groups]
    useful = int(target[indices].sum()) if len(indices) else 0
    opportunity_count = int((day_mask & opportunities).sum())
    return {
        "selected": int(len(indices)),
        "useful": useful,
        "opportunities": opportunity_count,
        "precision_pct": _pct(useful, len(indices)),
        "recall_pct": _pct(useful, opportunity_count),
        "avg_ret_5_pct": _avg(arrays["ret_5"][indices]),
        "avg_ret_10_pct": _avg(arrays["ret_10"][indices]),
    }


def load_canonical_top_movers(
    reports_dir: Path,
    days: Iterable[str],
) -> tuple[set[tuple[str, str]], list[str]]:
    keys: set[tuple[str, str]] = set()
    loaded_days: list[str] = []
    for day in sorted(set(days)):
        path = reports_dir / f"top_gainer_critic_{day}_final.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        loaded_days.append(day)
        for item in payload.get("exchange_top_gainers") or []:
            if item.get("in_watchlist") and item.get("symbol"):
                keys.add((day, str(item["symbol"])))
    return keys, loaded_days


def top_mover_metrics(
    first: np.ndarray,
    baseline_first: np.ndarray,
    top_keys: set[tuple[str, str]],
    loaded_days: Iterable[str],
    arrays: dict[str, Any],
) -> dict[str, Any]:
    loaded = set(loaded_days)
    selected_groups = np.flatnonzero(
        np.isin(arrays["group_day"], tuple(loaded)) & (first < len(arrays["day"]))
    )
    selected_keys = {
        (str(arrays["group_day"][group]), str(arrays["group_symbol"][group]))
        for group in selected_groups
    }
    observed_keys = {
        (str(arrays["group_day"][group]), str(arrays["group_symbol"][group]))
        for group in range(arrays["group_count"])
        if str(arrays["group_day"][group]) in loaded
    }
    captured = top_keys & selected_keys
    observable_top = top_keys & observed_keys
    lead_minutes: list[float] = []
    newly_captured = 0
    for group in selected_groups:
        key = (str(arrays["group_day"][group]), str(arrays["group_symbol"][group]))
        if key not in top_keys:
            continue
        current_idx = int(first[group])
        baseline_idx = int(baseline_first[group])
        if baseline_idx >= len(arrays["day"]):
            newly_captured += 1
        elif current_idx < baseline_idx:
            lead_minutes.append(
                (arrays["bar_ts"][baseline_idx] - arrays["bar_ts"][current_idx]) / 60000.0
            )
    return {
        "critic_days": len(loaded),
        "canonical_top": len(top_keys),
        "observable_top": len(observable_top),
        "captured_top": len(captured),
        "selected": len(selected_groups),
        "recall_pct": _pct(len(captured), len(top_keys)),
        "observable_recall_pct": _pct(len(captured), len(observable_top)),
        "precision_pct": _pct(len(captured), len(selected_groups)),
        "newly_captured_top": newly_captured,
        "earlier_top_count": len(lead_minutes),
        "avg_earlier_by_min": _avg(np.asarray(lead_minutes, dtype=float)),
    }


def _fit_scores(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], Any]:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.07,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        thread_count=4,
        l2_leaf_reg=5,
    )
    model.fit(features[train_mask], target[train_mask])
    scores = model.predict_proba(features)[:, 1]
    importance = sorted(
        (
            {"feature": name, "importance": round(float(value), 6)}
            for name, value in zip(FEATURE_NAMES, model.get_feature_importance())
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return scores, importance, model


def _passes_validation(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    primary = candidate["primary"]
    strict = candidate["strict"]
    base_primary = baseline["primary"]
    base_strict = baseline["strict"]
    pressure = primary["selected"] / max(1, base_primary["selected"])
    return bool(
        primary["recall_pct"] >= base_primary["recall_pct"] + 2.0
        and primary["precision_pct"] >= base_primary["precision_pct"] - 2.0
        and primary["avg_ret_10_pct"] >= base_primary["avg_ret_10_pct"]
        and pressure <= 1.5
        and strict["recall_pct"] >= base_strict["recall_pct"]
        and strict["precision_pct"] >= base_strict["precision_pct"] - 2.0
    )


def _passes_holdout(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    primary = candidate["primary"]
    strict = candidate["strict"]
    base_primary = baseline["primary"]
    base_strict = baseline["strict"]
    pressure = primary["selected"] / max(1, base_primary["selected"])
    return bool(
        primary["recall_pct"] >= base_primary["recall_pct"] + 2.0
        and primary["precision_pct"] >= base_primary["precision_pct"]
        and primary["avg_ret_10_pct"] >= base_primary["avg_ret_10_pct"]
        and pressure <= 1.5
        and strict["recall_pct"] >= base_strict["recall_pct"]
        and strict["precision_pct"] >= base_strict["precision_pct"]
    )


def _validation_score(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    primary = candidate["primary"]
    base = baseline["primary"]
    return (
        primary["recall_pct"]
        - base["recall_pct"]
        + 2.0 * min(0.0, primary["precision_pct"] - base["precision_pct"])
        + 10.0 * min(0.0, primary["avg_ret_10_pct"] - base["avg_ret_10_pct"])
    )


def run_replay(
    *,
    dataset: Path = DEFAULT_DATASET,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    output: Path = DEFAULT_OUTPUT,
    iterations: int = 280,
    seed: int = 20260803,
    export_shadow_model: Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    rows, load_stats = load_rows(dataset)
    if not rows:
        return {"status": "no_data", "decision": "keep_production_unchanged", "load": load_stats}
    splits = chronological_day_splits([row.local_day for row in rows])
    arrays = _arrays(rows)
    primary = (arrays["ret_5"] >= 0.5) & (arrays["ret_10"] >= 1.0)
    strict = (arrays["ret_5"] >= 1.0) & (arrays["ret_10"] >= 2.0)
    train_mask = np.isin(arrays["day"], splits["train"])
    scores, importance, model = _fit_scores(
        arrays["features"], primary, train_mask, iterations=iterations, seed=seed
    )
    baseline_first = first_signal_indices(
        (arrays["rule"] != "none") & arrays["watch"],
        arrays["group_index"],
        arrays["group_count"],
    )
    primary_opportunities = _opportunity_groups(primary, arrays["watch"], arrays)
    strict_opportunities = _opportunity_groups(strict, arrays["watch"], arrays)

    baseline: dict[str, Any] = {}
    for split_name in ("train", "validation", "holdout"):
        baseline[split_name] = {
            "primary": policy_metrics(
                baseline_first, splits[split_name], primary, primary_opportunities, arrays
            ),
            "strict": policy_metrics(
                baseline_first, splits[split_name], strict, strict_opportunities, arrays
            ),
        }

    validation_pool = scores[
        np.isin(arrays["day"], splits["validation"])
        & arrays["watch"]
        & (arrays["rule"] == "none")
    ]
    thresholds = (
        sorted(
            {
                round(float(np.quantile(validation_pool, quantile)), 12)
                for quantile in VALIDATION_QUANTILES
            }
        )
        if len(validation_pool)
        else []
    )
    candidates: list[dict[str, Any]] = []
    first_by_threshold: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        model_first = first_signal_indices(
            (arrays["rule"] == "none") & arrays["watch"] & (scores >= threshold),
            arrays["group_index"],
            arrays["group_count"],
        )
        union_first = np.minimum(baseline_first, model_first)
        first_by_threshold[threshold] = union_first
        metrics: dict[str, Any] = {}
        for split_name in ("train", "validation", "holdout"):
            metrics[split_name] = {
                "primary": policy_metrics(
                    union_first, splits[split_name], primary, primary_opportunities, arrays
                ),
                "strict": policy_metrics(
                    union_first, splits[split_name], strict, strict_opportunities, arrays
                ),
            }
        metrics["threshold"] = threshold
        metrics["passes_validation"] = _passes_validation(
            metrics["validation"], baseline["validation"]
        )
        metrics["validation_score"] = round(
            _validation_score(metrics["validation"], baseline["validation"]), 6
        )
        candidates.append(metrics)

    eligible = [candidate for candidate in candidates if candidate["passes_validation"]]
    selected = max(eligible, key=lambda item: item["validation_score"]) if eligible else None
    holdout_pass = bool(
        selected and _passes_holdout(selected["holdout"], baseline["holdout"])
    )

    north_star: dict[str, Any] = {"status": "not_evaluated"}
    top_direction_ok = False
    if selected:
        top_keys, loaded_days = load_canonical_top_movers(reports_dir, splits["holdout"])
        selected_first = first_by_threshold[float(selected["threshold"])]
        base_top = top_mover_metrics(
            baseline_first, baseline_first, top_keys, loaded_days, arrays
        )
        candidate_top = top_mover_metrics(
            selected_first, baseline_first, top_keys, loaded_days, arrays
        )
        enough = len(loaded_days) >= 5 and len(top_keys) >= 5
        top_direction_ok = bool(
            enough
            and candidate_top["recall_pct"] >= base_top["recall_pct"]
            and candidate_top["precision_pct"] >= base_top["precision_pct"] - 2.0
        )
        north_star = {
            "status": "complete" if enough else "insufficient",
            "baseline": base_top,
            "candidate": candidate_top,
            "directional_gate_passed": top_direction_ok,
            "limitation": "same-day membership/lead only; critic early-capture ratio is not reconstructable from research rows",
        }

    if holdout_pass and top_direction_ok:
        decision = "advance_to_independent_shadow_only"
    elif holdout_pass:
        decision = "proxy_pass_north_star_not_confirmed"
    else:
        decision = "reject_keep_production_unchanged"
    model_export: dict[str, Any] = {"saved": False}
    if decision == "advance_to_independent_shadow_only" and export_shadow_model is not None:
        export_shadow_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(export_shadow_model))
        model_sha256 = hashlib.sha256(export_shadow_model.read_bytes()).hexdigest()
        feature_schema_sha256 = hashlib.sha256(
            json.dumps(FEATURE_NAMES, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        dataset_stat = dataset.stat()
        metadata_file = export_shadow_model.with_suffix(".json")
        metadata = {
            "profile": "research_early_trend_catboost_v1",
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "threshold": float(selected["threshold"]),
            "feature_names": list(FEATURE_NAMES),
            "train_start": splits["train"][0],
            "train_end": splits["train"][-1],
            "validation_start": splits["validation"][0],
            "validation_end": splits["validation"][-1],
            "holdout_start": splits["holdout"][0],
            "holdout_end": splits["holdout"][-1],
            "seed": seed,
            "iterations": iterations,
            "decision": decision,
            "model_sha256": model_sha256,
            "feature_schema_sha256": feature_schema_sha256,
            "dataset": str(dataset),
            "dataset_size": dataset_stat.st_size,
            "dataset_mtime_ns": dataset_stat.st_mtime_ns,
            "mature_rows": len(rows),
        }
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        model_export = {
            "saved": True,
            "model": str(export_shadow_model),
            "metadata": str(metadata_file),
        }
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "complete",
        "decision": decision,
        "production_changed": False,
        "dataset": str(dataset),
        "load": {**load_stats, "mature_rows": len(rows)},
        "split": {
            name: {"days": len(values), "start": values[0], "end": values[-1]}
            for name, values in splits.items()
            if name != "embargo"
        },
        "embargo_days": list(splits["embargo"]),
        "labels": {
            "primary": "ret_5 >= 0.5% and ret_10 >= 1.0%",
            "strict": "ret_5 >= 1.0% and ret_10 >= 2.0%",
        },
        "model": {
            "type": "CatBoostClassifier",
            "iterations": iterations,
            "seed": seed,
            "features": list(FEATURE_NAMES),
            "feature_importance": importance,
            "artifact_saved": bool(model_export["saved"]),
            "export": model_export,
        },
        "baseline": baseline,
        "thresholds_tested": len(candidates),
        "validation_candidates_passed": len(eligible),
        "selected": selected,
        "holdout_gate_passed": holdout_pass,
        "north_star": north_star,
    }
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 6) if denominator else 0.0


def _avg(values: np.ndarray) -> float | None:
    return round(float(np.mean(values)), 6) if len(values) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the research early-trend discriminator")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=280)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--export-shadow-model",
        action="store_true",
        help="Save an uncommitted runtime model only when every shadow gate passes",
    )
    parser.add_argument("--shadow-model", type=Path, default=DEFAULT_SHADOW_MODEL)
    args = parser.parse_args()
    report = run_replay(
        dataset=args.dataset,
        reports_dir=args.reports_dir,
        output=args.output,
        iterations=max(20, args.iterations),
        seed=args.seed,
        export_shadow_model=args.shadow_model if args.export_shadow_model else None,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
