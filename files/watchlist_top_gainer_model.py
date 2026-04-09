from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

import ml_dataset
import top_gainer_critic
from ml_signal_model import (
    MLPModel,
    LogisticModel,
    StandardScaler,
    _parse_ts,
    build_feature_dict,
    evaluate_predictions,
    find_best_threshold,
    save_json,
    safe_feature_names,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_FILE = ROOT / "ml_dataset.jsonl"
DEFAULT_MODEL_FILE = ROOT / "watchlist_top_gainer_model.json"
DEFAULT_REPORT_FILE = ROOT / "watchlist_top_gainer_model_report.json"
DEFAULT_REPORT_DIR = ROOT.parent / ".runtime" / "reports"

DEFAULT_TEACHER_KEY = "watchlist_top_gainer_final_proxy"
DEFAULT_TIMEZONE = top_gainer_critic.DEFAULT_TZ
DEFAULT_TOP_N = top_gainer_critic.DEFAULT_TOP_N
DEFAULT_CHECKPOINT_HOURS = (1, 4, 8, 12, 18, 22)
DEFAULT_MAX_LOCAL_HOUR = 22


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if (out != out or np.isinf(out)) else out


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _record_local_dt(rec: Dict[str, Any], tz: ZoneInfo) -> Optional[datetime]:
    ts = _parse_ts(str(rec.get("ts_signal", "")).replace("+00:00", "Z")) if rec.get("ts_signal") else None
    if ts is None:
        bar_ts = _safe_int(rec.get("bar_ts"))
        if bar_ts is None:
            return None
        ts = datetime.fromtimestamp(bar_ts / 1000, tz=ZoneInfo("UTC"))
    return ts.astimezone(tz)


def _normalize_checkpoint_hours(values: Sequence[int]) -> List[int]:
    return sorted({max(0, min(23, int(v))) for v in values})


def _iter_local_reports(report_dir: Path, phase: str = "final") -> Iterable[Dict[str, Any]]:
    if not report_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for report_path in sorted(report_dir.glob(f"top_gainer_critic_*_{phase}.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def annotate_ml_dataset_from_report(
    report: Dict[str, Any],
    *,
    teacher_key: str,
    dataset_path: Path = DEFAULT_DATASET_FILE,
) -> Dict[str, Any]:
    if not dataset_path.exists():
        return {"teacher_key": teacher_key, "rows_scanned": 0, "rows_annotated": 0, "positive_rows": 0}

    target_day_text = str(report.get("target_day_local", "") or "").strip()
    if not target_day_text:
        raise ValueError("Report missing target_day_local")
    phase = str(report.get("phase", "") or "final").strip().lower()
    settings = report.get("settings") or {}
    timezone_name = str(settings.get("timezone", DEFAULT_TIMEZONE) or DEFAULT_TIMEZONE)
    tz = ZoneInfo(timezone_name)
    target_day = date.fromisoformat(target_day_text)
    start_local, end_local = top_gainer_critic._local_window(target_day, phase, tz)
    watchlist = top_gainer_critic._load_watchlist()

    watchlist_map: Dict[str, Dict[str, Any]] = {}
    watchlist_rank_map: Dict[str, int] = {}
    for idx, item in enumerate(report.get("watchlist_top_gainers") or [], start=1):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        watchlist_map[sym] = item
        watchlist_rank_map[sym] = idx

    original_ml_file = ml_dataset.ML_FILE
    ml_dataset.ML_FILE = dataset_path
    rows_scanned = 0
    rows_in_window = 0
    positive_rows = 0
    positive_symbols_with_rows: set[str] = set()
    symbols_in_window: set[str] = set()

    def _payload_for_row(rec: Dict[str, Any]) -> tuple[str, Dict[str, Any]] | None:
        local_dt = _record_local_dt(rec, tz)
        if local_dt is None or not (start_local <= local_dt <= end_local):
            return None
        sym = str(rec.get("sym", "")).strip()
        if not sym:
            return None
        if watchlist and sym not in watchlist:
            return None
        summary = watchlist_map.get(sym)
        payload = {
            "goal": "watchlist_top_gainer_end_of_day_proxy",
            "source": "top_gainer_critic_local_report",
            "source_phase": phase,
            "target_day_local": target_day_text,
            "timezone": timezone_name,
            "watchlist_top_gainer": summary is not None,
            "watchlist_top_rank": watchlist_rank_map.get(sym),
            "status": None if summary is None else summary.get("status"),
            "day_change_pct": None if summary is None else summary.get("day_change_pct"),
            "quote_volume_24h": None if summary is None else summary.get("quote_volume_24h"),
        }
        return sym, payload

    try:
        def _preview_mutate(rec: Dict[str, Any]) -> bool:
            nonlocal rows_scanned, rows_in_window, positive_rows
            rows_scanned += 1
            built = _payload_for_row(rec)
            if built is None:
                return False
            sym, payload = built
            rows_in_window += 1
            symbols_in_window.add(sym)
            if payload["watchlist_top_gainer"]:
                positive_rows += 1
                positive_symbols_with_rows.add(sym)
            teacher = rec.get("teacher") or {}
            return teacher.get(teacher_key) != payload

        _, maybe_changed, maybe_bad_rows = ml_dataset._collect_mutated_lines(_preview_mutate)
        rows_annotated = 0
        if maybe_changed or maybe_bad_rows:
            def _commit_mutate(rec: Dict[str, Any]) -> bool:
                nonlocal rows_annotated
                built = _payload_for_row(rec)
                if built is None:
                    return False
                _, payload = built
                teacher = rec.setdefault("teacher", {})
                if teacher.get(teacher_key) == payload:
                    return False
                teacher[teacher_key] = payload
                rows_annotated += 1
                return True

            with ml_dataset._dataset_io_lock():
                updated, changed, had_bad_rows = ml_dataset._collect_mutated_lines(_commit_mutate)
                if changed or had_bad_rows:
                    dataset_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dataset_path.with_name(
                        f"{dataset_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                    )
                    tmp.write_text("\n".join(updated) + "\n", encoding="utf-8")
                    ml_dataset._atomic_replace_with_retry(tmp, dataset_path)
        else:
            rows_annotated = 0
    finally:
        ml_dataset.ML_FILE = original_ml_file

    missing_positive_symbols = sorted(sym for sym in watchlist_map if sym not in positive_symbols_with_rows)
    return {
        "teacher_key": teacher_key,
        "target_day_local": target_day_text,
        "source_phase": phase,
        "rows_scanned": rows_scanned,
        "rows_in_window": rows_in_window,
        "rows_annotated": rows_annotated,
        "positive_rows": positive_rows,
        "symbols_in_window": len(symbols_in_window),
        "positive_symbols_with_rows": len(positive_symbols_with_rows),
        "positive_symbols_missing_rows": missing_positive_symbols,
        "mandatory_positive_coverage_pct": round(
            len(positive_symbols_with_rows) / len(watchlist_map) * 100.0, 2
        ) if watchlist_map else 0.0,
    }


def backfill_from_local_reports(
    *,
    dataset_path: Path = DEFAULT_DATASET_FILE,
    report_dir: Path = DEFAULT_REPORT_DIR,
    phase: str = "final",
    teacher_key: str = DEFAULT_TEACHER_KEY,
) -> Dict[str, Any]:
    reports = list(_iter_local_reports(report_dir, phase=phase))
    summaries: List[Dict[str, Any]] = []
    rows_annotated_total = 0
    for report in reports:
        summary = annotate_ml_dataset_from_report(
            report,
            teacher_key=teacher_key,
            dataset_path=dataset_path,
        )
        summaries.append(summary)
        rows_annotated_total += int(summary.get("rows_annotated", 0) or 0)
    return {
        "dataset_file": str(dataset_path),
        "report_dir": str(report_dir),
        "phase": phase,
        "teacher_key": teacher_key,
        "reports_processed": len(reports),
        "rows_annotated_total": rows_annotated_total,
        "summaries": summaries,
    }


@dataclass
class DatasetBundle:
    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_train: List[dict]
    meta_val: List[dict]
    meta_test: List[dict]


def load_training_rows(
    dataset_path: Path,
    *,
    teacher_key: str = DEFAULT_TEACHER_KEY,
    timezone_name: str = DEFAULT_TIMEZONE,
    max_local_hour: int = DEFAULT_MAX_LOCAL_HOUR,
    min_date: Optional[datetime] = None,
    timeframes: Sequence[str] = ("15m", "1h"),
) -> List[dict]:
    tz = ZoneInfo(timezone_name)
    allowed_tfs = {str(tf) for tf in timeframes}
    rows: List[dict] = []
    for rec in _iter_jsonl(dataset_path):
        tf = str(rec.get("tf", ""))
        if allowed_tfs and tf not in allowed_tfs:
            continue
        teacher = (rec.get("teacher") or {}).get(teacher_key)
        if not isinstance(teacher, dict):
            continue
        dt = _parse_ts(rec.get("ts_signal")) if rec.get("ts_signal") else None
        if dt is None:
            continue
        if min_date and dt < min_date:
            continue
        local_dt = dt.astimezone(tz)
        if local_dt.hour > int(max_local_hour):
            continue
        rec = dict(rec)
        rec["_dt"] = dt
        rec["_dt_local"] = local_dt
        rec["_target_day_local"] = str(teacher.get("target_day_local", ""))
        rec["_y"] = 1.0 if bool(teacher.get("watchlist_top_gainer")) else 0.0
        rows.append(rec)
    rows.sort(key=lambda item: item["_dt"])
    return rows


def _split_days(days: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    ordered = [day for day in days if day]
    n = len(ordered)
    if n < 3:
        raise RuntimeError("Need at least 3 labeled days for train/val/test split")
    train_end = max(1, int(n * 0.6))
    val_end = max(train_end + 1, int(n * 0.8))
    if val_end >= n:
        val_end = n - 1
    train_days = list(ordered[:train_end])
    val_days = list(ordered[train_end:val_end])
    test_days = list(ordered[val_end:])
    if not val_days:
        val_days = [train_days.pop()]
    if not test_days:
        test_days = [val_days.pop()]
    return train_days, val_days, test_days


def build_dataset(rows: List[dict]) -> Tuple[DatasetBundle, Dict[str, List[str]]]:
    feature_names = safe_feature_names()
    unique_days = sorted({str(rec.get("_target_day_local", "")) for rec in rows if rec.get("_target_day_local")})
    train_days, val_days, test_days = _split_days(unique_days)
    by_split = {
        "train": [rec for rec in rows if rec.get("_target_day_local") in set(train_days)],
        "val": [rec for rec in rows if rec.get("_target_day_local") in set(val_days)],
        "test": [rec for rec in rows if rec.get("_target_day_local") in set(test_days)],
    }

    def _pack(items: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((len(items), len(feature_names)), dtype=float)
        y = np.zeros(len(items), dtype=float)
        for idx, rec in enumerate(items):
            fmap = build_feature_dict(rec)
            X[idx] = np.array([_safe_float(fmap.get(name)) for name in feature_names], dtype=float)
            y[idx] = float(rec.get("_y", 0.0))
        return X, y

    X_train, y_train = _pack(by_split["train"])
    X_val, y_val = _pack(by_split["val"])
    X_test, y_test = _pack(by_split["test"])
    bundle = DatasetBundle(
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        meta_train=by_split["train"],
        meta_val=by_split["val"],
        meta_test=by_split["test"],
    )
    return bundle, {"train": train_days, "val": val_days, "test": test_days}


def _ranking_summary(
    rows: Sequence[dict],
    scores: np.ndarray,
    *,
    checkpoint_hours: Sequence[int],
    top_n: int,
    timezone_name: str,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Tuple[dict, float]]]] = {}
    for rec, score in zip(rows, scores):
        day = str(rec.get("_target_day_local", ""))
        sym = str(rec.get("sym", ""))
        if not day or not sym:
            continue
        grouped.setdefault(day, {}).setdefault(sym, []).append((rec, float(score)))

    checkpoints = _normalize_checkpoint_hours(checkpoint_hours)
    checkpoint_metrics: List[Dict[str, Any]] = []
    day_details: List[Dict[str, Any]] = []
    primary_values: List[float] = []
    for hour in checkpoints:
        recalls: List[float] = []
        precisions: List[float] = []
        days_evaluated = 0
        for day, sym_rows in sorted(grouped.items()):
            positives = {
                sym
                for sym, items in sym_rows.items()
                if any(float(item[0].get("_y", 0.0)) > 0.5 for item in items)
            }
            if not positives:
                continue
            latest: List[Tuple[str, float]] = []
            for sym, items in sym_rows.items():
                eligible = [
                    (item_rec, item_score)
                    for item_rec, item_score in items
                    if int(item_rec["_dt_local"].hour) <= int(hour)
                ]
                if not eligible:
                    continue
                chosen = max(eligible, key=lambda pair: pair[0]["_dt_local"])
                latest.append((sym, float(chosen[1])))
            if not latest:
                continue
            latest.sort(key=lambda item: item[1], reverse=True)
            predicted = [sym for sym, _ in latest[:top_n]]
            hits = sum(1 for sym in predicted if sym in positives)
            recall = hits / max(1, len(positives)) * 100.0
            precision = hits / max(1, len(predicted)) * 100.0
            recalls.append(recall)
            precisions.append(precision)
            days_evaluated += 1
            day_details.append(
                {
                    "target_day_local": day,
                    "checkpoint_hour_local": hour,
                    "positives_total": len(positives),
                    "predicted_top_n": top_n,
                    "hits": hits,
                    "recall_pct": round(recall, 2),
                    "precision_pct": round(precision, 2),
                    "predicted_symbols": predicted,
                }
            )
        mean_recall = round(float(np.mean(recalls)) if recalls else 0.0, 2)
        mean_precision = round(float(np.mean(precisions)) if precisions else 0.0, 2)
        checkpoint_metrics.append(
            {
                "hour_local": hour,
                "days_evaluated": days_evaluated,
                "mean_recall_pct": mean_recall,
                "mean_precision_pct": mean_precision,
            }
        )
        if recalls:
            primary_values.append(float(np.mean(recalls)))
    return {
        "timezone": timezone_name,
        "top_n": int(top_n),
        "checkpoint_metrics": checkpoint_metrics,
        "days": sorted(grouped),
        "primary_score": round(float(np.mean(primary_values)) if primary_values else 0.0, 2),
        "day_details": day_details,
    }


def train_and_evaluate(
    dataset_path: Path,
    *,
    teacher_key: str = DEFAULT_TEACHER_KEY,
    timezone_name: str = DEFAULT_TIMEZONE,
    max_local_hour: int = DEFAULT_MAX_LOCAL_HOUR,
    checkpoint_hours: Sequence[int] = DEFAULT_CHECKPOINT_HOURS,
    top_n: int = DEFAULT_TOP_N,
    min_rows: int = 500,
    min_days: int = 3,
    min_date: Optional[datetime] = None,
) -> dict:
    rows = load_training_rows(
        dataset_path,
        teacher_key=teacher_key,
        timezone_name=timezone_name,
        max_local_hour=max_local_hour,
        min_date=min_date,
    )
    unique_days = sorted({rec["_target_day_local"] for rec in rows if rec.get("_target_day_local")})
    if len(rows) < min_rows:
        raise RuntimeError(f"Not enough rows for top-gainer model training: {len(rows)} < {min_rows}")
    if len(unique_days) < min_days:
        raise RuntimeError(f"Not enough labeled days for top-gainer model training: {len(unique_days)} < {min_days}")

    bundle, split_days = build_dataset(rows)
    scaler = StandardScaler().fit(bundle.X_train)
    X_train = scaler.transform(bundle.X_train)
    X_val = scaler.transform(bundle.X_val)
    X_test = scaler.transform(bundle.X_test)

    models = {
        "logistic": LogisticModel(X_train.shape[1]).fit(X_train, bundle.y_train),
        "mlp": MLPModel(X_train.shape[1]).fit(X_train, bundle.y_train),
    }

    validation: Dict[str, Dict[str, Any]] = {}
    best_name = ""
    best_score = -1e9
    best_threshold = 0.5
    for name, model in models.items():
        val_score = model.predict_proba(X_val)
        threshold, val_metrics = find_best_threshold(bundle.y_val, val_score, bundle.y_val)
        val_ranking = _ranking_summary(
            bundle.meta_val,
            val_score,
            checkpoint_hours=checkpoint_hours,
            top_n=top_n,
            timezone_name=timezone_name,
        )
        validation[name] = {
            "threshold_metrics": val_metrics,
            "ranking": {
                "primary_score": val_ranking["primary_score"],
                "checkpoint_metrics": val_ranking["checkpoint_metrics"],
            },
        }
        score = float(val_ranking["primary_score"])
        if score > best_score:
            best_score = score
            best_name = name
            best_threshold = float(threshold)

    best_model = models[best_name]
    test_score = best_model.predict_proba(X_test)
    test_threshold_metrics = evaluate_predictions(bundle.y_test, test_score, bundle.y_test, best_threshold)
    test_ranking = _ranking_summary(
        bundle.meta_test,
        test_score,
        checkpoint_hours=checkpoint_hours,
        top_n=top_n,
        timezone_name=timezone_name,
    )

    model_payload = {
        "payload_version": 1,
        "model_name": best_name,
        "teacher_key": teacher_key,
        "timezone": timezone_name,
        "max_local_hour": int(max_local_hour),
        "top_n": int(top_n),
        "checkpoint_hours_local": list(_normalize_checkpoint_hours(checkpoint_hours)),
        "feature_names": bundle.feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model": best_model.to_dict(),
    }

    return {
        "dataset_file": str(dataset_path),
        "teacher_key": teacher_key,
        "timezone": timezone_name,
        "max_local_hour": int(max_local_hour),
        "top_n": int(top_n),
        "rows_total": len(rows),
        "days_total": len(unique_days),
        "positive_rate_pct": round(float(np.mean([rec["_y"] for rec in rows])) * 100.0, 2),
        "split_days": split_days,
        "train_rows": int(bundle.X_train.shape[0]),
        "val_rows": int(bundle.X_val.shape[0]),
        "test_rows": int(bundle.X_test.shape[0]),
        "chosen_model": best_name,
        "validation": validation,
        "test_threshold_metrics": test_threshold_metrics,
        "test_ranking": test_ranking,
        "model_payload": model_payload,
    }


def render_text(report: dict) -> str:
    ranking = report.get("test_ranking") or {}
    lines = [
        "Watchlist Top-Gainer Model",
        f"Rows: {report.get('rows_total')} days={report.get('days_total')} positive_rate={report.get('positive_rate_pct')}%",
        f"Split rows: train={report.get('train_rows')} val={report.get('val_rows')} test={report.get('test_rows')}",
        f"Chosen model: {report.get('chosen_model')}",
        f"Teacher key: {report.get('teacher_key')}",
        f"Primary ranking score: {ranking.get('primary_score')}",
        "",
        "Test checkpoint ranking:",
    ]
    for item in ranking.get("checkpoint_metrics") or []:
        lines.append(
            f"  {int(item.get('hour_local', 0)):02d}:00 recall={item.get('mean_recall_pct')}% precision={item.get('mean_precision_pct')}% days={item.get('days_evaluated')}"
        )
    threshold = report.get("test_threshold_metrics") or {}
    lines.extend(
        [
            "",
            "Threshold view on test rows:",
            f"  precision={threshold.get('precision')} recall={threshold.get('recall')} f1={threshold.get('f1')} coverage={threshold.get('coverage')}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a watchlist-wide model for future top gainers from ml_dataset teacher labels.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--teacher-key", type=str, default=DEFAULT_TEACHER_KEY)
    parser.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE)
    parser.add_argument("--max-local-hour", type=int, default=DEFAULT_MAX_LOCAL_HOUR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--checkpoint-hours", type=str, default="1,4,8,12,18,22")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-days", type=int, default=3)
    parser.add_argument("--min-date", type=str, default="")
    parser.add_argument("--phase", type=str, default="final")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    checkpoint_hours = tuple(
        int(part.strip())
        for part in args.checkpoint_hours.split(",")
        if part.strip()
    )
    min_date = _parse_ts(args.min_date) if args.min_date else None

    backfill_summary = None
    if not args.skip_backfill:
        backfill_summary = backfill_from_local_reports(
            dataset_path=args.dataset,
            report_dir=args.report_dir,
            phase=args.phase,
            teacher_key=args.teacher_key,
        )

    report = train_and_evaluate(
        args.dataset,
        teacher_key=args.teacher_key,
        timezone_name=args.timezone,
        max_local_hour=args.max_local_hour,
        checkpoint_hours=checkpoint_hours,
        top_n=args.top_n,
        min_rows=args.min_rows,
        min_days=args.min_days,
        min_date=min_date,
    )
    if backfill_summary is not None:
        report["backfill"] = backfill_summary
    save_json(args.model_out, report["model_payload"])
    save_json(args.report_out, {k: v for k, v in report.items() if k != "model_payload"})

    if args.as_json:
        print(json.dumps({k: v for k, v in report.items() if k != "model_payload"}, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print("")
        print(f"Model saved to: {args.model_out}")
        print(f"Report saved to: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
