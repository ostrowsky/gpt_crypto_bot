from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_DATASET = FILES / "critic_dataset.jsonl"
DEFAULT_OUTPUT = REPORTS / "watchlist_rescue_admission_replay_latest.json"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        return value if value == value else default
    except Exception:
        return default


def _bool(value: Any) -> bool:
    return bool(value)


def _day(row: dict[str, Any]) -> str:
    ts = str(row.get("ts_signal") or "")
    return ts[:10]


def _teacher_final(row: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher") or {}
    final = teacher.get("final") if isinstance(teacher, dict) else {}
    return final if isinstance(final, dict) else {}


def _flags(row: dict[str, Any]) -> dict[str, Any]:
    decision = row.get("decision") or {}
    flags = decision.get("signal_flags") if isinstance(decision, dict) else {}
    return flags if isinstance(flags, dict) else {}


def _features(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("f") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {str(k): _num(v) for k, v in raw.items()}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        labels = row.get("labels") or {}
        if labels.get("ret_5") is None:
            continue
        if str(row.get("signal_type") or "none") == "none":
            continue
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r.get("ts_signal") or ""), str(r.get("sym") or "")))


def _profile_entry_ok_trend(row: dict[str, Any]) -> bool:
    return str(row.get("signal_type") or "") in {"trend", "strong_trend"} and _bool(_flags(row).get("entry_ok"))


def _profile_alignment_structural(row: dict[str, Any]) -> bool:
    f = _features(row)
    flags = _flags(row)
    return (
        (str(row.get("signal_type") or "") == "alignment" or _bool(flags.get("alignment_ok")))
        and f.get("slope", 0.0) >= 0.45
        and 52.0 <= f.get("rsi", 0.0) <= 74.0
        and f.get("vol_x", 0.0) >= 0.9
        and 3.0 <= f.get("daily_range", 0.0) <= 14.0
        and f.get("macd_hist_norm", 0.0) > 0.0
        and f.get("close_vs_ema20", 0.0) > 0.0
    )


def _profile_surge_followthrough(row: dict[str, Any]) -> bool:
    f = _features(row)
    return (
        _bool(_flags(row).get("surge_ok"))
        and f.get("slope", 0.0) >= 0.70
        and f.get("vol_x", 0.0) >= 1.25
        and 55.0 <= f.get("rsi", 0.0) <= 75.0
        and f.get("daily_range", 0.0) <= 14.0
        and f.get("macd_hist_norm", 0.0) > 0.0
    )


def _profile_near_miss_momentum(row: dict[str, Any]) -> bool:
    f = _features(row)
    return (
        str(row.get("signal_type") or "") in {"trend", "strong_trend", "impulse", "impulse_speed"}
        and f.get("slope", 0.0) >= 0.50
        and f.get("vol_x", 0.0) >= 1.15
        and f.get("adx", 0.0) >= 20.0
        and 55.0 <= f.get("rsi", 0.0) <= 74.0
        and 3.0 <= f.get("daily_range", 0.0) <= 14.0
        and f.get("close_vs_ema20", 0.0) > 0.0
    )


def _profile_watchlist_mover_rescue_v1(row: dict[str, Any]) -> bool:
    f = _features(row)
    flags = _flags(row)
    structural = _bool(flags.get("entry_ok")) or _bool(flags.get("surge_ok")) or _bool(flags.get("alignment_ok"))
    return (
        structural
        and f.get("slope", 0.0) >= 0.60
        and f.get("vol_x", 0.0) >= 1.20
        and 56.0 <= f.get("rsi", 0.0) <= 73.0
        and 4.0 <= f.get("daily_range", 0.0) <= 13.5
        and f.get("macd_hist_norm", 0.0) > 0.0
        and f.get("close_vs_ema20", 0.0) > 0.50
    )


def _profile_watchlist_mover_rescue_strict(row: dict[str, Any]) -> bool:
    f = _features(row)
    return (
        _profile_watchlist_mover_rescue_v1(row)
        and f.get("adx", 0.0) >= 24.0
        and f.get("daily_range", 0.0) <= 12.0
        and f.get("vol_x", 0.0) >= 1.40
    )


PROFILES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "entry_ok_trend": _profile_entry_ok_trend,
    "alignment_structural": _profile_alignment_structural,
    "surge_followthrough": _profile_surge_followthrough,
    "near_miss_momentum": _profile_near_miss_momentum,
    "watchlist_mover_rescue_v1": _profile_watchlist_mover_rescue_v1,
    "watchlist_mover_rescue_strict": _profile_watchlist_mover_rescue_strict,
}


def build(dataset: Path = DEFAULT_DATASET, output: Path = DEFAULT_OUTPUT, train_fraction: float = 0.70, focus_symbols: list[str] | None = None) -> dict[str, Any]:
    rows = _load_rows(dataset)
    train, holdout = _split_by_day(rows, train_fraction)
    all_results = {name: _evaluate(rows, predicate) for name, predicate in PROFILES.items()}
    holdout_results = {name: _evaluate(holdout, predicate) for name, predicate in PROFILES.items()}
    ranked = sorted(
        [{"profile": name, **metrics} for name, metrics in holdout_results.items()],
        key=lambda item: (item["passes_research_gate"], item["ret5_precision"], item["avg_ret5"], item["selected_count"]),
        reverse=True,
    )
    payload = {
        "dataset": str(dataset),
        "rows": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_days": _day_range(train),
        "holdout_days": _day_range(holdout),
        "baseline_all": _evaluate(rows, lambda _row: True),
        "baseline_holdout": _evaluate(holdout, lambda _row: True),
        "profiles_all": all_results,
        "profiles_holdout": holdout_results,
        "selected_profile": ranked[0] if ranked else None,
        "decision": _decision(ranked[0] if ranked else None),
        "focus_symbols": _focus(rows, focus_symbols or []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _split_by_day(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    days = sorted({day for row in rows if (day := _day(row))})
    if len(days) < 2:
        return rows, []
    split = max(1, min(len(days) - 1, int(len(days) * train_fraction)))
    train_days = set(days[:split])
    return [row for row in rows if _day(row) in train_days], [row for row in rows if _day(row) not in train_days]


def _evaluate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    unique = {(str(row.get("sym") or ""), _day(row)) for row in selected}
    top = [row for row in selected if bool(_teacher_final(row).get("watchlist_top_gainer"))]
    missed_top = [row for row in top if str(_teacher_final(row).get("status") or "") != "bought"]
    bought_top = [row for row in top if str(_teacher_final(row).get("status") or "") == "bought"]
    ret5_pos = [row for row in selected if _num((row.get("labels") or {}).get("ret_5")) > 0.0]
    ret5_ge_1 = [row for row in selected if _num((row.get("labels") or {}).get("ret_5")) >= 1.0]
    negative = [row for row in selected if _num((row.get("labels") or {}).get("ret_5")) < 0.0]
    metrics = {
        "selected_count": len(selected),
        "unique_day_symbols": len(unique),
        "avg_ret3": _avg_label(selected, "ret_3"),
        "avg_ret5": _avg_label(selected, "ret_5"),
        "avg_ret10": _avg_label(selected, "ret_10"),
        "ret5_precision": _ratio(len(ret5_pos), len(selected)),
        "ret5_ge_1_rate": _ratio(len(ret5_ge_1), len(selected)),
        "negative_ret5_rate": _ratio(len(negative), len(selected)),
        "teacher_top15_count": len(top),
        "teacher_top15_precision": _ratio(len(top), len(selected)),
        "teacher_missed_top15_count": len(missed_top),
        "teacher_bought_top15_count": len(bought_top),
        "examples_positive": [_compact(row) for row in sorted(ret5_ge_1, key=lambda r: _num((r.get("labels") or {}).get("ret_5")), reverse=True)[:10]],
        "examples_missed_top15": [_compact(row) for row in missed_top[:10]],
    }
    metrics["passes_research_gate"] = (
        metrics["selected_count"] >= 20
        and metrics["ret5_precision"] >= 0.55
        and metrics["avg_ret5"] > 0.15
        and metrics["negative_ret5_rate"] <= 0.45
    )
    return metrics


def _focus(rows: list[dict[str, Any]], symbols: list[str]) -> list[dict[str, Any]]:
    wanted = {s.strip().upper().replace("/", "") for s in symbols if s.strip()}
    if not wanted:
        return []
    out = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sym = str(row.get("sym") or "").upper()
        if sym in wanted:
            grouped[sym].append(row)
    for sym in sorted(wanted):
        symbol_rows = grouped.get(sym, [])
        profile_hits = {
            name: _evaluate(symbol_rows, predicate)
            for name, predicate in PROFILES.items()
        }
        out.append({
            "symbol": sym,
            "rows": len(symbol_rows),
            "latest_rows": [_compact(row) for row in symbol_rows[-5:]],
            "profile_hits": profile_hits,
        })
    return out


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels") or {}
    f = _features(row)
    return {
        "day": _day(row),
        "ts_signal": row.get("ts_signal"),
        "symbol": row.get("sym"),
        "tf": row.get("tf"),
        "signal_type": row.get("signal_type"),
        "flags": _flags(row),
        "ret_3": labels.get("ret_3"),
        "ret_5": labels.get("ret_5"),
        "ret_10": labels.get("ret_10"),
        "teacher_top15": bool(_teacher_final(row).get("watchlist_top_gainer")),
        "teacher_status": _teacher_final(row).get("status"),
        "features": {k: f.get(k) for k in ("slope", "rsi", "adx", "vol_x", "daily_range", "macd_hist_norm", "close_vs_ema20")},
    }


def _avg_label(rows: list[dict[str, Any]], key: str) -> float:
    values = [_num((row.get("labels") or {}).get(key), None) for row in rows]
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else 0.0


def _ratio(a: int, b: int) -> float:
    return round(a / b, 6) if b else 0.0


def _day_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    days = sorted({day for row in rows if (day := _day(row))})
    return {"count": len(days), "start": days[0] if days else None, "end": days[-1] if days else None}


def _decision(selected: dict[str, Any] | None) -> str:
    if not selected:
        return "research_only_no_profile"
    if selected.get("passes_research_gate"):
        return "advance_selected_profile_to_fee_slippage_behavior_replay"
    return "research_only_no_profile_passed_gate"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--focus-symbols", default="")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    focus = [s for s in args.focus_symbols.split(",") if s.strip()]
    payload = build(args.dataset, args.output, args.train_fraction, focus)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"decision": payload["decision"], "selected_profile": payload["selected_profile"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
