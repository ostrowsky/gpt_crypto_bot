from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import research_universe_shadow_collector as collector


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
DATASET_FILE = collector.DATASET_FILE
DEFAULT_OUTPUT_JSON = REPORT_DIR / "research_universe_shadow_scorecard_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "research_universe_shadow_scorecard_latest.txt"
LOCAL_ZONE = ZoneInfo("Europe/Budapest")


def build_scorecard(
    *,
    dataset_file: Path = DATASET_FILE,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    horizon: int = 5,
    days: int = 14,
    min_mature: int = 3,
    min_positive_rate_pct: float = 55.0,
    min_avg_ret_pct: float = 0.05,
    high_return_threshold_pct: float = 1.0,
    save: bool = True,
) -> dict[str, Any]:
    rows_all = _load_rows(dataset_file)
    rows = _filter_last_days(rows_all, days=days)
    label_key = f"ret_{int(horizon)}"
    mature = [row for row in rows if _label_value(row, label_key) is not None]
    immature = len(rows) - len(mature)

    summary = _summarize_rows(mature, label_key, high_return_threshold_pct)
    inside = _summarize_rows([row for row in mature if bool(row.get("in_trade_watchlist"))], label_key, high_return_threshold_pct)
    outside = _summarize_rows([row for row in mature if not bool(row.get("in_trade_watchlist"))], label_key, high_return_threshold_pct)
    by_rule_signal = _group_rows(mature, key_fn=lambda row: str(row.get("rule_signal") or "none"), label_key=label_key, high_return_threshold_pct=high_return_threshold_pct)
    top_symbols = _symbol_rows(
        [row for row in mature if not bool(row.get("in_trade_watchlist"))],
        label_key=label_key,
        high_return_threshold_pct=high_return_threshold_pct,
        min_mature=min_mature,
        min_positive_rate_pct=min_positive_rate_pct,
        min_avg_ret_pct=min_avg_ret_pct,
    )
    feature_patterns = _feature_pattern_rows(mature, label_key=label_key, high_return_threshold_pct=high_return_threshold_pct)
    early_trend_shadow = _early_trend_shadow_cohort(rows)
    recommendation = _recommendation(total_rows=len(rows), mature_rows=len(mature), promotion_candidates=top_symbols)
    report = {
        "mode": "research_only",
        "dataset_file": str(dataset_file),
        "generated_at_utc": _utc_now_iso(),
        "window": _window(rows),
        "horizon": horizon,
        "label_key": label_key,
        "days_requested": days,
        "thresholds": {
            "min_mature": min_mature,
            "min_positive_rate_pct": min_positive_rate_pct,
            "min_avg_ret_pct": min_avg_ret_pct,
            "high_return_threshold_pct": high_return_threshold_pct,
        },
        "coverage": {
            "rows_total_loaded": len(rows_all),
            "rows_in_window": len(rows),
            "mature_rows": len(mature),
            "immature_rows": immature,
            "outside_watchlist_mature_rows": outside["count"],
            "inside_watchlist_mature_rows": inside["count"],
        },
        "summary": summary,
        "inside_watchlist": inside,
        "outside_watchlist": outside,
        "by_rule_signal": by_rule_signal,
        "promotion_candidates": top_symbols[:20],
        "feature_patterns": feature_patterns[:20],
        "early_trend_shadow": early_trend_shadow,
        "recommendation": recommendation,
        "guardrails": [
            "research_only",
            "live_watchlist_unchanged",
            "buy_sell_gates_unchanged",
            "promotion_requires_separate_replay_gate",
        ],
    }
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(render_text(report), encoding="utf-8")
        report["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return report


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    summary = report.get("summary") or {}
    outside = report.get("outside_watchlist") or {}
    early = report.get("early_trend_shadow") or {}
    rec = report.get("recommendation") or {}
    lines = [
        "Research Universe Shadow Scorecard",
        "",
        f"Mode: {report.get('mode', 'research_only')}",
        f"Window: {(report.get('window') or {}).get('first_ts_utc', '')} -> {(report.get('window') or {}).get('last_ts_utc', '')}",
        f"Label: {report.get('label_key')}  days={report.get('days_requested')}",
        "",
        "Coverage:",
        f"  rows loaded/window: {coverage.get('rows_total_loaded', 0)}/{coverage.get('rows_in_window', 0)}",
        f"  mature/immature: {coverage.get('mature_rows', 0)}/{coverage.get('immature_rows', 0)}",
        f"  outside mature: {coverage.get('outside_watchlist_mature_rows', 0)}  inside mature: {coverage.get('inside_watchlist_mature_rows', 0)}",
        "",
        "Quality:",
        f"  mature avg/median: {_fmt(summary.get('avg_ret_pct'))}% / {_fmt(summary.get('median_ret_pct'))}%",
        f"  positive rate: {_fmt(summary.get('positive_rate_pct'))}%  high-return rate: {_fmt(summary.get('high_return_rate_pct'))}%",
        f"  outside avg/positive: {_fmt(outside.get('avg_ret_pct'))}% / {_fmt(outside.get('positive_rate_pct'))}%",
        "",
        "Early-trend independent shadow:",
        f"  first candidates/mature: {early.get('first_candidates', 0)}/{early.get('mature_both', 0)}",
        f"  primary precision: {_fmt(early.get('primary_precision_pct'))}%  strict: {_fmt(early.get('strict_precision_pct'))}%",
        f"  avg T+5/T+10: {_fmt(early.get('avg_ret_5_pct'))}% / {_fmt(early.get('avg_ret_10_pct'))}%",
        f"  decision: {early.get('decision', 'collect_forward_labels')}",
        "",
        f"Recommendation: {rec.get('decision', 'unknown')}",
        f"Reason: {rec.get('reason', '')}",
        "",
        "Top rule signals:",
    ]
    for row in (report.get("by_rule_signal") or [])[:8]:
        lines.append(
            f"  {row['key']}: n={row['count']} avg={_fmt(row.get('avg_ret_pct'))}% "
            f"pos={_fmt(row.get('positive_rate_pct'))}% high={_fmt(row.get('high_return_rate_pct'))}%"
        )
    lines.append("")
    lines.append("Promotion candidates for replay gate:")
    candidates = report.get("promotion_candidates") or []
    if candidates:
        for idx, row in enumerate(candidates[:10], start=1):
            lines.append(
                f"  {idx}. {row['symbol']} n={row['count']} avg={_fmt(row.get('avg_ret_pct'))}% "
                f"pos={_fmt(row.get('positive_rate_pct'))}% score={_fmt(row.get('score'), 3)}"
            )
    else:
        lines.append("  none")
    lines.extend([
        "",
        "Guardrails:",
        "  - research-only; live watchlist unchanged",
        "  - outside-watchlist rows are learning coverage, not live misses",
        "  - promotion requires separate replay/liquidity/operator gate",
    ])
    return "\n".join(lines)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _filter_last_days(rows: list[dict[str, Any]], *, days: int) -> list[dict[str, Any]]:
    if days <= 0 or not rows:
        return rows
    timestamps = [_parse_ts(row.get("ts_utc")) for row in rows]
    valid = [ts for ts in timestamps if ts is not None]
    if not valid:
        return rows
    last_ts = max(valid)
    cutoff = last_ts - timedelta(days=days)
    return [row for row in rows if (_parse_ts(row.get("ts_utc")) or last_ts) >= cutoff]


def _window(rows: list[dict[str, Any]]) -> dict[str, str]:
    timestamps = sorted(ts for ts in (_parse_ts(row.get("ts_utc")) for row in rows) if ts is not None)
    return {
        "first_ts_utc": _fmt_ts(timestamps[0]) if timestamps else "",
        "last_ts_utc": _fmt_ts(timestamps[-1]) if timestamps else "",
    }


def _summarize_rows(rows: list[dict[str, Any]], label_key: str, high_return_threshold_pct: float) -> dict[str, Any]:
    values = [_label_value(row, label_key) for row in rows]
    values = [float(v) for v in values if v is not None]
    positives = [v for v in values if v > 0]
    high = [v for v in values if v >= high_return_threshold_pct]
    return {
        "count": len(values),
        "avg_ret_pct": _avg(values),
        "median_ret_pct": _median(values),
        "positive_count": len(positives),
        "positive_rate_pct": _pct(len(positives), len(values)),
        "high_return_count": len(high),
        "high_return_rate_pct": _pct(len(high), len(values)),
    }


def _group_rows(rows: list[dict[str, Any]], *, key_fn: Any, label_key: str, high_return_threshold_pct: float) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row) or "unknown")].append(row)
    out = []
    for key, group in groups.items():
        item = {"key": key, **_summarize_rows(group, label_key, high_return_threshold_pct)}
        item["score"] = _score(item)
        out.append(item)
    out.sort(key=lambda row: (float(row.get("score") or 0.0), int(row.get("count") or 0)), reverse=True)
    return out


def _symbol_rows(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    high_return_threshold_pct: float,
    min_mature: int,
    min_positive_rate_pct: float,
    min_avg_ret_pct: float,
) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, key_fn=lambda row: str(row.get("sym") or ""), label_key=label_key, high_return_threshold_pct=high_return_threshold_pct)
    out = []
    for row in grouped:
        if not _is_safe_symbol(str(row.get("key") or "")):
            continue
        item = dict(row)
        item["symbol"] = item.pop("key")
        if (
            int(item.get("count") or 0) >= min_mature
            and float(item.get("positive_rate_pct") or 0.0) >= min_positive_rate_pct
            and float(item.get("avg_ret_pct") or 0.0) >= min_avg_ret_pct
        ):
            out.append(item)
    out.sort(key=lambda row: (float(row.get("score") or 0.0), int(row.get("count") or 0)), reverse=True)
    return out


def _feature_pattern_rows(rows: list[dict[str, Any]], *, label_key: str, high_return_threshold_pct: float) -> list[dict[str, Any]]:
    return _group_rows(rows, key_fn=_feature_pattern_key, label_key=label_key, high_return_threshold_pct=high_return_threshold_pct)


def _early_trend_shadow_cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = [
        row
        for row in rows
        if bool((row.get("early_trend_shadow") or {}).get("candidate"))
    ]
    annotated.sort(key=lambda row: str(row.get("ts_utc") or ""))
    first_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in annotated:
        symbol = str(row.get("sym") or "")
        local_day = _local_day(row.get("ts_utc"))
        if not symbol or not local_day:
            continue
        key = (symbol, local_day)
        if key in seen:
            continue
        seen.add(key)
        first_rows.append(row)
    mature = [
        row
        for row in first_rows
        if _label_value(row, "ret_5") is not None and _label_value(row, "ret_10") is not None
    ]
    primary = [
        row
        for row in mature
        if float(_label_value(row, "ret_5") or 0.0) >= 0.5
        and float(_label_value(row, "ret_10") or 0.0) >= 1.0
    ]
    strict = [
        row
        for row in mature
        if float(_label_value(row, "ret_5") or 0.0) >= 1.0
        and float(_label_value(row, "ret_10") or 0.0) >= 2.0
    ]
    ret_5 = [float(_label_value(row, "ret_5") or 0.0) for row in mature]
    ret_10 = [float(_label_value(row, "ret_10") or 0.0) for row in mature]
    if len(mature) < 30:
        decision = "collect_forward_labels"
    elif _pct(len(primary), len(mature)) is not None and float(_pct(len(primary), len(mature)) or 0.0) >= 10.0:
        decision = "ready_for_frozen_forward_audit_not_production"
    else:
        decision = "forward_gate_failed_keep_shadow_only"
    return {
        "annotated_candidates": len(annotated),
        "first_candidates": len(first_rows),
        "mature_both": len(mature),
        "pending": len(first_rows) - len(mature),
        "primary_useful": len(primary),
        "primary_precision_pct": _pct(len(primary), len(mature)),
        "strict_useful": len(strict),
        "strict_precision_pct": _pct(len(strict), len(mature)),
        "avg_ret_5_pct": _avg(ret_5),
        "median_ret_5_pct": _median(ret_5),
        "avg_ret_10_pct": _avg(ret_10),
        "median_ret_10_pct": _median(ret_10),
        "decision": decision,
        "production_eligible": False,
    }


def _feature_pattern_key(row: dict[str, Any]) -> str:
    f = row.get("f") or {}
    return "|".join(
        [
            str(row.get("rule_signal") or "none"),
            _bin("adx", _float(f.get("adx")), [(15, "low"), (25, "mid"), (35, "strong")]),
            _bin("vol", _float(f.get("vol_x")), [(1.0, "base"), (2.0, "active"), (4.0, "hot")]),
            _bin("slope", _float(f.get("slope")), [(0.0, "flat"), (0.25, "up"), (0.75, "steep")]),
        ]
    )


def _bin(name: str, value: float, thresholds: list[tuple[float, str]]) -> str:
    label = "below"
    for threshold, candidate in thresholds:
        if value >= threshold:
            label = candidate
    return f"{name}:{label}"


def _recommendation(*, total_rows: int, mature_rows: int, promotion_candidates: list[dict[str, Any]]) -> dict[str, str]:
    if total_rows <= 0:
        return {"decision": "insufficient_data", "reason": "research universe shadow dataset is empty"}
    if promotion_candidates:
        caution = "" if mature_rows >= 20 else f"; caution: only {mature_rows} mature labels overall"
        return {
            "decision": "advance_candidates_to_replay_gate",
            "reason": f"{len(promotion_candidates)} outside-watchlist symbols passed research thresholds{caution}",
        }
    if mature_rows < 20:
        return {"decision": "collect_more_labels", "reason": f"only {mature_rows} mature labels; avoid drawing conclusions"}
    return {"decision": "continue_shadow_collection", "reason": "no outside-watchlist symbol passed promotion thresholds yet"}


def _label_value(row: dict[str, Any], label_key: str) -> float | None:
    labels = row.get("labels") or {}
    value = labels.get(label_key)
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _score(row: dict[str, Any]) -> float:
    avg_ret = float(row.get("avg_ret_pct") or 0.0)
    positive = float(row.get("positive_rate_pct") or 0.0) / 100.0
    high = float(row.get("high_return_rate_pct") or 0.0) / 100.0
    count_bonus = min(1.0, math.log1p(max(0, int(row.get("count") or 0))) / 5.0)
    return round((avg_ret * 0.55) + (positive * 0.25) + (high * 0.35) + (count_bonus * 0.05), 6)


def _is_safe_symbol(symbol: str) -> bool:
    symbol = symbol.strip().upper()
    return bool(symbol) and symbol.isascii() and symbol.isalnum() and symbol.endswith("USDT")


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _local_day(value: Any) -> str:
    parsed = _parse_ts(value)
    return parsed.astimezone(LOCAL_ZONE).date().isoformat() if parsed else ""


def _fmt_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _avg(values: Iterable[float]) -> float | None:
    data = [float(x) for x in values]
    if not data:
        return None
    return round(sum(data) / len(data), 6)


def _median(values: Iterable[float]) -> float | None:
    data = [float(x) for x in values]
    if not data:
        return None
    return round(float(median(data)), 6)


def _pct(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(float(num) / float(den) * 100.0, 4)


def _float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        out = float(value)
    except Exception:
        return "n/a"
    return f"{out:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build research-universe shadow scorecard.")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--min-mature", type=int, default=3)
    parser.add_argument("--min-positive-rate-pct", type=float, default=55.0)
    parser.add_argument("--min-avg-ret-pct", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = build_scorecard(
        dataset_file=args.dataset,
        output_json=args.output_json,
        output_txt=args.output_txt,
        horizon=args.horizon,
        days=args.days,
        min_mature=args.min_mature,
        min_positive_rate_pct=args.min_positive_rate_pct,
        min_avg_ret_pct=args.min_avg_ret_pct,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
