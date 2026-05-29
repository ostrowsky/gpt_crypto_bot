from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import report_exit_quality as exit_quality

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_CONTINUATION_MARGIN_PCT = 0.75
DEFAULT_MIN_TRAIN_DAYS = 3


def _num(value: Any) -> float | None:
    return exit_quality._num(value)  # type: ignore[attr-defined]


def _bucket(value: Any, cuts: list[float], labels: list[str]) -> str:
    number = _num(value)
    if number is None:
        return "unknown"
    for cut, label in zip(cuts, labels):
        if number <= cut:
            return label
    return labels[-1]


def _rank_bucket(value: Any) -> str:
    rank = _num(value)
    if rank is None or rank <= 0:
        return "not_top"
    if rank <= 3:
        return "top1_3"
    if rank <= 7:
        return "top4_7"
    if rank <= 15:
        return "top8_15"
    return "below_top15"


def _case_features(case: dict[str, Any]) -> dict[str, str]:
    return {
        "reason": str(case.get("exit_reason_bucket") or "unknown"),
        "source": str(case.get("source") or "unknown"),
        "mode": str(case.get("mode") or "unknown"),
        "tf": str(case.get("tf") or "unknown"),
        "entry_timing": str(case.get("entry_timing") or "unknown"),
        "exit_timing": str(case.get("exit_timing") or "unknown"),
        "pnl_bucket": _bucket(case.get("pnl_pct"), [-2.0, -0.25, 0.0, 0.75, 2.0], ["loss_gt2", "loss", "flat", "small_win", "win", "big_win"]),
        "mfe_bucket": _bucket(case.get("max_favorable_pct"), [0.0, 0.5, 1.5, 3.0, 6.0], ["none", "tiny", "small", "medium", "large", "huge"]),
        "giveback_bucket": _bucket(case.get("giveback_pct"), [0.0, 0.5, 1.5, 3.0, 6.0], ["none", "low", "medium", "high", "very_high", "extreme"]),
        "eff_bucket": _bucket(case.get("exit_efficiency"), [-1.0, 0.0, 0.25, 0.5, 0.8], ["very_bad", "bad", "weak", "ok", "good", "excellent"]),
        "capture_bucket": _bucket(case.get("capture_ratio_at_entry"), [0.2, 0.4, 0.6, 0.8], ["early", "good", "mid", "late", "very_late"]),
        "top_rank_bucket": _rank_bucket(case.get("top_mover_rank")),
    }


def _wrong_exit_label(case: dict[str, Any], *, continuation_margin_pct: float) -> tuple[int | None, float | None]:
    future = _num(case.get("future_favorable_pct"))
    mfe = _num(case.get("max_favorable_pct"))
    pnl = _num(case.get("pnl_pct"))
    if future is None or mfe is None:
        return None, None
    anchor = max(mfe, pnl or 0.0)
    continuation = future - anchor
    label = 1 if continuation >= continuation_margin_pct else 0
    return label, round(max(continuation, 0.0), 4)


def _load_cases(days: int, reports_dir: Path, *, continuation_margin_pct: float) -> list[dict[str, Any]]:
    cfg = exit_quality.ExitAuditConfig(days=days)
    _, cases = exit_quality._load_reports(reports_dir, cfg)  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []
    for case in cases:
        label, continuation = _wrong_exit_label(case, continuation_margin_pct=continuation_margin_pct)
        if label is None:
            continue
        features = _case_features(case)
        rows.append({
            **case,
            "label_wrong_exit_continuation": label,
            "post_exit_continuation_extra_pct": continuation,
            "features": features,
        })
    rows.sort(key=lambda row: (str(row.get("day") or ""), str(row.get("exit_ts") or row.get("entry_ts") or "")))
    return rows


def _split_by_day(rows: list[dict[str, Any]], min_train_days: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    days = sorted({str(row.get("day")) for row in rows if row.get("day")})
    if len(days) <= 1:
        return rows, [], "insufficient_days"
    split_idx = max(min_train_days, int(math.floor(len(days) * 0.7)))
    split_idx = min(split_idx, len(days) - 1)
    train_days = set(days[:split_idx])
    train = [row for row in rows if str(row.get("day")) in train_days]
    test = [row for row in rows if str(row.get("day")) not in train_days]
    return train, test, "chronological"


def _rate(rows: Iterable[dict[str, Any]]) -> float | None:
    labels = [int(row.get("label_wrong_exit_continuation") or 0) for row in rows]
    if not labels:
        return None
    return round(sum(labels) / len(labels), 4)


def _train_feature_rates(train: list[dict[str, Any]], *, min_support: int = 3) -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        for name, value in (row.get("features") or {}).items():
            buckets[(str(name), str(value))].append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in buckets.items():
        if len(rows) < min_support:
            continue
        wrong = sum(int(row.get("label_wrong_exit_continuation") or 0) for row in rows)
        loss_values = [_num(row.get("post_exit_continuation_extra_pct")) or 0.0 for row in rows]
        out[key] = {
            "support": len(rows),
            "wrong": wrong,
            "wrong_rate": wrong / len(rows),
            "avg_continuation_extra_pct": sum(loss_values) / len(loss_values),
        }
    return out


def _score(row: dict[str, Any], feature_rates: dict[tuple[str, str], dict[str, Any]], baseline: float) -> float:
    rates: list[float] = []
    for name, value in (row.get("features") or {}).items():
        stat = feature_rates.get((str(name), str(value)))
        if not stat:
            continue
        support_weight = min(float(stat.get("support") or 0) / 10.0, 1.0)
        downside_weight = 1.0 + min(float(stat.get("avg_continuation_extra_pct") or 0.0) / 3.0, 1.0)
        rates.append(float(stat.get("wrong_rate") or baseline) * support_weight * downside_weight)
    if not rates:
        return round(baseline, 4)
    return round(sum(rates) / len(rates), 4)


def _precision_at_fraction(scored: list[dict[str, Any]], fraction: float) -> dict[str, Any]:
    if not scored:
        return {"n": 0, "precision": None, "avg_extra_pct": None}
    n = max(1, int(math.ceil(len(scored) * fraction)))
    top = sorted(scored, key=lambda row: row.get("risk_score") or 0.0, reverse=True)[:n]
    labels = [int(row.get("label_wrong_exit_continuation") or 0) for row in top]
    extras = [_num(row.get("post_exit_continuation_extra_pct")) or 0.0 for row in top]
    return {
        "n": n,
        "precision": round(sum(labels) / len(labels), 4),
        "avg_extra_pct": round(sum(extras) / len(extras), 4),
        "symbols": [row.get("sym") for row in top[:10]],
    }


def _top_segments(rows: list[dict[str, Any]], *, min_support: int = 2) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for name, value in (row.get("features") or {}).items():
            buckets[(str(name), str(value))].append(row)
    segments: list[dict[str, Any]] = []
    for (name, value), seg_rows in buckets.items():
        if len(seg_rows) < min_support:
            continue
        wrong = sum(int(row.get("label_wrong_exit_continuation") or 0) for row in seg_rows)
        extras = [_num(row.get("post_exit_continuation_extra_pct")) or 0.0 for row in seg_rows]
        segments.append({
            "feature": name,
            "value": value,
            "support": len(seg_rows),
            "wrong": wrong,
            "wrong_rate": round(wrong / len(seg_rows), 4),
            "avg_extra_pct": round(sum(extras) / len(extras), 4),
            "score": round((wrong / len(seg_rows)) * math.log1p(len(seg_rows)) * (1.0 + sum(extras) / max(len(extras), 1)), 4),
            "examples": [f"{r.get('day')} {r.get('sym')}" for r in seg_rows[:5]],
        })
    return sorted(segments, key=lambda s: s["score"], reverse=True)


def build(days: int = 0, *, reports_dir: Path = REPORTS, continuation_margin_pct: float = DEFAULT_CONTINUATION_MARGIN_PCT, min_train_days: int = DEFAULT_MIN_TRAIN_DAYS) -> dict[str, Any]:
    rows = _load_cases(days, reports_dir, continuation_margin_pct=continuation_margin_pct)
    train, test, split_status = _split_by_day(rows, min_train_days)
    train_baseline = _rate(train) or 0.0
    test_baseline = _rate(test)
    feature_rates = _train_feature_rates(train)
    scored_test = [{**row, "risk_score": _score(row, feature_rates, train_baseline)} for row in test]
    status = "empty" if not rows else ("insufficient_test" if not test else "ok")
    precision_top_20 = _precision_at_fraction(scored_test, 0.20)
    decision = "research_only"
    if status == "ok" and test_baseline is not None and precision_top_20["precision"] is not None:
        if precision_top_20["precision"] >= test_baseline + 0.15 and precision_top_20["n"] >= 3:
            decision = "promising_shadow_segments_only"
        else:
            decision = "inconclusive_or_weak"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "decision": decision,
        "config": {
            "days": days,
            "reports_dir": str(reports_dir),
            "continuation_margin_pct": continuation_margin_pct,
            "min_train_days": min_train_days,
        },
        "summary": {
            "cases_labeled": len(rows),
            "train_cases": len(train),
            "test_cases": len(test),
            "split_status": split_status,
            "train_wrong_exit_rate": train_baseline,
            "test_wrong_exit_rate": test_baseline,
            "test_precision_top_20pct": precision_top_20,
        },
        "top_train_segments": _top_segments(train)[:25],
        "top_test_segments": _top_segments(test)[:25],
        "scored_test_cases": sorted(scored_test, key=lambda row: row.get("risk_score") or 0.0, reverse=True)[:50],
        "recommendation": "Use high-risk segments only to form replay hypotheses. Do not change live SELL logic from this report alone.",
    }


def format_text(report: dict[str, Any]) -> str:
    s = report.get("summary") or {}
    p20 = s.get("test_precision_top_20pct") or {}
    lines = [
        "Exit failure discriminator",
        f"status: {report.get('status')} | decision: {report.get('decision')}",
        f"cases: labeled={s.get('cases_labeled')} train={s.get('train_cases')} test={s.get('test_cases')} split={s.get('split_status')}",
        f"wrong-exit rate: train={s.get('train_wrong_exit_rate')} test={s.get('test_wrong_exit_rate')} | top20 precision={p20.get('precision')} n={p20.get('n')} avg_extra={p20.get('avg_extra_pct')}",
        "",
        "Top train segments:",
    ]
    for seg in (report.get("top_train_segments") or [])[:10]:
        lines.append(f"- {seg.get('feature')}={seg.get('value')} support={seg.get('support')} wrong_rate={seg.get('wrong_rate')} avg_extra={seg.get('avg_extra_pct')}")
    lines += ["", "Top scored test cases:"]
    for row in (report.get("scored_test_cases") or [])[:10]:
        lines.append(f"- {row.get('day')} {row.get('sym')} {row.get('tf')} score={row.get('risk_score')} label={row.get('label_wrong_exit_continuation')} extra={row.get('post_exit_continuation_extra_pct')} reason={row.get('exit_reason_bucket')} mode={row.get('mode')}")
    lines.append("")
    lines.append("Recommendation: " + str(report.get("recommendation")))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only discriminator for likely wrong exits.")
    parser.add_argument("--days", type=int, default=0, help="Latest daily final reports to load; <=0 means all.")
    parser.add_argument("--reports-dir", default=str(REPORTS))
    parser.add_argument("--continuation-margin-pct", type=float, default=DEFAULT_CONTINUATION_MARGIN_PCT)
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument("--output")
    parser.add_argument("--text-output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build(
        args.days,
        reports_dir=Path(args.reports_dir),
        continuation_margin_pct=args.continuation_margin_pct,
        min_train_days=args.min_train_days,
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output) if args.output else REPORTS / "exit_failure_discriminator_latest.json"
    text_path = Path(args.text_output) if args.text_output else REPORTS / "exit_failure_discriminator_latest.txt"
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    text = format_text(payload)
    json_path.write_text(json_text, encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    stdout_encoding = sys.stdout.encoding or "utf-8"
    print((json_text if args.json else text).encode(stdout_encoding, errors="replace").decode(stdout_encoding, errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
