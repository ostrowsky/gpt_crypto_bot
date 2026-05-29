from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = REPORTS / "protected_trailing_exit_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "protected_trailing_exit_replay_latest.txt"


@dataclass(frozen=True)
class ReplayConfig:
    days: int = 0
    min_continuation_edge_pct: float = 0.75
    fractions: tuple[float, ...] = (0.25, 0.50, 0.75)


def build_replay(
    *,
    reports_dir: Path = REPORTS,
    cfg: ReplayConfig = ReplayConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    cases = _load_cases(reports_dir, cfg.days)
    eligible = [case for case in cases if _eligible(case, cfg)]
    policies = {"baseline": _summarize(cases, lambda case: _num(case.get("pnl_pct")))}
    for frac in cfg.fractions:
        name = f"protected_{int(frac * 100)}"
        policies[name] = _summarize(cases, lambda case, f=frac: _protected_pnl(case, f, cfg))
        policies[name]["eligible_count"] = len(eligible)
        policies[name]["estimated_total_uplift_pct"] = round(
            sum((_protected_pnl(case, frac, cfg) or 0.0) - (_num(case.get("pnl_pct")) or 0.0) for case in eligible),
            4,
        )
        policies[name]["estimated_median_uplift_pct"] = _median(
            (_protected_pnl(case, frac, cfg) or 0.0) - (_num(case.get("pnl_pct")) or 0.0)
            for case in eligible
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "coverage": {
            "reports_loaded": len({case.get("day") for case in cases}),
            "case_rows": len(cases),
            "eligible_rows": len(eligible),
            "case_coverage_status": "partial" if any(case.get("_source_bucket") != "trades" for case in cases) else "full",
        },
        "config": {
            "days": cfg.days,
            "min_continuation_edge_pct": cfg.min_continuation_edge_pct,
            "fractions": list(cfg.fractions),
        },
        "policies": policies,
        "eligible_reason_buckets": _counts(case.get("exit_reason_bucket") for case in eligible),
        "eligible_tag_counts": _counts(tag for case in eligible for tag in case.get("tags", [])),
        "top_opportunities": _top_opportunities(eligible, cfg),
        "decision": _decision(eligible, policies),
    }
    text = render_text(payload)
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output), "txt": str(text_output)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    lines = [
        "Protected trailing exit replay (research-only)",
        f"coverage: reports={coverage.get('reports_loaded')} cases={coverage.get('case_rows')} eligible={coverage.get('eligible_rows')} status={coverage.get('case_coverage_status')}",
        f"decision: {report.get('decision')}",
        "",
        "Policies:",
    ]
    for name, metrics in (report.get("policies") or {}).items():
        lines.append(
            f"  {name}: n={metrics.get('n')} median={metrics.get('median_pnl_pct')} "
            f"avg={metrics.get('avg_pnl_pct')} win={metrics.get('win_rate_pct')}% "
            f"uplift_total={metrics.get('estimated_total_uplift_pct')}"
        )
    lines.extend(["", "Top opportunities:"])
    for row in (report.get("top_opportunities") or [])[:12]:
        lines.append(
            f"  {row.get('day')} {row.get('sym')} {row.get('tf')} {row.get('source')}/{row.get('mode')} "
            f"pnl={row.get('pnl_pct')} future={row.get('future_favorable_pct')} "
            f"edge={row.get('continuation_edge_pct')} reason={row.get('exit_reason_bucket')}"
        )
    return "\n".join(lines) + "\n"


def _load_cases(reports_dir: Path, days: int) -> list[dict[str, Any]]:
    paths = sorted(reports_dir.glob("signal_quality_*_final.json"), key=lambda p: (_day_from_path(p) or "", p.name))
    if days > 0:
        paths = paths[-days:]
    cases: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        day = _day_from_path(path)
        if not day:
            continue
        data = _read_json(path)
        for bucket in ("trades", "early_exits", "false_positive_buys", "late_entries"):
            for row in data.get(bucket) or []:
                if not isinstance(row, dict):
                    continue
                case = _compact_case(day, bucket, row)
                key = _case_key(case)
                if key in seen:
                    continue
                seen.add(key)
                cases.append(case)
    return cases


def _compact_case(day: str, bucket: str, row: dict[str, Any]) -> dict[str, Any]:
    reason = _normalize_reason(row.get("exit_reason"))
    pnl = _num(row.get("pnl_pct"))
    future = _num(row.get("future_favorable_pct"))
    return {
        "day": day,
        "_source_bucket": bucket,
        "sym": row.get("sym"),
        "tf": row.get("tf"),
        "source": row.get("source"),
        "mode": row.get("mode"),
        "entry_ts": row.get("entry_ts"),
        "exit_ts": row.get("exit_ts"),
        "exit_timing": row.get("exit_timing"),
        "entry_timing": row.get("entry_timing"),
        "pnl_pct": pnl,
        "max_favorable_pct": _num(row.get("max_favorable_pct")),
        "future_favorable_pct": future,
        "exit_efficiency": _num(row.get("exit_efficiency")),
        "giveback_pct": _num(row.get("giveback_pct")),
        "exit_reason_bucket": reason,
        "tags": _tags(row, reason),
        "continuation_edge_pct": None if pnl is None or future is None else round(future - pnl, 4),
    }


def _eligible(case: dict[str, Any], cfg: ReplayConfig) -> bool:
    pnl = _num(case.get("pnl_pct"))
    future = _num(case.get("future_favorable_pct"))
    if pnl is None or future is None:
        return False
    if future - pnl < cfg.min_continuation_edge_pct:
        return False
    reason = str(case.get("exit_reason_bucket") or "")
    tags = set(case.get("tags") or [])
    return (
        reason in {"ema_break", "stop_loss_or_trail", "atr_trail", "weak_signal", "unknown"}
        or bool(tags & {"early_exit", "late_exit", "post_exit_continuation", "high_giveback"})
    )


def _protected_pnl(case: dict[str, Any], fraction: float, cfg: ReplayConfig) -> float | None:
    pnl = _num(case.get("pnl_pct"))
    future = _num(case.get("future_favorable_pct"))
    if pnl is None:
        return None
    if not _eligible(case, cfg) or future is None:
        return pnl
    return round(pnl + max(0.0, future - pnl) * fraction, 4)


def _summarize(cases: list[dict[str, Any]], value_fn) -> dict[str, Any]:
    values = [value_fn(case) for case in cases]
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    wins = [v for v in nums if v > 0]
    return {
        "n": len(nums),
        "median_pnl_pct": _median(nums),
        "avg_pnl_pct": _avg(nums),
        "total_pnl_pct": round(sum(nums), 4) if nums else None,
        "win_rate_pct": round(len(wins) / len(nums) * 100.0, 2) if nums else 0.0,
    }


def _top_opportunities(cases: list[dict[str, Any]], cfg: ReplayConfig) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        best = _protected_pnl(case, 0.50, cfg)
        pnl = _num(case.get("pnl_pct"))
        if best is None or pnl is None:
            continue
        rows.append({**case, "protected_50_pnl_pct": best, "protected_50_uplift_pct": round(best - pnl, 4)})
    rows.sort(key=lambda row: row.get("protected_50_uplift_pct") or 0.0, reverse=True)
    return rows[:30]


def _decision(eligible: list[dict[str, Any]], policies: dict[str, Any]) -> str:
    p50 = policies.get("protected_50") or {}
    baseline = policies.get("baseline") or {}
    if len(eligible) < 20:
        return "insufficient_independent_cases_keep_research_only"
    uplift = (p50.get("total_pnl_pct") or 0.0) - (baseline.get("total_pnl_pct") or 0.0)
    if uplift > 0 and (p50.get("win_rate_pct") or 0.0) >= (baseline.get("win_rate_pct") or 0.0):
        return "advance_to_candle_path_replay_required_before_sell_change"
    return "reject_or_refine_no_estimated_edge"


def _tags(row: dict[str, Any], reason: str) -> list[str]:
    tags: list[str] = []
    exit_timing = str(row.get("exit_timing") or "").lower()
    pnl = _num(row.get("pnl_pct"))
    mfe = _num(row.get("max_favorable_pct")) or 0.0
    future = _num(row.get("future_favorable_pct"))
    giveback = _num(row.get("giveback_pct")) or 0.0
    if exit_timing == "early":
        tags.append("early_exit")
    if exit_timing == "late":
        tags.append("late_exit")
    if pnl is not None and pnl <= 0 and mfe > 0:
        tags.append("negative_after_mfe")
    if giveback >= 1.0:
        tags.append("high_giveback")
    if future is not None and future > max(mfe, pnl or 0.0):
        tags.append("post_exit_continuation")
    if reason == "weak_signal":
        tags.append("weak_signal")
    return tags


def _normalize_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    lowered = text.lower()
    if not text:
        return "unknown"
    if "weak:" in lowered or "exhaustion" in lowered or "диверген" in lowered:
        return "weak_signal"
    if "atr" in lowered and "trail" in lowered:
        return "atr_trail"
    if "ema20" in lowered or "ema" in lowered:
        return "ema_break"
    if "stop" in lowered or "стоп" in lowered:
        return "stop_loss_or_trail"
    return text[:80]


def _case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    if case.get("exit_ts"):
        return (case.get("day"), case.get("sym"), case.get("tf"), case.get("source"), case.get("entry_ts"), case.get("exit_ts"))
    return (
        case.get("day"),
        case.get("sym"),
        case.get("tf"),
        case.get("source"),
        case.get("mode"),
        case.get("pnl_pct"),
        case.get("max_favorable_pct"),
        case.get("future_favorable_pct"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _day_from_path(path: Path) -> str | None:
    match = re.search(r"(20\d\d-\d\d-\d\d)", path.name)
    return match.group(1) if match else None


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _median(values: Iterable[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return round(float(median(nums)), 4) if nums else None


def _avg(values: Iterable[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def _counts(values: Iterable[Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"key": key, "count": count} for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only protected trailing exit replay from signal-quality cases.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--min-continuation-edge-pct", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_replay(
        reports_dir=args.reports_dir,
        cfg=ReplayConfig(days=args.days, min_continuation_edge_pct=args.min_continuation_edge_pct),
        output=args.output,
        text_output=args.text_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

