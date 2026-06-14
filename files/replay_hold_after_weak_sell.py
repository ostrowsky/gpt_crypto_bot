from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
CACHE_DIR = ROOT / ".runtime" / "signal_quality_cache"
DEFAULT_OUTPUT = REPORTS / "hold_after_weak_sell_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "hold_after_weak_sell_replay_latest.txt"
HORIZONS = (2, 5, 10)


@dataclass(frozen=True)
class ReplayConfig:
    days: int = 14
    min_mfe_pct: float = 0.75
    min_giveback_pct: float = 0.5
    horizons: tuple[int, ...] = HORIZONS


def build_replay(
    *,
    reports_dir: Path = REPORTS,
    cache_dir: Path = CACHE_DIR,
    cfg: ReplayConfig = ReplayConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    cases = _load_cases(reports_dir, cfg)
    candle_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], list[int]]] = {}
    labeled = [_label_case(case, cache_dir, cfg, candle_cache=candle_cache) for case in cases]
    eligible = [row for row in labeled if row.get("eligible")]
    complete = [row for row in eligible if row.get("label_status") == "labeled"]
    policies = {"baseline": _policy_summary(complete, "pnl_pct")}
    for horizon in cfg.horizons:
        key = f"hold_{horizon}"
        policies[key] = _hold_summary(complete, horizon)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": {"days": cfg.days, "min_mfe_pct": cfg.min_mfe_pct, "min_giveback_pct": cfg.min_giveback_pct, "horizons": list(cfg.horizons)},
        "coverage": {
            "cases_total": len(cases),
            "eligible_total": len(eligible),
            "labeled_total": len(complete),
            "pending_or_missing": len(eligible) - len(complete),
            "reports_loaded": len({case.get("day") for case in cases}),
        },
        "policies": policies,
        "breakdowns": _breakdowns(complete, cfg),
        "top_improvements": _top_rows(complete, best=True),
        "top_harms": _top_rows(complete, best=False),
        "decision": _decision(complete, policies),
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
        "Hold-after-weak-sell replay (research-only)",
        f"coverage: cases={coverage.get('cases_total')} eligible={coverage.get('eligible_total')} labeled={coverage.get('labeled_total')} missing={coverage.get('pending_or_missing')}",
        f"decision: {report.get('decision')}",
        "",
        "Policies:",
    ]
    for name, metrics in (report.get("policies") or {}).items():
        lines.append(
            f"  {name}: n={metrics.get('n')} avg={metrics.get('avg_pnl_pct')} median={metrics.get('median_pnl_pct')} "
            f"win={metrics.get('win_rate_pct')}% delta_avg={metrics.get('avg_delta_pct')} "
            f"worse={metrics.get('worse_rate_pct')}% adverse={metrics.get('median_adverse_pct')}"
        )
    lines.extend(["", "Top improvements:"])
    for row in (report.get("top_improvements") or [])[:8]:
        lines.append(_row_line(row))
    lines.extend(["", "Top harms:"])
    for row in (report.get("top_harms") or [])[:8]:
        lines.append(_row_line(row))
    return "\n".join(lines) + "\n"


def _load_cases(reports_dir: Path, cfg: ReplayConfig) -> list[dict[str, Any]]:
    paths = sorted(reports_dir.glob("signal_quality_*_final.json"), key=lambda p: (_day_from_path(p) or "", p.name))
    if cfg.days > 0:
        paths = paths[-cfg.days:]
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        day = _day_from_path(path)
        if not day:
            continue
        data = _read_json(path)
        for bucket in ("trades", "early_exits", "late_exits", "false_positive_buys"):
            for row in data.get(bucket) or []:
                if not isinstance(row, dict):
                    continue
                case = _compact_case(day, bucket, row)
                key = _case_key(case)
                if key in seen:
                    continue
                seen.add(key)
                out.append(case)
    return out


def _compact_case(day: str, bucket: str, row: dict[str, Any]) -> dict[str, Any]:
    reason = _reason_bucket(row.get("exit_reason"))
    tags = _tags(row, reason)
    return {
        "day": day,
        "bucket": bucket,
        "sym": row.get("sym"),
        "tf": row.get("tf") or "15m",
        "source": row.get("source"),
        "mode": row.get("mode"),
        "entry_ts": row.get("entry_ts"),
        "exit_ts": row.get("exit_ts"),
        "entry_price": _num(row.get("entry_price")),
        "exit_price": _num(row.get("exit_price")),
        "pnl_pct": _num(row.get("pnl_pct")),
        "max_favorable_pct": _num(row.get("max_favorable_pct")),
        "future_favorable_pct": _num(row.get("future_favorable_pct")),
        "exit_efficiency": _num(row.get("exit_efficiency")),
        "giveback_pct": _num(row.get("giveback_pct")),
        "exit_timing": row.get("exit_timing"),
        "exit_reason_bucket": reason,
        "tags": tags,
    }


def _label_case(
    case: dict[str, Any],
    cache_dir: Path,
    cfg: ReplayConfig,
    candle_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], list[int]]] | None = None,
) -> dict[str, Any]:
    eligible, reason = _eligible(case, cfg)
    row = {**case, "eligible": eligible, "eligibility_reason": reason, "label_status": "not_eligible" if not eligible else "missing"}
    if not eligible:
        return row
    entry_price = _num(case.get("entry_price"))
    if not entry_price or entry_price <= 0:
        row["label_reason"] = "missing entry_price"
        return row
    exit_ts_ms = _parse_ts_ms(case.get("exit_ts"))
    if exit_ts_ms is None:
        row["label_reason"] = "missing exit_ts"
        return row
    candles, candle_ts = _load_cached_candle_series(
        cache_dir,
        str(case.get("sym") or ""),
        str(case.get("tf") or "15m"),
        candle_cache,
    )
    if not candles:
        row["label_reason"] = "missing cached candles"
        return row
    idx = _first_idx_at_or_after_ts(candle_ts, exit_ts_ms)
    if idx is None:
        row["label_reason"] = "exit candle not found"
        return row
    max_h = max(cfg.horizons)
    if idx + max_h >= len(candles):
        row["label_status"] = "pending"
        row["label_reason"] = "not enough future candles"
        return row
    baseline = _num(case.get("pnl_pct")) or 0.0
    future_lows = [float(c["l"]) for c in candles[idx + 1: idx + max_h + 1] if _num(c.get("l")) is not None]
    row["label_status"] = "labeled"
    row["post_exit_adverse_pct"] = _pnl_pct(min(future_lows), entry_price) - baseline if future_lows else None
    for h in cfg.horizons:
        close = _num(candles[idx + h].get("c"))
        low_slice = [_num(c.get("l")) for c in candles[idx + 1: idx + h + 1]]
        low_vals = [x for x in low_slice if x is not None]
        hold_pnl = _pnl_pct(close, entry_price) if close is not None else None
        row[f"hold_{h}_pnl_pct"] = hold_pnl
        row[f"hold_{h}_delta_pct"] = None if hold_pnl is None else round(hold_pnl - baseline, 4)
        row[f"hold_{h}_adverse_pct"] = None if not low_vals else round(_pnl_pct(min(low_vals), entry_price) - baseline, 4)
    return row


def _eligible(case: dict[str, Any], cfg: ReplayConfig) -> tuple[bool, str]:
    pnl = _num(case.get("pnl_pct"))
    mfe = _num(case.get("max_favorable_pct")) or 0.0
    giveback = _num(case.get("giveback_pct")) or 0.0
    reason = str(case.get("exit_reason_bucket") or "")
    tags = set(case.get("tags") or [])
    if pnl is None:
        return False, "missing pnl"
    if mfe < cfg.min_mfe_pct and giveback < cfg.min_giveback_pct:
        return False, "insufficient mfe/giveback"
    if reason in {"weak_signal", "atr_trail", "ema_break", "stop_loss_or_trail"}:
        return True, "weak_or_trailing_exit"
    if tags & {"early_exit", "high_giveback", "post_exit_continuation", "negative_after_mfe"}:
        return True, "exit_quality_tag"
    return False, "reason_not_targeted"


def _policy_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [_num(row.get(key)) for row in rows]
    nums = [x for x in vals if x is not None]
    return {"n": len(nums), "avg_pnl_pct": _avg(nums), "median_pnl_pct": _median(nums), "win_rate_pct": _win_rate(nums)}


def _hold_summary(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    pnl_key = f"hold_{horizon}_pnl_pct"
    delta_key = f"hold_{horizon}_delta_pct"
    adverse_key = f"hold_{horizon}_adverse_pct"
    pnls = [_num(row.get(pnl_key)) for row in rows]
    deltas = [_num(row.get(delta_key)) for row in rows]
    adverse = [_num(row.get(adverse_key)) for row in rows]
    pnl_nums = [x for x in pnls if x is not None]
    delta_nums = [x for x in deltas if x is not None]
    worse = [x for x in delta_nums if x < 0]
    return {
        "n": len(pnl_nums),
        "avg_pnl_pct": _avg(pnl_nums),
        "median_pnl_pct": _median(pnl_nums),
        "win_rate_pct": _win_rate(pnl_nums),
        "avg_delta_pct": _avg(delta_nums),
        "median_delta_pct": _median(delta_nums),
        "total_delta_pct": round(sum(delta_nums), 4) if delta_nums else None,
        "worse_rate_pct": round(len(worse) / len(delta_nums) * 100.0, 2) if delta_nums else 0.0,
        "median_adverse_pct": _median(adverse),
    }


def _breakdowns(rows: list[dict[str, Any]], cfg: ReplayConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("exit_reason_bucket", "mode", "bucket"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field) or "unknown")].append(row)
        out[field] = {
            name: {f"hold_{h}": _hold_summary(items, h) for h in cfg.horizons}
            for name, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]
        }
    return out


def _top_rows(rows: list[dict[str, Any]], *, best: bool) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        deltas = [_num(row.get(f"hold_{h}_delta_pct")) for h in HORIZONS]
        nums = [x for x in deltas if x is not None]
        if not nums:
            continue
        score = max(nums) if best else min(nums)
        best_h = HORIZONS[nums.index(score)] if score in nums else None
        scored.append({**row, "best_hold_delta_pct": score, "best_hold_horizon": best_h})
    scored.sort(key=lambda r: r.get("best_hold_delta_pct") or 0.0, reverse=best)
    keep = []
    for row in scored[:30]:
        keep.append({k: row.get(k) for k in ("day", "sym", "tf", "mode", "bucket", "exit_reason_bucket", "pnl_pct", "max_favorable_pct", "giveback_pct", "best_hold_horizon", "best_hold_delta_pct", "hold_2_delta_pct", "hold_5_delta_pct", "hold_10_delta_pct")})
    return keep


def _decision(rows: list[dict[str, Any]], policies: dict[str, Any]) -> str:
    if len(rows) < 20:
        return "insufficient_labeled_cases_keep_collecting"
    candidates = []
    for h in HORIZONS:
        m = policies.get(f"hold_{h}") or {}
        avg_delta = _num(m.get("avg_delta_pct")) or 0.0
        med_delta = _num(m.get("median_delta_pct")) or 0.0
        worse = _num(m.get("worse_rate_pct")) or 100.0
        adverse = _num(m.get("median_adverse_pct")) or 0.0
        if avg_delta > 0.15 and med_delta >= 0.0 and worse <= 45.0 and adverse >= -1.5:
            candidates.append((h, avg_delta, med_delta, worse, adverse))
    if candidates:
        best = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)[0]
        return f"advance_hold_{best[0]}_to_partial_exit_replay_before_production"
    return "reject_or_refine_hold_policy_no_safe_edge"


def _load_cached_candles(cache_dir: Path, sym: str, tf: str) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in cache_dir.glob(f"{sym}_{tf}_*.json"):
        data = _read_json(path)
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict) or row.get("t") is None:
                continue
            rows[int(row["t"])] = row
    return [rows[t] for t in sorted(rows)]


def _load_cached_candle_series(
    cache_dir: Path,
    sym: str,
    tf: str,
    candle_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], list[int]]] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    key = (sym, tf)
    if candle_cache is not None and key in candle_cache:
        return candle_cache[key]
    candles = _load_cached_candles(cache_dir, sym, tf)
    timestamps = [int(row.get("t") or 0) for row in candles]
    series = (candles, timestamps)
    if candle_cache is not None:
        candle_cache[key] = series
    return series


def _first_idx_at_or_after(candles: list[dict[str, Any]], ts_ms: int) -> int | None:
    for i, row in enumerate(candles):
        if int(row.get("t") or 0) >= ts_ms:
            return i
    return None


def _first_idx_at_or_after_ts(timestamps: list[int], ts_ms: int) -> int | None:
    idx = bisect.bisect_left(timestamps, ts_ms)
    return idx if idx < len(timestamps) else None


def _tags(row: dict[str, Any], reason: str) -> list[str]:
    tags = []
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


def _reason_bucket(reason: Any) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"
    if "weak:" in text or "diverg" in text or "диверг" in text:
        return "weak_signal"
    if "atr" in text or "trail" in text or "трейл" in text:
        return "atr_trail"
    if "ema20" in text or "ema" in text:
        return "ema_break"
    if "stop" in text or "стоп" in text:
        return "stop_loss_or_trail"
    if "time" in text or "время" in text:
        return "time_exit"
    return "other"


def _case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (case.get("day"), case.get("sym"), case.get("tf"), case.get("source"), case.get("entry_ts"), case.get("exit_ts"), case.get("pnl_pct"))


def _day_from_path(path: Path) -> str | None:
    m = re.search(r"(20\d\d-\d\d-\d\d)", path.name)
    return m.group(1) if m else None


def _parse_ts_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    try:
        return int(float(text))
    except Exception:
        pass
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _pnl_pct(price: float | None, entry: float | None) -> float | None:
    if price is None or entry is None or entry <= 0:
        return None
    return round((float(price) / float(entry) - 1.0) * 100.0, 4)


def _avg(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(median(vals), 4) if vals else None


def _win_rate(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return round(sum(1 for v in vals if v > 0) / len(vals) * 100.0, 2) if vals else 0.0


def _row_line(row: dict[str, Any]) -> str:
    return (
        f"  {row.get('day')} {row.get('sym')} {row.get('tf')} {row.get('mode')} "
        f"pnl={row.get('pnl_pct')} mfe={row.get('max_favorable_pct')} giveback={row.get('giveback_pct')} "
        f"best=+{row.get('best_hold_horizon')}b delta={row.get('best_hold_delta_pct')} reason={row.get('exit_reason_bucket')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only hold-after-weak-sell replay")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--min-mfe-pct", type=float, default=0.75)
    parser.add_argument("--min-giveback-pct", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    payload = build_replay(
        reports_dir=args.reports_dir,
        cache_dir=args.cache_dir,
        cfg=ReplayConfig(days=args.days, min_mfe_pct=args.min_mfe_pct, min_giveback_pct=args.min_giveback_pct),
        output=args.output,
        text_output=args.text_output,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
