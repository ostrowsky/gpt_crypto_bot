from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import replay_hold_after_weak_sell as hold_replay


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
CACHE_DIR = ROOT / ".runtime" / "signal_quality_cache"
DEFAULT_OUTPUT = REPORTS / "trailing_tail_after_partial_exit_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "trailing_tail_after_partial_exit_replay_latest.txt"
HORIZONS = (10,)


@dataclass(frozen=True)
class TailPolicy:
    name: str
    sell_fraction: float
    max_horizon: int
    adverse_cap_pct: float
    ema_period: int = 20
    min_hold_bars: int = 1


@dataclass(frozen=True)
class TrailingTailConfig:
    days: int = 14
    min_mfe_pct: float = 0.75
    min_giveback_pct: float = 0.5
    policies: tuple[TailPolicy, ...] = (
        TailPolicy("tail50_h10_ema20_cap100", 0.50, 10, 1.00),
        TailPolicy("tail50_h10_ema20_cap150", 0.50, 10, 1.50),
        TailPolicy("tail70_h10_ema20_cap100", 0.70, 10, 1.00),
    )


def build_replay(
    *,
    reports_dir: Path = REPORTS,
    cache_dir: Path = CACHE_DIR,
    cfg: TrailingTailConfig = TrailingTailConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    rows = _labeled_rows(reports_dir, cache_dir, cfg)
    complete = [row for row in rows if row.get("label_status") == "labeled"]
    policies = {"baseline": _policy_summary(complete, "pnl_pct")}
    for policy in cfg.policies:
        _apply_tail_policy(complete, policy)
        policies[policy.name] = _tail_summary(complete, policy.name)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": {
            "days": cfg.days,
            "min_mfe_pct": cfg.min_mfe_pct,
            "min_giveback_pct": cfg.min_giveback_pct,
            "policies": [policy.__dict__ for policy in cfg.policies],
        },
        "coverage": _coverage(rows),
        "policies": policies,
        "breakdowns": _breakdowns(complete, cfg),
        "exit_reasons": _exit_reason_counts(complete, cfg),
        "top_improvements": _top_rows(complete, best=True),
        "top_harms": _top_rows(complete, best=False),
        "decision": _decision(complete, policies, cfg),
    }
    text = render_text(payload)
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output), "txt": str(text_output)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    c = report.get("coverage") or {}
    lines = [
        "Trailing-tail-after-partial-exit replay (research-only)",
        f"coverage: eligible={c.get('eligible_total')} labeled={c.get('labeled_total')} missing={c.get('pending_or_missing')}",
        f"decision: {report.get('decision')}",
        "",
        "Policies:",
    ]
    for name, metrics in (report.get("policies") or {}).items():
        lines.append(
            f"  {name}: n={metrics.get('n')} avg={metrics.get('avg_pnl_pct')} median={metrics.get('median_pnl_pct')} "
            f"win={metrics.get('win_rate_pct')}% delta_avg={metrics.get('avg_delta_pct')} "
            f"worse={metrics.get('worse_rate_pct')}% adverse={metrics.get('median_tail_adverse_pct')} "
            f"tail_bars={metrics.get('median_tail_bars')}"
        )
    lines.extend(["", "Top improvements:"])
    for row in (report.get("top_improvements") or [])[:8]:
        lines.append(_row_line(row))
    lines.extend(["", "Top harms:"])
    for row in (report.get("top_harms") or [])[:8]:
        lines.append(_row_line(row))
    return "\n".join(lines) + "\n"


def _labeled_rows(reports_dir: Path, cache_dir: Path, cfg: TrailingTailConfig) -> list[dict[str, Any]]:
    max_h = max(policy.max_horizon for policy in cfg.policies)
    hold_cfg = hold_replay.ReplayConfig(days=cfg.days, min_mfe_pct=cfg.min_mfe_pct, min_giveback_pct=cfg.min_giveback_pct, horizons=(max_h,))
    cases = hold_replay._load_cases(reports_dir, hold_cfg)
    rows = [hold_replay._label_case(case, cache_dir, hold_cfg) for case in cases]
    for row in rows:
        if not row.get("eligible") or row.get("label_status") != "labeled":
            continue
        _attach_candle_path(row, cache_dir, max_h)
    return [row for row in rows if row.get("eligible")]


def _attach_candle_path(row: dict[str, Any], cache_dir: Path, max_horizon: int) -> None:
    sym = str(row.get("sym") or "")
    tf = str(row.get("tf") or "15m")
    exit_ts_ms = hold_replay._parse_ts_ms(row.get("exit_ts"))
    candles = hold_replay._load_cached_candles(cache_dir, sym, tf)
    if exit_ts_ms is None or not candles:
        row["tail_path_status"] = "missing_path"
        return
    idx = hold_replay._first_idx_at_or_after(candles, exit_ts_ms)
    if idx is None or idx + max_horizon >= len(candles):
        row["tail_path_status"] = "insufficient_future"
        return
    start = max(0, idx - 60)
    path = candles[start: idx + max_horizon + 1]
    closes = [hold_replay._num(c.get("c")) for c in path]
    emas = _ema(closes, 20)
    offset = idx - start
    row["tail_path_status"] = "ready"
    row["tail_path"] = []
    for local_i in range(offset, offset + max_horizon + 1):
        c = path[local_i]
        row["tail_path"].append({
            "bar": local_i - offset,
            "t": c.get("t"),
            "c": hold_replay._num(c.get("c")),
            "l": hold_replay._num(c.get("l")),
            "ema20": emas[local_i],
        })


def _apply_tail_policy(rows: list[dict[str, Any]], policy: TailPolicy) -> None:
    tail_fraction = 1.0 - policy.sell_fraction
    for row in rows:
        baseline = _num(row.get("pnl_pct"))
        entry = _num(row.get("entry_price"))
        path = row.get("tail_path") or []
        if baseline is None or entry is None or entry <= 0 or not path:
            continue
        exit_bar = policy.max_horizon
        exit_reason = "max_horizon"
        tail_exit_pnl = None
        min_tail_adverse = 0.0
        for step in path[1: policy.max_horizon + 1]:
            bar = int(step.get("bar") or 0)
            close = _num(step.get("c"))
            low = _num(step.get("l"))
            ema20 = _num(step.get("ema20"))
            if low is not None:
                adverse = hold_replay._pnl_pct(low, entry) - baseline
                min_tail_adverse = min(min_tail_adverse, adverse)
                if bar >= policy.min_hold_bars and adverse <= -policy.adverse_cap_pct:
                    exit_bar = bar
                    exit_reason = "adverse_cap"
                    tail_exit_pnl = hold_replay._pnl_pct(close, entry) if close is not None else None
                    break
            if close is not None and ema20 is not None and bar >= policy.min_hold_bars and close < ema20:
                exit_bar = bar
                exit_reason = "ema20_loss"
                tail_exit_pnl = hold_replay._pnl_pct(close, entry)
                break
            if bar == policy.max_horizon and close is not None:
                tail_exit_pnl = hold_replay._pnl_pct(close, entry)
        if tail_exit_pnl is None:
            continue
        pnl = policy.sell_fraction * baseline + tail_fraction * tail_exit_pnl
        row[f"{policy.name}_pnl_pct"] = round(pnl, 4)
        row[f"{policy.name}_delta_pct"] = round(pnl - baseline, 4)
        row[f"{policy.name}_tail_exit_bar"] = exit_bar
        row[f"{policy.name}_tail_exit_reason"] = exit_reason
        row[f"{policy.name}_tail_adverse_pct"] = round(tail_fraction * min_tail_adverse, 4)


def _ema(values: list[float | None], period: int) -> list[float | None]:
    out: list[float | None] = []
    alpha = 2.0 / (period + 1.0)
    current: float | None = None
    for value in values:
        if value is None:
            out.append(current)
            continue
        current = value if current is None else alpha * value + (1.0 - alpha) * current
        out.append(round(current, 10))
    return out


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("label_status") == "labeled"]
    return {
        "eligible_total": len(rows),
        "labeled_total": len(complete),
        "pending_or_missing": len(rows) - len(complete),
        "by_status": _counts(row.get("label_status") for row in rows),
        "by_tail_path_status": _counts(row.get("tail_path_status") for row in complete),
        "by_reason_bucket": _counts(row.get("exit_reason_bucket") for row in complete),
        "by_bucket": _counts(row.get("bucket") for row in complete),
    }


def _policy_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [_num(row.get(key)) for row in rows]
    nums = [x for x in vals if x is not None]
    return {"n": len(nums), "avg_pnl_pct": _avg(nums), "median_pnl_pct": _median(nums), "win_rate_pct": _win_rate(nums)}


def _tail_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    pnls = [_num(row.get(f"{name}_pnl_pct")) for row in rows]
    deltas = [_num(row.get(f"{name}_delta_pct")) for row in rows]
    adverse = [_num(row.get(f"{name}_tail_adverse_pct")) for row in rows]
    bars = [_num(row.get(f"{name}_tail_exit_bar")) for row in rows]
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
        "median_tail_adverse_pct": _median(adverse),
        "median_tail_bars": _median(bars),
    }


def _breakdowns(rows: list[dict[str, Any]], cfg: TrailingTailConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("exit_reason_bucket", "mode", "bucket"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
        out[field] = {}
        for group, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]:
            out[field][group] = {policy.name: _tail_summary(items, policy.name) for policy in cfg.policies}
    return out


def _exit_reason_counts(rows: list[dict[str, Any]], cfg: TrailingTailConfig) -> dict[str, dict[str, int]]:
    return {policy.name: _counts(row.get(f"{policy.name}_tail_exit_reason") for row in rows if row.get(f"{policy.name}_pnl_pct") is not None) for policy in cfg.policies}


def _decision(rows: list[dict[str, Any]], policies: dict[str, Any], cfg: TrailingTailConfig) -> str:
    if len(rows) < 20:
        return "insufficient_labeled_cases_keep_collecting"
    viable = []
    for name, metrics in policies.items():
        if name == "baseline":
            continue
        avg_delta = _num(metrics.get("avg_delta_pct")) or 0.0
        med_delta = _num(metrics.get("median_delta_pct")) or 0.0
        worse = _num(metrics.get("worse_rate_pct")) or 100.0
        adverse = _num(metrics.get("median_tail_adverse_pct")) or 0.0
        if avg_delta > 0.10 and med_delta >= 0.0 and worse <= 35.0 and adverse >= -0.75:
            viable.append((name, avg_delta, med_delta, worse, adverse))
    if viable:
        best = sorted(viable, key=lambda x: (x[1], x[2], -x[3]), reverse=True)[0]
        return f"advance_{best[0]}_to_shadow_tail_selector_before_production"

    early_rows = [row for row in rows if row.get("bucket") == "early_exits"]
    false_rows = [row for row in rows if row.get("bucket") == "false_positive_buys"]
    gated = []
    for policy in cfg.policies:
        early = _tail_summary(early_rows, policy.name)
        false_pos = _tail_summary(false_rows, policy.name)
        early_avg = _num(early.get("avg_delta_pct")) or 0.0
        early_med = _num(early.get("median_delta_pct")) or 0.0
        early_worse = _num(early.get("worse_rate_pct")) or 100.0
        false_avg = _num(false_pos.get("avg_delta_pct")) or 0.0
        false_med = _num(false_pos.get("median_delta_pct")) or 0.0
        if early_avg > 0.30 and early_med > 0.0 and early_worse <= 38.0 and false_avg >= -0.15 and false_med >= -0.05:
            gated.append((policy.name, early_avg, early_med, early_worse, false_avg))
    if gated:
        best = sorted(gated, key=lambda x: (x[1], x[2], -x[3]), reverse=True)[0]
        return f"advance_{best[0]}_as_early_exit_gated_tail_selector_before_production"
    return "reject_or_refine_trailing_tail_policy_no_safe_edge"


def _top_rows(rows: list[dict[str, Any]], *, best: bool) -> list[dict[str, Any]]:
    names = sorted({key[:-10] for row in rows for key in row if key.endswith("_delta_pct") and key.startswith("tail")})
    scored = []
    for row in rows:
        candidates = [(name, _num(row.get(f"{name}_delta_pct"))) for name in names]
        candidates = [(name, val) for name, val in candidates if val is not None]
        if not candidates:
            continue
        name, score = (max if best else min)(candidates, key=lambda kv: kv[1])
        scored.append({**row, "best_policy": name, "best_tail_delta_pct": score})
    scored.sort(key=lambda row: row.get("best_tail_delta_pct") or 0.0, reverse=best)
    return [
        {k: row.get(k) for k in ("day", "sym", "tf", "mode", "bucket", "exit_reason_bucket", "pnl_pct", "max_favorable_pct", "giveback_pct", "best_policy", "best_tail_delta_pct")}
        for row in scored[:30]
    ]


def _row_line(row: dict[str, Any]) -> str:
    return (
        f"  {row.get('day')} {row.get('sym')} {row.get('tf')} {row.get('mode')} "
        f"pnl={row.get('pnl_pct')} mfe={row.get('max_favorable_pct')} giveback={row.get('giveback_pct')} "
        f"best={row.get('best_policy')} delta={row.get('best_tail_delta_pct')} reason={row.get('exit_reason_bucket')}"
    )


def _counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _num(value: Any) -> float | None:
    return hold_replay._num(value)


def _avg(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(median(vals), 4) if vals else None


def _win_rate(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return round(sum(1 for v in vals if v > 0) / len(vals) * 100.0, 2) if vals else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only trailing-tail-after-partial-exit replay")
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
        cfg=TrailingTailConfig(days=args.days, min_mfe_pct=args.min_mfe_pct, min_giveback_pct=args.min_giveback_pct),
        output=args.output,
        text_output=args.text_output,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
