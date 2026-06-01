from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

import replay_trailing_tail_after_partial_exit as tail_replay


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
CACHE_DIR = ROOT / ".runtime" / "signal_quality_cache"
DEFAULT_OUTPUT = REPORTS / "early_exit_gated_tail_selector_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "early_exit_gated_tail_selector_replay_latest.txt"
TAIL_POLICY = tail_replay.TailPolicy("tail50_h10_ema20_cap150", 0.50, 10, 1.50)


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    description: str


@dataclass(frozen=True)
class GatedTailConfig:
    days: int = 14
    min_mfe_pct: float = 0.75
    min_giveback_pct: float = 0.5
    tail_policy: tail_replay.TailPolicy = TAIL_POLICY
    selectors: tuple[SelectorSpec, ...] = (
        SelectorSpec("gate_oracle_early_exit", "allow only evaluator early_exits bucket; upper-bound diagnostic"),
        SelectorSpec("gate_early_weak_signal", "allow early_exits with weak/divergence exit reason"),
        SelectorSpec("gate_early_non_ema_break", "allow early_exits except EMA-break cleanup"),
        SelectorSpec("gate_weak_signal_only", "allow weak/divergence reason without hindsight bucket; caution baseline"),
    )


def build_replay(
    *,
    reports_dir: Path = REPORTS,
    cache_dir: Path = CACHE_DIR,
    cfg: GatedTailConfig = GatedTailConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    rows = _labeled_rows(reports_dir, cache_dir, cfg)
    policies = {"baseline": _summary(rows, "pnl_pct")}
    for selector in cfg.selectors:
        _apply_selector(rows, selector.name, cfg.tail_policy.name, _selector_fn(selector.name))
        policies[selector.name] = _gated_summary(rows, selector.name)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": {
            "days": cfg.days,
            "min_mfe_pct": cfg.min_mfe_pct,
            "min_giveback_pct": cfg.min_giveback_pct,
            "tail_policy": cfg.tail_policy.__dict__,
            "selectors": [s.__dict__ for s in cfg.selectors],
        },
        "coverage": _coverage(rows),
        "policies": policies,
        "breakdowns": _breakdowns(rows, cfg),
        "top_improvements": _top_rows(rows, best=True),
        "top_harms": _top_rows(rows, best=False),
        "decision": _decision(rows, policies),
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
        "Early-exit gated tail selector replay (research-only)",
        f"coverage: labeled={c.get('labeled_total')} missing={c.get('pending_or_missing')}",
        f"decision: {report.get('decision')}",
        "",
        "Policies:",
    ]
    for name, metrics in (report.get("policies") or {}).items():
        lines.append(
            f"  {name}: n={metrics.get('n')} avg={metrics.get('avg_pnl_pct')} median={metrics.get('median_pnl_pct')} "
            f"win={metrics.get('win_rate_pct')}% delta_avg={metrics.get('avg_delta_pct')} "
            f"worse={metrics.get('worse_rate_pct')}% allow={metrics.get('allowed_rate_pct')}% "
            f"fp_allow={metrics.get('false_positive_allowed_rate_pct')}%"
        )
    lines.extend(["", "Top improvements:"])
    for row in (report.get("top_improvements") or [])[:8]:
        lines.append(_row_line(row))
    lines.extend(["", "Top harms:"])
    for row in (report.get("top_harms") or [])[:8]:
        lines.append(_row_line(row))
    return "\n".join(lines) + "\n"


def _labeled_rows(reports_dir: Path, cache_dir: Path, cfg: GatedTailConfig) -> list[dict[str, Any]]:
    tail_cfg = tail_replay.TrailingTailConfig(
        days=cfg.days,
        min_mfe_pct=cfg.min_mfe_pct,
        min_giveback_pct=cfg.min_giveback_pct,
        policies=(cfg.tail_policy,),
    )
    rows = tail_replay._labeled_rows(reports_dir, cache_dir, tail_cfg)
    complete = [row for row in rows if row.get("label_status") == "labeled"]
    tail_replay._apply_tail_policy(complete, cfg.tail_policy)
    return complete


def _apply_selector(rows: list[dict[str, Any]], selector_name: str, tail_policy_name: str, fn: Callable[[dict[str, Any]], bool]) -> None:
    for row in rows:
        baseline = _num(row.get("pnl_pct"))
        tail_pnl = _num(row.get(f"{tail_policy_name}_pnl_pct"))
        allowed = bool(fn(row)) and tail_pnl is not None and baseline is not None
        selected_pnl = tail_pnl if allowed else baseline
        row[f"{selector_name}_allowed"] = allowed
        row[f"{selector_name}_pnl_pct"] = None if selected_pnl is None else round(selected_pnl, 4)
        row[f"{selector_name}_delta_pct"] = None if selected_pnl is None or baseline is None else round(selected_pnl - baseline, 4)


def _selector_fn(name: str) -> Callable[[dict[str, Any]], bool]:
    if name == "gate_oracle_early_exit":
        return lambda row: row.get("bucket") == "early_exits"
    if name == "gate_early_weak_signal":
        return lambda row: row.get("bucket") == "early_exits" and row.get("exit_reason_bucket") == "weak_signal"
    if name == "gate_early_non_ema_break":
        return lambda row: row.get("bucket") == "early_exits" and row.get("exit_reason_bucket") != "ema_break"
    if name == "gate_weak_signal_only":
        return lambda row: row.get("exit_reason_bucket") == "weak_signal"
    raise ValueError(f"unknown selector: {name}")


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "labeled_total": len(rows),
        "pending_or_missing": 0,
        "by_bucket": _counts(row.get("bucket") for row in rows),
        "by_reason_bucket": _counts(row.get("exit_reason_bucket") for row in rows),
    }


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [_num(row.get(key)) for row in rows]
    nums = [x for x in vals if x is not None]
    return {"n": len(nums), "avg_pnl_pct": _avg(nums), "median_pnl_pct": _median(nums), "win_rate_pct": _win_rate(nums)}


def _gated_summary(rows: list[dict[str, Any]], selector_name: str) -> dict[str, Any]:
    pnls = [_num(row.get(f"{selector_name}_pnl_pct")) for row in rows]
    deltas = [_num(row.get(f"{selector_name}_delta_pct")) for row in rows]
    pnl_nums = [x for x in pnls if x is not None]
    delta_nums = [x for x in deltas if x is not None]
    worse = [x for x in delta_nums if x < 0]
    allowed = [row for row in rows if row.get(f"{selector_name}_allowed")]
    false_rows = [row for row in rows if row.get("bucket") == "false_positive_buys"]
    false_allowed = [row for row in false_rows if row.get(f"{selector_name}_allowed")]
    early_rows = [row for row in rows if row.get("bucket") == "early_exits"]
    early_allowed = [row for row in early_rows if row.get(f"{selector_name}_allowed")]
    return {
        "n": len(pnl_nums),
        "avg_pnl_pct": _avg(pnl_nums),
        "median_pnl_pct": _median(pnl_nums),
        "win_rate_pct": _win_rate(pnl_nums),
        "avg_delta_pct": _avg(delta_nums),
        "median_delta_pct": _median(delta_nums),
        "total_delta_pct": round(sum(delta_nums), 4) if delta_nums else None,
        "worse_rate_pct": round(len(worse) / len(delta_nums) * 100.0, 2) if delta_nums else 0.0,
        "allowed_total": len(allowed),
        "allowed_rate_pct": round(len(allowed) / len(rows) * 100.0, 2) if rows else 0.0,
        "early_allowed_rate_pct": round(len(early_allowed) / len(early_rows) * 100.0, 2) if early_rows else 0.0,
        "false_positive_allowed_rate_pct": round(len(false_allowed) / len(false_rows) * 100.0, 2) if false_rows else 0.0,
        "false_positive_delta_avg_pct": _avg([_num(row.get(f"{selector_name}_delta_pct")) for row in false_rows]),
        "early_exit_delta_avg_pct": _avg([_num(row.get(f"{selector_name}_delta_pct")) for row in early_rows]),
    }


def _breakdowns(rows: list[dict[str, Any]], cfg: GatedTailConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("bucket", "exit_reason_bucket", "mode"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
        out[field] = {}
        for group, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]:
            out[field][group] = {selector.name: _gated_summary(items, selector.name) for selector in cfg.selectors}
    return out


def _decision(rows: list[dict[str, Any]], policies: dict[str, Any]) -> str:
    if len(rows) < 20:
        return "insufficient_labeled_cases_keep_collecting"
    candidates = []
    for name, metrics in policies.items():
        if name == "baseline":
            continue
        avg_delta = _num(metrics.get("avg_delta_pct")) or 0.0
        med_delta = _num(metrics.get("median_delta_pct")) or 0.0
        worse = _num(metrics.get("worse_rate_pct")) or 100.0
        allowed_rate = _num(metrics.get("allowed_rate_pct")) or 0.0
        fp_allowed = _num(metrics.get("false_positive_allowed_rate_pct")) or 100.0
        fp_delta = _num(metrics.get("false_positive_delta_avg_pct")) or 0.0
        early_delta = _num(metrics.get("early_exit_delta_avg_pct")) or 0.0
        if avg_delta > 0.10 and med_delta >= 0.0 and worse <= 30.0 and allowed_rate >= 10.0 and fp_allowed <= 5.0 and fp_delta >= -0.05 and early_delta > 0.25:
            candidates.append((name, avg_delta, med_delta, worse, allowed_rate))
    if candidates:
        best = sorted(candidates, key=lambda x: (x[1], x[2], -x[3]), reverse=True)[0]
        return f"advance_{best[0]}_to_observable_feature_shadow_selector"
    return "no_selector_passed_observable_shadow_gate"


def _top_rows(rows: list[dict[str, Any]], *, best: bool) -> list[dict[str, Any]]:
    names = sorted({key[:-10] for row in rows for key in row if key.endswith("_delta_pct") and key.startswith("gate_")})
    scored = []
    for row in rows:
        candidates = [(name, _num(row.get(f"{name}_delta_pct"))) for name in names]
        candidates = [(name, val) for name, val in candidates if val is not None]
        if not candidates:
            continue
        name, score = (max if best else min)(candidates, key=lambda kv: kv[1])
        scored.append({**row, "best_selector": name, "best_selector_delta_pct": score})
    scored.sort(key=lambda row: row.get("best_selector_delta_pct") or 0.0, reverse=best)
    return [
        {k: row.get(k) for k in ("day", "sym", "tf", "mode", "bucket", "exit_reason_bucket", "pnl_pct", "max_favorable_pct", "giveback_pct", "best_selector", "best_selector_delta_pct")}
        for row in scored[:30]
    ]


def _row_line(row: dict[str, Any]) -> str:
    return (
        f"  {row.get('day')} {row.get('sym')} {row.get('tf')} {row.get('mode')} "
        f"{row.get('bucket')} reason={row.get('exit_reason_bucket')} "
        f"pnl={row.get('pnl_pct')} best={row.get('best_selector')} delta={row.get('best_selector_delta_pct')}"
    )


def _counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _num(value: Any) -> float | None:
    return tail_replay._num(value)


def _avg(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _median(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(median(vals), 4) if vals else None


def _win_rate(values: Iterable[float | None]) -> float:
    vals = [float(v) for v in values if v is not None]
    return round(sum(1 for v in vals if v > 0) / len(vals) * 100.0, 2) if vals else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only early-exit gated tail selector replay")
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
        cfg=GatedTailConfig(days=args.days, min_mfe_pct=args.min_mfe_pct, min_giveback_pct=args.min_giveback_pct),
        output=args.output,
        text_output=args.text_output,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
