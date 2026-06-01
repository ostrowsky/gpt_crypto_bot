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
DEFAULT_OUTPUT = REPORTS / "observable_tail_selector_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "observable_tail_selector_replay_latest.txt"
TAIL_POLICY = tail_replay.TailPolicy("tail50_h10_ema20_cap150", 0.50, 10, 1.50)


@dataclass(frozen=True)
class ObservableSelectorConfig:
    days: int = 14
    min_mfe_pct: float = 0.75
    min_giveback_pct: float = 0.5
    train_fraction: float = 0.7
    tail_policy: tail_replay.TailPolicy = TAIL_POLICY


def build_replay(*, reports_dir: Path = REPORTS, cache_dir: Path = CACHE_DIR, cfg: ObservableSelectorConfig = ObservableSelectorConfig(), output: Path = DEFAULT_OUTPUT, text_output: Path = DEFAULT_TEXT_OUTPUT, save: bool = True) -> dict[str, Any]:
    rows = _rows(reports_dir, cache_dir, cfg)
    train, test = _split_rows(rows, cfg.train_fraction)
    candidates = _candidate_selectors()
    results = []
    for name, desc, fn in candidates:
        results.append({
            "name": name,
            "description": desc,
            "train": _score(train, name, fn, cfg.tail_policy.name),
            "test": _score(test, name, fn, cfg.tail_policy.name),
            "all": _score(rows, name, fn, cfg.tail_policy.name),
        })
    ranked = sorted(results, key=lambda r: _rank_key(r), reverse=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": {"days": cfg.days, "train_fraction": cfg.train_fraction, "tail_policy": cfg.tail_policy.__dict__},
        "coverage": {"rows": len(rows), "train_rows": len(train), "test_rows": len(test), "days": sorted({str(r.get('day')) for r in rows})},
        "baseline": _baseline(rows),
        "ranked_selectors": ranked,
        "decision": _decision(ranked),
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
        "Observable tail selector replay (research-only)",
        f"coverage: rows={c.get('rows')} train={c.get('train_rows')} test={c.get('test_rows')}",
        f"decision: {report.get('decision')}",
        "",
        "Top selectors:",
    ]
    for item in (report.get("ranked_selectors") or [])[:8]:
        t = item.get("test") or {}
        a = item.get("all") or {}
        lines.append(
            f"  {item.get('name')}: test n={t.get('n')} allow={t.get('allowed_rate_pct')}% "
            f"avg={t.get('avg_delta_pct')} med={t.get('median_delta_pct')} worse={t.get('worse_rate_pct')}% "
            f"fp_allow={t.get('false_positive_allowed_rate_pct')}% | all avg={a.get('avg_delta_pct')} med={a.get('median_delta_pct')}"
        )
    return "\n".join(lines) + "\n"


def _rows(reports_dir: Path, cache_dir: Path, cfg: ObservableSelectorConfig) -> list[dict[str, Any]]:
    tail_cfg = tail_replay.TrailingTailConfig(days=cfg.days, min_mfe_pct=cfg.min_mfe_pct, min_giveback_pct=cfg.min_giveback_pct, policies=(cfg.tail_policy,))
    rows = tail_replay._labeled_rows(reports_dir, cache_dir, tail_cfg)
    complete = [row for row in rows if row.get("label_status") == "labeled"]
    tail_replay._apply_tail_policy(complete, cfg.tail_policy)
    return [row for row in complete if _num(row.get(f"{cfg.tail_policy.name}_pnl_pct")) is not None]


def _split_rows(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda r: (str(r.get("day") or ""), str(r.get("exit_ts") or ""), str(r.get("sym") or "")))
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction)))) if len(ordered) > 1 else len(ordered)
    return ordered[:cut], ordered[cut:]


def _candidate_selectors() -> list[tuple[str, str, Callable[[dict[str, Any]], bool]]]:
    modes_momentum = {"impulse_speed", "strong_trend", "trend", "retest"}
    return [
        ("weak_signal", "weak/divergence exits", lambda r: r.get("exit_reason_bucket") == "weak_signal"),
        ("weak_signal_mfe150", "weak signal with MFE >= 1.5", lambda r: r.get("exit_reason_bucket") == "weak_signal" and (_num(r.get("max_favorable_pct")) or 0) >= 1.5),
        ("weak_signal_giveback075", "weak signal with giveback >= 0.75", lambda r: r.get("exit_reason_bucket") == "weak_signal" and (_num(r.get("giveback_pct")) or 0) >= 0.75),
        ("weak_positive_pnl", "weak signal with positive realized PnL", lambda r: r.get("exit_reason_bucket") == "weak_signal" and (_num(r.get("pnl_pct")) or 0) > 0),
        ("non_ema_positive_giveback", "non-EMA exit with positive PnL and giveback >= 0.5", lambda r: r.get("exit_reason_bucket") != "ema_break" and (_num(r.get("pnl_pct")) or 0) > 0 and (_num(r.get("giveback_pct")) or 0) >= 0.5),
        ("non_ema_mfe150", "non-EMA exit with MFE >= 1.5", lambda r: r.get("exit_reason_bucket") != "ema_break" and (_num(r.get("max_favorable_pct")) or 0) >= 1.5),
        ("momentum_mfe150_giveback050", "momentum mode with MFE >= 1.5 and giveback >= 0.5", lambda r: r.get("mode") in modes_momentum and (_num(r.get("max_favorable_pct")) or 0) >= 1.5 and (_num(r.get("giveback_pct")) or 0) >= 0.5),
        ("profitable_high_giveback", "profitable exits with giveback >= 1.0", lambda r: (_num(r.get("pnl_pct")) or 0) > 0 and (_num(r.get("giveback_pct")) or 0) >= 1.0),
        ("exclude_ema_and_false_cleanup", "non-EMA, MFE >= 1.0, pnl > -0.5", lambda r: r.get("exit_reason_bucket") != "ema_break" and (_num(r.get("max_favorable_pct")) or 0) >= 1.0 and (_num(r.get("pnl_pct")) or 0) > -0.5),
    ]


def _score(rows: list[dict[str, Any]], name: str, fn: Callable[[dict[str, Any]], bool], tail_policy: str) -> dict[str, Any]:
    deltas=[]; pnls=[]; allowed=[]
    for row in rows:
        baseline=_num(row.get("pnl_pct")); tail=_num(row.get(f"{tail_policy}_pnl_pct"))
        if baseline is None or tail is None: continue
        use=bool(fn(row)); pnl=tail if use else baseline
        pnls.append(pnl); deltas.append(round(pnl-baseline,4))
        if use: allowed.append(row)
    worse=[d for d in deltas if d < 0]
    fp=[r for r in rows if r.get("bucket") == "false_positive_buys"]
    fp_allowed=[r for r in fp if r in allowed]
    early=[r for r in rows if r.get("bucket") == "early_exits"]
    early_allowed=[r for r in early if r in allowed]
    return {
        "n": len(pnls), "avg_pnl_pct": _avg(pnls), "median_pnl_pct": _median(pnls), "win_rate_pct": _win_rate(pnls),
        "avg_delta_pct": _avg(deltas), "median_delta_pct": _median(deltas), "total_delta_pct": round(sum(deltas),4) if deltas else None,
        "worse_rate_pct": round(len(worse)/len(deltas)*100,2) if deltas else 0.0,
        "allowed_total": len(allowed), "allowed_rate_pct": round(len(allowed)/len(rows)*100,2) if rows else 0.0,
        "false_positive_allowed_rate_pct": round(len(fp_allowed)/len(fp)*100,2) if fp else 0.0,
        "early_allowed_rate_pct": round(len(early_allowed)/len(early)*100,2) if early else 0.0,
    }


def _baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals=[_num(r.get("pnl_pct")) for r in rows]; vals=[v for v in vals if v is not None]
    return {"n": len(vals), "avg_pnl_pct": _avg(vals), "median_pnl_pct": _median(vals), "win_rate_pct": _win_rate(vals)}


def _rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    t=item.get("test") or {}; a=item.get("all") or {}
    return (_num(t.get("avg_delta_pct")) or -999, _num(t.get("median_delta_pct")) or -999, -(_num(t.get("worse_rate_pct")) or 999), _num(a.get("avg_delta_pct")) or -999)


def _decision(ranked: list[dict[str, Any]]) -> str:
    for item in ranked:
        t = item.get("test") or {}
        n = _num(t.get("n"))
        if (n or 0) < 10:
            continue
        avg_delta = _num(t.get("avg_delta_pct"))
        med_delta = _num(t.get("median_delta_pct"))
        worse = _num(t.get("worse_rate_pct"))
        allowed_rate = _num(t.get("allowed_rate_pct"))
        fp_allowed = _num(t.get("false_positive_allowed_rate_pct"))
        avg_delta = 0.0 if avg_delta is None else avg_delta
        med_delta = 0.0 if med_delta is None else med_delta
        worse = 100.0 if worse is None else worse
        allowed_rate = 0.0 if allowed_rate is None else allowed_rate
        fp_allowed = 100.0 if fp_allowed is None else fp_allowed
        if avg_delta > 0.10 and med_delta >= 0 and worse <= 30 and allowed_rate >= 5 and fp_allowed <= 10:
            return f"advance_{item.get('name')}_to_shadow_observable_tail_selector"
    return "no_observable_selector_passed_test_gate"


def _num(v: Any) -> float | None:
    return tail_replay._num(v)

def _avg(values: Iterable[float | None]) -> float | None:
    vals=[float(v) for v in values if v is not None]
    return round(mean(vals),4) if vals else None

def _median(values: Iterable[float | None]) -> float | None:
    vals=[float(v) for v in values if v is not None]
    return round(median(vals),4) if vals else None

def _win_rate(values: Iterable[float | None]) -> float:
    vals=[float(v) for v in values if v is not None]
    return round(sum(1 for v in vals if v>0)/len(vals)*100,2) if vals else 0.0


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description="Research-only observable tail selector replay")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS); ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--days", type=int, default=14); ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); ap.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    ap.add_argument("--json", action="store_true"); ap.add_argument("--no-save", action="store_true")
    args=ap.parse_args(argv)
    payload=build_replay(reports_dir=args.reports_dir, cache_dir=args.cache_dir, cfg=ObservableSelectorConfig(days=args.days), output=args.output, text_output=args.text_output, save=not args.no_save)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
