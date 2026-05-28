from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
EVENTS_FILE = ROOT / "v2_shadow_events.jsonl"
WATCHLIST_FILE = ROOT / "watchlist.json"
HISTORY_ROOT = ROOT / ".runtime" / "v2_history"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "v2_early_admission_full_backtest_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "v2_early_admission_full_backtest_latest.txt"
UP_STATES = {"emerging_move", "confirmed_trend", "mature_trend"}
CONFIRMED_STATES = {"confirmed_trend", "mature_trend"}


def run_backtest(
    *,
    reports_dir: Path = REPORT_DIR,
    events_file: Path = EVENTS_FILE,
    watchlist_file: Path = WATCHLIST_FILE,
    history_root: Path = HISTORY_ROOT,
    timezone_name: str = "Europe/Budapest",
    fee_bps: float = 20.0,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
) -> dict[str, Any]:
    tz = ZoneInfo(timezone_name)
    watchlist = _load_watchlist(watchlist_file)
    reports = _load_reports(reports_dir)
    events_by = _load_events(events_file, tz)
    days = sorted(set(reports) & {day for day, _sym in events_by})
    rows: list[dict[str, Any]] = []
    missing_history = 0
    candle_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for day in days:
        labels = reports[day]["labels"]
        for sym in sorted(watchlist):
            evs = events_by.get((day, sym), [])
            if not evs:
                continue
            cache_key = (day, sym)
            if cache_key not in candle_cache:
                candle_cache[cache_key] = _load_day_candles(history_root, sym, day, tz)
            candles = candle_cache[cache_key]
            if not candles:
                missing_history += 1
            first_up = next((ev for ev in evs if ev.get("state") in UP_STATES), None)
            first_conf = next((ev for ev in evs if ev.get("state") in CONFIRMED_STATES), None)
            later_confirmed = bool(first_up and any(ev.get("state") in CONFIRMED_STATES and str(ev.get("ts")) >= str(first_up.get("ts")) for ev in evs))
            for policy, ev in (
                ("v2_first_upside", first_up),
                ("v2_first_confirmed", first_conf),
                ("v2_first_upside_later_confirmed", first_up if later_confirmed else None),
            ):
                if ev is None:
                    continue
                row = _candidate_row(day, sym, policy, ev, labels.get(sym, {}), candles, fee_bps)
                if row:
                    rows.append(row)
    v1_rows = _v1_rows(reports, fee_bps)
    policies = sorted({row["policy"] for row in rows})
    policy_results = {policy: _summarize([row for row in rows if row["policy"] == policy]) for policy in policies}
    policy_results["v1_actual_first_buy"] = _summarize(v1_rows)
    ranked = sorted(
        [{"policy": name, **metrics} for name, metrics in policy_results.items() if name != "v1_actual_first_buy"],
        key=lambda item: (item["decision_score"], item["top_precision_pct"], item["median_hold_to_close_net_pct"]),
        reverse=True,
    )
    payload = {
        "generated_at_local": datetime.now(tz).isoformat(timespec="seconds"),
        "settings": {
            "timezone": timezone_name,
            "fee_bps_roundtrip": fee_bps,
            "watchlist_symbols": len(watchlist),
            "history_root": str(history_root),
        },
        "coverage": {
            "days": days,
            "days_count": len(days),
            "candidate_rows": len(rows),
            "v1_rows": len(v1_rows),
            "missing_history_symbol_days": missing_history,
        },
        "policies": policy_results,
        "ranked_v2_policies": ranked,
        "decision": _decision(ranked[0] if ranked else None, policy_results.get("v1_actual_first_buy", {})),
        "examples": _examples(rows),
    }
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    ranked = report.get("ranked_v2_policies") or []
    policies = report.get("policies") or {}
    lines = [
        "V2 early admission full-candidate backtest",
        f"days={coverage.get('days_count')} candidates={coverage.get('candidate_rows')} missing_history={coverage.get('missing_history_symbol_days')}",
        "",
        "Policies:",
    ]
    for name, metrics in policies.items():
        lines.append(
            f"  {name}: n={metrics.get('n')} top_precision={metrics.get('top_precision_pct')}% "
            f"top_recall={metrics.get('top_recall_pct')}% false={metrics.get('false_favorable_rate_pct')}% "
            f"return_cov={metrics.get('return_coverage_pct')}% "
            f"median_net={metrics.get('median_hold_to_close_net_pct')}% median_capture={metrics.get('median_capture_remaining')}"
        )
    if ranked:
        best = ranked[0]
        lines.extend(["", f"Best V2: {best['policy']} score={best['decision_score']} median_net={best['median_hold_to_close_net_pct']}% precision={best['top_precision_pct']}% recall={best['top_recall_pct']}%"])
    lines.extend(["", "Decision:", f"  {report.get('decision')}"])
    return "\n".join(lines)


def _candidate_row(day: str, sym: str, policy: str, ev: dict[str, Any], label: dict[str, Any], candles: list[dict[str, Any]], fee_bps: float) -> dict[str, Any] | None:
    features = ev.get("features") or {}
    entry = _num(features.get("price"))
    if entry is None or entry <= 0:
        return None
    after = [c for c in candles if int(c.get("open_ts_ms") or 0) >= _ts_ms(str(ev.get("ts") or ""))]
    close = _num(after[-1].get("close")) if after else None
    high = max((_num(c.get("high")) or 0.0 for c in after), default=0.0) if after else None
    day_open = _num(label.get("day_open"))
    day_close = _num(label.get("day_close")) or close
    is_top = bool(label.get("top_mover"))
    hold = _ret(entry, close)
    mfe = _ret(entry, high)
    return {
        "day": day,
        "symbol": sym,
        "policy": policy,
        "entry_ts": ev.get("ts"),
        "entry_price": entry,
        "state": ev.get("state"),
        "confidence": ev.get("confidence"),
        "top_mover": is_top,
        "v1_bought": bool(label.get("v1_bought")),
        "hold_to_close_pct": hold,
        "hold_to_close_net_pct": None if hold is None else round(hold - fee_bps / 100.0, 4),
        "mfe_to_high_pct": mfe,
        "capture_remaining": _capture_remaining(day_open, day_close, entry) if is_top else None,
    }


def _v1_rows(reports: dict[str, Any], fee_bps: float) -> list[dict[str, Any]]:
    rows = []
    for day, payload in reports.items():
        for sym, label in payload["labels"].items():
            if not label.get("v1_bought"):
                continue
            entry = _num(label.get("first_entry_price"))
            close = _num(label.get("day_close"))
            if not entry or not close:
                continue
            hold = _ret(entry, close)
            rows.append({
                "day": day,
                "symbol": sym,
                "policy": "v1_actual_first_buy",
                "entry_price": entry,
                "top_mover": bool(label.get("top_mover")),
                "v1_bought": True,
                "hold_to_close_pct": hold,
                "hold_to_close_net_pct": None if hold is None else round(hold - fee_bps / 100.0, 4),
                "mfe_to_high_pct": None,
                "capture_remaining": _capture_remaining(_num(label.get("day_open")), _num(label.get("day_close")), entry) if label.get("top_mover") else None,
                "actual_pnl_pct": label.get("latest_exit_pnl_pct"),
            })
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = [r for r in rows if r.get("top_mover")]
    false = [r for r in rows if not r.get("top_mover")]
    return_rows = [r for r in rows if isinstance(r.get("hold_to_close_net_pct"), (int, float))]
    top_keys = {(r["day"], r["symbol"]) for r in top}
    # recall denominator: unique top movers present in reports for all days where this policy emitted anything.
    days = {r["day"] for r in rows}
    denom = len(_GLOBAL_TOP_KEYS_BY_DAY(days)) if rows else 0
    med_net = _median(r.get("hold_to_close_net_pct") for r in rows)
    precision = _pct(len(top), len(rows))
    recall = _pct(len(top_keys), denom)
    false_rate = _pct(len(false), len(rows))
    return {
        "n": len(rows),
        "unique_day_symbols": len({(r["day"], r["symbol"]) for r in rows}),
        "top_count": len(top),
        "false_favorable_count": len(false),
        "top_precision_pct": precision,
        "top_recall_pct": recall,
        "false_favorable_rate_pct": false_rate,
        "return_coverage_count": len(return_rows),
        "return_coverage_pct": _pct(len(return_rows), len(rows)),
        "median_hold_to_close_net_pct": med_net,
        "avg_hold_to_close_net_pct": _avg(r.get("hold_to_close_net_pct") for r in rows),
        "median_mfe_to_high_pct": _median(r.get("mfe_to_high_pct") for r in rows),
        "median_capture_remaining": _median(r.get("capture_remaining") for r in top),
        "avg_capture_remaining": _avg(r.get("capture_remaining") for r in top),
        "decision_score": round((med_net or 0.0) + precision / 10.0 + recall / 20.0 - false_rate / 20.0, 4),
    }

_GLOBAL_REPORTS_CACHE: dict[str, Any] = {}

def _GLOBAL_TOP_KEYS_BY_DAY(days: set[str]) -> set[tuple[str, str]]:
    out = set()
    for day in days:
        for sym, label in (_GLOBAL_REPORTS_CACHE.get(day, {}).get("labels") or {}).items():
            if label.get("top_mover"):
                out.add((day, sym))
    return out


def _load_reports(reports_dir: Path) -> dict[str, Any]:
    out = {}
    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        summary = report.get("summary") or {}
        if summary.get("watchlist_top_denominator") != "exchange_top_filtered_to_watchlist":
            continue
        day = str(report.get("target_day_local") or "")
        labels = {}
        for row in report.get("watchlist_top_gainers") or []:
            sym = str(row.get("symbol") or "")
            if not sym:
                continue
            labels[sym] = {
                "top_mover": True,
                "v1_bought": row.get("status") == "bought",
                "day_open": row.get("day_open"),
                "day_close": row.get("day_close"),
                "day_high": row.get("day_high"),
                "first_entry_price": row.get("first_entry_price"),
                "latest_exit_pnl_pct": row.get("latest_exit_pnl_pct"),
            }
        out[day] = {"labels": labels, "summary": summary}
    _GLOBAL_REPORTS_CACHE.clear(); _GLOBAL_REPORTS_CACHE.update(out)
    return out


def _load_events(events_file: Path, tz: ZoneInfo) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not events_file.exists():
        return out
    for line in events_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("bootstrap") is True or ("previous_state" in row and row.get("previous_state") is None):
            continue
        sym = str(row.get("sym") or "")
        ts = str(row.get("ts") or "")
        if not sym or not ts:
            continue
        try:
            day = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz).date().isoformat()
        except Exception:
            continue
        out[(day, sym)].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: str(r.get("ts") or ""))
    return out


def _load_day_candles(history_root: Path, sym: str, day: str, tz: ZoneInfo) -> list[dict[str, Any]]:
    path = history_root / sym / "15m.jsonl"
    if not path.exists():
        return []
    target = date.fromisoformat(day)
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
            dt = datetime.fromtimestamp(int(row["open_ts_ms"]) / 1000, tz=ZoneInfo("UTC")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target:
            out.append(row)
    return out


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    top = sorted([r for r in rows if r.get("top_mover")], key=lambda r: r.get("hold_to_close_net_pct") or -999, reverse=True)
    false = sorted([r for r in rows if not r.get("top_mover")], key=lambda r: r.get("hold_to_close_net_pct") or -999)
    return {"best_top": top[:10], "worst_false": false[:10]}


def _decision(best: dict[str, Any] | None, v1: dict[str, Any]) -> str:
    if not best:
        return "insufficient_data"
    if best.get("top_precision_pct", 0) >= 25 and best.get("top_recall_pct", 0) >= 60 and (best.get("median_hold_to_close_net_pct") or -999) > (v1.get("median_hold_to_close_net_pct") or -999):
        return f"advance_to_portfolio_aware_replay: {best['policy']}"
    return "research_only_rejected_or_needs_stricter_discriminator"


def _load_watchlist(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return set()
    return {str(x).strip().upper() for x in data if str(x).strip()}


def _ts_ms(raw: str) -> int:
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return 0


def _ret(entry: float | None, exit_: float | None) -> float | None:
    if not entry or not exit_:
        return None
    return round((exit_ / entry - 1.0) * 100.0, 4)


def _capture_remaining(day_open: float | None, day_close: float | None, entry: float | None) -> float | None:
    if day_open is None or day_close is None or entry is None:
        return None
    move = day_close - day_open
    if move <= 0:
        return None
    return round(max(0.0, min(1.5, (day_close - entry) / move)), 4)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if value == value else None
    except Exception:
        return None


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _median(values) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(median(vals), 4) if vals else None


def _avg(values) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(mean(vals), 4) if vals else None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--events-file", type=Path, default=EVENTS_FILE)
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST_FILE)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--fee-bps", type=float, default=20.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = run_backtest(
        reports_dir=args.reports_dir,
        events_file=args.events_file,
        watchlist_file=args.watchlist,
        history_root=args.history_root,
        timezone_name=args.timezone,
        fee_bps=args.fee_bps,
        output_json=args.output_json,
        output_txt=args.output_txt,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
