from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
EVENTS_FILE = ROOT / "v2_shadow_events.jsonl"
CRITIC_DATASET = ROOT / "critic_dataset.jsonl"
WATCHLIST_FILE = ROOT / "watchlist.json"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "v2_wakeup_v1_bridge_replay_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "v2_wakeup_v1_bridge_replay_latest.txt"
UP_STATES = {"emerging_move", "confirmed_trend", "mature_trend"}


def run_replay(
    *,
    reports_dir: Path = REPORT_DIR,
    events_file: Path = EVENTS_FILE,
    critic_dataset: Path = CRITIC_DATASET,
    watchlist_file: Path = WATCHLIST_FILE,
    timezone_name: str = "Europe/Budapest",
    window_minutes: int = 360,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
) -> dict[str, Any]:
    tz = ZoneInfo(timezone_name)
    watchlist = _load_watchlist(watchlist_file)
    reports = _load_reports(reports_dir)
    v2 = _load_v2_first_upside(events_file, watchlist, tz)
    critic = _load_critic_rows(critic_dataset, watchlist)
    days = sorted(set(reports) & {d for d, _s in v2} & {d for d, _s in critic})
    profiles = {
        "v2_wakeup_v1_observation_window": _profile_observation,
        "v2_wakeup_v1_structural": _profile_structural,
        "v2_wakeup_v1_momentum": _profile_momentum,
    }
    rows_by_profile = {name: [] for name in profiles}
    naive_rows = []
    for day in days:
        labels = reports[day]
        for sym in sorted(watchlist):
            wake = v2.get((day, sym))
            if not wake:
                continue
            label = labels.get(sym, {"top_mover": False})
            naive_rows.append(_naive_row(day, sym, wake, label))
            cands = [r for r in critic.get((day, sym), []) if 0 <= int((r["ts_ms"] - wake["ts_ms"]) / 60000) <= window_minutes]
            for name, pred in profiles.items():
                selected = next((r for r in cands if pred(r)), None)
                if selected:
                    rows_by_profile[name].append(_bridge_row(day, sym, wake, selected, label))
    v1 = _v1_actual_rows(reports)
    policies = {"v2_naive_first_upside": _summarize(naive_rows, reports)}
    policies.update({name: _summarize(rows, reports) for name, rows in rows_by_profile.items()})
    policies["v1_actual"] = _summarize(v1, reports)
    ranked = sorted(
        [{"policy": k, **v} for k, v in policies.items() if k not in {"v1_actual", "v2_naive_first_upside"}],
        key=lambda x: (x["passes_gate"], x["top_precision_pct"], x["ret5_precision_pct"], -x["candidate_pressure_vs_v1"]),
        reverse=True,
    )
    payload = {
        "generated_at_local": datetime.now(tz).isoformat(timespec="seconds"),
        "settings": {"timezone": timezone_name, "window_minutes": window_minutes, "watchlist_symbols": len(watchlist)},
        "coverage": {
            "days": days,
            "days_count": len(days),
            "v2_wakeup_symbol_days": len(naive_rows),
            "critic_symbol_days": len({k for k in critic if k[0] in days}),
            "v1_actual_rows": len(v1),
        },
        "policies": policies,
        "ranked_bridge_profiles": ranked,
        "decision": _decision(ranked[0] if ranked else None),
        "examples": {name: rows[:10] for name, rows in rows_by_profile.items()},
    }
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    lines = ["V2 wake-up → V1 bridge replay", f"coverage={report['coverage']}", "", "Policies:"]
    for name, m in report["policies"].items():
        lines.append(
            f"  {name}: n={m['n']} top_precision={m['top_precision_pct']}% top_recall={m['top_recall_pct']}% "
            f"ret5_precision={m['ret5_precision_pct']}% avg_ret5={m['avg_ret5']} neg_ret5={m['negative_ret5_rate_pct']}% "
            f"delay_med={m['median_v2_to_v1_delay_min']} pressure={m['candidate_pressure_vs_v1']}"
        )
    if report["ranked_bridge_profiles"]:
        b = report["ranked_bridge_profiles"][0]
        lines += ["", f"Best bridge: {b['policy']}", f"Decision: {report['decision']}"]
    return "\n".join(lines)


def _profile_observation(row: dict[str, Any]) -> bool:
    flags = row.get("flags") or {}
    return bool(flags.get("entry_ok") or flags.get("alignment_ok") or flags.get("surge_ok") or flags.get("breakout_ok") or flags.get("impulse_ok"))


def _profile_structural(row: dict[str, Any]) -> bool:
    f = row.get("f") or {}; flags = row.get("flags") or {}
    return (
        bool(flags.get("entry_ok") or flags.get("alignment_ok") or flags.get("surge_ok"))
        and _num(f.get("close_vs_ema20"), -99) > 0
        and _num(f.get("macd_hist_norm"), -99) > 0
        and 52 <= _num(f.get("rsi"), 0) <= 74
        and _num(f.get("daily_range"), 99) <= 14
    )


def _profile_momentum(row: dict[str, Any]) -> bool:
    f = row.get("f") or {}
    return (
        str(row.get("signal_type")) in {"trend", "strong_trend", "impulse", "impulse_speed", "alignment"}
        and _num(f.get("slope"), 0) >= 0.35
        and _num(f.get("vol_x"), 0) >= 1.1
        and _num(f.get("close_vs_ema20"), -99) > 0
        and 55 <= _num(f.get("rsi"), 0) <= 74
        and _num(f.get("daily_range"), 99) <= 14
    )


def _bridge_row(day: str, sym: str, wake: dict[str, Any], row: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    delay = int((row["ts_ms"] - wake["ts_ms"]) / 60000)
    labels = row.get("labels") or {}
    return {
        "day": day, "symbol": sym, "top_mover": bool(label.get("top_mover")), "v1_bought": bool(label.get("v1_bought")),
        "v2_ts": wake["ts"], "v1_ts": row["ts_signal"], "v2_to_v1_delay_min": delay,
        "signal_type": row.get("signal_type"), "ret_3": _maybe(labels.get("ret_3")), "ret_5": _maybe(labels.get("ret_5")), "ret_10": _maybe(labels.get("ret_10")),
        "flags": row.get("flags"), "f": row.get("f"),
    }


def _naive_row(day: str, sym: str, wake: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    return {"day": day, "symbol": sym, "top_mover": bool(label.get("top_mover")), "v1_bought": bool(label.get("v1_bought")), "v2_ts": wake["ts"]}


def _v1_actual_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for day, labels in reports.items():
        for sym, label in labels.items():
            if label.get("v1_bought"):
                out.append({"day": day, "symbol": sym, "top_mover": bool(label.get("top_mover")), "v1_bought": True, "ret_5": label.get("latest_exit_pnl_pct")})
    return out


def _summarize(rows: list[dict[str, Any]], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    days = {r["day"] for r in rows}
    denom = {(d, s) for d in days for s, lab in reports.get(d, {}).items() if lab.get("top_mover")}
    top = [r for r in rows if r.get("top_mover")]
    false = [r for r in rows if not r.get("top_mover")]
    ret5 = [_maybe(r.get("ret_5")) for r in rows if _maybe(r.get("ret_5")) is not None]
    pos = [x for x in ret5 if x > 0]
    neg = [x for x in ret5 if x < 0]
    delay = [_maybe(r.get("v2_to_v1_delay_min")) for r in rows if _maybe(r.get("v2_to_v1_delay_min")) is not None]
    v1_count = sum(1 for d in days for lab in reports.get(d, {}).values() if lab.get("v1_bought")) or 1
    precision = _pct(len(top), len(rows)); recall = _pct(len({(r["day"], r["symbol"]) for r in top}), len(denom))
    ret5_precision = _pct(len(pos), len(ret5)); neg_rate = _pct(len(neg), len(ret5))
    passes = len(rows) >= 10 and precision >= 15 and recall >= 50 and ret5_precision >= 50 and neg_rate <= 45
    return {
        "n": len(rows), "top_count": len(top), "false_favorable_count": len(false),
        "top_precision_pct": precision, "top_recall_pct": recall, "false_favorable_rate_pct": _pct(len(false), len(rows)),
        "ret5_rows": len(ret5), "ret5_precision_pct": ret5_precision, "avg_ret5": _avg(ret5), "median_ret5": _median(ret5), "negative_ret5_rate_pct": neg_rate,
        "median_v2_to_v1_delay_min": _median(delay), "avg_v2_to_v1_delay_min": _avg(delay),
        "candidate_pressure_vs_v1": round(len(rows) / v1_count, 3), "passes_gate": passes,
    }


def _decision(best: dict[str, Any] | None) -> str:
    if not best:
        return "insufficient_data"
    if best.get("passes_gate"):
        return f"advance_to_portfolio_aware_replay: {best['policy']}"
    return "research_only_no_bridge_profile_passed_gate"


def _load_reports(reports_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for p in reports_dir.glob("top_gainer_critic_*_final.json"):
        try: d = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception: continue
        if (d.get("summary") or {}).get("watchlist_top_denominator") != "exchange_top_filtered_to_watchlist": continue
        day = str(d.get("target_day_local") or ""); labels = {}
        for r in d.get("watchlist_top_gainers") or []:
            labels[str(r.get("symbol"))] = {"top_mover": True, "v1_bought": r.get("status") == "bought", "latest_exit_pnl_pct": r.get("latest_exit_pnl_pct")}
        out[day] = labels
    return out


def _load_v2_first_upside(path: Path, watchlist: set[str], tz: ZoneInfo) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if path.exists() else []:
        try: r = json.loads(line)
        except Exception: continue
        if r.get("bootstrap") is True or ("previous_state" in r and r.get("previous_state") is None): continue
        sym = str(r.get("sym") or "")
        if sym not in watchlist or r.get("state") not in UP_STATES: continue
        ts = str(r.get("ts") or "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")); day = dt.astimezone(tz).date().isoformat()
        except Exception: continue
        key = (day, sym)
        if key not in out or ts < out[key]["ts"]:
            out[key] = {"ts": ts, "ts_ms": int(dt.timestamp() * 1000)}
    return out


def _load_critic_rows(path: Path, watchlist: set[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if path.exists() else []:
        try: r = json.loads(line)
        except Exception: continue
        sym = str(r.get("sym") or "")
        ts = str(r.get("ts_signal") or "")
        if sym not in watchlist or not ts: continue
        try: dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception: continue
        dec = r.get("decision") or {}; flags = dec.get("signal_flags") or {}
        item = {"sym": sym, "ts_signal": ts, "ts_ms": int(dt.timestamp() * 1000), "signal_type": r.get("signal_type"), "f": r.get("f") or {}, "flags": flags, "labels": r.get("labels") or {}}
        out[(ts[:10], sym)].append(item)
    for rows in out.values(): rows.sort(key=lambda x: x["ts_ms"])
    return out


def _load_watchlist(path: Path) -> set[str]:
    try: return {str(x).strip().upper() for x in json.loads(path.read_text(encoding="utf-8")) if str(x).strip()}
    except Exception: return set()


def _num(v: Any, default: float) -> float:
    x = _maybe(v); return default if x is None else x

def _maybe(v: Any) -> float | None:
    try:
        if v is None: return None
        x = float(v); return x if x == x else None
    except Exception: return None

def _pct(a: int, b: int) -> float: return round(100.0 * a / b, 2) if b else 0.0

def _avg(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return round(mean(vals), 4) if vals else None

def _median(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return round(median(vals), 4) if vals else None


def main() -> int:
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-minutes", type=int, default=360)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    payload = run_replay(window_minutes=args.window_minutes)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else render_text(payload))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
