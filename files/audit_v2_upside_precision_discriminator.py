from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
EVENTS_FILE = ROOT / "v2_shadow_events.jsonl"
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "v2_upside_precision_discriminator_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "v2_upside_precision_discriminator_latest.txt"
UP_STATES = {"emerging_move", "confirmed_trend", "mature_trend"}
CONFIRMED_STATES = {"confirmed_trend", "mature_trend"}


@dataclass(frozen=True)
class FirstUpsideRow:
    day: str
    symbol: str
    ts: str
    state: str
    previous_state: str
    action: str
    confidence: float | None
    reason: str
    adx: float | None = None
    rsi: float | None = None
    slope: float | None = None
    vol_x: float | None = None
    daily_range: float | None = None
    macd_hist: float | None = None
    price_vs_ema20_pct: float | None = None
    later_confirmed_events: int = 0
    later_noise_events: int = 0
    top_mover: bool = False
    bought_by_v1: bool = False
    status: str = "not_top"
    day_change_pct: float | None = None
    capture_ratio_at_entry: float | None = None
    exit_efficiency: float | None = None


@dataclass(frozen=True)
class SliceResult:
    name: str
    description: str
    selected: int
    useful: int
    false_favorable: int
    precision_pct: float
    recall_pct: float
    false_favorable_reduction_pct: float
    avg_day_change_pct: float | None
    avg_capture_ratio_at_entry: float | None
    examples: tuple[str, ...]


def run_audit(
    *,
    target_day: date | None = None,
    days: int = 7,
    events_file: Path = EVENTS_FILE,
    reports_dir: Path = REPORT_DIR,
    timezone_name: str = "Europe/Budapest",
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
) -> dict[str, Any]:
    if target_day is None:
        target_day = _latest_outcome_day(reports_dir) or datetime.now(ZoneInfo(timezone_name)).date() - timedelta(days=1)
    start_day = target_day - timedelta(days=max(1, days) - 1)
    rows, coverage = build_dataset(start_day, target_day, events_file=events_file, reports_dir=reports_dir, timezone_name=timezone_name)
    baseline = _baseline(rows)
    slices = rank_slices(rows)
    best = slices[0] if slices else None
    decision = _decision(baseline, best, coverage)
    payload = {
        "generated_at_local": datetime.now(ZoneInfo(timezone_name)).isoformat(timespec="seconds"),
        "window": {"start_day": start_day.isoformat(), "end_day": target_day.isoformat(), "days_requested": days},
        "status": "complete" if not coverage["missing_outcome_days"] and rows else "partial",
        "coverage": coverage,
        "baseline": baseline,
        "best_slice": asdict(best) if best else None,
        "slices": [asdict(item) for item in slices[:20]],
        "decision": decision,
    }
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        stem = f"v2_upside_precision_discriminator_{start_day.isoformat()}_{target_day.isoformat()}"
        dated_json = output_json.parent / f"{stem}.json"
        dated_txt = output_txt.parent / f"{stem}.txt"
        dated_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        dated_txt.write_text(text, encoding="utf-8")
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(dated_json), "txt": str(dated_txt), "latest_json": str(output_json), "latest_txt": str(output_txt)}
    return payload


def build_dataset(
    start_day: date,
    end_day: date,
    *,
    events_file: Path = EVENTS_FILE,
    reports_dir: Path = REPORT_DIR,
    timezone_name: str = "Europe/Budapest",
) -> tuple[list[FirstUpsideRow], dict[str, Any]]:
    events_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _load_events(events_file, start_day, end_day, timezone_name):
        sym = str(row.get("sym") or "")
        if not sym:
            continue
        day = _local_day(str(row.get("ts") or ""), timezone_name)
        if day:
            events_by_key[(day, sym)].append(row)
    outcome_by_day: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    current = start_day
    while current <= end_day:
        report = _load_top_report(reports_dir, current)
        if not report:
            missing.append(current.isoformat())
            outcome_by_day[current.isoformat()] = {}
        else:
            outcome_by_day[current.isoformat()] = {
                str(item.get("symbol") or ""): item for item in (report.get("watchlist_top_gainers") or []) if item.get("symbol")
            }
        current += timedelta(days=1)
    rows: list[FirstUpsideRow] = []
    for (day, sym), events in sorted(events_by_key.items()):
        events.sort(key=lambda item: str(item.get("ts") or ""))
        first_idx = next((i for i, item in enumerate(events) if item.get("state") in UP_STATES), None)
        if first_idx is None:
            continue
        first = events[first_idx]
        later = events[first_idx + 1 :]
        features = first.get("features") or {}
        price = _maybe_float(features.get("price"))
        ema20 = _maybe_float(features.get("ema20"))
        outcome = outcome_by_day.get(day, {}).get(sym, {})
        rows.append(
            FirstUpsideRow(
                day=day,
                symbol=sym,
                ts=str(first.get("ts") or ""),
                state=str(first.get("state") or ""),
                previous_state=str(first.get("previous_state") or ""),
                action=str(first.get("action") or ""),
                confidence=_maybe_float(first.get("confidence")),
                reason=str(first.get("reason") or ""),
                adx=_maybe_float(features.get("adx")),
                rsi=_maybe_float(features.get("rsi")),
                slope=_maybe_float(features.get("slope")),
                vol_x=_maybe_float(features.get("vol_x")),
                daily_range=_maybe_float(features.get("daily_range")),
                macd_hist=_maybe_float(features.get("macd_hist")),
                price_vs_ema20_pct=(round((price - ema20) / ema20 * 100.0, 6) if price is not None and ema20 not in (None, 0) else None),
                later_confirmed_events=sum(1 for item in later if item.get("state") in CONFIRMED_STATES),
                later_noise_events=sum(1 for item in later if item.get("state") == "noise"),
                top_mover=bool(outcome),
                bought_by_v1=str(outcome.get("status") or "") == "bought",
                status=str(outcome.get("status") or "not_top"),
                day_change_pct=_maybe_float(outcome.get("day_change_pct")),
                capture_ratio_at_entry=_maybe_float(outcome.get("capture_ratio_at_entry")),
                exit_efficiency=_maybe_float(outcome.get("exit_efficiency")),
            )
        )
    return rows, {
        "events_file": str(events_file),
        "rows": len(rows),
        "missing_outcome_days": missing,
        "outcome_days_loaded": _count_days(start_day, end_day) - len(missing),
    }


def rank_slices(rows: list[FirstUpsideRow]) -> list[SliceResult]:
    candidates: list[tuple[str, str, Any]] = [
        ("state_confirmed_or_mature", "first state is confirmed_trend/mature_trend", lambda r: r.state in CONFIRMED_STATES),
        ("prev_noise", "previous state was noise", lambda r: r.previous_state == "noise"),
        ("prev_reversal", "previous state was reversal", lambda r: r.previous_state == "reversal"),
        ("confidence_ge_0_70", "confidence >= 0.70", lambda r: _ge(r.confidence, 0.70)),
        ("confidence_ge_0_64", "confidence >= 0.64", lambda r: _ge(r.confidence, 0.64)),
        ("adx_ge_20", "ADX >= 20", lambda r: _ge(r.adx, 20)),
        ("adx_ge_25", "ADX >= 25", lambda r: _ge(r.adx, 25)),
        ("adx_lt_25", "ADX < 25", lambda r: _lt(r.adx, 25)),
        ("rsi_55_72", "55 <= RSI <= 72", lambda r: _between(r.rsi, 55, 72)),
        ("rsi_ge_60", "RSI >= 60", lambda r: _ge(r.rsi, 60)),
        ("slope_ge_0_10", "slope >= 0.10", lambda r: _ge(r.slope, 0.10)),
        ("slope_ge_0_25", "slope >= 0.25", lambda r: _ge(r.slope, 0.25)),
        ("vol_x_ge_1_5", "volume multiple >= 1.5", lambda r: _ge(r.vol_x, 1.5)),
        ("vol_x_ge_2", "volume multiple >= 2.0", lambda r: _ge(r.vol_x, 2.0)),
        ("daily_range_lt_6", "daily range < 6% at first upside", lambda r: _lt(r.daily_range, 6)),
        ("daily_range_2_10", "2% <= daily range <= 10%", lambda r: _between(r.daily_range, 2, 10)),
        ("macd_positive", "MACD histogram > 0", lambda r: _gt(r.macd_hist, 0)),
        ("price_above_ema20", "price above EMA20", lambda r: _gt(r.price_vs_ema20_pct, 0)),
        ("later_confirmed", "later same-day V2 confirmation", lambda r: r.later_confirmed_events > 0),
        ("no_later_noise", "no later deescalation to noise", lambda r: r.later_noise_events == 0),
        ("slope_and_volume", "slope >= 0.10 and vol_x >= 1.5", lambda r: _ge(r.slope, 0.10) and _ge(r.vol_x, 1.5)),
        ("confirmed_or_later_confirmed", "first/later confirmed trend", lambda r: r.state in CONFIRMED_STATES or r.later_confirmed_events > 0),
        ("moderate_range_positive_structure", "daily range < 6%, slope >= 0.10, RSI >= 55", lambda r: _lt(r.daily_range, 6) and _ge(r.slope, 0.10) and _ge(r.rsi, 55)),
    ]
    return sorted((_score_slice(rows, name, desc, pred) for name, desc, pred in candidates), key=_slice_sort_key, reverse=True)


def render_text(report: dict[str, Any]) -> str:
    window = report.get("window") or {}
    baseline = report.get("baseline") or {}
    best = report.get("best_slice") or {}
    slices = report.get("slices") or []
    lines = [
        f"V2 upside precision discriminator — {window.get('start_day')} → {window.get('end_day')}",
        "",
        f"status={report.get('status')} rows={baseline.get('rows', 0)} useful={baseline.get('useful', 0)} false={baseline.get('false_favorable', 0)}",
        f"baseline precision={_fmt(baseline.get('precision_pct'))} recall=100.0%",
        "",
        "Лучший slice:",
        f"  • {best.get('name', 'none')}: precision={_fmt(best.get('precision_pct'))}, recall={_fmt(best.get('recall_pct'))}, false reduction={_fmt(best.get('false_favorable_reduction_pct'))}",
        f"    {best.get('description', '')}",
        "",
        "Top slices:",
    ]
    for item in slices[:8]:
        lines.append(
            f"  • {item['name']}: precision={_fmt(item['precision_pct'])}, recall={_fmt(item['recall_pct'])}, "
            f"selected={item['selected']}, false_reduction={_fmt(item['false_favorable_reduction_pct'])}"
        )
    lines.extend(["", "Decision:", f"  • {report.get('decision')}"])
    return "\n".join(lines).strip()


def _baseline(rows: list[FirstUpsideRow]) -> dict[str, Any]:
    useful = [r for r in rows if r.top_mover]
    false = [r for r in rows if not r.top_mover]
    return {
        "rows": len(rows),
        "useful": len(useful),
        "false_favorable": len(false),
        "precision_pct": _pct(len(useful), len(rows)),
        "bought_by_v1": sum(1 for r in useful if r.bought_by_v1),
        "avg_day_change_pct": _avg(r.day_change_pct for r in useful),
    }


def _score_slice(rows: list[FirstUpsideRow], name: str, description: str, pred: Any) -> SliceResult:
    useful_total = sum(1 for r in rows if r.top_mover)
    false_total = sum(1 for r in rows if not r.top_mover)
    selected_rows = [r for r in rows if pred(r)]
    useful = [r for r in selected_rows if r.top_mover]
    false = [r for r in selected_rows if not r.top_mover]
    false_reduction = _pct(false_total - len(false), false_total)
    return SliceResult(
        name=name,
        description=description,
        selected=len(selected_rows),
        useful=len(useful),
        false_favorable=len(false),
        precision_pct=_pct(len(useful), len(selected_rows)),
        recall_pct=_pct(len(useful), useful_total),
        false_favorable_reduction_pct=false_reduction,
        avg_day_change_pct=_avg(r.day_change_pct for r in useful),
        avg_capture_ratio_at_entry=_avg(r.capture_ratio_at_entry for r in useful if r.capture_ratio_at_entry is not None),
        examples=tuple(f"{r.day}:{r.symbol}" for r in useful[:8]),
    )


def _slice_sort_key(item: SliceResult) -> tuple[float, float, float, int]:
    # Favor precision lift, but do not let tiny one-row slices dominate.
    support_penalty = -1 if item.useful < 3 else 0
    return (support_penalty, item.precision_pct, item.recall_pct, item.false_favorable_reduction_pct, item.selected)


def _decision(baseline: dict[str, Any], best: SliceResult | None, coverage: dict[str, Any]) -> str:
    if coverage.get("missing_outcome_days"):
        return "partial_only: missing outcome days; do not trust discriminator decision yet"
    if not best or baseline.get("rows", 0) < 30 or baseline.get("useful", 0) < 5:
        return "insufficient_data: keep collecting V2 shadow rows"
    precision_lift = best.precision_pct - float(baseline.get("precision_pct") or 0)
    if best.recall_pct >= 50 and precision_lift >= 10 and best.false_favorable_reduction_pct >= 25:
        return f"advance_to_replay_candidate: {best.name}"
    return "research_only_no_slice_passed_gate"


def _load_events(events_file: Path, start_day: date, end_day: date, timezone_name: str) -> Iterable[dict[str, Any]]:
    if not events_file.exists():
        return []
    rows = []
    for line in events_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if _is_bootstrap(row):
            continue
        local_day = _local_day(str(row.get("ts") or ""), timezone_name)
        if not local_day:
            continue
        day = date.fromisoformat(local_day)
        if start_day <= day <= end_day:
            rows.append(row)
    return rows


def _load_top_report(reports_dir: Path, target_day: date) -> dict[str, Any]:
    for suffix in ("final", "22h"):
        path = reports_dir / f"top_gainer_critic_{target_day.isoformat()}_{suffix}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                return {}
    return {}


def _latest_outcome_day(reports_dir: Path) -> date | None:
    days = []
    for path in reports_dir.glob("top_gainer_critic_*_final.json"):
        parts = path.name.split("_")
        for part in parts:
            if len(part) == 10 and part[4] == "-":
                try:
                    days.append(date.fromisoformat(part))
                except ValueError:
                    pass
    return max(days) if days else None


def _local_day(raw: str, timezone_name: str) -> str | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ZoneInfo(timezone_name)).date().isoformat()
    except Exception:
        return None


def _is_bootstrap(row: dict[str, Any]) -> bool:
    return bool(row.get("bootstrap") is True or ("previous_state" in row and row.get("previous_state") is None))


def _count_days(start_day: date, end_day: date) -> int:
    return (end_day - start_day).days + 1


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if value == value else None
    except Exception:
        return None


def _avg(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if _maybe_float(v) is not None]
    return round(mean(vals), 6) if vals else None


def _pct(num: int | float, den: int | float) -> float:
    return round(100.0 * float(num) / float(den), 2) if den else 0.0


def _fmt(value: Any) -> str:
    val = _maybe_float(value)
    return "н/д" if val is None else f"{val:.1f}%"


def _ge(value: Any, threshold: float) -> bool:
    val = _maybe_float(value)
    return val is not None and val >= threshold


def _gt(value: Any, threshold: float) -> bool:
    val = _maybe_float(value)
    return val is not None and val > threshold


def _lt(value: Any, threshold: float) -> bool:
    val = _maybe_float(value)
    return val is not None and val < threshold


def _between(value: Any, low: float, high: float) -> bool:
    val = _maybe_float(value)
    return val is not None and low <= val <= high


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", dest="target_date")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--events-file", type=Path, default=EVENTS_FILE)
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    target = date.fromisoformat(args.target_date) if args.target_date else None
    report = run_audit(
        target_day=target,
        days=args.days,
        events_file=args.events_file,
        reports_dir=args.reports_dir,
        timezone_name=args.timezone,
        output_json=args.output_json,
        output_txt=args.output_txt,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
