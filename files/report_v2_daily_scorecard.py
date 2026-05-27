from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
DEFAULT_OUTPUT_JSON = REPORT_DIR / "v2_scorecard_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "v2_scorecard_latest.txt"
UP_STATES = {"emerging_move", "confirmed_trend", "mature_trend"}
CONFIRMED_STATES = {"confirmed_trend", "mature_trend"}


@dataclass(frozen=True)
class V2DayMetrics:
    day: str
    coverage_status: str
    coverage_reasons: tuple[str, ...]
    events_total: int = 0
    upside_events: int = 0
    confirmed_events: int = 0
    deescalation_events: int = 0
    unique_upside_symbols: int = 0
    unique_confirmed_symbols: int = 0
    top_count: int = 0
    top_with_v2_upside: int = 0
    top_with_v2_confirmed: int = 0
    top_with_v2_upside_bought: int = 0
    v1_top_bought: int = 0
    v2_false_favorable_symbols: int = 0
    confirmation_ratio: float = 0.0
    v2_top_recall_pct: float = 0.0
    v2_confirmed_top_recall_pct: float = 0.0
    v2_top_precision_pct: float = 0.0
    v2_confirmed_top_precision_pct: float = 0.0
    v2_handoff_bought_pct: float = 0.0
    v1_top_capture_pct: float = 0.0
    state_counts: dict[str, int] | None = None
    action_counts: dict[str, int] | None = None
    top_symbols_seen_by_v2: tuple[str, ...] = ()
    top_symbols_missed_by_v2: tuple[str, ...] = ()
    false_favorable_examples: tuple[str, ...] = ()


def build_scorecard(
    target_day: date,
    *,
    events_file: Path = EVENTS_FILE,
    reports_dir: Path = REPORT_DIR,
    timezone_name: str = "Europe/Budapest",
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
) -> dict[str, Any]:
    days = _build_history(target_day, events_file=events_file, reports_dir=reports_dir, timezone_name=timezone_name)
    latest = next((d for d in days if d.day == target_day.isoformat()), None)
    if latest is None:
        latest = _build_day(target_day, events_file=events_file, reports_dir=reports_dir, timezone_name=timezone_name)
        days.append(latest)
        days.sort(key=lambda d: d.day)
    payload = {
        "generated_at_local": datetime.now(ZoneInfo(timezone_name)).isoformat(timespec="seconds"),
        "target_day_local": target_day.isoformat(),
        "timezone": timezone_name,
        "status": latest.coverage_status,
        "coverage_reasons": list(latest.coverage_reasons),
        "latest": asdict(latest),
        "progress": _progress(days, target_day.isoformat()),
        "interpretation": _interpretation(latest),
        "recommendation": _recommendation(latest),
    }
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        stem = f"v2_scorecard_{target_day.isoformat()}"
        dated_json = output_json.parent / f"{stem}.json"
        dated_txt = output_txt.parent / f"{stem}.txt"
        dated_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        dated_txt.write_text(text, encoding="utf-8")
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        payload["files"] = {
            "json": str(dated_json),
            "txt": str(dated_txt),
            "latest_json": str(output_json),
            "latest_txt": str(output_txt),
        }
    return payload


def render_text(scorecard: dict[str, Any]) -> str:
    latest = scorecard.get("latest") or {}
    progress = scorecard.get("progress") or {}
    day = scorecard.get("target_day_local") or latest.get("day") or "unknown"
    status = scorecard.get("status") or "unknown"
    status_icon = "🟢" if status == "complete" else "🟡"
    dod = progress.get("day_over_day") or {}
    wow = progress.get("week_over_week") or {}
    lines = [
        f"V2 Markov/RL scorecard — {day}",
        "",
        f"{status_icon} status={status}" + _coverage_suffix(scorecard.get("coverage_reasons") or []),
        "",
        "Главное:",
        f"  • V2 увидел {latest.get('top_with_v2_upside', 0)}/{latest.get('top_count', 0)} top movers "
        f"({ _fmt(latest.get('v2_top_recall_pct')) } recall).",
        f"  • Precision V2 upside: { _fmt(latest.get('v2_top_precision_pct')) }; "
        f"confirmation ratio: { _fmt_ratio(latest.get('confirmation_ratio')) }.",
        f"  • Handoff в V1 BUY: { _fmt(latest.get('v2_handoff_bought_pct')) } среди top movers, которые V2 видел.",
        f"  • False-favorable pressure: {latest.get('v2_false_favorable_symbols', 0)} символ(ов) вне same-day top movers.",
        "",
        "Прогресс:",
        f"  • День ко дню recall: { _delta_line(dod, 'v2_top_recall_pct') }; precision: { _delta_line(dod, 'v2_top_precision_pct') }; handoff: { _delta_line(dod, 'v2_handoff_bought_pct') }.",
        f"  • Неделя к неделе recall: { _delta_line(wow, 'v2_top_recall_pct') }; precision: { _delta_line(wow, 'v2_top_precision_pct') }; confirmation: { _delta_line(wow, 'confirmation_ratio', ratio=True) }.",
        "",
        "Где V2 промахнулся:",
        "  • top movers без V2 upside: " + (_join(latest.get("top_symbols_missed_by_v2")) or "none"),
        "  • V2 upside не top movers: " + (_join(latest.get("false_favorable_examples")) or "none"),
        "",
        "Решение:",
        f"  • {scorecard.get('recommendation')}",
    ]
    return "\n".join(lines).strip()


def save_report(scorecard: dict[str, Any]) -> dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target_day = str(scorecard.get("target_day_local") or scorecard.get("latest", {}).get("day") or date.today().isoformat())
    text = render_text(scorecard)
    json_payload = json.dumps(scorecard, ensure_ascii=False, indent=2)
    dated_json = REPORT_DIR / f"v2_scorecard_{target_day}.json"
    dated_txt = REPORT_DIR / f"v2_scorecard_{target_day}.txt"
    dated_json.write_text(json_payload, encoding="utf-8")
    dated_txt.write_text(text, encoding="utf-8")
    DEFAULT_OUTPUT_JSON.write_text(json_payload, encoding="utf-8")
    DEFAULT_OUTPUT_TXT.write_text(text, encoding="utf-8")
    return {"json": str(dated_json), "txt": str(dated_txt), "latest_json": str(DEFAULT_OUTPUT_JSON), "latest_txt": str(DEFAULT_OUTPUT_TXT)}


def _build_history(target_day: date, *, events_file: Path, reports_dir: Path, timezone_name: str) -> list[V2DayMetrics]:
    start = target_day - timedelta(days=20)
    days = []
    current = start
    while current <= target_day:
        days.append(_build_day(current, events_file=events_file, reports_dir=reports_dir, timezone_name=timezone_name))
        current += timedelta(days=1)
    return days


def _build_day(target_day: date, *, events_file: Path, reports_dir: Path, timezone_name: str) -> V2DayMetrics:
    rows = _load_v2_rows(events_file, target_day, timezone_name)
    state_counts = Counter(str(row.get("state") or "") for row in rows)
    action_counts = Counter(str(row.get("action") or "") for row in rows)
    upside_symbols = {str(row.get("sym") or "") for row in rows if row.get("state") in UP_STATES and row.get("sym")}
    confirmed_symbols = {str(row.get("sym") or "") for row in rows if row.get("state") in CONFIRMED_STATES and row.get("sym")}
    upside_events = sum(1 for row in rows if row.get("state") in UP_STATES)
    confirmed_events = sum(1 for row in rows if row.get("state") in CONFIRMED_STATES)
    deescalation_events = sum(1 for row in rows if row.get("state") == "noise")
    top_report = _load_top_report(reports_dir, target_day)
    top_rows = top_report.get("watchlist_top_gainers") or []
    top_symbols = {str(row.get("symbol") or "") for row in top_rows if row.get("symbol")}
    bought_top_symbols = {str(row.get("symbol") or "") for row in top_rows if row.get("status") == "bought"}
    seen_top = top_symbols & upside_symbols
    confirmed_top = top_symbols & confirmed_symbols
    false_favorable = sorted(sym for sym in upside_symbols - top_symbols if sym)
    coverage_reasons = []
    if not events_file.exists():
        coverage_reasons.append("missing v2 shadow event file")
    if not top_report:
        coverage_reasons.append("missing top-gainer outcome report")
    if not rows:
        coverage_reasons.append("no v2 events for target day")
    coverage_status = "complete" if not coverage_reasons else "partial"
    return V2DayMetrics(
        day=target_day.isoformat(),
        coverage_status=coverage_status,
        coverage_reasons=tuple(coverage_reasons),
        events_total=len(rows),
        upside_events=upside_events,
        confirmed_events=confirmed_events,
        deescalation_events=deescalation_events,
        unique_upside_symbols=len(upside_symbols),
        unique_confirmed_symbols=len(confirmed_symbols),
        top_count=len(top_symbols),
        top_with_v2_upside=len(seen_top),
        top_with_v2_confirmed=len(confirmed_top),
        top_with_v2_upside_bought=len(seen_top & bought_top_symbols),
        v1_top_bought=len(bought_top_symbols),
        v2_false_favorable_symbols=len(false_favorable),
        confirmation_ratio=_safe_ratio(confirmed_events, upside_events),
        v2_top_recall_pct=_pct(len(seen_top), len(top_symbols)),
        v2_confirmed_top_recall_pct=_pct(len(confirmed_top), len(top_symbols)),
        v2_top_precision_pct=_pct(len(seen_top), len(upside_symbols)),
        v2_confirmed_top_precision_pct=_pct(len(confirmed_top), len(confirmed_symbols)),
        v2_handoff_bought_pct=_pct(len(seen_top & bought_top_symbols), len(seen_top)),
        v1_top_capture_pct=_pct(len(bought_top_symbols), len(top_symbols)),
        state_counts=dict(state_counts),
        action_counts=dict(action_counts),
        top_symbols_seen_by_v2=tuple(sorted(seen_top)),
        top_symbols_missed_by_v2=tuple(sorted(top_symbols - upside_symbols)),
        false_favorable_examples=tuple(false_favorable[:12]),
    )


def _load_v2_rows(events_file: Path, target_day: date, timezone_name: str) -> list[dict[str, Any]]:
    out = []
    if not events_file.exists():
        return out
    for line in events_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if _is_bootstrap(row):
            continue
        try:
            local_day = datetime.fromisoformat(str(row.get("ts") or "").replace("Z", "+00:00")).astimezone(ZoneInfo(timezone_name)).date()
        except Exception:
            continue
        if local_day == target_day:
            out.append(row)
    return out


def _load_top_report(reports_dir: Path, target_day: date) -> dict[str, Any]:
    path = reports_dir / f"top_gainer_critic_{target_day.isoformat()}_final.json"
    if not path.exists():
        path = reports_dir / f"top_gainer_critic_{target_day.isoformat()}_22h.json"
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _progress(days: list[V2DayMetrics], target_day: str) -> dict[str, Any]:
    complete = [d for d in days if d.coverage_status == "complete"]
    latest = next((d for d in days if d.day == target_day), None)
    prev = None
    if latest:
        prevs = [d for d in complete if d.day < latest.day]
        prev = prevs[-1] if prevs else None
    last7 = [d for d in complete if d.day <= target_day][-7:]
    prev7 = [d for d in complete if d.day < (last7[0].day if last7 else target_day)][-7:]
    keys = [
        "v2_top_recall_pct",
        "v2_top_precision_pct",
        "v2_handoff_bought_pct",
        "v1_top_capture_pct",
        "confirmation_ratio",
        "unique_upside_symbols",
        "v2_false_favorable_symbols",
    ]
    return {
        "day_over_day": _compare_days(latest, prev, keys),
        "week_over_week": _compare_windows(last7, prev7, keys),
        "windows": {
            "last7_days": [d.day for d in last7],
            "prev7_days": [d.day for d in prev7],
            "complete_days_loaded": len(complete),
        },
    }


def _compare_days(current: V2DayMetrics | None, previous: V2DayMetrics | None, keys: Iterable[str]) -> dict[str, Any]:
    out = {"current_day": getattr(current, "day", None), "previous_day": getattr(previous, "day", None)}
    for key in keys:
        cur = getattr(current, key, None) if current else None
        prev = getattr(previous, key, None) if previous else None
        out[key] = _delta(cur, prev)
    return out


def _compare_windows(current: list[V2DayMetrics], previous: list[V2DayMetrics], keys: Iterable[str]) -> dict[str, Any]:
    out = {"current_n": len(current), "previous_n": len(previous)}
    for key in keys:
        cur = _avg(getattr(day, key) for day in current)
        prev = _avg(getattr(day, key) for day in previous)
        out[key] = _delta(cur, prev)
    return out


def _delta(cur: Any, prev: Any) -> dict[str, float | None]:
    cur_f = _maybe_float(cur)
    prev_f = _maybe_float(prev)
    return {"current": cur_f, "previous": prev_f, "delta": None if cur_f is None or prev_f is None else round(cur_f - prev_f, 6)}


def _interpretation(latest: V2DayMetrics) -> str:
    if latest.coverage_status != "complete":
        return "incomplete: first fix coverage before judging V2 progress"
    if latest.v2_top_recall_pct >= 50 and latest.v2_top_precision_pct >= 20:
        return "useful radar candidate, still shadow-only until replayed as an admission policy"
    if latest.v2_top_recall_pct >= 50:
        return "broad radar with weak precision; use for diagnostics, not entries"
    return "insufficient objective coverage; V2 is missing too many same-day top movers"


def _recommendation(latest: V2DayMetrics) -> str:
    if latest.coverage_status != "complete":
        return "Не продвигать V2: scorecard partial, сначала восстановить coverage."
    if latest.v2_top_precision_pct < 15:
        return "Не продвигать V2 в BUY: слишком много false-favorable upside; продолжать causal discriminator / reward labels."
    if latest.v2_handoff_bought_pct < 50:
        return "Не менять gates: сначала понять, почему V2 увидел movers, но V1 не купил."
    return "Оставить shadow-only и копить 7–14 дней scorecard перед replay-повышением."


def _is_bootstrap(row: dict[str, Any]) -> bool:
    return bool(row.get("bootstrap") is True or ("previous_state" in row and row.get("previous_state") is None))


def _safe_ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _avg(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if _maybe_float(v) is not None]
    return round(mean(vals), 6) if vals else None


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if value == value else None
    except Exception:
        return None


def _fmt(value: Any) -> str:
    val = _maybe_float(value)
    return "н/д" if val is None else f"{val:.1f}%"


def _fmt_ratio(value: Any) -> str:
    val = _maybe_float(value)
    return "н/д" if val is None else f"{val * 100:.1f}%"


def _delta_line(block: dict[str, Any], key: str, *, ratio: bool = False) -> str:
    item = block.get(key) or {}
    cur = item.get("current")
    prev = item.get("previous")
    delta = item.get("delta")
    if cur is None:
        return "н/д"
    fmt = _fmt_ratio if ratio else _fmt
    cur_s = fmt(cur)
    if prev is None or delta is None:
        return f"{cur_s} (нет базы)"
    sign = "+" if delta >= 0 else ""
    delta_s = f"{sign}{delta * 100:.1f}pp" if ratio else f"{sign}{delta:.1f}pp"
    return f"{fmt(prev)} → {cur_s} ({delta_s})"


def _coverage_suffix(reasons: list[str]) -> str:
    return "" if not reasons else ": " + "; ".join(str(r) for r in reasons[:3])


def _join(items: Any) -> str:
    return ", ".join(str(x) for x in (items or [])[:12])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", dest="target_date", default=date.today().isoformat())
    parser.add_argument("--events-file", type=Path, default=EVENTS_FILE)
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    target = date.fromisoformat(args.target_date)
    scorecard = build_scorecard(
        target,
        events_file=args.events_file,
        reports_dir=args.reports_dir,
        timezone_name=args.timezone,
        output_json=args.output_json,
        output_txt=args.output_txt,
    )
    print(json.dumps(scorecard, ensure_ascii=False, indent=2) if args.as_json else render_text(scorecard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
