from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
STATUS_FILE = WORKSPACE_ROOT / ".runtime" / "rl_worker_status.json"
FEEDBACK_FILE = WORKSPACE_ROOT / ".runtime" / "signal_quality_feedback.json"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "learning_progress_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "learning_progress_latest.txt"


@dataclass(frozen=True)
class DayMetrics:
    day: str
    watchlist_top_count: int = 0
    bought: int = 0
    missed: int = 0
    early: int = 0
    false_positive_buys: int = 0
    capture_pct: float = 0.0
    early_pct: float = 0.0
    blocked_winner_count: int = 0
    blocked_symbols: tuple[str, ...] = ()
    blocked_reasons: dict[str, int] | None = None
    miss_rate: float | None = None
    false_positive_rate: float | None = None
    median_capture_ratio: float | None = None
    median_exit_efficiency: float | None = None
    median_giveback_pct: float | None = None
    coverage_status: str = "unknown"
    coverage_reasons: tuple[str, ...] = ()


def build_report(
    reports_dir: Path = REPORT_DIR,
    status_file: Path = STATUS_FILE,
    feedback_file: Path = FEEDBACK_FILE,
    focus_symbols: Iterable[str] = (),
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
) -> dict[str, Any]:
    days = _load_day_metrics(reports_dir)
    latest = days[-1] if days else DayMetrics(day="unknown")
    previous = days[-8:-1]
    older = days[-15:-8]
    status = _load_json(status_file)
    feedback = _load_json(feedback_file)
    focus = sorted({str(s).upper().replace("/", "").replace("USDT", "") + "USDT" for s in focus_symbols if str(s).strip()})
    focus_findings = _focus_findings(reports_dir, latest.day, focus)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "latest_day": latest.day,
        "days_loaded": len(days),
        "verdict": _verdict(latest, previous, older, status),
        "latest": latest.__dict__,
        "rolling": _rolling_summary(days),
        "learning_components": _learning_components(status, feedback, latest.day),
        "previous_decisions": _previous_decisions(feedback, status, latest.day),
        "alerts": _alerts(latest, status, feedback, focus_findings),
        "focus_symbols": focus_findings,
        "next_actions": _next_actions(latest, status, feedback, focus_findings),
    }
    text = render_text(payload)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_txt.write_text(text, encoding="utf-8")
    payload["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    latest = report.get("latest") or {}
    rolling = report.get("rolling") or {}
    verdict = report.get("verdict") or {}
    alerts = report.get("alerts") or []
    decisions = report.get("previous_decisions") or []
    actions = report.get("next_actions") or []
    components = report.get("learning_components") or {}
    day = report.get("latest_day") or "unknown"
    emoji = verdict.get("emoji", "⚪")
    title = verdict.get("label", "СТАТУС НЕЯСЕН")
    ask = verdict.get("operator_hint", "проверь полный отчёт")
    early_now = _fmt(rolling.get("early_last7_pct"), 1)
    early_prev = _fmt(rolling.get("early_prev7_pct"), 1)
    capture = _fmt(latest.get("capture_pct"), 1)
    early = _fmt(latest.get("early_pct"), 1)
    median_capture = latest.get("median_capture_ratio")
    capture_piece = "нет данных" if median_capture is None else f"взял медианно {float(median_capture) * 100:.0f}% движения на входе"
    blocked = latest.get("blocked_winner_count") or 0
    miss_rate = latest.get("miss_rate")
    miss_piece = "miss-rate нет" if miss_rate is None else f"miss-rate {float(miss_rate) * 100:.0f}%"
    lines = [
        f"Бот — {day}",
        "",
        f"{emoji} {title}   ·   👉 {ask}",
        "",
        f"Главное: early-capture ~{early_now}% за 7д ({early_prev}% → {early_now}%). Цель — 25%+.",
        f"Вчера: поймал {capture}% top movers, вовремя {early}%; {capture_piece}.",
        f"Где теряем: blocked winners {blocked}; {miss_piece}; exit efficiency median {_fmt(latest.get('median_exit_efficiency'), 2)}.",
        "",
        "📋 Прошлые решения:",
    ]
    for item in decisions[:4]:
        lines.append(f"  • {item['name']} — {item['status']} · {item['impact']}")
    if not decisions:
        lines.append("  • нет зафиксированных решений с измеримым эффектом")
    lines.extend(["", f"🚨 {len(alerts)} сигнал(ов) тревоги" + (_serious_suffix(alerts))])
    for alert in alerts[:4]:
        lines.append(f"  • {alert['severity']}: {alert['text']}")
    lines.extend(["", "🧠 Контур обучения:"])
    for name, comp in components.items():
        lines.append(f"  • {comp['label']}: {comp['status']} — {comp['detail']}")
    lines.extend(["", "🎯 ЧТО ДАЛЬШЕ"])
    for action in actions[:4]:
        lines.append(f"{action}")
    return "\n".join(lines).strip()


def _load_day_metrics(reports_dir: Path) -> list[DayMetrics]:
    by_day: dict[str, dict[str, Any]] = {}
    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        data = _load_json(path)
        day = str(data.get("target_day_local") or "")
        if not day:
            continue
        summary = data.get("summary") or {}
        by_day.setdefault(day, {})["critic"] = summary
    for path in sorted(reports_dir.glob("signal_quality_*_final.json")):
        data = _load_json(path)
        day = _day_from_signal_quality_path(path, data)
        if not day:
            continue
        by_day.setdefault(day, {})["signal_quality"] = data
    out = []
    for day in sorted(by_day):
        critic = by_day[day].get("critic") or {}
        sq = by_day[day].get("signal_quality") or {}
        sq_summary = sq.get("summary") or {}
        coverage = sq.get("coverage") or {}
        out.append(
            DayMetrics(
                day=day,
                watchlist_top_count=int(critic.get("watchlist_top_count") or 0),
                bought=int(critic.get("watchlist_top_bought") or 0),
                missed=int(critic.get("watchlist_top_missed") or 0),
                early=int(critic.get("watchlist_top_early_captured") or 0),
                false_positive_buys=int(critic.get("bot_false_positive_buys") or 0),
                capture_pct=float(critic.get("watchlist_top_capture_rate_pct") or 0.0),
                early_pct=float(critic.get("watchlist_top_early_capture_rate_pct") or 0.0),
                blocked_winner_count=int(critic.get("blocked_winner_count") or 0),
                blocked_symbols=tuple(critic.get("blocked_winner_symbols") or ()),
                blocked_reasons=dict(critic.get("blocked_winner_reason_counts") or {}),
                miss_rate=_maybe_float(sq_summary.get("miss_rate")),
                false_positive_rate=_maybe_float(sq_summary.get("false_positive_rate")),
                median_capture_ratio=_metric_median(sq_summary, "capture_ratio_at_entry"),
                median_exit_efficiency=_metric_median(sq_summary, "exit_efficiency"),
                median_giveback_pct=_metric_median(sq_summary, "giveback_pct"),
                coverage_status=str(coverage.get("status") or "unknown"),
                coverage_reasons=tuple(str(x) for x in (coverage.get("reasons") or ())),
            )
        )
    return out


def _rolling_summary(days: list[DayMetrics]) -> dict[str, Any]:
    last7 = days[-7:]
    prev7 = days[-14:-7]
    return {
        "early_last7_pct": _avg([d.early_pct for d in last7]),
        "early_prev7_pct": _avg([d.early_pct for d in prev7]),
        "capture_last7_pct": _avg([d.capture_pct for d in last7]),
        "capture_prev7_pct": _avg([d.capture_pct for d in prev7]),
        "miss_rate_last7_pct": _avg([d.miss_rate * 100 for d in last7 if d.miss_rate is not None]),
        "blocked_winners_last7": sum(d.blocked_winner_count for d in last7),
        "n_last7": len(last7),
        "n_prev7": len(prev7),
    }


def _verdict(latest: DayMetrics, previous: list[DayMetrics], older: list[DayMetrics], status: dict[str, Any]) -> dict[str, str]:
    early_recent = _avg([d.early_pct for d in previous[-7:]] + [latest.early_pct])
    early_old = _avg([d.early_pct for d in older])
    training = (((status.get("training") or {}).get("last_finished_at")) or "")
    stale_training = bool(training and training[:10] < latest.day)
    if latest.coverage_status not in {"ok", "complete"}:
        return {"label": "СТАТУС НЕПОЛНЫЙ", "emoji": "🟡", "operator_hint": "сначала проверь покрытие данных"}
    if stale_training and early_recent <= early_old + 1.0:
        return {"label": "СТОИТ НА МЕСТЕ", "emoji": "🟠", "operator_hint": "нужно чинить обучение/гейты"}
    if early_recent >= early_old + 2.0:
        return {"label": "РАЗВИВАЕТСЯ (медленно)", "emoji": "📈", "operator_hint": "от тебя ничего не требуется"}
    if early_recent + 2.0 < early_old:
        return {"label": "ДЕГРАДИРУЕТ", "emoji": "📉", "operator_hint": "нужно остановить авто-изменения"}
    return {"label": "СТОИТ НА МЕСТЕ", "emoji": "➡️", "operator_hint": "ждать нельзя, нужны узкие проверки"}


def _learning_components(status: dict[str, Any], feedback: dict[str, Any], latest_day: str) -> dict[str, dict[str, str]]:
    training = status.get("training") or {}
    critic = status.get("top_gainer_critic") or {}
    sq = status.get("signal_quality_evaluator") or {}
    fb_policy = feedback.get("policy") or feedback
    last_train = str(training.get("last_finished_at") or "")
    return {
        "measurement": {
            "label": "measurement",
            "status": "ok" if critic.get("last_target_day_local") == latest_day and sq.get("last_target_day_local") == latest_day else "stale/partial",
            "detail": f"critic={critic.get('last_target_day_local')}, signal_quality={sq.get('last_target_day_local')}",
        },
        "feedback": {
            "label": "feedback",
            "status": "active" if fb_policy.get("apply_cooldown_relaxation") else "watch-only",
            "detail": str(feedback.get("reason") or fb_policy.get("reason") or "no active policy"),
        },
        "ranker_training": {
            "label": "ML ranker training",
            "status": "stale" if last_train[:10] < latest_day else "fresh",
            "detail": f"last_finished={last_train or 'never'}, rows={training.get('last_rows_total')}",
        },
    }


def _previous_decisions(feedback: dict[str, Any], status: dict[str, Any], latest_day: str) -> list[dict[str, str]]:
    out = []
    if (feedback.get("policy") or feedback).get("apply_cooldown_relaxation"):
        out.append({"name": "cooldown relaxation", "status": "применено", "impact": "рано судить; нужно сравнить cooldown_harm и false positives"})
    out.append({"name": "top-gainer score gate 34", "status": "оставлен", "impact": "защищает precision, но блокирует часть winners; нужен blocked-winner audit"})
    out.append({"name": "delayed +120m confirmation", "status": "отклонено", "impact": "правильно: top15 предсказывает, но вход после +120m убыточен"})
    training = (status.get("training") or {})
    if str(training.get("last_finished_at") or "")[:10] < latest_day:
        out.append({"name": "ML ranker retraining", "status": "не обновляется", "impact": "это не self-improvement; нужен repair"})
    return out


def _alerts(latest: DayMetrics, status: dict[str, Any], feedback: dict[str, Any], focus_findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    if latest.coverage_status not in {"ok", "complete"}:
        out.append({"severity": "serious", "text": f"coverage={latest.coverage_status}: {', '.join(latest.coverage_reasons[:2])}"})
    if latest.early_pct < 15.0:
        out.append({"severity": "serious", "text": f"early capture only {latest.early_pct:.1f}% vs 25%+ target"})
    if latest.miss_rate is not None and latest.miss_rate > 0.80:
        out.append({"severity": "serious", "text": f"trend miss-rate {latest.miss_rate * 100:.1f}%"})
    if latest.blocked_winner_count >= 5:
        out.append({"severity": "warn", "text": f"blocked winners={latest.blocked_winner_count}: {', '.join(latest.blocked_symbols[:6])}"})
    training = status.get("training") or {}
    if str(training.get("last_finished_at") or "")[:10] < latest.day:
        out.append({"severity": "serious", "text": f"ML ranker training stale: last {training.get('last_finished_at')}"})
    blocked_focus = [x["symbol"] for x in focus_findings if x.get("status") in {"blocked_rule", "missed", "not_bought"}]
    if blocked_focus:
        out.append({"severity": "warn", "text": "focus top movers blocked/missed: " + ", ".join(blocked_focus[:8])})
    return out


def _next_actions(latest: DayMetrics, status: dict[str, Any], feedback: dict[str, Any], focus_findings: list[dict[str, Any]]) -> list[str]:
    actions = []
    training = status.get("training") or {}
    if str(training.get("last_finished_at") or "")[:10] < latest.day:
        actions.append("▶️ Починить ML/ranker retraining freshness: сейчас модель не доказывает ежедневное обучение.")
    if latest.blocked_winner_count:
        actions.append("▶️ Запустить blocked-winner audit по top_gainer_score_gate / agent_mode_disabled / cooldown, без production relax.")
    if latest.median_exit_efficiency is not None and latest.median_exit_efficiency < 0:
        actions.append("▶️ Продолжить exit-quality auditor: входы монетизируются плохо, менять SELL только через replay.")
    if not actions:
        actions.append("⏸️ Пока одобрять нечего — ждём следующего final report и replay evidence.")
    return actions


def _focus_findings(reports_dir: Path, latest_day: str, focus: list[str]) -> list[dict[str, Any]]:
    if not focus or latest_day == "unknown":
        return []
    path = reports_dir / f"top_gainer_critic_{latest_day}_final.json"
    data = _load_json(path)
    rows = []
    for section in ("watchlist_top_gainers", "exchange_top_gainers"):
        for row in data.get(section) or []:
            if str(row.get("symbol")) in focus:
                rows.append({
                    "symbol": row.get("symbol"),
                    "section": section,
                    "status": row.get("status"),
                    "reason": row.get("reason") or row.get("missed_reason_code"),
                    "entries_count": row.get("entries_count"),
                    "blocked_count": row.get("blocked_count"),
                    "capture_ratio_at_entry": row.get("capture_ratio_at_entry"),
                })
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _day_from_signal_quality_path(path: Path, data: dict[str, Any]) -> str:
    name = path.name
    parts = name.split("_")
    for part in parts:
        if len(part) == 10 and part[4] == "-":
            return part
    window = data.get("window") or {}
    end = str(window.get("end") or "")[:10]
    return end


def _metric_median(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict) and value.get("median") is not None:
        return _maybe_float(value.get("median"))
    return None


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if value == value else None
    except Exception:
        return None


def _avg(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 6) if vals else 0.0


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "н/д"


def _serious_suffix(alerts: list[dict[str, str]]) -> str:
    serious = sum(1 for a in alerts if a.get("severity") == "serious")
    return f" ({serious} серьёзн.)" if serious else ""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--status-file", type=Path, default=STATUS_FILE)
    parser.add_argument("--feedback-file", type=Path, default=FEEDBACK_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--focus-symbols", default="")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    focus = [s.strip() for s in args.focus_symbols.replace("/", "").split(",") if s.strip()]
    report = build_report(args.reports_dir, args.status_file, args.feedback_file, focus, args.output_json, args.output_txt)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
