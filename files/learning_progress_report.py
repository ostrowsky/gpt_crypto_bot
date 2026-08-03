from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import report_signal_quality_coverage
import report_entry_admission_shadow_reward
import report_blocked_winner_causal_reward
import report_portfolio_replacement_shadow_reward
import replay_observable_tail_selector
import research_artifact_provenance as artifact_provenance


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
STATUS_FILE = WORKSPACE_ROOT / ".runtime" / "rl_worker_status.json"
FEEDBACK_FILE = WORKSPACE_ROOT / ".runtime" / "signal_quality_feedback.json"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "learning_progress_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "learning_progress_latest.txt"
SHADOW_REENTRY_SCORECARD_LATEST = REPORT_DIR / "suspicious_reentry_scorecard_latest.json"
TAIL_SELECTOR_RESEARCH_CONFIG = replay_observable_tail_selector.ObservableSelectorConfig()
ENTRY_ADMISSION_RESEARCH_CONFIG = report_entry_admission_shadow_reward.RewardConfig()
BLOCKER_REWARD_RESEARCH_CONFIG = report_blocked_winner_causal_reward.BlockerRewardConfig()
PORTFOLIO_REPLACEMENT_RESEARCH_CONFIG = report_portfolio_replacement_shadow_reward.ReplacementConfig()


@dataclass(frozen=True)
class DayMetrics:
    day: str
    critic_present: bool = True
    signal_quality_present: bool = True
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
    coverage_assessment: str = "unknown"
    missing_series_count: int = 0
    missing_symbol_status_counts: dict[str, int] | None = None


def build_report(
    reports_dir: Path = REPORT_DIR,
    status_file: Path = STATUS_FILE,
    feedback_file: Path = FEEDBACK_FILE,
    focus_symbols: Iterable[str] = (),
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
) -> dict[str, Any]:
    days = _load_day_metrics(reports_dir)
    days = _apply_latest_coverage_triage(days, reports_dir)
    latest = days[-1] if days else DayMetrics(day="unknown")
    previous = days[-8:-1]
    older = days[-15:-8]
    status = _load_json(status_file)
    feedback = _load_json(feedback_file)
    shadow_reentry = _load_json(reports_dir / SHADOW_REENTRY_SCORECARD_LATEST.name)
    shadow_tail_selector = _build_shadow_tail_selector_summary(reports_dir)
    shadow_entry_admission = _build_shadow_entry_admission_summary(reports_dir)
    blocker_reward = _build_blocker_reward_summary(reports_dir)
    portfolio_replacement = _build_portfolio_replacement_summary(reports_dir)
    focus = sorted({str(s).upper().replace("/", "").replace("USDT", "") + "USDT" for s in focus_symbols if str(s).strip()})
    focus_findings = _focus_findings(reports_dir, latest.day, focus)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "latest_day": latest.day,
        "days_loaded": len(days),
        "verdict": _verdict(latest, previous, older, status),
        "latest": latest.__dict__,
        "rolling": _rolling_summary(days),
        "learning_components": _learning_components(status, feedback, latest.day, reports_dir),
        "shadow_reentry": _shadow_reentry_summary(shadow_reentry),
        "shadow_tail_selector": shadow_tail_selector,
        "shadow_entry_admission": shadow_entry_admission,
        "blocker_reward": blocker_reward,
        "portfolio_replacement": portfolio_replacement,
        "data_confidence": _data_confidence(
            latest,
            reports_dir,
            shadow_tail_selector,
            shadow_entry_admission,
            blocker_reward,
            portfolio_replacement,
        ),
        "previous_decisions": _previous_decisions(feedback, status, latest.day),
        "alerts": _alerts(latest, status, feedback, focus_findings, shadow_reentry, shadow_tail_selector, shadow_entry_admission, blocker_reward, portfolio_replacement),
        "focus_symbols": focus_findings,
        "next_actions": _next_actions(latest, status, feedback, focus_findings, shadow_reentry, shadow_tail_selector, shadow_entry_admission, blocker_reward, portfolio_replacement),
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
    shadow_reentry = report.get("shadow_reentry") or {}
    shadow_tail_selector = report.get("shadow_tail_selector") or {}
    shadow_entry_admission = report.get("shadow_entry_admission") or {}
    blocker_reward = report.get("blocker_reward") or {}
    portfolio_replacement = report.get("portfolio_replacement") or {}
    data_confidence = report.get("data_confidence") or {}
    day = report.get("latest_day") or "unknown"
    emoji = verdict.get("emoji", "⚪")
    title = verdict.get("label", "СТАТУС НЕЯСЕН")
    ask = verdict.get("operator_hint", "проверь полный отчёт")
    confidence = verdict.get("confidence", "unknown")
    confidence_reason = verdict.get("confidence_reason", "")
    early_now = _fmt(rolling.get("early_last7_pct"), 1)
    early_prev = _fmt(rolling.get("early_prev7_pct"), 1)
    capture = _fmt(latest.get("capture_pct"), 1)
    early = _fmt(latest.get("early_pct"), 1)
    median_capture = latest.get("median_capture_ratio")
    capture_piece = "нет данных" if median_capture is None else f"оставалось медианно {float(median_capture) * 100:.0f}% дневного движения после входа"
    blocked = latest.get("blocked_winner_count") or 0
    miss_rate = latest.get("miss_rate")
    miss_piece = "miss-rate нет" if miss_rate is None else f"miss-rate {float(miss_rate) * 100:.0f}%"
    if not bool(latest.get("critic_present", True)):
        yesterday_line = (
            "Вчера: top-mover denominator недоступен — final critic не создан; "
            f"{capture_piece}."
        )
    elif int(latest.get("watchlist_top_count") or 0) == 0:
        yesterday_line = (
            "Вчера: watchlist top movers: 0 — метрика дня не применима; "
            f"{capture_piece}."
        )
    else:
        yesterday_line = f"Вчера: поймал {capture}% top movers, вовремя {early}%; {capture_piece}."
    lines = [
        f"Бот — {day}",
        "",
        f"{emoji} {title}   ·   confidence={confidence}   ·   👉 {ask}",
        "",
        f"Главное: early-capture ~{early_now}% за 7д ({early_prev}% → {early_now}%). Цель — 25%+.",
        yesterday_line,
        f"Где теряем: blocked winners {blocked}; {miss_piece}; exit efficiency median {_fmt(latest.get('median_exit_efficiency'), 2)}.",
        "",
    ]
    if confidence_reason:
        lines.insert(4, f"Основание: {confidence_reason}")
    if data_confidence:
        lines.extend(["", "🔎 Достоверность данных:"])
        for item in data_confidence.get("items", [])[:5]:
            lines.append(f"  • {item.get('label')}: {item.get('status')} — {item.get('detail')}")
    lines.extend(["", "📋 Прошлые решения:"])
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
    lines.append(
        "  • shadow re-entry: "
        f"{shadow_reentry.get('status', 'unknown')} — "
        f"{shadow_reentry.get('detail', 'нет отчёта')}"
    )
    lines.append(
        "  • shadow tail selector: "
        f"{shadow_tail_selector.get('status', 'unknown')} — "
        f"{shadow_tail_selector.get('detail', 'нет отчёта')}"
    )
    lines.append(
        "  • shadow entry admission: "
        f"{shadow_entry_admission.get('status', 'unknown')} — "
        f"{shadow_entry_admission.get('detail', 'нет отчёта')}"
    )
    lines.append(
        "  • blocker reward: "
        f"{blocker_reward.get('status', 'unknown')} — "
        f"{blocker_reward.get('detail', 'нет отчёта')}"
    )
    lines.append(
        "  • portfolio replacement: "
        f"{portfolio_replacement.get('status', 'unknown')} — "
        f"{portfolio_replacement.get('detail', 'нет отчёта')}"
    )
    lines.extend(["", "🎯 ЧТО ДАЛЬШЕ"])
    for action in actions[:6]:
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
        critic_present = "critic" in by_day[day]
        signal_quality_present = "signal_quality" in by_day[day]
        critic = by_day[day].get("critic") or {}
        sq = by_day[day].get("signal_quality") or {}
        sq_summary = sq.get("summary") or {}
        coverage = sq.get("coverage") or {}
        out.append(
            DayMetrics(
                day=day,
                critic_present=critic_present,
                signal_quality_present=signal_quality_present,
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


def _apply_latest_coverage_triage(days: list[DayMetrics], reports_dir: Path) -> list[DayMetrics]:
    if not days:
        return days
    latest = days[-1]
    path = reports_dir / f"signal_quality_{latest.day}_final.json"
    if not path.exists() or latest.coverage_status in {"ok", "complete"}:
        return days
    runtime_dir = reports_dir.parent
    workspace_dir = runtime_dir.parent
    watchlist_path = workspace_dir / "files" / "watchlist.json"
    try:
        triage = report_signal_quality_coverage.build_report(
            signal_report=path,
            watchlist_path=watchlist_path if watchlist_path.exists() else report_signal_quality_coverage.WATCHLIST,
            cache_dir=runtime_dir / "signal_quality_cache",
            exchange_status_cache=runtime_dir / "exchange_symbol_status.json",
            output_json=reports_dir / "signal_quality_coverage_latest.json",
            output_txt=reports_dir / "signal_quality_coverage_latest.txt",
            save=True,
        )
    except Exception:
        return days
    updated = replace(
        latest,
        coverage_assessment=str(triage.get("assessment") or "unknown"),
        missing_series_count=int(triage.get("missing_series_count") or 0),
        missing_symbol_status_counts=dict(triage.get("missing_symbol_status_counts") or {}),
    )
    return [*days[:-1], updated]


def _rolling_summary(days: list[DayMetrics]) -> dict[str, Any]:
    last7 = days[-7:]
    prev7 = days[-14:-7]
    last7_top = _days_with_top_denominator(last7)
    prev7_top = _days_with_top_denominator(prev7)
    return {
        "early_last7_pct": _top_rate(last7_top, "early_pct"),
        "early_prev7_pct": _top_rate(prev7_top, "early_pct"),
        "capture_last7_pct": _top_rate(last7_top, "capture_pct"),
        "capture_prev7_pct": _top_rate(prev7_top, "capture_pct"),
        "miss_rate_last7_pct": _avg([d.miss_rate * 100 for d in last7 if d.miss_rate is not None]),
        "blocked_winners_last7": sum(d.blocked_winner_count for d in last7),
        "n_last7": len(last7),
        "n_prev7": len(prev7),
        "n_last7_top_days": len(last7_top),
        "n_prev7_top_days": len(prev7_top),
    }


def _days_with_top_denominator(days: Iterable[DayMetrics]) -> list[DayMetrics]:
    return [d for d in days if int(d.watchlist_top_count or 0) > 0]


def _top_rate(days: Iterable[DayMetrics], rate_field: str) -> float | None:
    rows = list(days)
    denominator = sum(int(d.watchlist_top_count or 0) for d in rows)
    if denominator <= 0:
        return None
    weighted = sum(float(getattr(d, rate_field, 0.0) or 0.0) * int(d.watchlist_top_count or 0) for d in rows)
    return round(weighted / denominator, 2)


def _verdict(latest: DayMetrics, previous: list[DayMetrics], older: list[DayMetrics], status: dict[str, Any]) -> dict[str, str]:
    recent_top_days = _days_with_top_denominator([*previous[-7:], latest])
    older_top_days = _days_with_top_denominator(older)
    early_recent = _top_rate(recent_top_days, "early_pct")
    early_old = _top_rate(older_top_days, "early_pct")
    training = (((status.get("training") or {}).get("last_finished_at")) or "")
    stale_training = bool(training and training[:10] < latest.day)
    confidence, confidence_reason = _verdict_confidence(latest, previous, older)
    extra = {"confidence": confidence, "confidence_reason": confidence_reason}
    if not latest.critic_present:
        return {"label": "СТАТУС НЕПОЛНЫЙ", "emoji": "🟡", "operator_hint": "починить final top-gainer critic", **extra}
    if latest.coverage_status not in {"ok", "complete"} and not _coverage_is_safe_partial(latest):
        return {"label": "СТАТУС НЕПОЛНЫЙ", "emoji": "🟡", "operator_hint": "сначала проверь покрытие данных", **extra}
    if latest.watchlist_top_count <= 0:
        if early_recent is not None and early_old is not None and early_recent >= early_old + 2.0:
            return {"label": 'ROLLING УЛУЧШАЕТСЯ, ДЕНЬ НЕИНФОРМАТИВЕН', "emoji": "🟡", "operator_hint": 'не принимать решений по пустому дню; фокус — exit', **extra}
        return {"label": 'ДЕНЬ НЕИНФОРМАТИВЕН', "emoji": "🟡", "operator_hint": 'ждать валидный denominator; проверять exit', **extra}
    if early_recent is None or early_old is None:
        return {"label": 'СТАТУС НЕПОЛНЫЙ', "emoji": "🟡", "operator_hint": 'нужно больше валидных top-mover дней', **extra}
    if stale_training and early_recent <= early_old + 1.0:
        return {"label": "СТОИТ НА МЕСТЕ", "emoji": "🟠", "operator_hint": "нужно чинить обучение/гейты", **extra}
    if early_recent >= early_old + 2.0:
        return {"label": "РАЗВИВАЕТСЯ ПО ЦЕЛЕВОЙ МЕТРИКЕ", "emoji": "📈", "operator_hint": "фокус — монетизация выходов", **extra}
    if early_recent + 2.0 < early_old:
        if confidence == "high":
            return {"label": "ДЕГРАДИРУЕТ", "emoji": "📉", "operator_hint": "остановить авто-изменения и проверить replay", **extra}
        return {"label": "УХУДШИЛСЯ ПО EARLY-CAPTURE", "emoji": "🟠", "operator_hint": "нужны replay-проверки, не авто-изменения", **extra}
    return {"label": "СТОИТ НА МЕСТЕ", "emoji": "➡️", "operator_hint": "ждать нельзя, нужны узкие проверки", **extra}


def _verdict_confidence(latest: DayMetrics, previous: list[DayMetrics], older: list[DayMetrics]) -> tuple[str, str]:
    reasons: list[str] = []
    if not latest.critic_present:
        reasons.append("final top-gainer critic missing")
    if latest.watchlist_top_count < 5:
        reasons.append(f"малый denominator дня: watchlist_top={latest.watchlist_top_count}")
    if len(previous) < 6 or len(older) < 6:
        reasons.append(f"короткое rolling окно: prev={len(previous)}, older={len(older)}")
    report_days = [d.day for d in [*older, *previous, latest] if d.day and d.day != "unknown"]
    if len(report_days) >= 2:
        try:
            parsed = [date.fromisoformat(day) for day in report_days]
            span = (max(parsed) - min(parsed)).days + 1
            if span > len(set(parsed)) + 3:
                reasons.append(f"разреженные отчётные дни: reports={len(set(parsed))}/{span} calendar days")
        except Exception:
            pass
    if latest.coverage_status not in {"ok", "complete"} and not _coverage_is_safe_partial(latest):
        reasons.append(f"coverage={latest.coverage_status}")
    if reasons:
        return "medium" if len(reasons) == 1 else "low", "; ".join(reasons)
    return "high", "достаточный denominator и плотное rolling окно"


def _learning_components(
    status: dict[str, Any],
    feedback: dict[str, Any],
    latest_day: str,
    reports_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    training = status.get("training") or {}
    critic = status.get("top_gainer_critic") or {}
    sq = status.get("signal_quality_evaluator") or {}
    fb_policy = feedback.get("policy") or feedback
    last_train = str(training.get("last_finished_at") or "")
    critic_day = str(critic.get("last_target_day_local") or "")
    sq_day = str(sq.get("last_target_day_local") or "")
    if reports_dir is not None:
        if not critic_day and (reports_dir / f"top_gainer_critic_{latest_day}_final.json").exists():
            critic_day = latest_day
        if not sq_day and (reports_dir / f"signal_quality_{latest_day}_final.json").exists():
            sq_day = latest_day
    critic_piece = critic_day or "missing"
    sq_piece = sq_day or "missing"
    return {
        "measurement": {
            "label": "measurement",
            "status": "ok" if _day_not_older(critic_day, latest_day) and _day_not_older(sq_day, latest_day) else "stale/partial",
            "detail": f"critic={critic_piece}, signal_quality={sq_piece}",
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


def _data_confidence(
    latest: DayMetrics,
    reports_dir: Path,
    shadow_tail_selector: dict[str, Any],
    shadow_entry_admission: dict[str, Any],
    blocker_reward: dict[str, Any],
    portfolio_replacement: dict[str, Any],
) -> dict[str, Any]:
    critic_path = reports_dir / f"top_gainer_critic_{latest.day}_final.json"
    signal_path = reports_dir / f"signal_quality_{latest.day}_final.json"
    if not latest.critic_present:
        denominator_status = "unknown"
        denominator_detail = "final critic missing; watchlist_top_count нельзя трактовать как 0"
    elif latest.watchlist_top_count <= 0:
        denominator_status = "empty"
        denominator_detail = "critic fresh enough, но watchlist top movers отсутствуют"
    else:
        denominator_status = "ok"
        denominator_detail = f"watchlist_top={latest.watchlist_top_count}"
    research_stale = [
        name
        for name, component in (
            ("tail_selector", shadow_tail_selector),
            ("entry_admission", shadow_entry_admission),
            ("blocker_reward", blocker_reward),
            ("portfolio_replacement", portfolio_replacement),
        )
        if (component or {}).get("stale")
    ]
    items = [
        {
            "label": "top-gainer critic",
            "status": "fresh" if latest.critic_present else "missing",
            "detail": str(critic_path.name if latest.critic_present else "нет final critic за день"),
        },
        {
            "label": "signal-quality",
            "status": str(latest.coverage_status or "unknown"),
            "detail": str(signal_path.name if latest.signal_quality_present else "нет signal-quality final за день"),
        },
        {
            "label": "top-mover denominator",
            "status": denominator_status,
            "detail": denominator_detail,
        },
        {
            "label": "research replays",
            "status": "stale" if research_stale else "fresh",
            "detail": ", ".join(research_stale) if research_stale else "optional research artifacts fresh enough",
        },
    ]
    return {
        "status": "decision_grade" if latest.critic_present and latest.signal_quality_present and denominator_status == "ok" and not research_stale else "diagnostic_only",
        "items": items,
        "research_stale_components": research_stale,
    }


def _alerts(
    latest: DayMetrics,
    status: dict[str, Any],
    feedback: dict[str, Any],
    focus_findings: list[dict[str, Any]],
    shadow_reentry: dict[str, Any],
    shadow_tail_selector: dict[str, Any] | None = None,
    shadow_entry_admission: dict[str, Any] | None = None,
    blocker_reward: dict[str, Any] | None = None,
    portfolio_replacement: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    out = []
    if not latest.critic_present:
        out.append({"severity": "serious", "text": "top-gainer critic final missing; watchlist_top denominator unknown"})
    if latest.coverage_status not in {"ok", "complete"}:
        if _coverage_is_safe_partial(latest):
            counts = latest.missing_symbol_status_counts or {}
            out.append({
                "severity": "warn",
                "text": (
                    f"coverage={latest.coverage_status}, но triage={latest.coverage_assessment}; "
                    f"missing={latest.missing_series_count}, statuses={counts}"
                ),
            })
        else:
            out.append({"severity": "serious", "text": f"coverage={latest.coverage_status}: {', '.join(latest.coverage_reasons[:2])}"})
    if latest.watchlist_top_count > 0 and latest.early_pct < 15.0:
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
    reentry_summary = shadow_reentry.get("summary") or {}
    reentry_labeled = int(reentry_summary.get("labeled_ret5") or 0)
    reentry_avg = _maybe_float(reentry_summary.get("avg_ret5"))
    if reentry_labeled >= 5 and reentry_avg is not None and reentry_avg < 0:
        out.append({"severity": "warn", "text": f"shadow re-entry noisy: labeled={reentry_labeled}, avg_ret5={reentry_avg:+.2f}%"})
    tail = shadow_tail_selector or {}
    if tail.get("status") == "error":
        out.append({"severity": "warn", "text": f"shadow tail selector report failed: {tail.get('detail', 'unknown error')}"})
    entry = shadow_entry_admission or {}
    if entry.get("status") == "error":
        out.append({"severity": "warn", "text": f"shadow entry admission report failed: {entry.get('detail', 'unknown error')}"})
    blocker = blocker_reward or {}
    if blocker.get("status") == "error":
        out.append({"severity": "warn", "text": f"blocker reward report failed: {blocker.get('detail', 'unknown error')}"})
    replacement = portfolio_replacement or {}
    if replacement.get("status") == "error":
        out.append({"severity": "warn", "text": f"portfolio replacement report failed: {replacement.get('detail', 'unknown error')}"})
    return out


def _next_actions(
    latest: DayMetrics,
    status: dict[str, Any],
    feedback: dict[str, Any],
    focus_findings: list[dict[str, Any]],
    shadow_reentry: dict[str, Any],
    shadow_tail_selector: dict[str, Any] | None = None,
    shadow_entry_admission: dict[str, Any] | None = None,
    blocker_reward: dict[str, Any] | None = None,
    portfolio_replacement: dict[str, Any] | None = None,
) -> list[str]:
    actions = []
    training = status.get("training") or {}
    if not latest.critic_present:
        actions.append("▶️ Починить/перезапустить final top-gainer critic: без него watchlist_top=0 недостоверен.")
    if str(training.get("last_finished_at") or "")[:10] < latest.day:
        actions.append("▶️ Починить ML/ranker retraining freshness: сейчас модель не доказывает ежедневное обучение.")
    if latest.blocked_winner_count:
        actions.append("▶️ Запустить blocked-winner audit по top_gainer_score_gate / agent_mode_disabled / cooldown, без production relax.")
    if latest.median_exit_efficiency is not None and latest.median_exit_efficiency < 0:
        actions.append("▶️ Продолжить exit-quality auditor: входы монетизируются плохо, менять SELL только через replay.")
    if (
        latest.early_pct >= 25.0
        and latest.median_exit_efficiency is not None
        and latest.median_giveback_pct is not None
        and (latest.median_exit_efficiency < 0.25 or latest.median_giveback_pct >= 3.0)
    ):
        actions.append("▶️ Запустить exit monetization replay по вчерашним watchlist top movers: high-MFE/giveback, re-entry и partial-exit гипотезы без production SELL changes.")
    reentry_summary = shadow_reentry.get("summary") or {}
    reentry_labeled = int(reentry_summary.get("labeled_ret5") or 0)
    reentry_avg = _maybe_float(reentry_summary.get("avg_ret5"))
    reentry_positive = _maybe_float(reentry_summary.get("ret5_positive_rate"))
    if reentry_labeled < 10:
        actions.append("⏳ Shadow re-entry: продолжать сбор labels; production re-entry не включать.")
    elif reentry_avg is not None and reentry_positive is not None and reentry_avg > 0.25 and reentry_positive >= 0.55:
        actions.append("▶️ Shadow re-entry выглядит promising: подготовить replay-gated production spec, но не включать напрямую.")
    elif reentry_avg is not None and reentry_avg < 0:
        actions.append("⏸️ Shadow re-entry шумит: разобрать false re-entry before any production policy.")
    tail = shadow_tail_selector or {}
    if tail.get("status") == "passed_shadow_gate":
        actions.append("⏳ Shadow tail selector прошёл replay-gate: собирать daily labels, production SELL не менять.")
    elif tail.get("status") == "failed_gate":
        actions.append("▶️ Shadow tail selector пока не проходит gate: продолжить observable feature search для exit monetization.")
    elif tail.get("status") == "stale":
        actions.append("▶️ Пересобрать shadow tail selector с current-policy provenance; stale evidence не использовать для решений.")
    elif tail.get("status") in {"missing", "error"}:
        actions.append("▶️ Починить shadow tail selector report: exit-learning контур неполный.")
    entry = shadow_entry_admission or {}
    if entry.get("status") == "passed_shadow_gate":
        actions.append("▶️ Entry admission shadow reward положительный: готовить candle-level behavior replay, BUY не менять.")
    elif entry.get("status") == "no_positive_reward":
        actions.append("⏸️ Entry admission: текущие rescue-гипотезы не дают positive reward; BUY-гейты не расширять.")
    elif entry.get("status") == "stale":
        actions.append("▶️ Пересобрать entry admission на текущей policy; stale evidence не расширяет BUY-гейты.")
    elif entry.get("status") in {"missing", "error"}:
        actions.append("▶️ Починить entry admission shadow reward report: admission-learning контур неполный.")
    blocker = blocker_reward or {}
    if blocker.get("status") == "passed_harm_gate":
        top = blocker.get("top") or {}
        net_harm = _maybe_float(top.get("net_harm_pct"))
        if net_harm is not None and net_harm < 5.0:
            actions.append(
                "▶️ Blocker reward нашёл слабый blocker-кандидат: только targeted replay; "
                "гейты не расслаблять напрямую."
            )
        else:
            actions.append(
                "▶️ Blocker reward нашёл сильный blocker-кандидат: готовить targeted behavior replay; "
                "гейты не расслаблять напрямую."
            )
    elif blocker.get("status") == "monitor":
        actions.append("⏸️ Blocker reward: явного вредного blocker-а нет; не расслаблять blockers без targeted replay.")
    elif blocker.get("status") == "stale":
        actions.append("▶️ Пересобрать blocker reward на текущей policy; stale evidence не расслабляет blockers.")
    elif blocker.get("status") in {"missing", "error"}:
        actions.append("▶️ Починить blocker reward report: blocker-learning контур неполный.")
    replacement = portfolio_replacement or {}
    if replacement.get("status") == "policy_candidate":
        actions.append("▶️ Portfolio replacement: есть policy-кандидат; готовить behavior replay, live rotation не менять.")
    elif replacement.get("status") == "passed_shadow_gate":
        actions.append("▶️ Portfolio replacement shadow reward положительный: готовить counterfactual replay, live rotation не менять.")
    elif replacement.get("status") == "hurting":
        actions.append("⚠️ Portfolio replacement выглядит вредным: разобрать rotation outcomes перед любыми новыми заменами.")
    elif replacement.get("status") == "collecting":
        actions.append("⏳ Portfolio replacement: мало закрытых replacement outcomes; продолжить сбор без изменения live rotation.")
    elif replacement.get("status") == "stale":
        actions.append("▶️ Пересобрать portfolio replacement на текущей policy; stale evidence не меняет live rotation.")
    elif replacement.get("status") in {"missing", "error"}:
        actions.append("▶️ Починить portfolio replacement report: rotation-learning контур неполный.")
    if not actions:
        actions.append("⏸️ Пока одобрять нечего — ждём следующего final report и replay evidence.")
    return actions


def _build_shadow_tail_selector_summary(reports_dir: Path) -> dict[str, Any]:
    cached, stale = _load_cached_json_with_staleness(
        reports_dir / "observable_tail_selector_replay_latest.json",
        expected_builder="observable_tail_selector_replay_v1",
        expected_research_config=TAIL_SELECTOR_RESEARCH_CONFIG,
    )
    if cached:
        return _with_cache_staleness(_shadow_tail_selector_summary_from_report(cached), stale)
    try:
        report = replay_observable_tail_selector.build_replay(
            reports_dir=reports_dir,
            cache_dir=reports_dir.parent / "signal_quality_cache",
            output=reports_dir / "observable_tail_selector_replay_latest.json",
            text_output=reports_dir / "observable_tail_selector_replay_latest.txt",
            save=True,
        )
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return _shadow_tail_selector_summary_from_report(report)


def _shadow_tail_selector_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    ranked = report.get("ranked_selectors") or []
    if not ranked:
        return {"status": "missing", "detail": "нет selector candidates"}
    top = ranked[0]
    test = top.get("test") or {}
    decision = str(report.get("decision") or "")
    passed = decision.startswith("advance_")
    avg = _maybe_float(test.get("avg_delta_pct"))
    med = _maybe_float(test.get("median_delta_pct"))
    worse = _maybe_float(test.get("worse_rate_pct"))
    allow = _maybe_float(test.get("allowed_rate_pct"))
    fp_allow = _maybe_float(test.get("false_positive_allowed_rate_pct"))
    def pct(value: float | None, digits: int = 2) -> str:
        return "н/д" if value is None else f"{value:.{digits}f}"
    detail = (
        f"top={top.get('name')}, test_avg={pct(avg)}%, med={pct(med)}%, "
        f"worse={pct(worse, 1)}%, allow={pct(allow, 1)}%, fp_allow={pct(fp_allow, 1)}%"
    )
    return {
        "status": "passed_shadow_gate" if passed else "failed_gate",
        "detail": detail,
        "decision": decision,
        "top_selector": top.get("name"),
        "test": test,
        "files": report.get("files") or {},
        "artifact_freshness": report.get("_artifact_freshness") or {},
    }


def _build_shadow_entry_admission_summary(reports_dir: Path) -> dict[str, Any]:
    workspace_dir = reports_dir.parent.parent
    cached, stale = _load_cached_json_with_staleness(
        reports_dir / "entry_admission_shadow_reward_latest.json",
        expected_builder="entry_admission_shadow_reward_v1",
        expected_research_config=ENTRY_ADMISSION_RESEARCH_CONFIG,
    )
    if cached:
        return _with_cache_staleness(_shadow_entry_admission_summary_from_report(cached), stale)
    try:
        report = report_entry_admission_shadow_reward.build_report(
            reports_dir=reports_dir,
            files_dir=workspace_dir / "files",
            output_json=reports_dir / "entry_admission_shadow_reward_latest.json",
            output_txt=reports_dir / "entry_admission_shadow_reward_latest.txt",
            save=True,
        )
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return _shadow_entry_admission_summary_from_report(report)


def _shadow_entry_admission_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_variant") or {}
    if not best:
        return {"status": "missing", "detail": "нет admission candidates"}
    decision = str(report.get("decision") or "")
    net = _maybe_float(best.get("net_reward_pct"))
    precision = _maybe_float(best.get("top_precision"))
    candidates = int(best.get("candidate_count") or 0)
    top = int(best.get("top_candidates") or 0)
    false = int(best.get("false_candidates") or 0)
    status = "passed_shadow_gate" if decision.startswith("advance_") else "no_positive_reward"
    detail = (
        f"best={best.get('reason_set')}, net={_fmt(net, 2)}%, "
        f"precision={_fmt((precision or 0.0) * 100, 1)}%, candidates={candidates}, top={top}, false={false}"
    )
    return {
        "status": status,
        "detail": detail,
        "decision": decision,
        "best_variant": best,
        "files": report.get("files") or {},
        "artifact_freshness": report.get("_artifact_freshness") or {},
    }


def _build_blocker_reward_summary(reports_dir: Path) -> dict[str, Any]:
    workspace_dir = reports_dir.parent.parent
    cached, stale = _load_cached_json_with_staleness(
        reports_dir / "blocked_winner_causal_reward_latest.json",
        expected_builder="blocked_winner_causal_reward_v1",
        expected_research_config=BLOCKER_REWARD_RESEARCH_CONFIG,
    )
    if cached:
        return _with_cache_staleness(_blocker_reward_summary_from_report(cached), stale)
    try:
        report = report_blocked_winner_causal_reward.build_report(
            reports_dir=reports_dir,
            files_dir=workspace_dir / "files",
            output_json=reports_dir / "blocked_winner_causal_reward_latest.json",
            output_txt=reports_dir / "blocked_winner_causal_reward_latest.txt",
            save=True,
        )
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return _blocker_reward_summary_from_report(report)


def _blocker_reward_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    table = report.get("reason_table") or []
    if not table:
        return {"status": "missing", "detail": "нет blocker rows"}
    top = table[0]
    decision = str(report.get("decision") or "")
    status = "passed_harm_gate" if decision.startswith("advance_") else "monitor"
    detail = (
        f"top={top.get('reason_code')}, net_harm={_fmt(top.get('net_harm_pct'), 2)}%, "
        f"harm={_fmt(top.get('harm_pct'), 2)}%, protect={_fmt(top.get('protection_credit_pct'), 2)}%, "
        f"decision={top.get('decision')}"
    )
    net_harm = _maybe_float(top.get("net_harm_pct"))
    evidence_strength = "weak" if net_harm is not None and net_harm < 5.0 else "strong"
    return {
        "status": status,
        "detail": detail,
        "evidence_strength": evidence_strength,
        "decision": decision,
        "top_reason": top.get("reason_code"),
        "top": top,
        "files": report.get("files") or {},
        "artifact_freshness": report.get("_artifact_freshness") or {},
    }


def _build_portfolio_replacement_summary(reports_dir: Path) -> dict[str, Any]:
    workspace_dir = reports_dir.parent.parent
    cached, stale = _load_cached_json_with_staleness(
        reports_dir / "portfolio_replacement_shadow_reward_latest.json",
        expected_builder="portfolio_replacement_shadow_reward_v1",
        expected_research_config=PORTFOLIO_REPLACEMENT_RESEARCH_CONFIG,
    )
    if cached:
        return _with_cache_staleness(_portfolio_replacement_summary_from_report(cached), stale)
    try:
        report = report_portfolio_replacement_shadow_reward.build_report(
            files_dir=workspace_dir / "files",
            reports_dir=reports_dir,
            output_json=reports_dir / "portfolio_replacement_shadow_reward_latest.json",
            output_txt=reports_dir / "portfolio_replacement_shadow_reward_latest.txt",
            save=True,
        )
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return _portfolio_replacement_summary_from_report(report)


def _portfolio_replacement_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") or {}
    policies = report.get("policy_simulations") or []
    decision = str(report.get("decision") or "")
    replacements = int(summary.get("replacement_count") or 0)
    closed = int(summary.get("closed_incoming_count") or 0)
    if replacements <= 0:
        return {"status": "missing", "detail": "нет replacement events", "decision": decision}
    advanced_policy = next(
        (
            row for row in policies
            if row.get("kind") == "causal" and str(row.get("decision") or "").startswith("advance_")
        ),
        None,
    )
    if advanced_policy:
        status = "policy_candidate"
    elif decision.startswith("advance_"):
        status = "passed_shadow_gate"
    elif "hurting" in decision:
        status = "hurting"
    elif "collect_more" in decision:
        status = "collecting"
    else:
        status = "monitor"
    detail = (
        f"closed={closed}/{replacements}, avg_delta={_fmt(summary.get('avg_replacement_delta_pct'), 2)}%, "
        f"med_delta={_fmt(summary.get('median_replacement_delta_pct'), 2)}%, "
        f"positive={_fmt(summary.get('positive_delta_rate_pct'), 1)}%"
    )
    if advanced_policy:
        detail += (
            f"; candidate={advanced_policy.get('policy')}, "
            f"saved={_fmt(advanced_policy.get('net_saved_delta_pct'), 2)}%, "
            f"regret={_fmt(advanced_policy.get('regret_rate_pct'), 1)}%"
        )
    return {
        "status": status,
        "detail": detail,
        "decision": decision,
        "summary": summary,
        "advanced_policy": advanced_policy or {},
        "files": report.get("files") or {},
        "artifact_freshness": report.get("_artifact_freshness") or {},
    }


def _shadow_reentry_summary(scorecard: dict[str, Any]) -> dict[str, str]:
    if not scorecard:
        return {"status": "missing", "detail": "scorecard ещё не создан"}
    summary = scorecard.get("summary") or {}
    alerts = int(summary.get("alerts_total") or 0)
    labeled = int(summary.get("labeled_ret5") or 0)
    pending = int(summary.get("pending") or 0)
    avg_ret5 = _maybe_float(summary.get("avg_ret5"))
    pos_rate = _maybe_float(summary.get("ret5_positive_rate"))
    status = str(scorecard.get("status") or "unknown")
    if alerts == 0:
        detail = "alerts=0; данных для оценки re-entry пока нет"
    elif labeled == 0:
        detail = f"alerts={alerts}, pending={pending}; ждём mature T+5/T+10 labels"
    else:
        pos_piece = "н/д" if pos_rate is None else f"{pos_rate * 100:.1f}%"
        ret_piece = "н/д" if avg_ret5 is None else f"{avg_ret5:+.2f}%"
        detail = f"alerts={alerts}, labeled={labeled}, avg_ret5={ret_piece}, positive={pos_piece}"
    return {"status": status, "detail": detail}


def _coverage_is_safe_partial(day: DayMetrics) -> bool:
    return (
        day.coverage_status == "partial"
        and day.coverage_assessment == "partial_safe_inactive_symbols_only"
    )


def _day_not_older(value: Any, reference: str) -> bool:
    text = str(value or "")
    return bool(text) and text[:10] >= str(reference)


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


def _load_fresh_cached_json(path: Path, max_age_hours: float = 36.0) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        if age_seconds > max_age_hours * 3600:
            return {}
        return _load_json(path)
    except Exception:
        return {}


def _load_cached_json_with_staleness(
    path: Path,
    max_age_hours: float = 36.0,
    *,
    expected_builder: str = "",
    expected_research_config: Any = None,
) -> tuple[dict[str, Any], bool]:
    """Load cached research artifact without blocking the daily report on recompute.

    The morning learning report must be a fast aggregation layer. If an expensive
    research artifact is stale, keep the report moving and mark that component
    stale instead of recomputing heavy replay inline.
    """
    try:
        if not path.exists():
            return {}, False
        data = _load_json(path)
        if expected_builder:
            freshness = artifact_provenance.artifact_freshness(
                data,
                expected_builder=expected_builder,
                expected_research_config=expected_research_config,
                max_age_hours=max_age_hours,
            )
            data["_artifact_freshness"] = freshness
            return data, freshness.get("status") != "fresh"
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        return data, age_seconds > max_age_hours * 3600
    except Exception:
        return {}, False


def _with_cache_staleness(summary: dict[str, Any], stale: bool) -> dict[str, Any]:
    if not stale:
        return summary
    out = dict(summary)
    out["cached_status"] = out.get("status")
    out["status"] = "stale"
    out["stale"] = True
    freshness = out.get("artifact_freshness") or {}
    reasons = list(freshness.get("reasons") or [])
    out["stale_reasons"] = reasons
    detail = str(out.get("detail") or "")
    reason_piece = f" ({', '.join(reasons)})" if reasons else ""
    out["detail"] = f"stale cache{reason_piece}; {detail}" if detail else f"stale cache{reason_piece}"
    return out


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
