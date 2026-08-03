from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import config


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
EVENTS_FILE = ROOT / "bot_events.jsonl"
WATCHLIST_FILE = ROOT / "watchlist.json"
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "suspicious_reentry_scorecard_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "suspicious_reentry_scorecard_latest.txt"
HORIZONS = (2, 5, 10)


@dataclass(frozen=True)
class LabeledShadowReentry:
    sym: str
    tf: str
    ts: str
    price: float
    mode: str
    candidate_score: float
    exit_score: float
    exit_pnl_pct: float
    mfe_pct: float
    bars_since_exit: int
    cooldown_bars_left: int
    cohort: str
    label_status: str
    label_reason: str = ""
    ret_2: float | None = None
    ret_5: float | None = None
    ret_10: float | None = None
    max_runup_10: float | None = None
    max_drawdown_10: float | None = None


def build_scorecard(
    target_day: date,
    *,
    events_file: Path = EVENTS_FILE,
    reports_dir: Path = REPORT_DIR,
    timezone_name: str = "Europe/Budapest",
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
    kline_loader: Callable[[str, str, int, int], list[list[Any]]] | None = None,
    now_utc: datetime | None = None,
    valid_symbols: set[str] | None = None,
) -> dict[str, Any]:
    rows, watch_decisions = _load_day_events(events_file, target_day, timezone_name)
    if valid_symbols is None and events_file.resolve() == EVENTS_FILE.resolve():
        valid_symbols = _load_runtime_watchlist()
    rows, excluded_alerts = _filter_valid_symbols(rows, valid_symbols)
    watch_decisions, excluded_watch_decisions = _filter_valid_symbols(watch_decisions, valid_symbols)
    labeled = [
        _label_event(row, kline_loader=kline_loader, now_utc=now_utc)
        for row in rows
    ]
    registered_labeled = [
        _label_registered_watch(row, kline_loader=kline_loader, now_utc=now_utc)
        for row in watch_decisions
        if row.get("decision") == "registered"
    ]
    payload = _payload(
        target_day,
        timezone_name,
        labeled,
        registered_labeled=registered_labeled,
        watch_decisions=watch_decisions,
        excluded_alerts=excluded_alerts,
        excluded_watch_decisions=excluded_watch_decisions,
        events_file=events_file,
        now_utc=now_utc,
    )
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        dated_json = output_json.parent / f"suspicious_reentry_scorecard_{target_day.isoformat()}.json"
        dated_txt = output_txt.parent / f"suspicious_reentry_scorecard_{target_day.isoformat()}.txt"
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
    day = scorecard.get("target_day_local") or "unknown"
    summary = scorecard.get("summary") or {}
    status = scorecard.get("status") or "unknown"
    icon = "🟢" if status == "complete" else "🟡" if status == "partial" else "⚪"
    lines = [
        f"Shadow re-entry scorecard — {day}",
        "",
        f"{icon} status={status}" + _coverage_suffix(scorecard.get("coverage_reasons") or []),
        "",
        "Главное:",
        f"  • shadow re-entry alerts: {summary.get('alerts_total', 0)}; labeled T+5: {summary.get('labeled_ret5', 0)}.",
        f"  • upstream watch funnel: total={summary.get('watch_decisions_total', 0)}; "
        f"registered={summary.get('watch_registered', 0)}; rejected exit score={summary.get('watch_rejected_exit_score', 0)}; "
        f"rejected MFE={summary.get('watch_rejected_mfe', 0)}.",
        f"  • data quality: raw watch decisions={summary.get('watch_decisions_total_raw', 0)}; "
        f"excluded non-watchlist telemetry={summary.get('excluded_non_watchlist_events', 0)}.",
        f"  • same-day alert/registered pressure: {_fmt_pct(summary.get('same_day_alert_per_registered_ratio'))}; "
        f"pending watch windows={summary.get('watch_pending', 0)}.",
        f"  • registered cohort T+5: labeled={summary.get('registered_labeled_ret5', 0)}; "
        f"avg/median={_fmt(summary.get('registered_avg_ret5'))} / {_fmt(summary.get('registered_median_ret5'))}; "
        f"positive={_fmt_pct(summary.get('registered_ret5_positive_rate'))}.",
        f"  • ret5 avg/median: {_fmt(summary.get('avg_ret5'))} / {_fmt(summary.get('median_ret5'))}; positive={_fmt_pct(summary.get('ret5_positive_rate'))}.",
        f"  • ret10 avg/median: {_fmt(summary.get('avg_ret10'))} / {_fmt(summary.get('median_ret10'))}; downside p50={_fmt(summary.get('median_drawdown10'))}.",
        f"  • exit context: avg exit pnl={_fmt(summary.get('avg_exit_pnl_pct'))}; avg prior MFE={_fmt(summary.get('avg_mfe_pct'))}.",
        "",
        "Интерпретация:",
        f"  • {scorecard.get('interpretation')}",
        "",
        "Решение:",
        f"  • {scorecard.get('recommendation')}",
    ]
    examples = scorecard.get("examples") or {}
    if examples.get("best") or examples.get("worst"):
        lines.extend(["", "Примеры:"])
        if examples.get("best"):
            lines.append("  • best: " + _join_examples(examples["best"]))
        if examples.get("worst"):
            lines.append("  • worst: " + _join_examples(examples["worst"]))
    return "\n".join(lines).strip()


def _payload(
    target_day: date,
    timezone_name: str,
    labeled: list[LabeledShadowReentry],
    *,
    registered_labeled: list[LabeledShadowReentry],
    watch_decisions: list[dict[str, Any]],
    excluded_alerts: list[dict[str, Any]],
    excluded_watch_decisions: list[dict[str, Any]],
    events_file: Path,
    now_utc: datetime | None,
) -> dict[str, Any]:
    mature = [x for x in labeled if x.label_status == "labeled"]
    pending = [x for x in labeled if x.label_status == "pending"]
    failed = [x for x in labeled if x.label_status == "missing"]
    ret2 = _vals(x.ret_2 for x in mature)
    ret5 = _vals(x.ret_5 for x in mature)
    ret10 = _vals(x.ret_10 for x in mature)
    drawdown10 = _vals(x.max_drawdown_10 for x in mature)
    runup10 = _vals(x.max_runup_10 for x in mature)
    registered_mature = [x for x in registered_labeled if x.label_status == "labeled"]
    registered_pending = [x for x in registered_labeled if x.label_status == "pending"]
    registered_failed = [x for x in registered_labeled if x.label_status == "missing"]
    registered_ret5 = _vals(x.ret_5 for x in registered_mature)
    registered_ret10 = _vals(x.ret_10 for x in registered_mature)
    registered_drawdown10 = _vals(x.max_drawdown_10 for x in registered_mature)
    watch_counts = Counter(str(row.get("decision") or "unknown") for row in watch_decisions)
    watch_registered = int(watch_counts.get("registered", 0))
    watch_pending = len(registered_pending)
    coverage_reasons: list[str] = []
    if not events_file.exists():
        coverage_reasons.append("missing bot_events.jsonl")
    if pending:
        coverage_reasons.append(f"pending labels: {len(pending)}")
    if failed:
        coverage_reasons.append(f"missing market data: {len(failed)}")
    if registered_failed:
        coverage_reasons.append(f"missing registered-watch market data: {len(registered_failed)}")
    if not labeled and not watch_decisions and events_file.exists():
        coverage_reasons.append("no re-entry watch decisions or alerts for target day")
    if watch_pending:
        coverage_reasons.append(f"pending registered watch windows: {watch_pending}")
    status = "complete" if not coverage_reasons else "partial"
    summary = {
        "alerts_total": len(labeled),
        "alerts_total_raw": len(labeled) + len(excluded_alerts),
        "labeled_ret5": len(ret5),
        "pending": len(pending),
        "missing": len(failed),
        "watch_decisions_total": len(watch_decisions),
        "watch_decisions_total_raw": len(watch_decisions) + len(excluded_watch_decisions),
        "excluded_non_watchlist_events": len(excluded_alerts) + len(excluded_watch_decisions),
        "watch_registered": watch_registered,
        "watch_pending": watch_pending,
        "registered_labeled_ret5": len(registered_ret5),
        "registered_pending": len(registered_pending),
        "registered_missing": len(registered_failed),
        "registered_avg_ret5": _avg(registered_ret5),
        "registered_median_ret5": _median(registered_ret5),
        "registered_avg_ret10": _avg(registered_ret10),
        "registered_median_ret10": _median(registered_ret10),
        "registered_ret5_positive_rate": _ratio(
            sum(1 for value in registered_ret5 if value > 0.0), len(registered_ret5)
        ),
        "registered_ret5_gt_fee_rate": _ratio(
            sum(1 for value in registered_ret5 if value > 0.15), len(registered_ret5)
        ),
        "registered_median_drawdown10": _median(registered_drawdown10),
        "watch_rejected_exit_score": int(watch_counts.get("rejected_exit_score", 0)),
        "watch_rejected_mfe": int(watch_counts.get("rejected_mfe", 0)),
        "watch_other": len(watch_decisions)
        - watch_registered
        - int(watch_counts.get("rejected_exit_score", 0))
        - int(watch_counts.get("rejected_mfe", 0)),
        "registration_rate": _ratio(watch_registered, len(watch_decisions)),
        "same_day_alert_per_registered_ratio": _ratio(len(labeled), watch_registered),
        "avg_ret2": _avg(ret2),
        "avg_ret5": _avg(ret5),
        "avg_ret10": _avg(ret10),
        "median_ret2": _median(ret2),
        "median_ret5": _median(ret5),
        "median_ret10": _median(ret10),
        "ret5_positive_rate": _ratio(sum(1 for x in ret5 if x > 0.0), len(ret5)),
        "ret5_gt_fee_rate": _ratio(sum(1 for x in ret5 if x > 0.15), len(ret5)),
        "avg_runup10": _avg(runup10),
        "median_drawdown10": _median(drawdown10),
        "avg_exit_pnl_pct": _avg([x.exit_pnl_pct for x in labeled]),
        "avg_mfe_pct": _avg([x.mfe_pct for x in labeled]),
    }
    return {
        "generated_at_local": datetime.now(ZoneInfo(timezone_name)).isoformat(timespec="seconds"),
        "target_day_local": target_day.isoformat(),
        "timezone": timezone_name,
        "status": status,
        "coverage_reasons": coverage_reasons,
        "summary": summary,
        "rows": [asdict(x) for x in labeled],
        "registered_rows": [asdict(x) for x in registered_labeled],
        "watch_funnel": {
            "decision_counts": dict(sorted(watch_counts.items())),
            "registered_examples": [
                {
                    "sym": str(row.get("sym") or ""),
                    "tf": str(row.get("tf") or ""),
                    "ts": str(row.get("ts") or ""),
                    "exit_score": _float(row.get("exit_score")),
                    "mfe_pct": _float(row.get("mfe_pct")),
                }
                for row in watch_decisions
                if row.get("decision") == "registered"
            ][:12],
        },
        "data_quality": {
            "excluded_non_watchlist_alerts": [
                {"sym": str(row.get("sym") or ""), "ts": str(row.get("ts") or "")}
                for row in excluded_alerts[:12]
            ],
            "excluded_non_watchlist_watch_decisions": [
                {
                    "sym": str(row.get("sym") or ""),
                    "decision": str(row.get("decision") or ""),
                    "ts": str(row.get("ts") or ""),
                }
                for row in excluded_watch_decisions[:12]
            ],
        },
        "examples": _examples(mature),
        "interpretation": _interpretation(summary),
        "recommendation": _recommendation(summary, status),
    }


def _load_day_events(
    events_file: Path,
    target_day: date,
    timezone_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not events_file.exists():
        return [], []
    alerts: list[dict[str, Any]] = []
    watch_decisions: list[dict[str, Any]] = []
    timezone = ZoneInfo(timezone_name)
    with events_file.open(encoding="utf-8", errors="ignore") as lines:
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            event = str(row.get("event") or "")
            if event not in {"suspicious_reentry_shadow", "suspicious_reentry_watch_decision"}:
                continue
            ts = str(row.get("ts") or "")
            if not ts:
                continue
            try:
                local_day = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone).date()
            except Exception:
                continue
            if local_day == target_day:
                if event == "suspicious_reentry_shadow":
                    alerts.append(row)
                else:
                    watch_decisions.append(row)
    alerts.sort(key=lambda r: str(r.get("ts") or ""))
    watch_decisions.sort(key=lambda r: str(r.get("ts") or ""))
    return alerts, watch_decisions


def _load_runtime_watchlist() -> set[str] | None:
    try:
        raw = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    symbols = {str(value) for value in raw if value}
    return symbols or None


def _filter_valid_symbols(
    rows: list[dict[str, Any]],
    valid_symbols: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not valid_symbols:
        return rows, []
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        target = included if str(row.get("sym") or "") in valid_symbols else excluded
        target.append(row)
    return included, excluded


def _label_event(
    row: dict[str, Any], *,
    kline_loader: Callable[[str, str, int, int], list[list[Any]]] | None,
    now_utc: datetime | None,
) -> LabeledShadowReentry:
    sym = str(row.get("sym") or "")
    tf = str(row.get("tf") or "15m")
    ts = str(row.get("ts") or "")
    price = _float(row.get("price"))
    event_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    event_ms = int(event_dt.timestamp() * 1000)
    bar_ms = _bar_ms(tf)
    max_h = max(HORIZONS)
    now = now_utc or datetime.now(timezone.utc)
    if int(now.timestamp() * 1000) < event_ms + max_h * bar_ms:
        return _row(row, label_status="pending", label_reason="forward horizon not mature")
    loader = kline_loader or _fetch_binance_klines
    try:
        candles = loader(sym, tf, event_ms - bar_ms, event_ms + (max_h + 2) * bar_ms)
    except Exception as exc:
        return _row(row, label_status="missing", label_reason=str(exc)[:160])
    if not candles:
        return _row(row, label_status="missing", label_reason="no candles")
    idx = _first_candle_at_or_after(candles, event_ms - bar_ms // 2)
    if idx is None:
        return _row(row, label_status="missing", label_reason="event candle not found")
    if idx + max_h >= len(candles):
        return _row(row, label_status="pending", label_reason="not enough future candles")
    closes = [_float(c[4]) for c in candles]
    highs = [_float(c[2]) for c in candles]
    lows = [_float(c[3]) for c in candles]
    ret = {h: _pct_return(closes[idx + h], price) for h in HORIZONS}
    future_high = max(highs[idx + 1: idx + max_h + 1])
    future_low = min(lows[idx + 1: idx + max_h + 1])
    return _row(
        row,
        label_status="labeled",
        ret_2=ret[2],
        ret_5=ret[5],
        ret_10=ret[10],
        max_runup_10=_pct_return(future_high, price),
        max_drawdown_10=_pct_return(future_low, price),
    )


def _label_registered_watch(
    row: dict[str, Any],
    *,
    kline_loader: Callable[[str, str, int, int], list[list[Any]]] | None,
    now_utc: datetime | None,
) -> LabeledShadowReentry:
    """Label the actionable counterfactual: enter at the next candle open."""
    sym = str(row.get("sym") or "")
    tf = str(row.get("tf") or "15m")
    ts = str(row.get("ts") or "")
    event_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    event_ms = int(event_dt.timestamp() * 1000)
    bar_ms = _bar_ms(tf)
    max_h = max(HORIZONS)
    now = now_utc or datetime.now(timezone.utc)
    if int(now.timestamp() * 1000) < event_ms + max_h * bar_ms:
        return _row(
            row,
            cohort="registered_watch",
            label_status="pending",
            label_reason="forward horizon not mature",
        )
    loader = kline_loader or _fetch_binance_klines
    try:
        candles = loader(sym, tf, event_ms, event_ms + (max_h + 2) * bar_ms)
    except Exception as exc:
        return _row(
            row,
            cohort="registered_watch",
            label_status="missing",
            label_reason=str(exc)[:160],
        )
    if not candles:
        return _row(
            row,
            cohort="registered_watch",
            label_status="missing",
            label_reason="no candles",
        )
    idx = _first_candle_at_or_after(candles, event_ms)
    if idx is None or idx + max_h - 1 >= len(candles):
        return _row(
            row,
            cohort="registered_watch",
            label_status="pending",
            label_reason="not enough future candles",
        )
    entry = _float(candles[idx][1])
    if entry <= 0:
        return _row(
            row,
            cohort="registered_watch",
            label_status="missing",
            label_reason="invalid next-candle open",
        )
    closes = [_float(candle[4]) for candle in candles]
    highs = [_float(candle[2]) for candle in candles]
    lows = [_float(candle[3]) for candle in candles]
    future = slice(idx, idx + max_h)
    labeled_row = dict(row)
    labeled_row["price"] = entry
    return _row(
        labeled_row,
        cohort="registered_watch",
        label_status="labeled",
        ret_2=_pct_return(closes[idx + 1], entry),
        ret_5=_pct_return(closes[idx + 4], entry),
        ret_10=_pct_return(closes[idx + 9], entry),
        max_runup_10=_pct_return(max(highs[future]), entry),
        max_drawdown_10=_pct_return(min(lows[future]), entry),
    )


def _row(row: dict[str, Any], *, cohort: str = "alert", label_status: str, label_reason: str = "", ret_2: float | None = None, ret_5: float | None = None, ret_10: float | None = None, max_runup_10: float | None = None, max_drawdown_10: float | None = None) -> LabeledShadowReentry:
    return LabeledShadowReentry(
        sym=str(row.get("sym") or ""),
        tf=str(row.get("tf") or "15m"),
        ts=str(row.get("ts") or ""),
        price=_float(row.get("price")),
        mode=str(row.get("mode") or ""),
        candidate_score=_float(row.get("candidate_score")),
        exit_score=_float(row.get("exit_score")),
        exit_pnl_pct=_float(row.get("exit_pnl_pct")),
        mfe_pct=_float(row.get("mfe_pct")),
        bars_since_exit=int(_float(row.get("bars_since_exit"))),
        cooldown_bars_left=int(_float(row.get("cooldown_bars_left"))),
        cohort=cohort,
        label_status=label_status,
        label_reason=label_reason,
        ret_2=ret_2,
        ret_5=ret_5,
        ret_10=ret_10,
        max_runup_10=max_runup_10,
        max_drawdown_10=max_drawdown_10,
    )


def _fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list[Any]]:
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": 1000,
    })
    url = f"{getattr(config, 'BINANCE_REST', 'https://api.binance.com')}/api/v3/klines?{params}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _first_candle_at_or_after(candles: list[list[Any]], ts_ms: int) -> int | None:
    for i, candle in enumerate(candles):
        if int(candle[0]) >= ts_ms:
            return i
    return None


def _examples(rows: list[LabeledShadowReentry]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted([r for r in rows if r.ret_5 is not None], key=lambda r: float(r.ret_5 or 0.0))
    def compact(items: Iterable[LabeledShadowReentry]) -> list[dict[str, Any]]:
        return [
            {"sym": x.sym, "tf": x.tf, "ret5": x.ret_5, "ret10": x.ret_10, "exit_pnl": x.exit_pnl_pct, "mfe": x.mfe_pct}
            for x in items
        ]
    return {"worst": compact(ordered[:5]), "best": compact(reversed(ordered[-5:]))}


def _interpretation(summary: dict[str, Any]) -> str:
    n = int(summary.get("labeled_ret5") or 0)
    if n == 0:
        watch_total = int(summary.get("watch_decisions_total") or 0)
        registered = int(summary.get("watch_registered") or 0)
        alerts = int(summary.get("alerts_total") or 0)
        pending = int(summary.get("watch_pending") or 0)
        registered_labeled = int(summary.get("registered_labeled_ret5") or 0)
        if watch_total == 0:
            return "нет upstream watch decisions и mature labels; coverage контура за день не доказан."
        if pending:
            return (
                f"upstream работал, но pending registered watch windows: {pending}; "
                "оценка final confirmation преждевременна."
            )
        if registered_labeled:
            avg = summary.get("registered_avg_ret5")
            positive = summary.get("registered_ret5_positive_rate")
            if avg is not None and avg > 0.25 and positive is not None and positive >= 0.55:
                return (
                    f"final alerts отсутствуют, но counterfactual registered cohort n={registered_labeled} "
                    "имеет положительный T+5; нужен максимальный replay final confirmation."
                )
            return (
                f"counterfactual registered cohort размечен (n={registered_labeled}), но не проходит "
                "promotion gate; пороги re-entry не расслаблять."
            )
        if registered > 0 and alerts == 0:
            return (
                f"upstream работал: {registered}/{watch_total} watch зарегистрировано, "
                "но ни один не прошёл final candidate confirmation; решений о production принимать нельзя."
            )
        return (
            f"upstream работал, но все {watch_total} решений отклонены до регистрации watch; "
            "mature labels нет."
        )
    avg = summary.get("avg_ret5")
    pos = summary.get("ret5_positive_rate")
    if avg is not None and avg > 0.25 and pos is not None and pos >= 0.55:
        return "shadow re-entry выглядит перспективно: средний T+5 положительный и большинство кандидатов не проваливаются."
    if avg is not None and avg < 0.0:
        return "shadow re-entry пока выглядит шумным: средний T+5 отрицательный."
    return "сигнал неоднозначный; нужно больше дней и разрез по режимам рынка."


def _recommendation(summary: dict[str, Any], status: str) -> str:
    n = int(summary.get("labeled_ret5") or 0)
    registered_n = int(summary.get("registered_labeled_ret5") or 0)
    if n < 10 and registered_n < 10:
        return "продолжать shadow-only сбор; production re-entry не включать."
    if n < 10:
        avg = summary.get("registered_avg_ret5") or 0.0
        pos = summary.get("registered_ret5_positive_rate") or 0.0
        if status == "complete" and avg > 0.25 and pos >= 0.55:
            return "проверить final confirmation на максимальном replay; production re-entry пока не включать."
        return "registered cohort не проходит gate; production re-entry не включать и пороги не расслаблять."
    avg = summary.get("avg_ret5") or 0.0
    pos = summary.get("ret5_positive_rate") or 0.0
    if status == "complete" and avg > 0.25 and pos >= 0.55:
        return "подготовить replay-gated production spec для ограниченного re-entry; пока не включать автоматически."
    return "оставить shadow-only и добавить разбор причин false re-entry."


def _bar_ms(tf: str) -> int:
    if tf == "1h":
        return 60 * 60 * 1000
    if tf == "4h":
        return 4 * 60 * 60 * 1000
    return 15 * 60 * 1000


def _pct_return(value: float, base: float) -> float:
    if base <= 0:
        return 0.0
    return round((value / base - 1.0) * 100.0, 4)


def _float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


def _vals(values: Iterable[float | None]) -> list[float]:
    return [float(x) for x in values if x is not None and math.isfinite(float(x))]


def _avg(values: Iterable[float]) -> float | None:
    vals = list(values)
    return round(statistics.mean(vals), 4) if vals else None


def _median(values: Iterable[float]) -> float | None:
    vals = list(values)
    return round(statistics.median(vals), 4) if vals else None


def _ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2f}%"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _coverage_suffix(reasons: list[str]) -> str:
    return "" if not reasons else " — " + "; ".join(str(x) for x in reasons[:3])


def _join_examples(rows: list[dict[str, Any]]) -> str:
    return ", ".join(f"{r.get('sym')} ret5={_fmt(r.get('ret5'))}" for r in rows[:5])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build suspicious re-entry shadow daily scorecard")
    parser.add_argument("--target-day", default=(datetime.now(ZoneInfo("Europe/Budapest")).date() - timedelta(days=1)).isoformat())
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--events", default=str(EVENTS_FILE))
    parser.add_argument("--json", action="store_true", help="print JSON payload")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    payload = build_scorecard(
        date.fromisoformat(args.target_day),
        events_file=Path(args.events),
        timezone_name=args.timezone,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
