from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


BTC_SYMBOL = "BTCUSDT"
WATCH_TIMEFRAME = "15m"
WATCH_STATE = "emerging_move"
WATCH_ACTION = "elevate_priority"
WATCH_REASON = "early positive structure"
MAIN_BLOCK_REASON = "top_gainer_score_gate"


def parse_utc(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iter_tail_jsonl(path: Path, *, max_bytes: int) -> Iterable[dict[str, Any]]:
    if not path.exists() or max_bytes <= 0:
        return
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - int(max_bytes))
        handle.seek(start)
        if start:
            handle.readline()
        for raw_line in handle:
            try:
                row = json.loads(raw_line.decode("utf-8", errors="ignore"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                yield row


def is_btc_early_trend_event(event: Mapping[str, object]) -> bool:
    return (
        str(event.get("sym") or "").upper() == BTC_SYMBOL
        and str(event.get("tf") or "") == WATCH_TIMEFRAME
        and str(event.get("state") or "") == WATCH_STATE
        and str(event.get("action") or "") == WATCH_ACTION
        and str(event.get("reason") or "") == WATCH_REASON
        and event.get("bootstrap") is not True
        and event.get("previous_state") is not None
    )


def find_recent_main_block(
    path: Path,
    *,
    signal_ts: object,
    lookback_minutes: int,
    min_candidate_score: float,
    max_bytes: int,
) -> dict[str, Any] | None:
    signal_dt = parse_utc(signal_ts)
    if signal_dt is None:
        return None
    lower_bound = signal_dt - timedelta(minutes=max(1, int(lookback_minutes)))
    matches: list[dict[str, Any]] = []
    for row in iter_tail_jsonl(path, max_bytes=max_bytes):
        if str(row.get("event") or "") != "blocked":
            continue
        if str(row.get("sym") or "").upper() != BTC_SYMBOL:
            continue
        if str(row.get("tf") or "") != WATCH_TIMEFRAME:
            continue
        if str(row.get("reason_code") or "") != MAIN_BLOCK_REASON:
            continue
        row_dt = parse_utc(row.get("ts"))
        if row_dt is None or not lower_bound <= row_dt <= signal_dt:
            continue
        score = _number(row.get("candidate_score"))
        if score is None or score < float(min_candidate_score):
            continue
        matches.append(row)
    if not matches:
        return None
    return max(
        matches,
        key=lambda row: (
            _number(row.get("candidate_score")) or 0.0,
            parse_utc(row.get("ts")) or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )


def find_latest_btc_watch_event(
    path: Path,
    *,
    now: datetime,
    max_age_minutes: int,
    max_bytes: int,
) -> dict[str, Any] | None:
    now_utc = now.astimezone(timezone.utc)
    lower_bound = now_utc - timedelta(minutes=max(1, int(max_age_minutes)))
    matches = []
    for row in iter_tail_jsonl(path, max_bytes=max_bytes):
        if not is_btc_early_trend_event(row):
            continue
        event_dt = parse_utc(row.get("ts"))
        if event_dt is not None and lower_bound <= event_dt <= now_utc:
            matches.append(row)
    if not matches:
        return None
    return max(matches, key=lambda row: parse_utc(row.get("ts")) or lower_bound)


def event_key(event: Mapping[str, object]) -> str:
    return "|".join(
        (
            str(event.get("sym") or ""),
            str(event.get("tf") or ""),
            str(event.get("bar_ts") or ""),
            str(event.get("ts") or ""),
        )
    )


def was_sent(state_path: Path, event: Mapping[str, object]) -> bool:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(state.get("last_event_key") or "") == event_key(event)


def mark_sent(state_path: Path, event: Mapping[str, object], *, sent_at: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "last_event_key": event_key(event),
                "last_event_ts": event.get("ts"),
                "last_bar_ts": event.get("bar_ts"),
                "sent_at": sent_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def format_watch_message(event: Mapping[str, object], main_block: Mapping[str, object]) -> str:
    features = event.get("features") if isinstance(event.get("features"), Mapping) else {}
    price = _number(features.get("price"))
    slope = _number(features.get("slope"))
    rsi = _number(features.get("rsi"))
    adx = _number(features.get("adx"))
    vol_x = _number(features.get("vol_x"))
    raw_score = _number(main_block.get("candidate_score"))
    observed = parse_utc(event.get("ts"))
    observed_text = observed.strftime("%Y-%m-%d %H:%M UTC") if observed else str(event.get("ts") or "")
    return (
        "🟡 WATCH: возможное начало тренда BTC\n"
        f"BTCUSDT · 15m · {observed_text}\n"
        f"Цена {_fmt(price, 2)} | slope {_fmt(slope, 2, signed=True)}% | "
        f"RSI {_fmt(rsi, 1)} | ADX {_fmt(adx, 1)} | volume {_fmt(vol_x, 2)}x\n"
        f"V2: {WATCH_REASON}; V1 raw score {_fmt(raw_score, 2)}.\n"
        "Статус: наблюдение, не BUY; торговые гейты не ослаблены."
    )


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _fmt(value: float | None, digits: int, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"
