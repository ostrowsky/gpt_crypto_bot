from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import numpy as np

import agentlog
import config
from unified_portfolio import load_main_positions_raw, ranked_unified_positions
def _entry_signal_score(
    mode: str,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> float:
    score = 0.0
    score += max(0.0, slope) * 0.8
    score += max(0.0, adx - 10.0) * 0.05
    score += max(0.0, vol_x - 1.0) * 0.8
    if rsi > 85:
        score -= 1.0
    if rsi < 40:
        score -= 0.5
    if daily_range > 35:
        score -= 1.0
    if mode in ("strong_trend",):
        score += 0.4
    if mode in ("breakout", "impulse_speed"):
        score += 0.3
    if mode == "4h_leader_watch":
        score += 0.8
    if mode == "alignment":
        score -= 0.2
    return float(score)


def _forecast_return_score_bonus(forecast_return_pct: float) -> float:
    return max(min(forecast_return_pct, 10.0), -10.0) * 0.05


def _top_mover_score_bonus(change_pct: float) -> float:
    return max(min(change_pct, 10.0), -10.0) * 0.04


def _time_block_1h_continuation_profile(**_kwargs) -> dict:
    return {}


def _time_block_1h_continuation_bonus(**_kwargs) -> float:
    return 0.0


def _late_1h_continuation_guard(**_kwargs) -> bool:
    return False


def _impulse_speed_entry_guard(**_kwargs) -> str:
    return ""


def _mtf_soft_penalty_from_reason(_reason: str) -> float:
    return 0.0


def _entry_score_floor(_tf: str) -> float:
    return 0.0


def _entry_score_borderline_bypass_ok(**kwargs) -> bool:
    score = float(kwargs.get("candidate_score", 0.0))
    floor = float(kwargs.get("min_score", 0.0))
    return score >= floor


def _continuation_profit_lock_active(**_kwargs) -> bool:
    return False


def _short_mode_profit_lock_active(**_kwargs) -> bool:
    return False


def _continuation_micro_exit_reason(*_args, **_kwargs) -> str:
    return ""


def _cooldown_bars_after_exit(_mode: str, _reason: Optional[str], *_args, **_kwargs) -> int:
    return getattr(config, "COOLDOWN_BARS", 8)


def _fast_loss_ema_exit_reason(*_args, **_kwargs) -> str:
    return ""


def _time_exit_should_wait(*_args, **_kwargs) -> bool:
    return False


def _post_entry_quality_recheck_reason(*_args, **_kwargs) -> str:
    return ""


def _min_weak_exit_bars(_mode: Optional[str]) -> int:
    return 3
from strategy import (
    analyze_coin,
    check_alignment_conditions,
    check_breakout_conditions,
    check_entry_conditions,
    check_exit_conditions,
    check_impulse_conditions,
    check_retest_conditions,
    check_trend_surge_conditions,
    compute_features,
    detect_market_regime,
    fetch_klines,
    get_entry_mode,
    is_bull_day,
)

try:
    from monitor import _check_mtf as _check_mtf_from_monitor
except Exception:
    _check_mtf_from_monitor = None

async def _check_mtf(*args, **kwargs):
    if _check_mtf_from_monitor is None:
        return True, "mtf disabled"
    return await _check_mtf_from_monitor(*args, **kwargs)


log = logging.getLogger("market_agent")
POSITIONS_FILE = Path("agent_positions.json")
CHAT_IDS_FILE = Path(".chat_ids")
STATE_FILE = Path(".runtime") / "market_agent_state.json"
STATUS_FILE = Path(".runtime") / "market_agent_status.json"
_SOFT_BLOCK_WATCH_ALERT_DAY: Dict[str, str] = {}


def _unified_portfolio_limit() -> int:
    return int(
        getattr(
            config,
            "UNIFIED_PORTFOLIO_MAX_POSITIONS",
            getattr(config, "MAX_OPEN_POSITIONS", getattr(config, "AGENT_MAX_POSITIONS", 2)),
        )
        or 0
    )


def _resolve_local_tz():
    try:
        return ZoneInfo("Europe/Budapest")
    except ZoneInfoNotFoundError:
        return timezone.utc


LOCAL_TZ = _resolve_local_tz()


@dataclass
class AgentPosition:
    symbol: str
    tf: str
    entry_price: float
    entry_bar: int
    entry_ts: int
    entry_ema20: float
    entry_slope: float
    entry_adx: float
    entry_rsi: float
    entry_vol_x: float
    forecast_return_pct: float = 0.0
    today_change_pct: float = 0.0
    leader_score: float = 0.0
    four_h_context_score: float = 0.0
    four_h_context_label: str = ""
    predictions: Dict[int, Optional[bool]] = field(default_factory=dict)
    bars_elapsed: int = 0
    signal_mode: str = "trend"
    trail_k: float = 2.0
    max_hold_bars: int = 16
    trail_stop: float = 0.0
    last_bar_ts: int = 0
    mark_price: float = 0.0

    def pnl_pct(self, current_price: float) -> float:
        return (current_price / self.entry_price - 1.0) * 100.0


def _position_key(symbol: str, tf: str) -> str:
    return f"{symbol}|{tf}"


def _tf_bar_ms(tf: str) -> int:
    return 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000


def _load_chat_ids() -> list[int]:
    for candidate in (CHAT_IDS_FILE, Path(__file__).resolve().parent / ".chat_ids"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [int(x) for x in payload]
        except Exception:
            continue
    return []


def _save_positions(positions: Dict[str, AgentPosition]) -> None:
    payload = {}
    for key, pos in positions.items():
        payload[key] = {
            "symbol": pos.symbol,
            "tf": pos.tf,
            "entry_price": pos.entry_price,
            "entry_bar": pos.entry_bar,
            "entry_ts": pos.entry_ts,
            "entry_ema20": pos.entry_ema20,
            "entry_slope": pos.entry_slope,
            "entry_adx": pos.entry_adx,
            "entry_rsi": pos.entry_rsi,
            "entry_vol_x": pos.entry_vol_x,
            "forecast_return_pct": pos.forecast_return_pct,
            "today_change_pct": pos.today_change_pct,
            "leader_score": pos.leader_score,
            "four_h_context_score": pos.four_h_context_score,
            "four_h_context_label": pos.four_h_context_label,
            "predictions": pos.predictions,
            "bars_elapsed": pos.bars_elapsed,
            "signal_mode": pos.signal_mode,
            "trail_k": pos.trail_k,
            "max_hold_bars": pos.max_hold_bars,
            "trail_stop": pos.trail_stop,
            "last_bar_ts": pos.last_bar_ts,
            "mark_price": pos.mark_price,
        }
    POSITIONS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_positions() -> Dict[str, AgentPosition]:
    if not POSITIONS_FILE.exists():
        return {}
    try:
        raw = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    positions: Dict[str, AgentPosition] = {}
    if not isinstance(raw, dict):
        return positions
    for key, data in raw.items():
        if not isinstance(data, dict):
            continue
        positions[key] = AgentPosition(
            symbol=str(data.get("symbol", "")),
            tf=str(data.get("tf", "15m")),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_bar=int(data.get("entry_bar", 0)),
            entry_ts=int(data.get("entry_ts", 0)),
            entry_ema20=float(data.get("entry_ema20", 0.0)),
            entry_slope=float(data.get("entry_slope", 0.0)),
            entry_adx=float(data.get("entry_adx", 0.0)),
            entry_rsi=float(data.get("entry_rsi", 0.0)),
            entry_vol_x=float(data.get("entry_vol_x", 0.0)),
            forecast_return_pct=float(data.get("forecast_return_pct", 0.0)),
            today_change_pct=float(data.get("today_change_pct", 0.0)),
            leader_score=float(data.get("leader_score", 0.0)),
            four_h_context_score=float(data.get("four_h_context_score", 0.0)),
            four_h_context_label=str(data.get("four_h_context_label", "")),
            predictions={int(k): v for k, v in dict(data.get("predictions", {})).items()},
            bars_elapsed=int(data.get("bars_elapsed", 0)),
            signal_mode=str(data.get("signal_mode", "trend")),
            trail_k=float(data.get("trail_k", 2.0)),
            max_hold_bars=int(data.get("max_hold_bars", 16)),
            trail_stop=float(data.get("trail_stop", 0.0)),
            last_bar_ts=int(data.get("last_bar_ts", 0)),
            mark_price=float(data.get("mark_price", 0.0)),
        )
    return positions


def _agent_position_to_raw(pos: AgentPosition) -> dict:
    return {
        "symbol": pos.symbol,
        "tf": pos.tf,
        "entry_price": pos.entry_price,
        "entry_bar": pos.entry_bar,
        "entry_ts": pos.entry_ts,
        "entry_ema20": pos.entry_ema20,
        "entry_slope": pos.entry_slope,
        "entry_adx": pos.entry_adx,
        "entry_rsi": pos.entry_rsi,
        "entry_vol_x": pos.entry_vol_x,
        "forecast_return_pct": pos.forecast_return_pct,
        "today_change_pct": pos.today_change_pct,
        "leader_score": pos.leader_score,
        "four_h_context_score": pos.four_h_context_score,
        "four_h_context_label": pos.four_h_context_label,
        "predictions": pos.predictions,
        "bars_elapsed": pos.bars_elapsed,
        "signal_mode": pos.signal_mode,
        "trail_k": pos.trail_k,
        "max_hold_bars": pos.max_hold_bars,
        "trail_stop": pos.trail_stop,
        "last_bar_ts": pos.last_bar_ts,
        "mark_price": pos.mark_price,
    }


def _agent_allowed_slots(positions: Dict[str, AgentPosition]) -> int:
    if not bool(getattr(config, "UNIFIED_PORTFOLIO_ENABLED", True)):
        return int(getattr(config, "AGENT_MAX_POSITIONS", 2))
    limit = _unified_portfolio_limit()
    main_positions = load_main_positions_raw()
    main_symbols = {
        str(pos.get("symbol") or key)
        for key, pos in main_positions.items()
        if isinstance(pos, dict)
    }
    agent_symbols = {pos.symbol for pos in positions.values()}
    main_only_count = len(main_symbols - agent_symbols)
    return max(0, limit - main_only_count)


def _save_state(
    last_exit_bar: Dict[str, int],
    symbol_cooldown_until: Dict[str, int],
    soft_block_watch_alert_day: Dict[str, str] | None = None,
) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {
                "last_exit_bar": last_exit_bar,
                "symbol_cooldown_until": symbol_cooldown_until,
                "soft_block_watch_alert_day": soft_block_watch_alert_day or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_state() -> tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    if not STATE_FILE.exists():
        return {}, {}, {}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}, {}
    raw_exit = payload.get("last_exit_bar", {})
    raw_symbol_cd = payload.get("symbol_cooldown_until", {})
    raw_soft_block_watch = payload.get("soft_block_watch_alert_day", {})
    if not isinstance(raw_exit, dict):
        raw_exit = {}
    if not isinstance(raw_symbol_cd, dict):
        raw_symbol_cd = {}
    if not isinstance(raw_soft_block_watch, dict):
        raw_soft_block_watch = {}
    return (
        {str(k): int(v) for k, v in raw_exit.items()},
        {str(k): int(v) for k, v in raw_symbol_cd.items()},
        {str(k): str(v) for k, v in raw_soft_block_watch.items()},
    )


def _status_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(
    *,
    started_at: str,
    poll_sec: int,
    cycle_running: bool,
    n_open_positions: int,
    last_cycle_started_at: Optional[str] = None,
    last_cycle_finished_at: Optional[str] = None,
    last_cycle_stats: Optional[dict] = None,
    last_error: str = "",
) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "worker": {
            "started_at": started_at,
            "last_heartbeat": _status_now(),
            "mode": "market_agent_headless",
            "poll_sec": int(poll_sec),
        },
        "collector": {
            "running": bool(cycle_running),
            "last_cycle_started_at": last_cycle_started_at,
            "last_cycle_finished_at": last_cycle_finished_at,
            "last_cycle_stats": last_cycle_stats or {},
            "last_error": str(last_error or ""),
            "open_positions": int(n_open_positions),
        },
    }
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _send_telegram(session: aiohttp.ClientSession, text: str) -> None:
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    chat_ids = _load_chat_ids()
    if not chat_ids:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    timeout = aiohttp.ClientTimeout(total=12)
    for chat_id in chat_ids:
        try:
            async with session.post(
                url,
                json={"chat_id": chat_id, "text": text},
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                await resp.text()
        except Exception as exc:
            log.warning("telegram send failed for %s: %s", chat_id, exc)


def _entry_params(mode: str, tf: str) -> tuple[float, int]:
    if mode == "4h_leader_watch":
        trail_k = float(getattr(config, "AGENT_4H_LEADER_TRAIL_K", 2.8))
        hold = int(
            getattr(config, "AGENT_4H_LEADER_MAX_HOLD_BARS_15M", 48)
            if tf == "15m"
            else getattr(config, "AGENT_4H_LEADER_MAX_HOLD_BARS_1H", 16)
        )
        return trail_k, hold
    if mode == "breakout":
        return float(getattr(config, "ATR_TRAIL_K_BREAKOUT", 1.5)), int(getattr(config, "MAX_HOLD_BARS_BREAKOUT", 6))
    if mode == "retest":
        return float(getattr(config, "ATR_TRAIL_K_RETEST", 1.8)), int(getattr(config, "MAX_HOLD_BARS_RETEST", 10))
    if mode in ("strong_trend", "impulse_speed"):
        return float(getattr(config, "ATR_TRAIL_K_STRONG", 2.5)), int(
            getattr(config, "MAX_HOLD_BARS_15M", 48) if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16)
        )
    if mode == "impulse":
        return float(getattr(config, "ATR_TRAIL_K", 2.0)), int(
            getattr(config, "MAX_HOLD_BARS_15M", 48) if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16)
        )
    return float(getattr(config, "ATR_TRAIL_K", 2.0)), int(
        getattr(config, "MAX_HOLD_BARS_15M", 48) if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16)
    )


def _prediction_summary(pos: AgentPosition) -> str:
    parts = []
    for h in _position_forward_horizons(pos.tf, pos.signal_mode):
        value = pos.predictions.get(h)
        if value is None:
            parts.append(f"T+{h}: ⏳")
        elif value:
            parts.append(f"T+{h}: ✅")
        else:
            parts.append(f"T+{h}: ❌")
    return "  ".join(parts)


def _normalize_forward_horizons(values) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for raw in values or ():
        try:
            horizon = int(raw)
        except Exception:
            continue
        if horizon <= 0 or horizon in seen:
            continue
        seen.add(horizon)
        result.append(horizon)
    if result:
        return tuple(result)
    return tuple(int(h) for h in getattr(config, "FORWARD_BARS", [3, 5, 10]))


def _position_forward_horizons(tf: str, mode: str) -> tuple[int, ...]:
    fast_modes = tuple(getattr(config, "FORWARD_BARS_15M_FAST_MODES", ("breakout", "retest", "impulse_speed")))
    if tf == "15m" and mode in fast_modes:
        return _normalize_forward_horizons(getattr(config, "FORWARD_BARS_15M_FAST", [2, 5, 7]))
    return _normalize_forward_horizons(getattr(config, "FORWARD_BARS", [3, 5, 10]))


def _mode_label(mode: str) -> str:
    labels = {
        "trend": "📈 Тренд",
        "strong_trend": "💪 Сильный тренд",
        "impulse_speed": "⚡️ Быстрое движение",
        "4h_leader_watch": "🧭 4h лидер",
        "retest": "🔄 Ретест EMA20",
        "breakout": "⚡️ Пробой флэта",
        "impulse": "🚀 Импульс",
        "impulse_cross": "🚀 Импульс (кросс)",
        "alignment": "🌊 Выравнивание тренда",
    }
    return labels.get(mode, "📈 Тренд")


async def _send_entry_alert(session: aiohttp.ClientSession, pos: AgentPosition) -> None:
    text = (
        f"🟢 СИГНАЛ ПОКУПКИ — {_mode_label(pos.signal_mode)}\n\n"
        f"{pos.symbol}  [{pos.tf}]\n"
        f"💰 Цена: {pos.entry_price:.6g}\n"
        f"📐 EMA20: {pos.entry_ema20:.6g}\n"
        f"📈 Наклон EMA20: {pos.entry_slope:+.2f}%\n"
        f"💪 ADX: {pos.entry_adx:.1f}\n"
        f"📊 RSI: {pos.entry_rsi:.1f}\n"
        f"🔊 Объём ×: {pos.entry_vol_x:.2f}\n"
        f"⚙️ Стоп: ATR×{pos.trail_k:g}  |  Лимит: {pos.max_hold_bars} баров\n\n"
        f"🎯 Буду проверять прогноз: {_prediction_summary(pos)}"
    )
    await _send_telegram(session, text)


def _format_exit_reason(reason: str) -> str:
    text = str(reason or "").strip()
    if text.startswith("ATR trail broken"):
        return text.replace("ATR trail broken", "ATR-трейл пробит", 1)
    return text


async def _send_exit_alert(session: aiohttp.ClientSession, pos: AgentPosition, exit_price: float, reason: str) -> None:
    pnl = pos.pnl_pct(exit_price)
    pnl_icon = "🟢" if pnl >= 0 else "🔴"
    text = (
        f"{pos.symbol}  [{pos.tf}]\n"
        f"💰 Выход: {exit_price:.6g}\n"
        f"📉 Причина: {_format_exit_reason(reason)}\n"
        f"{pnl_icon} Изменение от входа: {pnl:+.2f}%\n"
        f"🎯 Точность прогнозов: {_prediction_summary(pos)}\n"
        f"⏱️ Баров в позиции: {pos.bars_elapsed}"
    )
    await _send_telegram(session, text)


async def _compute_features(data) -> tuple[np.ndarray, dict]:
    return await asyncio.to_thread(
        lambda: (
            data["c"].astype(float),
            compute_features(data["o"], data["h"], data["l"], data["c"].astype(float), data["v"]),
        )
    )


def _agent_allowed_modes() -> tuple[str, ...]:
    return tuple(
        str(x) for x in getattr(config, "AGENT_ALLOWED_MODES", ("trend", "strong_trend", "impulse_speed"))
    )


def _mode_cluster(mode: str) -> str:
    if mode in ("trend", "strong_trend", "impulse_speed", "4h_leader_watch"):
        return "momentum"
    if mode in ("breakout", "retest"):
        return "breakout_retest"
    if mode == "alignment":
        return "alignment"
    return mode


def _symbol_group(symbol: str) -> str:
    groups = getattr(config, "COIN_GROUPS", {})
    for group_name, members in groups.items():
        if symbol in members:
            return str(group_name)
    return ""


def _local_day_start_utc_ms(ts_ms: int) -> int:
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    dt_local = dt_utc.astimezone(LOCAL_TZ)
    start_local = datetime.combine(dt_local.date(), time.min, tzinfo=LOCAL_TZ)
    return int(start_local.astimezone(timezone.utc).timestamp() * 1000)


def _next_local_day_start_utc_ms(ts_ms: int) -> int:
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    dt_local = dt_utc.astimezone(LOCAL_TZ)
    next_local = datetime.combine(dt_local.date(), time.min, tzinfo=LOCAL_TZ)
    next_local = next_local + timedelta(days=1)
    return int(next_local.astimezone(timezone.utc).timestamp() * 1000)


def _local_day_key(ts_ms: int) -> str:
    if ts_ms <= 0:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(LOCAL_TZ).strftime("%Y-%m-%d")


def _today_change_pct_from_data(data, c: np.ndarray, i: int) -> float:
    if i < 0 or i >= len(c):
        return 0.0
    bar_ts = int(data["t"][i])
    day_start_ms = _local_day_start_utc_ms(bar_ts)
    day_open = None
    for idx, ts in enumerate(data["t"]):
        if int(ts) >= day_start_ms:
            day_open = float(data["o"][idx])
            break
    if not day_open or day_open <= 0:
        return 0.0
    return (float(c[i]) / day_open - 1.0) * 100.0


def _forecast_proxy_pct(
    *,
    today_change_pct: float,
    slope: float,
    adx: float,
    vol_x: float,
    rsi: float,
) -> float:
    upside = (
        max(0.0, today_change_pct) * 0.35
        + max(0.0, slope) * 1.8
        + max(0.0, adx - 18.0) * 0.06
        + max(0.0, vol_x - 1.0) * 0.45
    )
    downside = max(0.0, rsi - 70.0) * 0.20
    return max(-2.0, min(12.0, upside - downside))


def _leader_score(
    *,
    mode: str,
    today_change_pct: float,
    forecast_proxy_pct: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    today_confirmed: bool,
    today_signals: int,
    best_accuracy: float,
) -> float:
    mode_bonus = {
        "strong_trend": 2.2,
        "4h_leader_watch": 3.0,
        "impulse_speed": 1.6,
        "trend": 0.9,
        "alignment": -1.0,
        "breakout": -1.5,
        "retest": -1.5,
        "impulse": -0.5,
    }.get(mode, -1.5)
    score = 0.0
    score += max(0.0, min(today_change_pct, 15.0)) * 2.4
    score += max(0.0, forecast_proxy_pct) * 2.0
    score += max(0.0, slope) * 0.9
    score += max(0.0, adx - 18.0) * 0.10
    score += max(0.0, vol_x - 1.0) * 1.4
    score += min(max(best_accuracy, 0.0), 100.0) * 0.06
    score += min(max(today_signals, 0), 6) * 0.45
    score += 6.0 if today_confirmed else 0.0
    score += mode_bonus
    if rsi > 73.0:
        score -= (rsi - 73.0) * 0.30
    if daily_range > 12.0:
        score -= (daily_range - 12.0) * 0.45
    return round(score, 4)


async def _four_h_context_score(
    session: aiohttp.ClientSession,
    symbol: str,
) -> tuple[float, str]:
    if not bool(getattr(config, "FOUR_H_CONTEXT_SCORE_ENABLED", True)):
        return 0.0, "disabled"
    try:
        data = await fetch_klines(session, symbol, "4h", limit=120)
        c = data["c"].astype(float)
        if len(c) < 60:
            return 0.0, "4h_insufficient"
        feat = compute_features(data["o"], data["h"], data["l"], c, data["v"])
        i = len(c) - 2
        price = float(c[i])
        ema20 = float(feat["ema_fast"][i]) if np.isfinite(feat["ema_fast"][i]) else 0.0
        ema50 = float(feat["ema_slow"][i]) if np.isfinite(feat["ema_slow"][i]) else 0.0
        slope = float(feat["slope"][i]) if np.isfinite(feat["slope"][i]) else 0.0
        rsi = float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0
        vol_x = float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 0.0
        macd_hist = float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else 0.0
        greens = 0
        for j in range(max(0, i - 2), i + 1):
            if float(data["c"][j]) > float(data["o"][j]):
                greens += 1

        score = 0.0
        score += 2.0 if price > ema20 > 0 else -1.5
        score += 1.2 if price > ema50 > 0 else -0.8
        score += 1.3 if ema20 > ema50 > 0 else -0.8
        score += max(-3.0, min(3.0, slope * 1.6))
        score += 0.8 * greens
        score += 1.0 if macd_hist > 0 else -1.0
        if 45.0 <= rsi <= 68.0:
            score += 0.8
        elif rsi < 42.0 or rsi > 75.0:
            score -= 0.8
        score += 0.5 if vol_x >= 0.8 else -0.5

        score = max(
            float(getattr(config, "FOUR_H_CONTEXT_MAX_PENALTY", -6.0)),
            min(float(getattr(config, "FOUR_H_CONTEXT_MAX_BONUS", 8.0)), score),
        )
        if score >= 4.0:
            label = "4h_bull_context"
        elif score >= 1.0:
            label = "4h_recovery_context"
        elif score <= -2.0:
            label = "4h_weak_context"
        else:
            label = "4h_neutral_context"
        return round(score, 4), label
    except Exception as exc:
        log.debug("4h context score failed for %s: %s", symbol, exc)
        return 0.0, "4h_error"


def _four_h_leader_watch_reason(
    *,
    tf: str,
    price: float,
    ema20: float,
    ema50: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    today_change_pct: float,
    macd_hist: float,
    four_h_score: float,
    c: np.ndarray,
    i: int,
) -> str:
    if not bool(getattr(config, "AGENT_4H_LEADER_WATCH_ENABLED", True)):
        return "4h leader watch disabled"
    if tf not in tuple(str(x) for x in getattr(config, "AGENT_ALLOWED_TIMEFRAMES", ("15m", "1h"))):
        return f"agent timeframe disabled: {tf}"
    min_context = float(getattr(config, "AGENT_4H_LEADER_MIN_CONTEXT_SCORE", 7.0))
    if four_h_score < min_context:
        return f"4h context {four_h_score:.2f} < {min_context:.2f}"
    min_day = float(getattr(config, "AGENT_4H_LEADER_MIN_TODAY_CHANGE_PCT", 4.0))
    if today_change_pct < min_day:
        return f"day change {today_change_pct:.2f}% < {min_day:.2f}%"
    max_range = float(getattr(config, "AGENT_4H_LEADER_MAX_DAILY_RANGE_PCT", 35.0))
    if daily_range > max_range:
        return f"daily range {daily_range:.2f}% > {max_range:.2f}%"
    min_adx = float(getattr(config, "AGENT_4H_LEADER_MIN_ADX", 30.0))
    if adx < min_adx:
        return f"ADX {adx:.1f} < {min_adx:.1f}"
    min_slope = float(getattr(config, "AGENT_4H_LEADER_MIN_SLOPE", 0.35))
    if slope < min_slope:
        return f"slope {slope:.2f}% < {min_slope:.2f}%"
    min_rsi = float(getattr(config, "AGENT_4H_LEADER_MIN_RSI", 50.0))
    max_rsi = float(getattr(config, "AGENT_4H_LEADER_MAX_RSI", 78.0))
    if rsi < min_rsi:
        return f"RSI {rsi:.1f} < {min_rsi:.1f}"
    if rsi > max_rsi:
        return f"RSI {rsi:.1f} > {max_rsi:.1f}"
    min_vol = float(getattr(config, "AGENT_4H_LEADER_MIN_VOL_X", 0.65))
    if vol_x < min_vol:
        return f"vol_x {vol_x:.2f} < {min_vol:.2f}"
    strength_reason = _four_h_leader_strength_reason(today_change_pct=today_change_pct, vol_x=vol_x)
    if strength_reason:
        return strength_reason
    min_macd = float(getattr(config, "AGENT_4H_LEADER_MIN_MACD_HIST", 0.0))
    if macd_hist <= min_macd:
        return f"MACD hist {macd_hist:.6g} <= {min_macd:.6g}"
    if price <= 0 or ema20 <= 0 or ema50 <= 0:
        return "invalid EMA context"
    if not (price > ema20 > ema50):
        return "15m/1h reclaim not confirmed"

    price_edge = ((price / ema20) - 1.0) * 100.0
    reclaim_max = float(getattr(config, "AGENT_4H_LEADER_RECLAIM_MAX_PRICE_EDGE_PCT", 8.0))
    if price_edge > reclaim_max:
        return f"price edge {price_edge:.2f}% > {reclaim_max:.2f}%"

    pullback_max = float(getattr(config, "AGENT_4H_LEADER_PULLBACK_MAX_PRICE_EDGE_PCT", 3.5))
    prev_close = float(c[i - 1]) if i > 0 else price
    fresh_close = price > prev_close
    controlled_pullback = price_edge <= pullback_max
    if not (fresh_close or controlled_pullback):
        return "no fresh 15m/1h reclaim or controlled pullback"
    return ""


def _four_h_leader_strength_reason(*, today_change_pct: float, vol_x: float) -> str:
    if not bool(getattr(config, "AGENT_4H_LEADER_STRENGTH_GATE_ENABLED", True)):
        return ""
    min_today = float(getattr(config, "AGENT_4H_LEADER_STRENGTH_MIN_TODAY_CHANGE_PCT", 10.0))
    min_vol = float(getattr(config, "AGENT_4H_LEADER_STRENGTH_MIN_VOL_X", 3.0))
    missing: list[str] = []
    if today_change_pct < min_today:
        missing.append(f"today {today_change_pct:.2f}% < {min_today:.2f}%")
    if vol_x < min_vol:
        missing.append(f"vol_x {vol_x:.2f} < {min_vol:.2f}")
    if missing:
        return "4h leader strength gate: " + ", ".join(missing)
    return ""


def _candidate_block_reason(
    *,
    symbol: str,
    tf: str,
    mode: str,
    today_change_pct: float,
    forecast_proxy_pct: float,
    leader_score: float,
    daily_range: float,
    adx: float,
    rsi: float,
    vol_x: float,
    report,
) -> str:
    if mode not in _agent_allowed_modes():
        return f"agent mode disabled: {mode}"
    if tf not in tuple(str(x) for x in getattr(config, "AGENT_ALLOWED_TIMEFRAMES", ("15m", "1h"))):
        return f"agent timeframe disabled: {tf}"
    min_day = float(
        getattr(config, "AGENT_4H_LEADER_MIN_TODAY_CHANGE_PCT", 4.0)
        if mode == "4h_leader_watch"
        else getattr(config, "AGENT_MIN_DAY_CHANGE_PCT", 1.25)
    )
    if today_change_pct < min_day:
        return f"day change {today_change_pct:.2f}% < {min_day:.2f}%"
    if forecast_proxy_pct < float(getattr(config, "AGENT_MIN_FORECAST_PROXY_PCT", 0.35)):
        return f"forecast proxy {forecast_proxy_pct:.2f}% < {float(getattr(config, 'AGENT_MIN_FORECAST_PROXY_PCT', 0.35)):.2f}%"
    if leader_score < float(getattr(config, "AGENT_MIN_LEADER_SCORE", 12.0)):
        return f"leader score {leader_score:.2f} < {float(getattr(config, 'AGENT_MIN_LEADER_SCORE', 12.0)):.2f}"
    if mode == "4h_leader_watch":
        max_range = float(getattr(config, "AGENT_4H_LEADER_MAX_DAILY_RANGE_PCT", 35.0))
        min_adx = float(getattr(config, "AGENT_4H_LEADER_MIN_ADX", 30.0))
        min_vol = float(getattr(config, "AGENT_4H_LEADER_MIN_VOL_X", 0.65))
        min_rsi = float(getattr(config, "AGENT_4H_LEADER_MIN_RSI", 50.0))
        max_rsi = float(getattr(config, "AGENT_4H_LEADER_MAX_RSI", 78.0))
        if daily_range > max_range:
            return f"daily range {daily_range:.2f}% > {max_range:.2f}%"
        if adx < min_adx:
            return f"ADX {adx:.1f} < {min_adx:.1f}"
        if vol_x < min_vol:
            return f"vol_x {vol_x:.2f} < {min_vol:.2f}"
        if rsi < min_rsi:
            return f"RSI {rsi:.1f} < {min_rsi:.1f}"
        if rsi > max_rsi:
            return f"RSI {rsi:.1f} > {max_rsi:.1f}"
        strength_reason = _four_h_leader_strength_reason(today_change_pct=today_change_pct, vol_x=vol_x)
        if strength_reason:
            return strength_reason
        return ""
    if daily_range > float(getattr(config, "AGENT_MAX_DAILY_RANGE_PCT", 14.0)):
        return f"daily range {daily_range:.2f}% > {float(getattr(config, 'AGENT_MAX_DAILY_RANGE_PCT', 14.0)):.2f}%"
    min_adx = float(
        getattr(config, "AGENT_TREND_MIN_ADX", 35.0)
        if mode == "trend"
        else getattr(config, "AGENT_MIN_ADX", 18.0)
    )
    if adx < min_adx:
        return f"ADX {adx:.1f} < {min_adx:.1f}"
    if vol_x < float(getattr(config, "AGENT_MIN_VOL_X", 1.0)):
        return f"vol_x {vol_x:.2f} < {float(getattr(config, 'AGENT_MIN_VOL_X', 1.0)):.2f}"
    if rsi > float(getattr(config, "AGENT_MAX_RSI", 72.5)):
        return f"RSI {rsi:.1f} > {float(getattr(config, 'AGENT_MAX_RSI', 72.5)):.1f}"
    min_signals = int(getattr(config, "AGENT_MIN_TODAY_SIGNALS", 2))
    min_accuracy = float(getattr(config, "AGENT_MIN_BEST_ACCURACY", 55.0))
    if not bool(getattr(report, "today_confirmed", False)):
        if int(getattr(report, "today_signals", 0)) < min_signals:
            return f"today signals {int(getattr(report, 'today_signals', 0))} < {min_signals}"
        if float(getattr(report, "best_accuracy", 0.0)) < min_accuracy:
            return f"best accuracy {float(getattr(report, 'best_accuracy', 0.0)):.1f} < {min_accuracy:.1f}"
    return ""


def _agent_soft_block_watch_rule(
    *,
    reason: str,
    rsi: float,
    adx: float,
    vol_x: float,
    daily_range: float,
) -> str:
    if not getattr(config, "AGENT_SOFT_BLOCK_WATCH_ALERTS_ENABLED", False):
        return ""
    if (
        reason.startswith("RSI ")
        and rsi <= float(getattr(config, "AGENT_SOFT_BLOCK_RSI_MAX", 75.0))
        and vol_x >= float(getattr(config, "AGENT_SOFT_BLOCK_RSI_MIN_VOL_X", 3.0))
    ):
        return "rsi_high_volume"
    if (
        reason == "agent mode disabled: impulse"
        and adx >= float(getattr(config, "AGENT_SOFT_BLOCK_IMPULSE_MIN_ADX", 15.0))
        and rsi <= float(getattr(config, "AGENT_SOFT_BLOCK_IMPULSE_MAX_RSI", 75.0))
        and vol_x >= float(getattr(config, "AGENT_SOFT_BLOCK_IMPULSE_MIN_VOL_X", 2.0))
    ):
        return "impulse_mode_watch"
    if reason.startswith("daily range ") and daily_range <= float(
        getattr(config, "AGENT_SOFT_BLOCK_DAILY_RANGE_MAX_PCT", 20.0)
    ):
        return "daily_range_watch"
    return ""


async def _maybe_send_soft_block_watch_alert(
    session: aiohttp.ClientSession,
    *,
    symbol: str,
    tf: str,
    mode: str,
    price: float,
    reason: str,
    rsi: float,
    adx: float,
    vol_x: float,
    daily_range: float,
    bar_ts: int,
) -> None:
    rule = _agent_soft_block_watch_rule(
        reason=reason,
        rsi=rsi,
        adx=adx,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    if not rule:
        return
    key = f"{symbol}|{tf}|{rule}"
    day_key = _local_day_key(bar_ts)
    if _SOFT_BLOCK_WATCH_ALERT_DAY.get(key) == day_key:
        return
    _SOFT_BLOCK_WATCH_ALERT_DAY[key] = day_key
    await _send_telegram(
        session,
        "WATCH ONLY - agent soft-block candidate\n\n"
        f"{symbol}  [{tf}]  {mode}\n"
        f"Price: {price:.6g}\n"
        f"RSI: {rsi:.1f}  ADX: {adx:.1f}  Vol x: {vol_x:.2f}\n"
        f"Daily range: {daily_range:.2f}%\n"
        f"Blocked: {reason}\n"
        f"Rule: {rule}\n"
        "No position opened.",
    )


def _determine_signal_mode(
    *,
    entry_ok: bool,
    brk_ok: bool,
    ret_ok: bool,
    surge_ok: bool,
    imp_ok: bool,
    tf: str,
    feat: dict,
    c: np.ndarray,
    i: int,
) -> Optional[str]:
    if brk_ok:
        return "breakout"
    if ret_ok:
        return "retest"
    if entry_ok:
        return get_entry_mode(feat, i)
    if surge_ok:
        return "impulse_speed"
    if imp_ok:
        return "impulse"
    aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)
    if aln_ok:
        return "alignment"
    return None


async def _entry_candidate(
    session: aiohttp.ClientSession,
    symbol: str,
    tf: str,
    data,
    c: np.ndarray,
    feat: dict,
) -> Optional[dict]:
    i = len(c) - 2
    entry_ok, _ = check_entry_conditions(feat, i, c, tf=tf)
    brk_ok, _ = check_breakout_conditions(feat, i)
    ret_ok, _ = check_retest_conditions(feat, i)
    surge_ok, _ = check_trend_surge_conditions(feat, i)
    imp_ok, _ = check_impulse_conditions(feat, i)
    aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)
    price = float(c[i])
    ema20 = float(feat["ema_fast"][i]) if np.isfinite(feat["ema_fast"][i]) else 0.0
    ema50 = float(feat["ema_slow"][i]) if np.isfinite(feat["ema_slow"][i]) else 0.0
    slope = float(feat["slope"][i]) if np.isfinite(feat["slope"][i]) else 0.0
    adx = float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 0.0
    rsi = float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0
    vol_x = float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 0.0
    daily_range = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
    macd_hist = float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else 0.0

    today_change_pct = _today_change_pct_from_data(data, c, i)
    forecast_proxy_pct = _forecast_proxy_pct(
        today_change_pct=today_change_pct,
        slope=slope,
        adx=adx,
        vol_x=vol_x,
        rsi=rsi,
    )
    mode = _determine_signal_mode(
        entry_ok=entry_ok,
        brk_ok=brk_ok,
        ret_ok=ret_ok,
        surge_ok=surge_ok,
        imp_ok=imp_ok,
        tf=tf,
        feat=feat,
        c=c,
        i=i,
    )
    min_context = float(getattr(config, "AGENT_4H_LEADER_MIN_CONTEXT_SCORE", 7.0))
    local_four_h_reason = _four_h_leader_watch_reason(
        tf=tf,
        price=price,
        ema20=ema20,
        ema50=ema50,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
        today_change_pct=today_change_pct,
        macd_hist=macd_hist,
        four_h_score=min_context,
        c=c,
        i=i,
    )
    if mode is None and local_four_h_reason:
        return None

    four_h_score, four_h_label = await _four_h_context_score(session, symbol)
    four_h_watch_reason = _four_h_leader_watch_reason(
        tf=tf,
        price=price,
        ema20=ema20,
        ema50=ema50,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
        today_change_pct=today_change_pct,
        macd_hist=macd_hist,
        four_h_score=four_h_score,
        c=c,
        i=i,
    )
    four_h_watch_ok = four_h_watch_reason == ""
    if four_h_watch_ok:
        normal_range_max = float(getattr(config, "AGENT_MAX_DAILY_RANGE_PCT", 14.0))
        normal_vol_min = float(getattr(config, "AGENT_MIN_VOL_X", 1.0))
        normal_rsi_max = float(getattr(config, "AGENT_MAX_RSI", 72.5))
        if (
            mode is None
            or mode not in _agent_allowed_modes()
            or daily_range > normal_range_max
            or vol_x < normal_vol_min
            or rsi > normal_rsi_max
        ):
            mode = "4h_leader_watch"

    if mode is None:
        return None

    report = analyze_coin(symbol, tf, data, from_scan=False)

    candidate_score = _entry_signal_score(
        mode=mode,
        price=price,
        ema20=ema20,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    candidate_score += _top_mover_score_bonus(today_change_pct)
    candidate_score += _forecast_return_score_bonus(forecast_proxy_pct)
    candidate_score += four_h_score * float(getattr(config, "FOUR_H_CONTEXT_SCORE_WEIGHT", 1.0))
    if mode == "4h_leader_watch":
        candidate_score += float(getattr(config, "AGENT_4H_LEADER_BONUS", 10.0))

    continuation_profile = _time_block_1h_continuation_profile(
        tf=tf,
        mode=mode,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    candidate_score += _time_block_1h_continuation_bonus(
        tf=tf,
        mode=mode,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )

    if mode != "4h_leader_watch" and _late_1h_continuation_guard(
        tf=tf,
        mode=mode,
        continuation_profile=continuation_profile,
        candidate_score=candidate_score,
        price=price,
        ema20=ema20,
        rsi=rsi,
        daily_range=daily_range,
    ):
        agentlog.log_blocked(symbol, tf, price, "late 1h continuation", signal_type="late_continuation", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
        return None

    impulse_guard_reason = "" if mode == "4h_leader_watch" else _impulse_speed_entry_guard(
        tf=tf,
        mode=mode,
        feat=feat,
        i=i,
        price=price,
        ema20=ema20,
        rsi=rsi,
        adx=adx,
        daily_range=daily_range,
    )
    if impulse_guard_reason:
        agentlog.log_blocked(symbol, tf, price, impulse_guard_reason, signal_type="impulse_guard", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
        return None

    if mode != "4h_leader_watch" and tf == "1h" and getattr(config, "MTF_ENABLED", True):
        mtf_ok, mtf_reason = await _check_mtf(
            session,
            symbol,
            mode=mode,
            candidate_score=candidate_score,
            slope=slope,
            adx=adx,
            rsi=rsi,
            vol_x=vol_x,
            daily_range=daily_range,
        )
        if not mtf_ok:
            agentlog.log_blocked(symbol, tf, price, f"MTF: {mtf_reason}", signal_type="mtf", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
            return None
        candidate_score -= _mtf_soft_penalty_from_reason(mtf_reason)

    if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False):
        min_score = _entry_score_floor(tf)
        if candidate_score < min_score:
            if not _entry_score_borderline_bypass_ok(
                tf=tf,
                mode=mode,
                candidate_score=candidate_score,
                min_score=min_score,
                price=price,
                ema20=ema20,
                slope=slope,
                adx=adx,
                rsi=rsi,
                vol_x=vol_x,
                daily_range=daily_range,
            ):
                agentlog.log_blocked(symbol, tf, price, f"entry score {candidate_score:.2f} < floor {min_score:.2f}", signal_type="entry_score", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
                return None

    leader_score = _leader_score(
        mode=mode,
        today_change_pct=today_change_pct,
        forecast_proxy_pct=forecast_proxy_pct,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
        today_confirmed=bool(getattr(report, "today_confirmed", False)),
        today_signals=int(getattr(report, "today_signals", 0)),
        best_accuracy=float(getattr(report, "best_accuracy", 0.0)),
    )
    leader_score = round(
        leader_score + four_h_score * float(getattr(config, "FOUR_H_CONTEXT_LEADER_WEIGHT", 0.8)),
        4,
    )
    block_reason = _candidate_block_reason(
        symbol=symbol,
        tf=tf,
        mode=mode,
        today_change_pct=today_change_pct,
        forecast_proxy_pct=forecast_proxy_pct,
        leader_score=leader_score,
        daily_range=daily_range,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        report=report,
    )
    if block_reason:
        agentlog.log_blocked(
            symbol,
            tf,
            price,
            block_reason,
            signal_type="agent_leader_filter",
            rsi=rsi,
            adx=adx,
            vol_x=vol_x,
            daily_range=daily_range,
        )
        try:
            await _maybe_send_soft_block_watch_alert(
                session,
                symbol=symbol,
                tf=tf,
                mode=mode,
                price=price,
                reason=block_reason,
                rsi=rsi,
                adx=adx,
                vol_x=vol_x,
                daily_range=daily_range,
                bar_ts=int(data["t"][i]),
            )
        except Exception as exc:
            log.warning("agent soft-block watch alert failed for %s [%s]: %s", symbol, tf, exc)
        return None

    trail_k, max_hold_bars = _entry_params(mode, tf)
    atr_val = float(feat["atr"][i]) if np.isfinite(feat["atr"][i]) else 0.0
    trail_stop = price - trail_k * atr_val if atr_val > 0 else 0.0
    return {
        "mode": mode,
        "price": price,
        "ema20": ema20,
        "slope": slope,
        "adx": adx,
        "rsi": rsi,
        "vol_x": vol_x,
        "daily_range": daily_range,
        "macd_hist": macd_hist,
        "trail_k": trail_k,
        "max_hold_bars": max_hold_bars,
        "trail_stop": trail_stop,
        "forecast_return_pct": float(forecast_proxy_pct),
        "today_change_pct": float(today_change_pct),
        "leader_score": float(leader_score),
        "four_h_context_score": float(four_h_score),
        "four_h_context_label": str(four_h_label),
        "today_confirmed": bool(getattr(report, "today_confirmed", False)),
        "today_signals": int(getattr(report, "today_signals", 0)),
        "best_accuracy": float(getattr(report, "best_accuracy", 0.0)),
        "mode_cluster": _mode_cluster(mode),
        "group": _symbol_group(symbol),
        "bar_ts": int(data["t"][i]),
        "bar_idx": i,
    }


async def _evaluate_open_position(
    session: aiohttp.ClientSession,
    pos: AgentPosition,
    data,
    c: np.ndarray,
    feat: dict,
) -> Optional[dict]:
    i = len(c) - 2
    live_report = analyze_coin(pos.symbol, pos.tf, data, from_scan=False)
    pos.today_change_pct = _today_change_pct_from_data(data, c, i)
    pos.forecast_return_pct = _forecast_proxy_pct(
        today_change_pct=pos.today_change_pct,
        slope=float(feat["slope"][i]) if np.isfinite(feat["slope"][i]) else 0.0,
        adx=float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 0.0,
        vol_x=float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 0.0,
        rsi=float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0,
    )
    pos.leader_score = _leader_score(
        mode=pos.signal_mode,
        today_change_pct=pos.today_change_pct,
        forecast_proxy_pct=pos.forecast_return_pct,
        slope=float(feat["slope"][i]) if np.isfinite(feat["slope"][i]) else 0.0,
        adx=float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 0.0,
        rsi=float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0,
        vol_x=float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 0.0,
        daily_range=float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0,
        today_confirmed=bool(getattr(live_report, "today_confirmed", False)),
        today_signals=int(getattr(live_report, "today_signals", 0)),
        best_accuracy=float(getattr(live_report, "best_accuracy", 0.0)),
    )

    entry_idx: Optional[int] = None
    for idx, ts in enumerate(data["t"]):
        if int(ts) == int(pos.entry_ts):
            entry_idx = idx
            break
    if entry_idx is None:
        pos.bars_elapsed = max(0, (int(data["t"][i]) - int(pos.entry_ts)) // _tf_bar_ms(pos.tf))
    else:
        pos.bars_elapsed = max(0, i - entry_idx)
    if pos.bars_elapsed <= 0:
        pos.last_bar_ts = int(data["t"][i])
        return None

    close_now = float(c[i])
    pos.mark_price = close_now
    current_pnl = pos.pnl_pct(close_now)
    atr_now = float(feat["atr"][i]) if np.isfinite(feat["atr"][i]) else 0.0
    effective_trail_k = pos.trail_k

    if _continuation_profit_lock_active(
        tf=pos.tf,
        mode=pos.signal_mode,
        entry_rsi=pos.entry_rsi,
        bars_elapsed=pos.bars_elapsed,
        current_pnl=current_pnl,
        predictions=pos.predictions,
    ):
        effective_trail_k = min(
            pos.trail_k,
            float(getattr(config, "CONTINUATION_PROFIT_LOCK_TRAIL_K", 1.4)),
        )
        protect_floor_pct = float(getattr(config, "CONTINUATION_PROFIT_LOCK_FLOOR_PCT", 0.10))
        protect_floor = pos.entry_price * (1.0 + protect_floor_pct / 100.0)
        if protect_floor > pos.trail_stop:
            pos.trail_stop = protect_floor
        if pos.tf == "1h":
            micro_data = await fetch_klines(session, pos.symbol, "15m", limit=config.LIVE_LIMIT)
            if micro_data is not None and len(micro_data) >= 60:
                micro_c, micro_feat = await _compute_features(micro_data)
                micro_reason = _continuation_micro_exit_reason(
                    tf=pos.tf,
                    mode=pos.signal_mode,
                    bars_elapsed=pos.bars_elapsed,
                    data_15m=micro_data,
                    feat_15m=micro_feat,
                )
                if micro_reason:
                    exit_idx = len(micro_c) - 2
                    exit_price = float(micro_c[exit_idx])
                    return {"price": exit_price, "reason": micro_reason, "bar_ts": int(micro_data["t"][exit_idx])}

    if _short_mode_profit_lock_active(
        tf=pos.tf,
        mode=pos.signal_mode,
        bars_elapsed=pos.bars_elapsed,
        current_pnl=current_pnl,
        predictions=pos.predictions,
    ):
        effective_trail_k = min(
            effective_trail_k,
            float(getattr(config, "SHORT_MODE_PROFIT_LOCK_TRAIL_K", 1.2)),
        )
        protect_floor_pct = float(getattr(config, "SHORT_MODE_PROFIT_LOCK_FLOOR_PCT", 0.05))
        protect_floor = pos.entry_price * (1.0 + protect_floor_pct / 100.0)
        if protect_floor > pos.trail_stop:
            pos.trail_stop = protect_floor

    if atr_now > 0:
        new_trail = close_now - pos.trail_k * atr_now
        if effective_trail_k < pos.trail_k:
            new_trail = max(new_trail, close_now - effective_trail_k * atr_now)
        if new_trail > pos.trail_stop:
            pos.trail_stop = new_trail

    if pos.trail_stop > 0 and close_now < pos.trail_stop:
        return {"price": close_now, "reason": f"ATR trail broken ({pos.trail_stop:.6g})", "bar_ts": int(data["t"][i])}

    if pos.bars_elapsed >= pos.max_hold_bars and not _time_exit_should_wait(feat, i, close_now):
        return {"price": close_now, "reason": f"time ({pos.max_hold_bars} bars)", "bar_ts": int(data["t"][i])}

    fast_loss_reason = _fast_loss_ema_exit_reason(
        tf=pos.tf,
        mode=pos.signal_mode,
        bars_elapsed=pos.bars_elapsed,
        current_pnl=current_pnl,
        close_now=close_now,
        ema20=float(feat["ema_fast"][i]) if np.isfinite(feat["ema_fast"][i]) else np.nan,
        rsi=float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else np.nan,
    )
    if fast_loss_reason:
        return {"price": close_now, "reason": fast_loss_reason, "bar_ts": int(data["t"][i])}

    quality_recheck_reason = _post_entry_quality_recheck_reason(pos, feat, i)
    if quality_recheck_reason:
        return {"price": close_now, "reason": quality_recheck_reason, "bar_ts": int(data["t"][i])}

    weak_bars = _min_weak_exit_bars(pos.signal_mode)
    reason = check_exit_conditions(feat, i, c)
    if reason:
        if reason.startswith("⚠️ WEAK:") and pos.bars_elapsed < weak_bars:
            reason = None
    if reason:
        return {"price": close_now, "reason": reason, "bar_ts": int(data["t"][i])}

    pos.last_bar_ts = int(data["t"][i])
    return None


async def _evaluate_existing_symbol_tf(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
    symbol_cooldown_until: Dict[str, int],
    symbol: str,
    tf: str,
) -> tuple[bool, Optional[str]]:
    key = _position_key(symbol, tf)
    pos = positions.get(key)
    if pos is None:
        return False, None
    data = await fetch_klines(session, symbol, tf, limit=config.LIVE_LIMIT)
    if data is None or len(data) < 60:
        return False, None
    c, feat = await _compute_features(data)
    i = len(c) - 2
    current_bar_ts = int(data["t"][i])
    exit_event = await _evaluate_open_position(session, pos, data, c, feat)
    if exit_event is None:
        return False, None

    exit_price = float(exit_event["price"])
    exit_reason = str(exit_event["reason"])
    agentlog.log_exit(
        sym=pos.symbol,
        tf=pos.tf,
        mode=pos.signal_mode,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        reason=exit_reason,
        bars_held=pos.bars_elapsed,
        trail_k=pos.trail_k,
    )
    await _send_exit_alert(session, pos, exit_price, exit_reason)
    last_exit_bar[key] = int(exit_event["bar_ts"])
    symbol_cooldown_until[symbol] = _next_local_day_start_utc_ms(int(exit_event["bar_ts"]))
    del positions[key]
    return False, None


async def _scan_symbol_tf_candidate(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
    symbol_cooldown_until: Dict[str, int],
    symbol: str,
    tf: str,
) -> Optional[dict]:
    key = _position_key(symbol, tf)
    if key in positions:
        return None
    if any(pos.symbol == symbol for pos in positions.values()):
        return None
    data = await fetch_klines(session, symbol, tf, limit=config.LIVE_LIMIT)
    if data is None or len(data) < 60:
        return None
    c, feat = await _compute_features(data)
    i = len(c) - 2
    current_bar_ts = int(data["t"][i])
    if last_exit_bar.get(key) == current_bar_ts:
        return None
    cooldown_until = int(symbol_cooldown_until.get(symbol, 0))
    candidate = await _entry_candidate(session, symbol, tf, data, c, feat)
    if candidate is None:
        return None
    if cooldown_until and current_bar_ts < cooldown_until and not _candidate_bypasses_symbol_cooldown(candidate):
        agentlog.log_blocked(
            symbol,
            tf,
            float(candidate["price"]),
            "symbol cooldown active",
            signal_type="symbol_cooldown",
            rsi=float(candidate["rsi"]),
            adx=float(candidate["adx"]),
            vol_x=float(candidate["vol_x"]),
            daily_range=float(candidate["daily_range"]),
        )
        return None
    candidate["symbol"] = symbol
    candidate["tf"] = tf
    candidate["key"] = key
    return candidate


def _candidate_fits_portfolio(candidate: dict, positions: Dict[str, AgentPosition], selected: list[dict]) -> tuple[bool, str]:
    symbol = str(candidate["symbol"])
    mode = str(candidate["mode"])
    cluster = str(candidate.get("mode_cluster", _mode_cluster(mode)))
    group = str(candidate.get("group", ""))

    current_symbols = {pos.symbol for pos in positions.values()}
    selected_symbols = {str(item["symbol"]) for item in selected}
    if symbol in current_symbols or symbol in selected_symbols:
        return False, f"symbol already selected/open: {symbol}"

    if bool(getattr(config, "AGENT_REQUIRE_DISTINCT_MODE_CLUSTERS", False)):
        occupied_clusters = {_mode_cluster(pos.signal_mode) for pos in positions.values()}
        occupied_clusters.update(str(item.get("mode_cluster", _mode_cluster(str(item["mode"])))) for item in selected)
        if cluster in occupied_clusters:
            return False, f"same pattern cluster already open: {cluster}"

    max_cluster = int(getattr(config, "AGENT_MAX_POSITIONS_PER_MODE_CLUSTER", 2))
    if max_cluster > 0:
        cluster_count = sum(1 for pos in positions.values() if _mode_cluster(pos.signal_mode) == cluster)
        cluster_count += sum(
            1
            for item in selected
            if str(item.get("mode_cluster", _mode_cluster(str(item["mode"])))) == cluster
        )
        if cluster_count >= max_cluster:
            return False, f"pattern cluster cap reached: {cluster}"

    max_group = int(getattr(config, "AGENT_MAX_POSITIONS_PER_GROUP", 1))
    if max_group > 0 and group:
        group_count = sum(1 for pos in positions.values() if _symbol_group(pos.symbol) == group)
        group_count += sum(1 for item in selected if str(item.get("group", "")) == group)
        if group_count >= max_group:
            return False, f"group cap reached: {group}"

    return True, ""


def _candidate_rank_key(candidate: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(candidate.get("leader_score", 0.0)),
        float(candidate.get("today_change_pct", 0.0)),
        float(candidate.get("forecast_return_pct", 0.0)),
        float(candidate.get("four_h_context_score", 0.0)),
        float(candidate.get("adx", 0.0)),
        float(candidate.get("vol_x", 0.0)),
    )


def _position_rank_key(pos: AgentPosition) -> tuple[float, float, float, float, float, float]:
    return (
        float(getattr(pos, "leader_score", 0.0)),
        float(getattr(pos, "today_change_pct", 0.0)),
        float(getattr(pos, "forecast_return_pct", 0.0)),
        float(getattr(pos, "four_h_context_score", 0.0)),
        float(getattr(pos, "entry_adx", 0.0)),
        float(getattr(pos, "entry_vol_x", 0.0)),
    )


def _candidate_better_than_position(candidate: dict, pos: AgentPosition) -> bool:
    min_delta = float(getattr(config, "AGENT_REPLACEMENT_MIN_LEADER_DELTA", 0.0))
    candidate_leader = float(candidate.get("leader_score", 0.0))
    position_leader = float(getattr(pos, "leader_score", 0.0))
    if candidate_leader < position_leader + min_delta:
        return False
    return _candidate_rank_key(candidate) > _position_rank_key(pos)


def _candidate_bypasses_symbol_cooldown(candidate: dict) -> bool:
    if not bool(getattr(config, "AGENT_4H_LEADER_BYPASS_SYMBOL_COOLDOWN", True)):
        return False
    if str(candidate.get("mode", "")) != "4h_leader_watch":
        return False
    min_leader = float(getattr(config, "AGENT_4H_LEADER_COOLDOWN_MIN_LEADER_SCORE", 55.0))
    return float(candidate.get("leader_score", 0.0)) >= min_leader


def _find_replacement_target(
    candidate: dict,
    positions: Dict[str, AgentPosition],
) -> tuple[Optional[str], Optional[AgentPosition], str]:
    symbol = str(candidate["symbol"])
    if any(pos.symbol == symbol for pos in positions.values()):
        return None, None, f"symbol already open: {symbol}"

    ranked_open = sorted(positions.items(), key=lambda item: _position_rank_key(item[1]))
    last_reject = ""
    for key, pos in ranked_open:
        if not _candidate_better_than_position(candidate, pos):
            last_reject = (
                f"candidate leader {float(candidate.get('leader_score', 0.0)):.2f} "
                f"<= {pos.symbol} leader {float(getattr(pos, 'leader_score', 0.0)):.2f}"
            )
            continue

        remaining = dict(positions)
        remaining.pop(key, None)
        ok, reason = _candidate_fits_portfolio(candidate, remaining, [])
        if ok:
            return key, pos, ""
        last_reject = reason

    return None, None, last_reject or "no weaker replaceable position"


async def _open_selected_candidate(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    candidate: dict,
) -> str:
    pos = AgentPosition(
        symbol=str(candidate["symbol"]),
        tf=str(candidate["tf"]),
        entry_price=float(candidate["price"]),
        entry_bar=int(candidate["bar_idx"]),
        entry_ts=int(candidate["bar_ts"]),
        entry_ema20=float(candidate["ema20"]),
        entry_slope=float(candidate["slope"]),
        entry_adx=float(candidate["adx"]),
        entry_rsi=float(candidate["rsi"]),
        entry_vol_x=float(candidate["vol_x"]),
        forecast_return_pct=float(candidate["forecast_return_pct"]),
        today_change_pct=float(candidate["today_change_pct"]),
        leader_score=float(candidate.get("leader_score", 0.0)),
        four_h_context_score=float(candidate.get("four_h_context_score", 0.0)),
        four_h_context_label=str(candidate.get("four_h_context_label", "")),
        predictions={h: None for h in _position_forward_horizons(str(candidate["tf"]), str(candidate["mode"]))},
        signal_mode=str(candidate["mode"]),
        trail_k=float(candidate["trail_k"]),
        max_hold_bars=int(candidate["max_hold_bars"]),
        trail_stop=float(candidate["trail_stop"]),
        last_bar_ts=int(candidate["bar_ts"]),
        mark_price=float(candidate["price"]),
    )
    positions[str(candidate["key"])] = pos
    _save_positions(positions)
    agentlog.log_entry(
        sym=pos.symbol,
        tf=pos.tf,
        mode=pos.signal_mode,
        price=pos.entry_price,
        ema20=pos.entry_ema20,
        slope=pos.entry_slope,
        rsi=pos.entry_rsi,
        adx=pos.entry_adx,
        vol_x=pos.entry_vol_x,
        macd_hist=float(candidate["macd_hist"]),
        daily_range=float(candidate["daily_range"]),
        trail_k=pos.trail_k,
        max_hold_bars=pos.max_hold_bars,
        forecast_return_pct=pos.forecast_return_pct,
        today_change_pct=pos.today_change_pct,
    )
    await _send_entry_alert(session, pos)
    return f"{pos.symbol} [{pos.tf}] {pos.signal_mode}"


async def _replace_position_with_candidate(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
    symbol_cooldown_until: Dict[str, int],
    old_key: str,
    old_pos: AgentPosition,
    candidate: dict,
) -> str:
    exit_price = float(old_pos.mark_price or old_pos.entry_price)
    exit_ts = int(old_pos.last_bar_ts or old_pos.entry_ts)
    reason = (
        "portfolio replacement: "
        f"{candidate['symbol']} leader {float(candidate.get('leader_score', 0.0)):.2f} "
        f"> {old_pos.symbol} leader {float(getattr(old_pos, 'leader_score', 0.0)):.2f}"
    )
    agentlog.log_exit(
        sym=old_pos.symbol,
        tf=old_pos.tf,
        mode=old_pos.signal_mode,
        entry_price=old_pos.entry_price,
        exit_price=exit_price,
        reason=reason,
        bars_held=old_pos.bars_elapsed,
        trail_k=old_pos.trail_k,
    )
    await _send_exit_alert(session, old_pos, exit_price, reason)
    last_exit_bar[old_key] = exit_ts
    symbol_cooldown_until[old_pos.symbol] = _next_local_day_start_utc_ms(exit_ts)
    positions.pop(old_key, None)
    _save_positions(positions)

    opened = await _open_selected_candidate(session, positions, candidate)
    return f"{opened} replaced {old_pos.symbol} [{old_pos.tf}]"


async def _prune_positions_to_limit(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
    symbol_cooldown_until: Dict[str, int],
) -> None:
    max_positions = _agent_allowed_slots(positions)
    if len(positions) <= max_positions:
        return

    if bool(getattr(config, "UNIFIED_PORTFOLIO_ENABLED", True)):
        agent_raw = {key: _agent_position_to_raw(pos) for key, pos in positions.items()}
        unified = ranked_unified_positions(
            load_main_positions_raw(),
            agent_raw,
            limit=_unified_portfolio_limit(),
        )
        unified_agent_keep = [
            str(row["key"])
            for row in unified
            if str(row.get("source")) == "agent"
        ]
        ranked = [(key, positions[key]) for key in unified_agent_keep if key in positions]
        ranked.extend(
            sorted(
                [(key, pos) for key, pos in positions.items() if key not in unified_agent_keep],
                key=lambda item: _position_rank_key(item[1]),
                reverse=True,
            )
        )
    else:
        ranked = sorted(positions.items(), key=lambda item: _position_rank_key(item[1]), reverse=True)
    keep_keys: list[str] = []
    keep_clusters: set[str] = set()
    require_distinct = bool(getattr(config, "AGENT_REQUIRE_DISTINCT_MODE_CLUSTERS", False))
    max_cluster = int(getattr(config, "AGENT_MAX_POSITIONS_PER_MODE_CLUSTER", 2))
    cluster_counts: dict[str, int] = {}
    for key, pos in ranked:
        cluster = _mode_cluster(pos.signal_mode)
        if require_distinct and cluster in keep_clusters:
            continue
        if max_cluster > 0 and cluster_counts.get(cluster, 0) >= max_cluster:
            continue
        keep_keys.append(key)
        keep_clusters.add(cluster)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        if len(keep_keys) >= max_positions:
            break
    for key, _ in ranked:
        if len(keep_keys) >= max_positions:
            break
        if key not in keep_keys:
            keep_keys.append(key)

    for key, pos in list(positions.items()):
        if key in keep_keys:
            continue
        exit_price = float(pos.mark_price or pos.entry_price)
        exit_reason = f"portfolio prune: keep top {max_positions} leader candidates"
        agentlog.log_exit(
            sym=pos.symbol,
            tf=pos.tf,
            mode=pos.signal_mode,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            reason=exit_reason,
            bars_held=pos.bars_elapsed,
            trail_k=pos.trail_k,
        )
        await _send_exit_alert(session, pos, exit_price, exit_reason)
        last_exit_bar[key] = int(pos.last_bar_ts or pos.entry_ts)
        symbol_cooldown_until[pos.symbol] = _next_local_day_start_utc_ms(int(pos.last_bar_ts or pos.entry_ts))
        del positions[key]
    _save_positions(positions)


async def _run_cycle(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
    symbol_cooldown_until: Dict[str, int],
) -> tuple[list[str], int]:
    symbols = list(config.load_watchlist())
    bull, btc_price, btc_ema50 = await is_bull_day(session)
    regime = await detect_market_regime(session)
    config._bull_day_active = bull
    config._current_regime = regime.name
    config._btc_vs_ema50 = ((btc_price / btc_ema50) - 1.0) * 100.0 if btc_ema50 > 0 else 0.0

    entries: list[str] = []

    for key, pos in list(positions.items()):
        try:
            await _evaluate_existing_symbol_tf(
                session,
                positions,
                last_exit_bar,
                symbol_cooldown_until,
                pos.symbol,
                pos.tf,
            )
        except Exception as exc:
            log.warning("agent existing-position error %s [%s]: %s", pos.symbol, pos.tf, exc)

    await _prune_positions_to_limit(session, positions, last_exit_bar, symbol_cooldown_until)

    max_positions = _agent_allowed_slots(positions)
    sem = asyncio.Semaphore(12)
    candidates: list[dict] = []

    async def _wrapped_candidate(sym: str, tf: str) -> None:
        async with sem:
            try:
                candidate = await _scan_symbol_tf_candidate(
                    session,
                    positions,
                    last_exit_bar,
                    symbol_cooldown_until,
                    sym,
                    tf,
                )
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as exc:
                log.warning("agent candidate scan error %s [%s]: %s", sym, tf, exc)

    # Always scan the watchlist, even when the portfolio is full. Full portfolios
    # are improved through replacement rather than becoming blind to new leaders.
    await asyncio.gather(*[_wrapped_candidate(sym, tf) for sym in symbols for tf in config.TIMEFRAMES])

    best_by_symbol: Dict[str, dict] = {}
    for candidate in candidates:
        symbol = str(candidate["symbol"])
        current = best_by_symbol.get(symbol)
        if current is None or _candidate_rank_key(candidate) > _candidate_rank_key(current):
            best_by_symbol[symbol] = candidate

    ranked_candidates = sorted(best_by_symbol.values(), key=_candidate_rank_key, reverse=True)

    selected: list[dict] = []
    for candidate in ranked_candidates:
        if len(positions) + len(selected) >= max_positions:
            break
        ok, reason = _candidate_fits_portfolio(candidate, positions, selected)
        if not ok:
            agentlog.log_blocked(
                str(candidate["symbol"]),
                str(candidate["tf"]),
                float(candidate["price"]),
                reason,
                signal_type="agent_portfolio_filter",
                rsi=float(candidate["rsi"]),
                adx=float(candidate["adx"]),
                vol_x=float(candidate["vol_x"]),
                daily_range=float(candidate["daily_range"]),
            )
            continue
        selected.append(candidate)

    for candidate in selected:
        entries.append(await _open_selected_candidate(session, positions, candidate))

    if (
        max_positions > 0
        and bool(getattr(config, "AGENT_REPLACEMENT_ENABLED", True))
        and len(positions) >= max_positions
    ):
        replacements_done = 0
        max_replacements = max(0, int(getattr(config, "AGENT_MAX_REPLACEMENTS_PER_CYCLE", max_positions)))
        for candidate in ranked_candidates:
            if replacements_done >= max_replacements:
                break
            if any(pos.symbol == str(candidate["symbol"]) for pos in positions.values()):
                continue

            old_key, old_pos, reason = _find_replacement_target(candidate, positions)
            if old_key is None or old_pos is None:
                if reason:
                    agentlog.log_blocked(
                        str(candidate["symbol"]),
                        str(candidate["tf"]),
                        float(candidate["price"]),
                        f"replacement skipped: {reason}",
                        signal_type="agent_replacement_filter",
                        rsi=float(candidate["rsi"]),
                        adx=float(candidate["adx"]),
                        vol_x=float(candidate["vol_x"]),
                        daily_range=float(candidate["daily_range"]),
                    )
                continue

            entries.append(
                await _replace_position_with_candidate(
                    session,
                    positions,
                    last_exit_bar,
                    symbol_cooldown_until,
                    old_key,
                    old_pos,
                    candidate,
                )
            )
            replacements_done += 1

    agentlog.log_analysis(
        n_scanned=len(symbols) * len(config.TIMEFRAMES),
        n_entries=len(entries),
        n_open_positions=len(positions),
    )
    _save_positions(positions)
    _save_state(last_exit_bar, symbol_cooldown_until, _SOFT_BLOCK_WATCH_ALERT_DAY)
    return entries, len(symbols) * len(config.TIMEFRAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone market signal agent for watchlist.")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit")
    parser.add_argument("--poll-sec", type=int, default=None, help="Override polling interval")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    positions = _load_positions()
    last_exit_bar, symbol_cooldown_until, restored_soft_block_watch = _load_state()
    _SOFT_BLOCK_WATCH_ALERT_DAY.update(restored_soft_block_watch)
    if args.poll_sec:
        config.POLL_SEC = int(args.poll_sec)

    log.info("market agent started with %d restored positions", len(positions))
    headers = {"User-Agent": "market-signal-agent/1.0"}
    started_at = _status_now()
    poll_sec = max(5, int(getattr(config, "POLL_SEC", 60)))
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            cycle_started_at = _status_now()
            _write_status(
                started_at=started_at,
                poll_sec=poll_sec,
                cycle_running=True,
                n_open_positions=len(positions),
                last_cycle_started_at=cycle_started_at,
                last_cycle_finished_at=None,
                last_cycle_stats={},
                last_error="",
            )
            try:
                entries, n_scanned = await _run_cycle(session, positions, last_exit_bar, symbol_cooldown_until)
                cycle_finished_at = _status_now()
                _write_status(
                    started_at=started_at,
                    poll_sec=poll_sec,
                    cycle_running=False,
                    n_open_positions=len(positions),
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_finished_at=cycle_finished_at,
                    last_cycle_stats={
                        "n_scanned": int(n_scanned),
                        "n_entries": int(len(entries)),
                        "n_open_positions": int(len(positions)),
                        "bull": bool(getattr(config, "_bull_day_active", False)),
                    },
                    last_error="",
                )
            except Exception as exc:
                cycle_finished_at = _status_now()
                log.exception("market agent cycle error: %s", exc)
                _write_status(
                    started_at=started_at,
                    poll_sec=poll_sec,
                    cycle_running=False,
                    n_open_positions=len(positions),
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_finished_at=cycle_finished_at,
                    last_cycle_stats={},
                    last_error=str(exc),
                )
                raise
            if cycle_started_at == started_at:
                log.info("first cycle done: entries=%d open_positions=%d", len(entries), len(positions))
            if args.once:
                return
            await asyncio.sleep(poll_sec)


def main() -> None:
    args = parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
