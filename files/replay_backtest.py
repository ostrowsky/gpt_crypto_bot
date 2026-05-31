from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import numpy as np

import config
from indicators import compute_features
from monitor import (
    _candidate_signal_flags,
    _entry_signal_score,
    _impulse_speed_entry_guard,
    _is_fresh_priority_candidate,
    _ml_candidate_ranker_components,
    _ml_candidate_ranker_runtime_bonus,
    _signal_priority,
    _top_gainer_chase_guard_reason,
)
from strategy import (
    _early_15m_continuation_entry_ok,
    check_alignment_conditions,
    check_breakout_conditions,
    check_entry_conditions,
    check_exit_conditions,
    check_impulse_conditions,
    check_retest_conditions,
    check_trend_surge_conditions,
    get_effective_entry_mode,
    get_entry_mode,
)


BINANCE_URL = "https://api.binance.com/api/v3/klines"
BAR_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}
_ML_MODEL_CACHE: Optional[dict] = None
OBJECTIVE_TZ = "Europe/Budapest"
OBJECTIVE_CUTOFF_HOUR = 22
BOT_EVENTS_PATH = Path(__file__).resolve().with_name("bot_events.jsonl")


@dataclass
class ReplayTrade:
    sym: str
    tf: str
    mode: str
    entry_ts: int
    entry_price: float
    entry_i: int
    trail_k: float
    max_hold_bars: int
    trail_stop: float
    entry_score: float = 0.0
    ranker_final_score: float = 0.0
    ranker_top_gainer_prob: float = 0.0
    top_gainer_score: float = 0.0
    entry_rsi: float = 50.0
    entry_daily_range: float = 0.0
    entry_intraday_change_pct: float = 0.0
    capture_ratio_at_entry: Optional[float] = None
    lead_time_to_final_top_min: Optional[float] = None
    day_open_price: float = 0.0
    day_final_price: float = 0.0
    exit_ts: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    ret_3: Optional[float] = None
    ret_5: Optional[float] = None
    ret_10: Optional[float] = None
    max_price_since_entry: float = 0.0
    min_price_since_entry: float = 0.0
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0
    exit_efficiency: Optional[float] = None
    giveback_pct: float = 0.0
    cooldown_blocked_count: int = 0
    cooldown_positive_blocked_count: int = 0
    cooldown_harm_pct: float = 0.0
    protected_exit_active: bool = False
    protected_exit_start_bar: int = -1
    protected_exit_original_reason: str = ""
    discriminator_exit_active: bool = False
    discriminator_exit_start_bar: int = -1
    discriminator_exit_original_reason: str = ""
    discriminator_exit_score: float = 0.0
    partial_exit_taken: bool = False
    partial_exit_fraction: float = 0.0
    partial_exit_price: float = 0.0
    partial_exit_ts: int = 0
    partial_exit_reason: str = ""

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0 or self.exit_price <= 0:
            return 0.0
        final_pnl = (self.exit_price / self.entry_price - 1.0) * 100.0
        if self.partial_exit_taken and self.partial_exit_price > 0 and self.partial_exit_fraction > 0:
            fraction = max(0.0, min(float(self.partial_exit_fraction), 1.0))
            partial_pnl = (self.partial_exit_price / self.entry_price - 1.0) * 100.0
            return partial_pnl * fraction + final_pnl * (1.0 - fraction)
        return final_pnl


@dataclass
class ReplayCandidate:
    sym: str
    tf: str
    mode: str
    ts_ms: int
    i: int
    price: float
    trail_k: float
    max_hold_bars: int
    score: float
    ranker_final_score: float = 0.0
    ranker_top_gainer_prob: float = 0.0
    top_gainer_score: float = 0.0
    four_h_context_score: float = 0.0
    four_h_context_label: str = ""
    rsi: float = 50.0
    daily_range: float = 0.0
    vol_x: float = 0.0
    adx: float = 0.0
    intraday_change_pct: float = 0.0


@dataclass
class ReplayRunStats:
    candidates_total: int = 0
    skipped_portfolio_full: int = 0
    replacements_total: int = 0
    replacements_improved: int = 0
    replacements_worsened: int = 0
    skipped_top_gainer_score: int = 0
    skipped_cluster_cap: int = 0
    cooldown_skipped_candidates: int = 0
    cooldown_positive_skips: int = 0
    cooldown_harm_pct: float = 0.0
    temporal_structural_events_loaded: int = 0
    temporal_wakeup_events_loaded: int = 0
    temporal_rescue_admitted: int = 0
    suspicious_reentry_windows: int = 0
    suspicious_reentry_admitted: int = 0


def _load_ml_model_payload() -> dict:
    global _ML_MODEL_CACHE
    if _ML_MODEL_CACHE is not None:
        return _ML_MODEL_CACHE
    try:
        raw = json.loads((Path(__file__).resolve().parent / "ml_signal_model.json").read_text(encoding="utf-8"))
        _ML_MODEL_CACHE = raw if isinstance(raw, dict) else {}
    except Exception:
        _ML_MODEL_CACHE = {}
    return _ML_MODEL_CACHE


def _load_ml_segment_payloads() -> dict:
    raw = _load_ml_model_payload()
    return raw.get("segment_model_payloads", {}) if isinstance(raw, dict) else {}


def _objective_tz() -> timezone:
    try:
        return ZoneInfo(OBJECTIVE_TZ)
    except ZoneInfoNotFoundError:
        return timezone.utc


def _finite_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f):
        return None
    return round(value_f, digits)


def _entry_day_metrics(data: np.ndarray, entry_i: int, entry_price: float) -> dict:
    if entry_i < 0 or entry_i >= len(data["t"]) or entry_price <= 0:
        return {
            "capture_ratio_at_entry": None,
            "lead_time_to_final_top_min": None,
            "day_open_price": 0.0,
            "day_final_price": 0.0,
        }
    tz = _objective_tz()
    entry_dt = datetime.fromtimestamp(int(data["t"][entry_i]) / 1000, tz=timezone.utc).astimezone(tz)
    start_local = datetime.combine(entry_dt.date(), datetime.min.time(), tzinfo=tz)
    cutoff_local = datetime.combine(
        entry_dt.date(),
        datetime.min.time().replace(hour=OBJECTIVE_CUTOFF_HOUR),
        tzinfo=tz,
    )
    final_local = cutoff_local if entry_dt <= cutoff_local else datetime.combine(
        entry_dt.date(),
        datetime.max.time(),
        tzinfo=tz,
    )
    start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
    final_ms = int(final_local.astimezone(timezone.utc).timestamp() * 1000)

    day_open_idx = _find_last_closed_index(data["t"], start_ms)
    if day_open_idx is None:
        candidates = np.where(data["t"] >= start_ms)[0]
        day_open_idx = int(candidates[0]) if len(candidates) else 0
    day_final_idx = _find_last_closed_index(data["t"], final_ms)
    if day_final_idx is None:
        day_final_idx = len(data["c"]) - 2
    day_open = float(data["o"][day_open_idx]) if "o" in data.dtype.names else float(data["c"][day_open_idx])
    day_final = float(data["c"][day_final_idx])
    move = day_final - day_open
    capture_ratio = None
    if move > 0:
        capture_ratio = max(0.0, min(1.5, (day_final - entry_price) / move))
    lead_min = max(0.0, (cutoff_local - entry_dt).total_seconds() / 60.0)
    return {
        "capture_ratio_at_entry": _finite_or_none(capture_ratio),
        "lead_time_to_final_top_min": _finite_or_none(lead_min, digits=2),
        "day_open_price": round(day_open, 10),
        "day_final_price": round(day_final, 10),
    }


def _load_ml_general_payload() -> Optional[dict]:
    raw = _load_ml_model_payload()
    if not isinstance(raw, dict):
        return None
    if "feature_names" not in raw or "model" not in raw:
        return None
    return raw


def _select_ml_payload(signal_type: str, is_bull_day: bool) -> Optional[dict]:
    if getattr(config, "ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE", True):
        segment_key = f"{signal_type}|{'bull' if is_bull_day else 'nonbull'}"
        segment_payload = _load_ml_segment_payloads().get(segment_key)
        if isinstance(segment_payload, dict) and "feature_names" in segment_payload and "model" in segment_payload:
            return segment_payload
    return _load_ml_general_payload()


def _build_bull_day_context(
    btc_data: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if btc_data is None or len(btc_data) < 55:
        return None
    c_btc = btc_data["c"].astype(float)
    feat = compute_features(
        btc_data["o"].astype(float),
        btc_data["h"].astype(float),
        btc_data["l"].astype(float),
        c_btc,
        btc_data["v"],
    )
    ema50 = feat["ema_slow"]
    ts_arr = btc_data["t"].astype(np.int64)
    state_arr = np.zeros(len(c_btc), dtype=bool)
    vs_arr = np.zeros(len(c_btc), dtype=float)
    confirm_bars = max(1, int(getattr(config, "BULL_DAY_CONFIRM_BARS", 2)))
    enter_band = 1.0 + getattr(config, "BULL_DAY_ENTER_PCT", 0.20) / 100.0
    exit_band = 1.0 - getattr(config, "BULL_DAY_EXIT_PCT", 0.20) / 100.0
    prev_state = False
    for i in range(len(c_btc)):
        btc_ema50 = float(ema50[i]) if np.isfinite(ema50[i]) else 0.0
        vs_arr[i] = round((float(c_btc[i]) / btc_ema50 - 1.0) * 100.0, 4) if btc_ema50 > 0 else 0.0
        if i < 50 + confirm_bars:
            state_arr[i] = prev_state
            continue
        slope_ref = i - 5
        if slope_ref >= 0 and np.isfinite(ema50[slope_ref]) and ema50[slope_ref] > 0:
            slope = (float(ema50[i]) - float(ema50[slope_ref])) / float(ema50[slope_ref]) * 100.0
        else:
            slope = 0.0
        start = i - confirm_bars + 1
        closes = c_btc[start:i + 1]
        ema_slice = ema50[start:i + 1]
        enter_confirmed = bool(np.all(np.isfinite(ema_slice)) and np.all(closes > ema_slice * enter_band) and slope > 0)
        exit_confirmed = bool(np.all(np.isfinite(ema_slice)) and np.all(closes < ema_slice * exit_band))
        is_bull = prev_state
        if prev_state:
            if exit_confirmed:
                is_bull = False
        elif enter_confirmed:
            is_bull = True
        state_arr[i] = is_bull
        prev_state = is_bull
    return ts_arr, state_arr, vs_arr


def _market_context_at(
    market_ctx: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ts_ms: int,
) -> Tuple[bool, float]:
    if market_ctx is None:
        return False, 0.0
    ts_arr, bull_arr, vs_arr = market_ctx
    idx = int(np.searchsorted(ts_arr, ts_ms, side="right")) - 1
    if idx < 0:
        return False, 0.0
    return bool(bull_arr[idx]), float(vs_arr[idx])


def _ml_trend_nonbull_score_replay(
    sym: str,
    tf: str,
    feat: dict,
    data: np.ndarray,
    i: int,
    btc_vs_ema50: float,
) -> Optional[float]:
    if not getattr(config, "ML_ENABLE_TREND_NONBULL_FILTER", False):
        return None
    if tf != "15m":
        return None
    segment_key = str(getattr(config, "ML_TREND_NONBULL_SEGMENT_KEY", "trend|nonbull"))
    payload = _load_ml_segment_payloads().get(segment_key)
    if isinstance(payload, dict) and "feature_names" in payload and "model" in payload:
        try:
            from ml_signal_model import build_runtime_record, predict_proba_from_payload

            rec = build_runtime_record(
                sym=sym,
                tf=tf,
                signal_type="trend",
                is_bull_day=False,
                bar_ts=int(data["t"][i]),
                feat=feat,
                data=data,
                i=i,
                btc_vs_ema50=btc_vs_ema50,
            )
            return float(predict_proba_from_payload(payload, rec))
        except Exception:
            return None
    return _ml_general_score_replay(
        sym,
        tf,
        "trend",
        feat,
        data,
        i,
        btc_vs_ema50=btc_vs_ema50,
        is_bull_day=False,
    )


def _ml_general_score_replay(
    sym: str,
    tf: str,
    signal_type: str,
    feat: dict,
    data: np.ndarray,
    i: int,
    *,
    btc_vs_ema50: float,
    is_bull_day: bool,
) -> Optional[float]:
    if not getattr(config, "ML_ENABLE_GENERAL_RANKING", False):
        return None
    payload = _select_ml_payload(signal_type, is_bull_day)
    if not payload:
        return None
    try:
        from ml_signal_model import build_runtime_record, predict_proba_from_payload

        rec = build_runtime_record(
            sym=sym,
            tf=tf,
            signal_type=signal_type,
            is_bull_day=is_bull_day,
            bar_ts=int(data["t"][i]),
            feat=feat,
            data=data,
            i=i,
            btc_vs_ema50=btc_vs_ema50,
        )
        return float(predict_proba_from_payload(payload, rec))
    except Exception:
        return None


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> Optional[np.ndarray]:
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            async with session.get(BINANCE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
                r.raise_for_status()
                batch = await r.json()
        except Exception:
            return None
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + BAR_MS[interval]
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break

    if len(rows) < 60:
        return None

    arr = np.zeros(len(rows), dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")])
    arr["t"] = [int(x[0]) for x in rows]
    arr["o"] = [float(x[1]) for x in rows]
    arr["h"] = [float(x[2]) for x in rows]
    arr["l"] = [float(x[3]) for x in rows]
    arr["c"] = [float(x[4]) for x in rows]
    arr["v"] = [float(x[5]) for x in rows]
    return arr


def _find_last_closed_index(t_arr: np.ndarray, ts_ms: int) -> Optional[int]:
    idx = int(np.searchsorted(t_arr, ts_ms, side="right")) - 1
    return idx if idx >= 0 else None


def _mtf_relaxed_1h_candidate_ok(
    *,
    mode: Optional[str],
    candidate_score: Optional[float],
    slope: Optional[float],
    adx: Optional[float],
    rsi: Optional[float],
    vol_x: Optional[float],
    daily_range: Optional[float],
) -> bool:
    if not getattr(config, "MTF_1H_CONTINUATION_RELAX_ENABLED", False):
        return False
    if mode not in tuple(
        getattr(
            config,
            "MTF_1H_CONTINUATION_RELAX_MODES",
            ("alignment", "trend_start", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    ):
        return False
    if candidate_score is None or candidate_score < float(
        getattr(config, "MTF_1H_CONTINUATION_RELAX_SCORE_MIN", 68.0)
    ):
        return False
    if slope is None or slope < float(getattr(config, "MTF_1H_CONTINUATION_RELAX_SLOPE_MIN", 0.18)):
        return False
    if adx is None or adx < float(getattr(config, "MTF_1H_CONTINUATION_RELAX_ADX_MIN", 22.0)):
        return False
    if rsi is None:
        return False
    rsi_min = float(getattr(config, "MTF_1H_CONTINUATION_RELAX_RSI_MIN", 56.0))
    rsi_max = float(getattr(config, "MTF_1H_CONTINUATION_RELAX_RSI_MAX", 78.0))
    if rsi < rsi_min or rsi > rsi_max:
        return False
    if vol_x is None or vol_x < float(getattr(config, "MTF_1H_CONTINUATION_RELAX_VOL_X_MIN", 0.80)):
        return False
    if daily_range is None or daily_range > float(
        getattr(config, "MTF_1H_CONTINUATION_RELAX_RANGE_MAX", 8.0)
    ):
        return False
    return True


async def _mtf_ok_for_replay(
    sym: str,
    tf: str,
    ts_ms: int,
    cache_15m: Dict[str, Tuple[np.ndarray, dict]],
    *,
    mode: Optional[str] = None,
    candidate_score: Optional[float] = None,
    slope: Optional[float] = None,
    adx: Optional[float] = None,
    rsi: Optional[float] = None,
    vol_x: Optional[float] = None,
    daily_range: Optional[float] = None,
) -> tuple[bool, str]:
    if tf != "1h" or not getattr(config, "MTF_ENABLED", True):
        return True, "MTF disabled"
    pack = cache_15m.get(sym)
    if not pack:
        return True, "no 15m data"

    data_15m, feat_15m = pack
    i = _find_last_closed_index(data_15m["t"], ts_ms)
    if i is None or i < 1:
        return True, "no closed 15m bar"

    c = data_15m["c"].astype(float)
    macd_hist = float(feat_15m["macd_hist"][i]) if np.isfinite(feat_15m["macd_hist"][i]) else 0.0
    macd_prev = float(feat_15m["macd_hist"][i - 1]) if np.isfinite(feat_15m["macd_hist"][i - 1]) else macd_hist
    rsi_val = float(feat_15m["rsi"][i]) if np.isfinite(feat_15m["rsi"][i]) else 50.0
    close_val = float(c[i]) if np.isfinite(c[i]) else 0.0
    ema_fast_15m = feat_15m.get("ema_fast") if isinstance(feat_15m, dict) else None
    if ema_fast_15m is not None and len(ema_fast_15m) > i and np.isfinite(ema_fast_15m[i]):
        ema20_15m = float(ema_fast_15m[i])
    else:
        ema20_15m = close_val
    macd_floor = close_val * float(getattr(config, "MTF_MACD_SOFT_FLOOR_REL", -0.00035))
    macd_hard_floor = close_val * float(getattr(config, "MTF_MACD_HARD_FLOOR_REL", -0.00120))
    macd_rising = macd_hist >= macd_prev
    relaxed_candidate = _mtf_relaxed_1h_candidate_ok(
        mode=mode,
        candidate_score=candidate_score,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    relaxed_floor = close_val * float(
        getattr(config, "MTF_1H_CONTINUATION_RELAX_MACD_FLOOR_REL", -0.00075)
    )
    mtf_rsi_min = float(getattr(config, "MTF_RSI_MIN", 45.0))
    soft_rsi_min = float(getattr(config, "MTF_RSI_SOFT_MIN", max(mtf_rsi_min, 48.0)))
    hard_rsi_min = float(getattr(config, "MTF_RSI_HARD_MIN", 38.0))
    deep_negative = (
        macd_hist < 0
        and rsi_val < hard_rsi_min
        and close_val < ema20_15m
    )
    retest_reason = _retest_1h_mtf_confirm_reason(
        mode=mode,
        close_val=close_val,
        ema20_15m=ema20_15m,
        macd_hist=macd_hist,
        macd_prev=macd_prev,
        rsi_val=rsi_val,
    )
    if retest_reason is not None:
        return False, retest_reason
    if getattr(config, "MTF_MACD_POSITIVE", True) and macd_hist <= 0:
        if (
            relaxed_candidate
            and macd_hist > relaxed_floor
            and (
                not getattr(config, "MTF_1H_CONTINUATION_RELAX_REQUIRE_MACD_RISING", True)
                or macd_rising
            )
            and rsi_val >= float(getattr(config, "MTF_1H_CONTINUATION_RELAX_15M_RSI_MIN", 46.0))
            and close_val >= ema20_15m * (
                1.0 - float(getattr(config, "MTF_1H_CONTINUATION_RELAX_15M_EMA20_SLIP_PCT", 0.30)) / 100.0
            )
        ):
            return True, "15m relax-pass"
        if deep_negative:
            return False, "15m deep correction"
        if macd_hist <= macd_floor:
            return True, "15m soft-pass: MACD below soft floor"
        if rsi_val < hard_rsi_min:
            return True, "15m soft-pass: RSI below hard floor"
        if getattr(config, "MTF_REQUIRE_MACD_RISING", True) and macd_hist < macd_prev:
            return True, "15m soft-pass: MACD not rising"
        if rsi_val < soft_rsi_min:
            return True, "15m soft-pass: RSI below soft floor"
        return True, "15m soft-pass: shallow negative MACD"
    if rsi_val < mtf_rsi_min:
        if deep_negative:
            return False, "15m deep correction"
        return True, "15m soft-pass: RSI below MTF floor"
    return True, "OK"


def _entry_candidate(feat: dict, i: int, c: np.ndarray, tf: str) -> Optional[Tuple[str, float, int, bool]]:
    entry_ok, _ = check_entry_conditions(feat, i, c, tf=tf)
    brk_ok, _ = check_breakout_conditions(feat, i)
    ret_ok, _ = check_retest_conditions(feat, i)
    surge_ok, _ = check_trend_surge_conditions(feat, i)
    imp_ok, _ = check_impulse_conditions(feat, i)
    aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)

    if brk_ok:
        return "breakout", getattr(config, "ATR_TRAIL_K_BREAKOUT", 1.5), getattr(config, "MAX_HOLD_BARS_BREAKOUT", 6), False
    if ret_ok:
        return "retest", getattr(config, "ATR_TRAIL_K_RETEST", 1.8), getattr(config, "MAX_HOLD_BARS_RETEST", 10), False
    if entry_ok:
        mode, early_15m_continuation = get_effective_entry_mode(feat, i, c, tf=tf)
        trail_k = getattr(config, "ATR_TRAIL_K_STRONG", 2.5) if mode in ("strong_trend", "impulse_speed") else config.ATR_TRAIL_K
        max_hold = getattr(config, "MAX_HOLD_BARS_15M", 48)
        return mode, trail_k, max_hold, early_15m_continuation
    if surge_ok:
        return "impulse_speed", getattr(config, "ATR_TRAIL_K_STRONG", 2.5), getattr(config, "MAX_HOLD_BARS_15M", 48), False
    if imp_ok:
        return "impulse", getattr(config, "ATR_TRAIL_K", 2.0), getattr(config, "MAX_HOLD_BARS_15M", 48), False
    if aln_ok and bool(getattr(config, "ALIGNMENT_BUY_ENABLED", False)):
        return "alignment", getattr(config, "ATR_TRAIL_K", 2.0), getattr(config, "MAX_HOLD_BARS_15M", 48), False
    return None


def _trend_start_replay_candidate(
    feat: dict,
    data: np.ndarray,
    i: int,
    c: np.ndarray,
    tf: str,
) -> Optional[Tuple[str, float, int, bool]]:
    if tf != "15m" or i <= 1:
        return None

    ema20_arr = feat.get("ema20", feat.get("ema_fast"))
    ema_slow_arr = feat.get("ema_slow")
    slope_arr = feat.get("ema20_slope", feat.get("slope"))
    macd_hist_arr = feat.get("macd_hist")
    if any(arr is None for arr in (ema20_arr, ema_slow_arr, slope_arr, macd_hist_arr)):
        return None
    if not all(np.isfinite(arr[i]) for arr in (ema20_arr, ema_slow_arr, slope_arr, macd_hist_arr)):
        return None

    price = float(c[i])
    ema20 = float(ema20_arr[i])
    ema_slow = float(ema_slow_arr[i])
    slope = float(slope_arr[i])
    slope_prev = float(slope_arr[i - 1]) if np.isfinite(slope_arr[i - 1]) else slope
    macd_hist = float(macd_hist_arr[i])
    adx = float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 0.0
    rsi = float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0
    vol_x = float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 0.0
    intraday_change_pct = _intraday_change_pct_from_data(data, i)
    if price <= 0 or ema20 <= 0 or ema_slow <= 0:
        return None

    price_edge = ((price / ema20) - 1.0) * 100.0
    ema_sep_pct = ((ema20 / ema_slow) - 1.0) * 100.0
    recent_high = float(np.max(data["h"][max(0, i - 10):i])) if i > 0 else price
    recent_high_gap_pct = ((recent_high / price) - 1.0) * 100.0 if recent_high > 0 else 0.0

    if intraday_change_pct < 0.35:
        return None
    if slope < 0.12 or slope < slope_prev:
        return None
    if adx < 14.0 or adx > 28.0:
        return None
    if rsi < 52.0 or rsi > 70.0:
        return None
    if vol_x < 1.10:
        return None
    if ema_sep_pct < -0.18:
        return None
    if price_edge < -0.10 or price_edge > 1.80:
        return None
    if recent_high_gap_pct > 1.10:
        return None
    if price < ema20 * 0.998:
        return None
    if macd_hist < price * -0.00008:
        return None

    return (
        "trend_start",
        float(getattr(config, "ATR_TRAIL_K", 2.0)),
        int(getattr(config, "MAX_HOLD_BARS_15M", 48)),
        False,
    )


def _is_fresh_priority_mode(mode: str) -> bool:
    priority_modes = tuple(
        getattr(config, "FRESH_SIGNAL_PRIORITY_MODES", ("breakout", "retest", "impulse_speed", "impulse"))
    )
    return mode in priority_modes


def _time_block_bypass_allowed(
    tf: str,
    mode: str,
    score: float,
    vol_x: float,
    *,
    continuation_profile: bool = False,
) -> bool:
    if not getattr(config, "TIME_BLOCK_BYPASS_ENABLED", True):
        return False
    if tf == "15m":
        bypass_modes = tuple(getattr(config, "TIME_BLOCK_BYPASS_MODES", ("breakout", "retest", "impulse_speed")))
        if mode not in bypass_modes:
            return False
        min_score = float(getattr(config, "TIME_BLOCK_BYPASS_SCORE_MIN", 48.0))
        min_vol = float(getattr(config, "TIME_BLOCK_BYPASS_VOL_X_MIN", 1.1))
    elif tf == "1h":
        if not getattr(config, "TIME_BLOCK_BYPASS_1H_ENABLED", False):
            return False
        bypass_modes = tuple(getattr(config, "TIME_BLOCK_BYPASS_1H_MODES", ("alignment", "trend", "strong_trend", "impulse_speed")))
        if mode not in bypass_modes and not continuation_profile:
            return False
        min_score = float(getattr(config, "TIME_BLOCK_BYPASS_1H_SCORE_MIN", 60.0))
        min_vol = float(getattr(config, "TIME_BLOCK_BYPASS_1H_VOL_X_MIN", 1.0))
    else:
        return False
    return (
        score >= min_score
        and vol_x >= min_vol
    )


def _time_block_1h_continuation_profile(
    *,
    tf: str,
    mode: str,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> bool:
    if tf != "1h" or not getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_ENABLED", False):
        return False
    allowed_modes = tuple(
        getattr(
            config,
            "TIME_BLOCK_BYPASS_1H_CONTINUATION_MODES",
            ("alignment", "trend_start", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes:
        return False
    return (
        slope >= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_SLOPE_MIN", 0.08))
        and adx >= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_ADX_MIN", 16.0))
        and float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MIN", 62.0))
        <= rsi
        <= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MAX", 78.0))
        and vol_x >= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_VOL_X_MIN", 1.0))
        and daily_range <= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_RANGE_MAX", 6.5))
    )


def _time_block_1h_continuation_bonus(
    *,
    tf: str,
    mode: str,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> float:
    if not _time_block_1h_continuation_profile(
        tf=tf,
        mode=mode,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    ):
        return 0.0
    return float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_SCORE_BONUS", 0.0))


def _time_block_price_edge_pct(*, price: float, ema20: float) -> float:
    if ema20 <= 0:
        return 0.0
    return max(0.0, ((price / ema20) - 1.0) * 100.0)


def _time_block_1h_prebypass_allowed(
    *,
    tf: str,
    mode: str,
    score: float,
    vol_x: float,
    price: float,
    ema20: float,
    continuation_profile: bool,
    repeat_count: int,
) -> bool:
    if tf != "1h" or not getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_ENABLED", False):
        return False
    allowed_modes = tuple(
        getattr(
            config,
            "TIME_BLOCK_BYPASS_1H_PREBYPASS_MODES",
            ("alignment", "trend_start", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes or not continuation_profile:
        return False
    if repeat_count < int(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_CONFIRMATIONS", 2)):
        return False
    if score < float(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_SCORE_MIN", 54.0)):
        return False
    if vol_x < float(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_VOL_X_MIN", 1.0)):
        return False
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    if price_edge > float(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_PRICE_EDGE_MAX_PCT", 2.2)):
        return False
    return True


def _late_1h_continuation_guard(
    *,
    tf: str,
    mode: str,
    continuation_profile: bool,
    candidate_score: float,
    price: float,
    ema20: float,
    rsi: float,
    daily_range: float,
) -> bool:
    if tf != "1h" or not getattr(config, "LATE_1H_CONTINUATION_GUARD_ENABLED", False):
        return False
    if not continuation_profile:
        return False
    guard_modes = tuple(
        getattr(config, "LATE_1H_CONTINUATION_GUARD_MODES", ("trend", "alignment"))
    )
    if mode not in guard_modes:
        return False
    if rsi < float(getattr(config, "LATE_1H_CONTINUATION_GUARD_RSI_MIN", 72.0)):
        return False
    if daily_range < float(getattr(config, "LATE_1H_CONTINUATION_GUARD_RANGE_MIN", 5.0)):
        return False
    if _time_block_price_edge_pct(price=price, ema20=ema20) < float(
        getattr(config, "LATE_1H_CONTINUATION_GUARD_PRICE_EDGE_MIN_PCT", 1.5)
    ):
        return False
    if candidate_score > float(getattr(config, "LATE_1H_CONTINUATION_GUARD_SCORE_MAX", 68.0)):
        return False
    return True


def _late_impulse_speed_rotation_reason(
    *,
    tf: str,
    mode: str,
    rsi: float,
    daily_range: float,
) -> Optional[str]:
    if mode != "impulse_speed":
        return None
    if not getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_ENABLED", False):
        return None
    if tf == "15m":
        rsi_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_15M_RSI_MIN", 76.0))
        range_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_15M_RANGE_MIN", 5.0))
    elif tf == "1h":
        rsi_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_1H_RSI_MIN", 64.0))
        range_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_1H_RANGE_MIN", 16.0))
    else:
        return None
    if rsi >= rsi_min and daily_range >= range_min:
        return (
            f"late impulse_speed rotation guard: RSI {rsi:.1f} >= {rsi_min:.1f}, "
            f"daily_range {daily_range:.2f}% >= {range_min:.2f}%"
        )
    return None


def _local_day_start_utc_ms(ts_ms: int) -> int:
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    dt_local = dt_utc.astimezone(_objective_tz())
    start_local = datetime.combine(dt_local.date(), datetime.min.time(), tzinfo=_objective_tz())
    return int(start_local.astimezone(timezone.utc).timestamp() * 1000)


def _intraday_change_pct_from_data(data: dict, i: int) -> float:
    try:
        if isinstance(data, dict):
            t_arr = data.get("t")
            o_arr = data.get("o")
            c_arr = data.get("c")
        else:
            names = getattr(getattr(data, "dtype", None), "names", ()) or ()
            t_arr = data["t"] if "t" in names else None
            o_arr = data["o"] if "o" in names else None
            c_arr = data["c"] if "c" in names else None
        if t_arr is None or c_arr is None or i <= 0:
            return 0.0
        ts = int(t_arr[i])
        day_start = _local_day_start_utc_ms(ts)
        start_idx = 0
        for j in range(i, -1, -1):
            if int(t_arr[j]) < day_start:
                start_idx = min(i, j + 1)
                break
        base_arr = o_arr if o_arr is not None else c_arr
        base = float(base_arr[start_idx])
        current = float(c_arr[i])
        if base <= 0 or not np.isfinite(base) or not np.isfinite(current):
            return 0.0
        return (current / base - 1.0) * 100.0
    except Exception:
        return 0.0


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


def _top_gainer_objective_gate_reason(
    *,
    tf: str,
    mode: str,
    intraday_change_pct: float,
    daily_range: float,
    vol_x: float,
    adx: float,
    candidate_score: float,
) -> Optional[str]:
    if not getattr(config, "TOP_GAINER_OBJECTIVE_GATE_ENABLED", False):
        return None
    allowed_modes = tuple(
        str(x)
        for x in getattr(
            config,
            "TOP_GAINER_OBJECTIVE_GATE_MODES",
            ("breakout", "retest", "trend_start", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if allowed_modes and mode not in allowed_modes:
        return None

    min_change = float(getattr(config, "TOP_GAINER_OBJECTIVE_MIN_INTRADAY_CHANGE_PCT", 0.35))
    if mode == "retest":
        min_change = float(getattr(config, "TOP_GAINER_OBJECTIVE_RETEST_MIN_INTRADAY_CHANGE_PCT", 0.75))
    elif mode in ("trend_start", "trend", "strong_trend", "impulse_speed", "impulse"):
        min_change = float(getattr(config, "TOP_GAINER_OBJECTIVE_MOMENTUM_MIN_INTRADAY_CHANGE_PCT", 1.0))

    strong_score = candidate_score >= float(getattr(config, "TOP_GAINER_OBJECTIVE_STRONG_SCORE_BYPASS", 115.0))
    strong_adx = adx >= float(getattr(config, "TOP_GAINER_OBJECTIVE_STRONG_ADX_BYPASS", 32.0))
    if strong_score and strong_adx and intraday_change_pct >= 0.0:
        return None

    if intraday_change_pct < min_change:
        return (
            f"top-gainer objective gate: intraday {intraday_change_pct:+.2f}% "
            f"< {min_change:.2f}%"
        )
    min_range = float(getattr(config, "TOP_GAINER_OBJECTIVE_MIN_DAILY_RANGE_PCT", 0.90))
    if daily_range < min_range:
        return f"top-gainer objective gate: daily_range {daily_range:.2f}% < {min_range:.2f}%"
    min_vol = float(getattr(config, "TOP_GAINER_OBJECTIVE_MIN_VOL_X", 1.20))
    if vol_x < min_vol:
        return f"top-gainer objective gate: vol_x {vol_x:.2f} < {min_vol:.2f}"
    min_adx = float(getattr(config, "TOP_GAINER_OBJECTIVE_MIN_ADX", 20.0))
    if adx < min_adx:
        return f"top-gainer objective gate: ADX {adx:.1f} < {min_adx:.1f}"
    return None


def _continuation_profit_lock_active(
    *,
    tf: str,
    mode: str,
    entry_rsi: float,
    bars_elapsed: int,
    current_pnl: float,
    ret_3: Optional[float],
) -> bool:
    if not getattr(config, "CONTINUATION_PROFIT_LOCK_ENABLED", False):
        return False
    allowed_tf = tuple(getattr(config, "CONTINUATION_PROFIT_LOCK_TF", ("1h",)))
    if tf not in allowed_tf:
        return False
    allowed_modes = tuple(
        getattr(
            config,
            "CONTINUATION_PROFIT_LOCK_MODES",
            ("trend_start", "trend", "alignment", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes:
        return False
    if bars_elapsed < int(getattr(config, "CONTINUATION_PROFIT_LOCK_MIN_BARS", 3)):
        return False
    entry_rsi_min = float(getattr(config, "CONTINUATION_PROFIT_LOCK_ENTRY_RSI_MIN", 70.0))
    continuation_pnl_min = float(
        getattr(config, "CONTINUATION_PROFIT_LOCK_CONTINUATION_PNL_PCT", 0.60)
    )
    if entry_rsi < entry_rsi_min and current_pnl < continuation_pnl_min:
        return False
    if ret_3 is not None and ret_3 > 0:
        return True
    return current_pnl >= float(getattr(config, "CONTINUATION_PROFIT_LOCK_ACTIVATE_PNL_PCT", 0.20))


def _continuation_micro_exit_signal(
    *,
    tf: str,
    mode: str,
    bars_elapsed: int,
    data_15m: np.ndarray,
    feat_15m: dict,
    ts_ms: int,
) -> Optional[Tuple[int, float, str]]:
    if not getattr(config, "CONTINUATION_MICRO_EXIT_ENABLED", False):
        return None
    allowed_tf = tuple(getattr(config, "CONTINUATION_MICRO_EXIT_TF", ("1h",)))
    if tf not in allowed_tf:
        return None
    allowed_modes = tuple(
        getattr(
            config,
            "CONTINUATION_MICRO_EXIT_MODES",
            ("trend_start", "trend", "alignment", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes:
        return None
    if bars_elapsed < int(getattr(config, "CONTINUATION_MICRO_EXIT_MIN_BARS", 3)):
        return None
    idx = _find_last_closed_index(data_15m["t"], ts_ms)
    neg_bars = int(getattr(config, "CONTINUATION_MICRO_EXIT_MACD_NEG_BARS", 4))
    if idx is None or idx < neg_bars - 1:
        return None
    macd_hist = feat_15m["macd_hist"]
    for j in range(idx - neg_bars + 1, idx + 1):
        if not np.isfinite(macd_hist[j]) or macd_hist[j] >= 0:
            return None
    close_15m = float(data_15m["c"][idx])
    ema20_15m = float(feat_15m["ema_fast"][idx]) if np.isfinite(feat_15m["ema_fast"][idx]) else np.nan
    rsi_15m = float(feat_15m["rsi"][idx]) if np.isfinite(feat_15m["rsi"][idx]) else np.nan
    if not all(np.isfinite([close_15m, ema20_15m, rsi_15m])):
        return None
    if rsi_15m > float(getattr(config, "CONTINUATION_MICRO_EXIT_RSI_MAX", 72.0)):
        return None
    if _time_block_price_edge_pct(price=close_15m, ema20=ema20_15m) > float(
        getattr(config, "CONTINUATION_MICRO_EXIT_PRICE_EDGE_MAX_PCT", 0.50)
    ):
        return None
    return (
        int(data_15m["t"][idx]),
        close_15m,
        f"15m micro-weakness after profit-lock: MACD<0 {neg_bars} bars, RSI {rsi_15m:.1f}",
    )


def _short_mode_profit_lock_active(
    *,
    tf: str,
    mode: str,
    bars_elapsed: int,
    current_pnl: float,
    ret_3: Optional[float],
) -> bool:
    if not getattr(config, "SHORT_MODE_PROFIT_LOCK_ENABLED", False):
        return False
    allowed_tf = tuple(getattr(config, "SHORT_MODE_PROFIT_LOCK_TF", ("15m",)))
    if tf not in allowed_tf:
        return False
    allowed_modes = tuple(getattr(config, "SHORT_MODE_PROFIT_LOCK_MODES", ("breakout", "retest")))
    if mode not in allowed_modes:
        return False
    if bars_elapsed < int(getattr(config, "SHORT_MODE_PROFIT_LOCK_MIN_BARS", 2)):
        return False
    if ret_3 is not None and ret_3 > 0:
        return True
    return current_pnl >= float(getattr(config, "SHORT_MODE_PROFIT_LOCK_ACTIVATE_PNL_PCT", 0.30))


def _time_block_retest_bonus(
    *,
    mode: str,
    tf: str,
    ts_ms: int,
    is_bull_day: bool,
    last_time_block_ts: Optional[int],
) -> float:
    if mode != "retest" or is_bull_day or last_time_block_ts is None:
        return 0.0
    grace_bars = int(getattr(config, "TIME_BLOCK_RETEST_GRACE_BARS", 0))
    if grace_bars <= 0:
        return 0.0
    bar_ms = BAR_MS[tf]
    if ts_ms <= last_time_block_ts:
        return 0.0
    if ts_ms - last_time_block_ts > grace_bars * bar_ms:
        return 0.0
    return float(getattr(config, "TIME_BLOCK_RETEST_SCORE_BONUS", 0.0))


def _is_weak_exit_reason(reason: Optional[str]) -> bool:
    return bool(reason) and "WEAK:" in str(reason)


def _is_protected_trailing_exit_reason(reason: Optional[str]) -> bool:
    if not reason:
        return False
    text = str(reason).lower()
    return (
        "weak:" in text
        or "ema20" in text
        or "ema" in text
        or "atr trail" in text
        or "stop" in text
        or "ниже ema" in text
    )


def _protected_trailing_exit_should_hold(
    *,
    trade: "ReplayTrade",
    reason: Optional[str],
    current_pnl: float,
    weak_only: bool = False,
) -> bool:
    if weak_only and not _is_weak_exit_reason(reason):
        return False
    if not _is_protected_trailing_exit_reason(reason):
        return False
    if trade.protected_exit_active:
        return True
    if int(trade.bars_held) < int(getattr(config, "PROTECTED_EXIT_MIN_BARS", 1)):
        return False
    if float(trade.max_favorable_pct) < float(getattr(config, "PROTECTED_EXIT_MIN_MFE_PCT", 0.50)):
        return False
    if current_pnl < float(getattr(config, "PROTECTED_EXIT_MIN_CURRENT_PNL_PCT", -1.50)):
        return False
    return True


def _exit_reason_bucket(reason: Optional[str]) -> str:
    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if "weak:" in text:
        return "weak"
    if "atr" in text and "trail" in text:
        return "atr_trail"
    if "ema20" in text or "ema" in text:
        return "ema_break"
    if "stop" in text or "стоп" in text:
        return "stop_loss_or_trail"
    if "rsi" in text:
        return "rsi"
    if "time" in text or "max hold" in text:
        return "time_exit"
    return "other"


def _exit_discriminator_shadow_score(
    *,
    trade: "ReplayTrade",
    reason: Optional[str],
    current_pnl: float,
) -> float:
    score = 0.0
    reason_bucket = _exit_reason_bucket(reason)
    if reason_bucket in {"weak", "stop_loss_or_trail", "atr_trail"}:
        score += 0.24
    if reason_bucket == "weak":
        score += 0.08
    if trade.mode in {"trend", "strong_trend", "impulse_speed", "alignment"}:
        score += 0.16
    capture = trade.capture_ratio_at_entry
    if capture is not None:
        if capture >= 0.60:
            score += 0.20
        elif capture >= 0.40:
            score += 0.12
    if trade.max_favorable_pct >= 3.0:
        score += 0.18
    elif trade.max_favorable_pct >= 1.5:
        score += 0.10
    giveback_now = max(0.0, trade.max_favorable_pct - current_pnl)
    if giveback_now >= 1.5:
        score += 0.14
    elif giveback_now >= 0.5:
        score += 0.07
    if current_pnl > 0.0:
        score += 0.08
    if trade.bars_held <= int(getattr(config, "EXIT_DISCRIMINATOR_EARLY_BARS", 6)):
        score += 0.10
    return round(min(score, 1.0), 4)


def _exit_discriminator_shadow_should_hold(
    *,
    trade: "ReplayTrade",
    reason: Optional[str],
    current_pnl: float,
) -> bool:
    if not reason:
        return False
    if trade.discriminator_exit_active:
        return True
    if _exit_reason_bucket(reason) in {"ema_break", "time_exit"}:
        return False
    if current_pnl < float(getattr(config, "EXIT_DISCRIMINATOR_MIN_CURRENT_PNL_PCT", -0.25)):
        return False
    if trade.max_favorable_pct < float(getattr(config, "EXIT_DISCRIMINATOR_MIN_MFE_PCT", 1.0)):
        return False
    score = _exit_discriminator_shadow_score(trade=trade, reason=reason, current_pnl=current_pnl)
    trade.discriminator_exit_score = score
    return score >= float(getattr(config, "EXIT_DISCRIMINATOR_HOLD_SCORE_MIN", 0.68))


def _suspicious_exit_reentry_score(trade: "ReplayTrade") -> float:
    current_pnl = trade.pnl_pct if trade.exit_price > 0 else 0.0
    return _exit_discriminator_shadow_score(
        trade=trade,
        reason=trade.exit_reason,
        current_pnl=current_pnl,
    )


def _suspicious_exit_reentry_candidate_ok(
    candidate: "ReplayCandidate",
    *,
    base_score_floor: float,
) -> bool:
    score_floor = max(
        float(base_score_floor),
        float(getattr(config, "SUSPICIOUS_REENTRY_TOP_GAINER_SCORE_MIN", 38.0)),
    )
    if float(candidate.top_gainer_score) < score_floor:
        return False
    if candidate.mode not in {"trend", "strong_trend", "impulse_speed", "alignment", "trend_start"}:
        return False
    if float(getattr(candidate, "adx", 0.0)) < float(getattr(config, "SUSPICIOUS_REENTRY_MIN_ADX", 18.0)):
        return False
    return True


def _suspicious_reentry_enabled(variant: str) -> bool:
    return variant in {"suspicious_exit_reentry", "baseline_suspicious_reentry"}


def _maybe_mark_partial_profit_take(
    *,
    trade: "ReplayTrade",
    close_now: float,
    ts_ms: int,
    current_pnl: float,
) -> None:
    if trade.partial_exit_taken:
        return
    trigger_pct = float(getattr(config, "PARTIAL_PROFIT_TAKE_TRIGGER_PCT", 3.0))
    min_mfe_pct = float(getattr(config, "PARTIAL_PROFIT_TAKE_MIN_MFE_PCT", trigger_pct))
    if current_pnl < trigger_pct or trade.max_favorable_pct < min_mfe_pct:
        return
    fraction = float(getattr(config, "PARTIAL_PROFIT_TAKE_FRACTION", 0.50))
    fraction = max(0.0, min(fraction, 1.0))
    if fraction <= 0.0:
        return
    trade.partial_exit_taken = True
    trade.partial_exit_fraction = fraction
    trade.partial_exit_price = close_now
    trade.partial_exit_ts = ts_ms
    trade.partial_exit_reason = f"partial_profit_take_{fraction:.2f}_at_{current_pnl:.2f}%"


def _min_weak_exit_bars(mode: Optional[str]) -> int:
    mode_name = str(mode or "")
    if mode_name == "impulse_speed":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_IMPULSE_SPEED", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    if mode_name == "retest":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_RETEST", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    if mode_name == "breakout":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_BREAKOUT", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    if mode_name == "trend":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_TREND", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    return int(getattr(config, "MIN_WEAK_EXIT_BARS", 2))


def _trend_hold_weak_exit_active(
    *,
    trade: "ReplayTrade",
    feat: dict,
    idx: int,
    close_now: float,
    current_pnl: float,
) -> bool:
    if not getattr(config, "TREND_HOLD_WEAK_EXIT_ENABLED", False):
        return False
    if trade.tf not in tuple(getattr(config, "TREND_HOLD_WEAK_EXIT_TF", ("15m",))):
        return False
    if trade.mode not in tuple(getattr(config, "TREND_HOLD_WEAK_EXIT_MODES", ("impulse_speed", "trend_start", "trend", "alignment"))):
        return False
    if int(trade.bars_held) < int(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_BARS", 5)):
        return False
    if current_pnl < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_PNL_PCT", 0.75)):
        return False
    if float(getattr(trade, "entry_score", 0.0)) < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_ENTRY_SCORE", 70.0)):
        return False

    ema20_arr = feat.get("ema_fast")
    ema50_arr = feat.get("ema_slow")
    ema200_arr = feat.get("ema200")
    adx_arr = feat.get("adx")
    slope_arr = feat.get("slope")
    if any(arr is None or idx >= len(arr) for arr in (ema20_arr, ema50_arr, ema200_arr, adx_arr, slope_arr)):
        return False

    ema20 = float(ema20_arr[idx])
    ema50 = float(ema50_arr[idx])
    ema200 = float(ema200_arr[idx])
    adx = float(adx_arr[idx])
    slope = float(slope_arr[idx])
    if not all(np.isfinite(v) for v in (ema20, ema50, ema200, adx, slope, close_now)):
        return False
    if close_now < ema20:
        return False
    if not (ema20 >= ema50 >= ema200):
        return False
    if adx < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_ADX", 24.0)):
        return False
    if slope < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_SLOPE_PCT", 0.10)):
        return False
    return True


def _cooldown_bars_after_exit(mode: str, reason: Optional[str]) -> int:
    base = int(getattr(config, "COOLDOWN_BARS", 8))
    if _is_weak_exit_reason(reason) and mode in ("trend_start", "trend", "alignment"):
        return max(base, int(getattr(config, "WEAK_REENTRY_COOLDOWN_BARS", base)))
    return base


def _fast_loss_ema_exit_reason(
    *,
    tf: str,
    mode: str,
    bars_elapsed: int,
    current_pnl: float,
    close_now: float,
    ema20: float,
    rsi: float,
) -> Optional[str]:
    if not getattr(config, "FAST_LOSS_EMA_EXIT_ENABLED", False):
        return None
    if tf not in tuple(getattr(config, "FAST_LOSS_EMA_EXIT_TF", ("15m",))):
        return None
    if mode not in tuple(getattr(config, "FAST_LOSS_EMA_EXIT_MODES", ("retest", "alignment", "trend", "breakout"))):
        return None
    if bars_elapsed < int(getattr(config, "FAST_LOSS_EMA_EXIT_MIN_BARS", 1)):
        return None
    if not np.isfinite(ema20) or close_now >= ema20:
        return None
    if current_pnl > float(getattr(config, "FAST_LOSS_EMA_EXIT_PNL_MAX", 0.0)):
        return None
    if not np.isfinite(rsi) or rsi > float(getattr(config, "FAST_LOSS_EMA_EXIT_RSI_MAX", 55.0)):
        return None
    return f"first close below EMA20 ({ema20:.6g}) in losing trade"


def _time_exit_should_wait(feat: dict, idx: int, close_now: float) -> bool:
    if not getattr(config, "TIME_EXIT_TREND_CONTINUATION_ENABLED", False):
        return False
    ema20_arr = feat.get("ema20", feat.get("ema_fast"))
    ema20 = float(ema20_arr[idx]) if ema20_arr is not None and np.isfinite(ema20_arr[idx]) else np.nan
    slope_arr = feat.get("ema20_slope", feat.get("slope"))
    slope = float(slope_arr[idx]) if slope_arr is not None and np.isfinite(slope_arr[idx]) else np.nan
    rsi = float(feat["rsi"][idx]) if np.isfinite(feat["rsi"][idx]) else np.nan
    macd_hist_arr = feat.get("macd_hist")
    macd_hist = float(macd_hist_arr[idx]) if macd_hist_arr is not None and np.isfinite(macd_hist_arr[idx]) else np.nan
    if bool(getattr(config, "TIME_EXIT_CONTINUE_CLOSE_ABOVE_EMA20", True)):
        if not np.isfinite(ema20) or close_now < ema20:
            return False
    if not np.isfinite(slope) or slope < float(getattr(config, "TIME_EXIT_CONTINUE_SLOPE_MIN", 0.0)):
        return False
    if not np.isfinite(rsi) or rsi < float(getattr(config, "TIME_EXIT_CONTINUE_RSI_MIN", 50.0)):
        return False
    if np.isfinite(macd_hist) and macd_hist < float(getattr(config, "TIME_EXIT_CONTINUE_MACD_HIST_MIN", 0.0)):
        return False
    return True


def _candidate_score(feat: dict, i: int, c: np.ndarray, mode: str) -> float:
    price = float(c[i]) if np.isfinite(c[i]) else 0.0
    ema20_arr = feat.get("ema20", feat.get("ema_fast"))
    slope_arr = feat.get("ema20_slope", feat.get("slope"))
    ema20 = float(ema20_arr[i]) if ema20_arr is not None and np.isfinite(ema20_arr[i]) else 0.0
    slope = float(slope_arr[i]) if slope_arr is not None and np.isfinite(slope_arr[i]) else 0.0
    adx = float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 0.0
    rsi = float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0
    vol_x = float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 1.0
    dr_arr = feat.get("daily_range", feat.get("daily_range_pct"))
    daily_range = float(dr_arr[i]) if dr_arr is not None and np.isfinite(dr_arr[i]) else 0.0
    return _entry_signal_score(
        mode=mode,
        price=price,
        ema20=ema20,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )


def _entry_score_floor(tf: str) -> float:
    if tf == "1h":
        return float(getattr(config, "ENTRY_SCORE_MIN_1H", 0.0))
    return float(getattr(config, "ENTRY_SCORE_MIN_15M", 0.0))


def _entry_score_borderline_bypass_ok(
    *,
    tf: str,
    mode: str,
    candidate_score: float,
    min_score: float,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> bool:
    if not getattr(config, "ENTRY_SCORE_BORDERLINE_BYPASS_ENABLED", False):
        return False
    if tf == "1h" and not getattr(config, "ENTRY_SCORE_BORDERLINE_ALLOW_1H", False):
        return False
    allowed_modes = tuple(str(x) for x in getattr(config, "ENTRY_SCORE_BORDERLINE_MODES", ()))
    if allowed_modes and mode not in allowed_modes:
        return False
    max_deficit = float(
        getattr(
            config,
            "ENTRY_SCORE_BORDERLINE_MAX_DEFICIT_1H" if tf == "1h" else "ENTRY_SCORE_BORDERLINE_MAX_DEFICIT_15M",
            0.0,
        )
    )
    if candidate_score < (min_score - max_deficit):
        return False
    if ema20 <= 0:
        return False
    price_edge = ((price / ema20) - 1.0) * 100.0
    if price_edge > float(getattr(config, "ENTRY_SCORE_BORDERLINE_PRICE_EDGE_MAX_PCT", 2.8)):
        return False
    if slope < float(getattr(config, "ENTRY_SCORE_BORDERLINE_SLOPE_MIN", 0.35)):
        return False
    if adx < float(getattr(config, "ENTRY_SCORE_BORDERLINE_ADX_MIN", 28.0)):
        return False
    if vol_x < float(getattr(config, "ENTRY_SCORE_BORDERLINE_VOL_MIN", 1.2)):
        return False
    if daily_range > float(getattr(config, "ENTRY_SCORE_BORDERLINE_DAILY_RANGE_MAX", 6.5)):
        return False
    if rsi < float(getattr(config, "ENTRY_SCORE_BORDERLINE_RSI_MIN", 52.0)):
        return False
    if rsi > float(getattr(config, "ENTRY_SCORE_BORDERLINE_RSI_MAX", 74.5)):
        return False
    return True


def _entry_score_continuation_bypass_ok(
    *,
    tf: str,
    mode: str,
    candidate_score: float,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    continuation_profile: bool,
    is_bull_day: bool,
) -> bool:
    if not getattr(config, "ENTRY_SCORE_CONTINUATION_BYPASS_ENABLED", False):
        return False
    if tf != "1h" or not getattr(config, "ENTRY_SCORE_CONTINUATION_1H_ENABLED", False):
        return False
    if not continuation_profile:
        return False
    if getattr(config, "ENTRY_SCORE_CONTINUATION_REQUIRE_BULL_DAY", True) and not is_bull_day:
        return False
    allowed_modes = tuple(str(x) for x in getattr(config, "ENTRY_SCORE_CONTINUATION_MODES", ()))
    if allowed_modes and mode not in allowed_modes:
        return False
    if candidate_score < float(getattr(config, "ENTRY_SCORE_CONTINUATION_SCORE_MIN_1H", 42.0)):
        return False
    if ema20 <= 0:
        return False
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    if price_edge > float(getattr(config, "ENTRY_SCORE_CONTINUATION_PRICE_EDGE_MAX_PCT", 2.0)):
        return False
    if slope < float(getattr(config, "ENTRY_SCORE_CONTINUATION_SLOPE_MIN", 0.30)):
        return False
    if adx < float(getattr(config, "ENTRY_SCORE_CONTINUATION_ADX_MIN", 16.0)):
        return False
    if vol_x < float(getattr(config, "ENTRY_SCORE_CONTINUATION_VOL_MIN", 0.90)):
        return False
    if daily_range > float(getattr(config, "ENTRY_SCORE_CONTINUATION_DAILY_RANGE_MAX", 8.5)):
        return False
    if rsi < float(getattr(config, "ENTRY_SCORE_CONTINUATION_RSI_MIN", 54.0)):
        return False
    if rsi > float(getattr(config, "ENTRY_SCORE_CONTINUATION_RSI_MAX", 72.0)):
        return False
    return True


def _retest_1h_mtf_confirm_reason(
    *,
    mode: Optional[str],
    close_val: float,
    ema20_15m: float,
    macd_hist: float,
    macd_prev: float,
    rsi_val: float,
) -> Optional[str]:
    if mode != "retest" or not getattr(config, "RETEST_1H_MTF_CONFIRM_ENABLED", False):
        return None
    if ema20_15m <= 0:
        return None
    ema_slip_pct = float(getattr(config, "RETEST_1H_MTF_EMA20_SLIP_PCT", 0.15))
    if close_val < ema20_15m * (1.0 - ema_slip_pct / 100.0):
        return (
            f"15m retest micro weak: close {close_val:.6g} below EMA20 {ema20_15m:.6g} "
            f"by > {ema_slip_pct:.2f}%"
        )
    macd_floor = close_val * float(getattr(config, "RETEST_1H_MTF_MACD_MIN_REL", 0.0))
    if macd_hist < macd_floor:
        return f"15m retest micro weak: MACD {macd_hist:.4g} < floor {macd_floor:.4g}"
    if bool(getattr(config, "RETEST_1H_MTF_REQUIRE_MACD_RISING", True)) and macd_hist < macd_prev:
        return f"15m retest micro weak: MACD {macd_hist:.4g} < prev {macd_prev:.4g}"
    rsi_min = float(getattr(config, "RETEST_1H_MTF_RSI_MIN", 48.0))
    rsi_max = float(getattr(config, "RETEST_1H_MTF_RSI_MAX", 72.0))
    if rsi_val < rsi_min:
        return f"15m retest micro weak: RSI {rsi_val:.1f} < {rsi_min:.1f}"
    if rsi_val > rsi_max:
        return f"15m retest micro weak: RSI {rsi_val:.1f} > {rsi_max:.1f}"
    return None


def _trend_entry_quality_guard_reason(
    *,
    tf: str,
    mode: str,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    forecast_return_pct: float,
) -> Optional[str]:
    if tf != "15m" or mode != "trend":
        return None
    if not getattr(config, "TREND_15M_QUALITY_GUARD_ENABLED", False):
        return None
    if ema20 <= 0:
        return None
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    price_edge_max = float(getattr(config, "TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT", 2.40))
    if price_edge > price_edge_max:
        return f"trend quality guard: price edge {price_edge:.2f}% > {price_edge_max:.2f}%"
    daily_range_max = float(getattr(config, "TREND_15M_QUALITY_DAILY_RANGE_MAX", 8.0))
    if daily_range > daily_range_max:
        return f"trend quality guard: daily_range {daily_range:.2f}% > {daily_range_max:.2f}%"
    rsi_max = float(getattr(config, "TREND_15M_QUALITY_RSI_MAX", 68.0))
    if rsi > rsi_max:
        return f"trend quality guard: RSI {rsi:.1f} > {rsi_max:.1f}"
    forecast_min = float(getattr(config, "TREND_15M_QUALITY_FORECAST_MIN", 0.25))
    if forecast_return_pct >= forecast_min:
        return None
    alt_vol_min = float(getattr(config, "TREND_15M_QUALITY_ALT_VOL_MIN", 1.20))
    alt_adx_min = float(getattr(config, "TREND_15M_QUALITY_ALT_ADX_MIN", 24.0))
    alt_slope_min = float(getattr(config, "TREND_15M_QUALITY_ALT_SLOPE_MIN", 0.35))
    if vol_x >= alt_vol_min and adx >= alt_adx_min and slope >= alt_slope_min:
        return None
    return (
        "trend quality guard: weak 15m trend "
        f"(forecast {forecast_return_pct:.3f} < {forecast_min:.3f}, "
        f"vol {vol_x:.2f}, ADX {adx:.1f}, slope {slope:.3f})"
    )


def _mtf_soft_penalty_from_reason(reason: str) -> float:
    if reason.startswith("15m soft-pass:") or reason.startswith("15Рј soft-pass:"):
        return float(getattr(config, "MTF_SOFT_PASS_PENALTY", 0.0))
    return 0.0


def _top_gainer_replay_score(
    *,
    tf: str,
    mode: str,
    intraday_change_pct: float,
    daily_range: float,
    vol_x: float,
    adx: float,
    rsi: float,
    ranker_final_score: float,
    ranker_top_gainer_prob: float,
) -> float:
    score = 0.0
    score += max(-8.0, min(45.0, intraday_change_pct * 6.0))
    score += max(0.0, min(18.0, daily_range * 1.1))
    score += max(0.0, min(14.0, (vol_x - 1.0) * 7.0))
    score += max(0.0, min(14.0, (adx - 18.0) * 0.7))
    score += max(-8.0, min(12.0, ranker_final_score * 6.0))
    score += max(0.0, min(18.0, ranker_top_gainer_prob * 45.0))
    if mode in ("trend_start", "impulse_speed", "strong_trend", "trend", "impulse"):
        score += 5.0
    elif mode in ("breakout", "retest"):
        score += 1.5
    if tf == "15m":
        score += 1.5
    if rsi > 76.0 and daily_range >= 8.0:
        score -= min(16.0, (rsi - 76.0) * 1.2 + (daily_range - 8.0) * 0.4)
    if daily_range > 25.0:
        score -= min(18.0, (daily_range - 25.0) * 0.6)
    if intraday_change_pct < 0.0:
        score -= 8.0
    return round(score, 4)


def _signal_cluster_bucket_replay(tf: str, mode: str) -> str:
    if tf == "15m" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES", ("breakout", "retest")):
        return "15m_short_bounce"
    if tf == "15m" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES", ("impulse_speed",)):
        return "15m_impulse"
    if tf == "15m" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_MOMENTUM_MODES", ("trend_start", "trend", "strong_trend", "impulse")):
        return "15m_momentum"
    if tf == "15m" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_ALIGNMENT_MODES", ("alignment",)):
        return "15m_alignment"
    if tf == "1h" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES", ("retest",)):
        return "1h_retest"
    if tf == "1h" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_1H_MOMENTUM_MODES", ("trend", "strong_trend", "impulse_speed", "impulse")):
        return "1h_momentum"
    if tf == "1h" and mode in getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_1H_ALIGNMENT_MODES", ("alignment",)):
        return "1h_alignment"
    return f"{tf}_{mode}"


def _top_gainer_score_min_for_mode(mode: str, default_min: float) -> float:
    mode_min_scores = getattr(config, "TOP_GAINER_SCORE_GATE_MODE_MIN_SCORE", {}) or {}
    try:
        return float(mode_min_scores.get(mode, default_min))
    except Exception:
        return float(default_min)


def _agent_allowed_mode_replay(mode: str) -> bool:
    return mode in tuple(str(x) for x in getattr(config, "AGENT_ALLOWED_MODES", ()))


def _agent_mode_rescue_replay_candidate_ok(candidate: ReplayCandidate, top_gainer_score_min: float) -> bool:
    if _agent_allowed_mode_replay(candidate.mode):
        return True
    if candidate.mode not in {"breakout", "retest"}:
        return False
    if candidate.tf != "15m":
        return False
    if candidate.top_gainer_score < max(34.0, float(top_gainer_score_min)):
        return False
    if candidate.adx < 22.0:
        return False
    if candidate.vol_x < 2.0:
        return False
    if candidate.daily_range > 8.0:
        return False
    if candidate.intraday_change_pct < 1.0:
        return False
    return True


def _parse_event_ts_ms(raw: object) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _load_temporal_scout_events(
    path: Path = BOT_EVENTS_PATH,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> tuple[Dict[Tuple[str, str], List[int]], Dict[Tuple[str, str], List[int]]]:
    structural: Dict[Tuple[str, str], List[int]] = {}
    wakeups: Dict[Tuple[str, str], List[int]] = {}
    if not path.exists():
        return structural, wakeups
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event") != "scout_shadow":
            continue
        ts_ms = _parse_event_ts_ms(row.get("ts"))
        sym = str(row.get("sym") or "")
        tf = str(row.get("tf") or "")
        if ts_ms is None or not sym or not tf:
            continue
        if start_ms is not None and ts_ms < start_ms:
            continue
        if end_ms is not None and ts_ms > end_ms:
            continue
        key = (sym, tf)
        if str(row.get("scout_profile") or "") == "wake_up_1m_light15_v1":
            wakeups.setdefault(key, []).append(ts_ms)
        else:
            structural.setdefault(key, []).append(ts_ms)
    for rows in structural.values():
        rows.sort()
    for rows in wakeups.values():
        rows.sort()
    return structural, wakeups


def _agent_wakeup_rescue_replay_candidate_ok(
    candidate: ReplayCandidate,
    top_gainer_score_min: float,
    structural_events: Dict[Tuple[str, str], List[int]],
    wakeup_events: Dict[Tuple[str, str], List[int]],
) -> bool:
    if _agent_allowed_mode_replay(candidate.mode):
        return True
    if not _agent_mode_rescue_replay_candidate_ok(candidate, top_gainer_score_min):
        return False
    key = (candidate.sym, candidate.tf)
    wakeups = [ts for ts in wakeup_events.get(key, []) if ts <= candidate.ts_ms]
    if not wakeups:
        return False
    latest_wakeup = wakeups[-1]
    if candidate.ts_ms - latest_wakeup > 6 * 60 * 60 * 1000:
        return False
    structurals = [
        ts
        for ts in structural_events.get(key, [])
        if ts <= latest_wakeup and latest_wakeup - ts <= 24 * 60 * 60 * 1000
    ]
    return bool(structurals)


def _signal_cluster_cap_replay(bucket: str) -> int:
    mapping = {
        "15m_short_bounce": "OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX",
        "15m_impulse": "OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX",
        "15m_momentum": "OPEN_SIGNAL_CLUSTER_CAP_15M_MOMENTUM_MAX",
        "15m_alignment": "OPEN_SIGNAL_CLUSTER_CAP_15M_ALIGNMENT_MAX",
        "1h_retest": "OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX",
        "1h_momentum": "OPEN_SIGNAL_CLUSTER_CAP_1H_MOMENTUM_MAX",
        "1h_alignment": "OPEN_SIGNAL_CLUSTER_CAP_1H_ALIGNMENT_MAX",
    }
    attr = mapping.get(bucket)
    return max(1, int(getattr(config, attr, 2))) if attr else 2


def _four_h_context_score_replay(
    sym: str,
    ts_ms: int,
    cache_4h: Dict[str, Tuple[np.ndarray, dict]],
) -> tuple[float, str]:
    if not bool(getattr(config, "FOUR_H_CONTEXT_SCORE_ENABLED", True)):
        return 0.0, "disabled"
    pack = cache_4h.get(sym)
    if not pack:
        return 0.0, "4h_missing"
    data, feat = pack
    idx = _find_last_closed_index(data["t"], ts_ms)
    if idx is None or idx < 50:
        return 0.0, "4h_insufficient"
    price = float(data["c"][idx])
    ema20 = float(feat["ema_fast"][idx]) if np.isfinite(feat["ema_fast"][idx]) else 0.0
    ema50 = float(feat["ema_slow"][idx]) if np.isfinite(feat["ema_slow"][idx]) else 0.0
    slope = float(feat["ema20_slope"][idx] if "ema20_slope" in feat else feat["slope"][idx]) if np.isfinite(feat["slope"][idx]) else 0.0
    rsi = float(feat["rsi"][idx]) if np.isfinite(feat["rsi"][idx]) else 50.0
    vol_x = float(feat["vol_x"][idx]) if np.isfinite(feat["vol_x"][idx]) else 0.0
    macd_hist = float(feat["macd_hist"][idx]) if np.isfinite(feat["macd_hist"][idx]) else 0.0
    greens = sum(1 for j in range(max(0, idx - 2), idx + 1) if float(data["c"][j]) > float(data["o"][j]))
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


async def _build_candidates_for_symbol(
    sym: str,
    tf: str,
    data: np.ndarray,
    feat: dict,
    cache_15m: Dict[str, Tuple[np.ndarray, dict]],
    cache_4h: Dict[str, Tuple[np.ndarray, dict]],
    market_ctx: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    include_trend_start: bool = False,
) -> List[ReplayCandidate]:
    candidates: List[ReplayCandidate] = []
    c = data["c"].astype(float)
    last_time_block_ts: Optional[int] = None
    time_block_streak_count = 0
    time_block_streak_mode: Optional[str] = None
    time_block_streak_ts: Optional[int] = None
    prev_bull_state = bool(getattr(config, "_bull_day_active", False))
    prev_btc_vs_ema50 = float(getattr(config, "_btc_vs_ema50", 0.0))
    try:
        for i in range(max(25, 5), len(c) - 1):
            ts_ms = int(data["t"][i])
            is_bull_day, btc_vs_ema50 = _market_context_at(market_ctx, ts_ms)
            config._bull_day_active = is_bull_day
            config._btc_vs_ema50 = btc_vs_ema50

            picked = _trend_start_replay_candidate(feat, data, i, c, tf) if include_trend_start else None
            picked = picked or _entry_candidate(feat, i, c, tf)
            if picked is None:
                continue
            mode, trail_k, max_hold, early_15m_continuation = picked
            entry_ok, _ = check_entry_conditions(feat, i, c, tf=tf)
            brk_ok, _ = check_breakout_conditions(feat, i)
            ret_ok, _ = check_retest_conditions(feat, i)
            surge_ok, _ = check_trend_surge_conditions(feat, i)
            imp_ok, _ = check_impulse_conditions(feat, i)
            aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)
            signal_flags = _candidate_signal_flags(
                entry_ok=entry_ok,
                brk_ok=brk_ok,
                ret_ok=ret_ok,
                surge_ok=surge_ok,
                imp_ok=imp_ok,
                aln_ok=aln_ok,
            )
            base_score = _candidate_score(feat, i, c, mode)
            candidate_score = base_score
            four_h_context_score, four_h_context_label = _four_h_context_score_replay(sym, ts_ms, cache_4h)
            candidate_score += four_h_context_score * float(getattr(config, "FOUR_H_CONTEXT_SCORE_WEIGHT", 1.0))
            if early_15m_continuation:
                candidate_score += float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_SCORE_BONUS", 10.0))
            slope_arr = feat.get("ema20_slope", feat.get("slope"))
            slope = float(slope_arr[i]) if slope_arr is not None and np.isfinite(slope_arr[i]) else 0.0
            adx = float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 0.0
            rsi = float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0
            preview_vol = float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 0.0
            dr_arr = feat.get("daily_range", feat.get("daily_range_pct"))
            daily_range = float(dr_arr[i]) if dr_arr is not None and np.isfinite(dr_arr[i]) else 0.0
            ema20_arr = feat.get("ema20", feat.get("ema_fast"))
            preview_ema20 = float(ema20_arr[i]) if ema20_arr is not None and np.isfinite(ema20_arr[i]) else 0.0
            mtf_ok, mtf_reason = await _mtf_ok_for_replay(
                sym,
                tf,
                ts_ms,
                cache_15m,
                mode=mode,
                candidate_score=candidate_score,
                slope=slope,
                adx=adx,
                rsi=rsi,
                vol_x=preview_vol,
                daily_range=daily_range,
            )
            if not mtf_ok:
                continue
            candidate_score -= _mtf_soft_penalty_from_reason(mtf_reason)
            continuation_profile = _time_block_1h_continuation_profile(
                tf=tf,
                mode=mode,
                slope=slope,
                adx=adx,
                rsi=rsi,
                vol_x=preview_vol,
                daily_range=daily_range,
            )
            candidate_score += _time_block_1h_continuation_bonus(
                tf=tf,
                mode=mode,
                slope=slope,
                adx=adx,
                rsi=rsi,
                vol_x=preview_vol,
                daily_range=daily_range,
            )
            if _late_1h_continuation_guard(
                tf=tf,
                mode=mode,
                continuation_profile=continuation_profile,
                candidate_score=candidate_score,
                price=float(c[i]),
                ema20=preview_ema20,
                rsi=rsi,
                daily_range=daily_range,
            ):
                continue
            if _impulse_speed_entry_guard(
                tf=tf,
                mode=mode,
                feat=feat,
                i=i,
                price=float(c[i]),
                ema20=preview_ema20,
                rsi=rsi,
                adx=adx,
                daily_range=daily_range,
            ):
                continue
            if _is_fresh_priority_mode(mode):
                candidate_score += float(getattr(config, "FRESH_SIGNAL_SCORE_BONUS", 7.0))
            ml_proba = _ml_general_score_replay(
                sym,
                tf,
                mode,
                feat,
                data,
                i,
                btc_vs_ema50=btc_vs_ema50,
                is_bull_day=is_bull_day,
            )
            if ml_proba is not None:
                candidate_score += (
                    ml_proba - float(getattr(config, "ML_GENERAL_NEUTRAL_PROBA", 0.50))
                ) * float(getattr(config, "ML_GENERAL_SCORE_WEIGHT", 10.0))
            if mode == "trend" and not is_bull_day:
                if ml_proba is None:
                    ml_proba = _ml_trend_nonbull_score_replay(sym, tf, feat, data, i, btc_vs_ema50)
                if ml_proba is not None:
                    min_proba = float(getattr(config, "ML_TREND_NONBULL_MIN_PROBA", 0.35))
                    if ml_proba < min_proba:
                        candidate_score -= float(
                            getattr(config, "ML_TREND_NONBULL_LOW_PROBA_PENALTY", 6.0)
                        )
                        if getattr(config, "ML_TREND_NONBULL_HARD_BLOCK", False):
                            continue
            score_floor = _entry_score_floor(tf) if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False) else 0.0
            mtf_soft_penalty = _mtf_soft_penalty_from_reason(mtf_reason)
            intraday_change_pct = _intraday_change_pct_from_data(data, i)
            forecast_return_pct = _forecast_proxy_pct(
                today_change_pct=intraday_change_pct,
                slope=slope,
                adx=adx,
                vol_x=preview_vol,
                rsi=rsi,
            )
            ranker_info = _ml_candidate_ranker_components(
                sym=sym,
                tf=tf,
                signal_type=mode,
                feat=feat,
                data=data,
                i=i,
                is_bull_day=is_bull_day,
                candidate_score=candidate_score,
                base_score=base_score,
                score_floor=score_floor,
                forecast_return_pct=forecast_return_pct,
                today_change_pct=intraday_change_pct,
                ml_proba=ml_proba,
                mtf_soft_penalty=mtf_soft_penalty,
                fresh_priority=_is_fresh_priority_candidate(mode, None),
                catchup=False,
                continuation_profile=continuation_profile,
                signal_flags=signal_flags,
            )
            candidate_score += _ml_candidate_ranker_runtime_bonus(ranker_info)
            if _top_gainer_chase_guard_reason(
                tf=tf,
                mode=mode,
                rsi=rsi,
                daily_range=daily_range,
            ):
                continue
            if _top_gainer_objective_gate_reason(
                tf=tf,
                mode=mode,
                intraday_change_pct=intraday_change_pct,
                daily_range=daily_range,
                vol_x=preview_vol,
                adx=adx,
                candidate_score=candidate_score,
            ):
                continue
            if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False):
                min_score = score_floor
                if candidate_score < min_score:
                    if not _entry_score_borderline_bypass_ok(
                        tf=tf,
                        mode=mode,
                        candidate_score=candidate_score,
                        min_score=min_score,
                        price=float(c[i]),
                        ema20=preview_ema20,
                        slope=slope,
                        adx=adx,
                        rsi=rsi,
                        vol_x=preview_vol,
                        daily_range=daily_range,
                    ) and not _entry_score_continuation_bypass_ok(
                        tf=tf,
                        mode=mode,
                        candidate_score=candidate_score,
                        price=float(c[i]),
                        ema20=preview_ema20,
                        slope=slope,
                        adx=adx,
                        rsi=rsi,
                        vol_x=preview_vol,
                        daily_range=daily_range,
                        continuation_profile=continuation_profile,
                        is_bull_day=is_bull_day,
                    ):
                        continue
            trend_guard_reason = _trend_entry_quality_guard_reason(
                tf=tf,
                mode=mode,
                price=float(c[i]),
                ema20=preview_ema20,
                slope=slope,
                adx=adx,
                rsi=rsi,
                vol_x=preview_vol,
                daily_range=daily_range,
                forecast_return_pct=forecast_return_pct,
            )
            if trend_guard_reason:
                continue
            hour_utc = int(ts_ms // 3_600_000) % 24
            block_hours = getattr(config, "ENTRY_BLOCK_HOURS", [])
            if block_hours and hour_utc in block_hours:
                bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
                if (
                    time_block_streak_mode == mode
                    and time_block_streak_ts is not None
                    and ts_ms <= time_block_streak_ts + bar_ms * 2
                ):
                    time_block_streak_count += 1
                else:
                    time_block_streak_count = 1
                time_block_streak_mode = mode
                time_block_streak_ts = ts_ms
                if not _time_block_bypass_allowed(
                    tf,
                    mode,
                    candidate_score,
                    preview_vol,
                    continuation_profile=continuation_profile,
                ) and not _time_block_1h_prebypass_allowed(
                    tf=tf,
                    mode=mode,
                    score=candidate_score,
                    vol_x=preview_vol,
                    price=float(c[i]),
                    ema20=preview_ema20,
                    continuation_profile=continuation_profile,
                    repeat_count=time_block_streak_count,
                ):
                    last_time_block_ts = ts_ms
                    continue
                time_block_streak_count = 0
                time_block_streak_mode = None
                time_block_streak_ts = None
            else:
                time_block_streak_count = 0
                time_block_streak_mode = None
                time_block_streak_ts = None
            candidate_score += _time_block_retest_bonus(
                mode=mode,
                tf=tf,
                ts_ms=ts_ms,
                is_bull_day=is_bull_day,
                last_time_block_ts=last_time_block_ts,
            )
            ranker_final_score = float((ranker_info or {}).get("final_score", 0.0))
            ranker_top_gainer_prob = float((ranker_info or {}).get("top_gainer_prob", 0.0))
            top_gainer_score = _top_gainer_replay_score(
                tf=tf,
                mode=mode,
                intraday_change_pct=intraday_change_pct,
                daily_range=daily_range,
                vol_x=preview_vol,
                adx=adx,
                rsi=rsi,
                ranker_final_score=ranker_final_score,
                ranker_top_gainer_prob=ranker_top_gainer_prob,
            )
            candidates.append(
                ReplayCandidate(
                    sym=sym,
                    tf=tf,
                    mode=mode,
                    ts_ms=ts_ms,
                    i=i,
                    price=float(c[i]),
                    trail_k=trail_k,
                    max_hold_bars=max_hold,
                    score=candidate_score,
                    ranker_final_score=ranker_final_score,
                    ranker_top_gainer_prob=ranker_top_gainer_prob,
                    top_gainer_score=top_gainer_score,
                    four_h_context_score=four_h_context_score,
                    four_h_context_label=four_h_context_label,
                    rsi=rsi,
                    daily_range=daily_range,
                    vol_x=preview_vol,
                    adx=adx,
                    intraday_change_pct=intraday_change_pct,
                )
            )
    finally:
        config._bull_day_active = prev_bull_state
        config._btc_vs_ema50 = prev_btc_vs_ema50
    return candidates


def _stats(values: List[float]) -> Optional[dict]:
    if not values:
        return None
    return {
        "n": len(values),
        "avg": round(mean(values), 4),
        "win_rate": round(sum(1 for v in values if v > 0) / len(values), 4),
    }


def _metric_stats(values: List[float]) -> Optional[dict]:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return None
    clean_sorted = sorted(clean)
    return {
        "n": len(clean_sorted),
        "avg": round(mean(clean_sorted), 4),
        "median": round(median(clean_sorted), 4),
        "min": round(clean_sorted[0], 4),
        "max": round(clean_sorted[-1], 4),
    }


def _update_trade_extrema(trade: ReplayTrade, data: np.ndarray, idx: int) -> None:
    if idx < 0 or idx >= len(data["c"]) or trade.entry_price <= 0:
        return
    high_now = float(data["h"][idx]) if "h" in data.dtype.names else float(data["c"][idx])
    low_now = float(data["l"][idx]) if "l" in data.dtype.names else float(data["c"][idx])
    if not np.isfinite(high_now):
        high_now = float(data["c"][idx])
    if not np.isfinite(low_now):
        low_now = float(data["c"][idx])
    if trade.max_price_since_entry <= 0:
        trade.max_price_since_entry = trade.entry_price
    if trade.min_price_since_entry <= 0:
        trade.min_price_since_entry = trade.entry_price
    trade.max_price_since_entry = max(trade.max_price_since_entry, high_now)
    trade.min_price_since_entry = min(trade.min_price_since_entry, low_now)
    trade.max_favorable_pct = max(0.0, (trade.max_price_since_entry / trade.entry_price - 1.0) * 100.0)
    trade.max_adverse_pct = min(0.0, (trade.min_price_since_entry / trade.entry_price - 1.0) * 100.0)


def _finalize_trade_metrics(trade: ReplayTrade) -> None:
    if trade.entry_price <= 0:
        return
    if trade.max_price_since_entry <= 0:
        trade.max_price_since_entry = max(trade.entry_price, trade.exit_price)
    if trade.min_price_since_entry <= 0:
        trade.min_price_since_entry = min(trade.entry_price, trade.exit_price or trade.entry_price)
    trade.max_favorable_pct = max(0.0, (trade.max_price_since_entry / trade.entry_price - 1.0) * 100.0)
    trade.max_adverse_pct = min(0.0, (trade.min_price_since_entry / trade.entry_price - 1.0) * 100.0)
    if trade.max_favorable_pct > 0:
        trade.exit_efficiency = trade.pnl_pct / trade.max_favorable_pct
    else:
        trade.exit_efficiency = None
    trade.giveback_pct = max(0.0, trade.max_favorable_pct - trade.pnl_pct)


def _candidate_forward_return_pct(
    candidate: ReplayCandidate,
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, dict]],
    *,
    horizon: int = 5,
) -> Optional[float]:
    pack = cache.get((candidate.sym, candidate.tf))
    if not pack:
        return None
    data, _ = pack
    future_i = candidate.i + int(horizon)
    if candidate.price <= 0 or future_i >= len(data["c"]):
        return None
    future_close = float(data["c"][future_i])
    if not np.isfinite(future_close):
        return None
    return (future_close / candidate.price - 1.0) * 100.0


def _record_cooldown_skip(
    trade: Optional[ReplayTrade],
    candidate: ReplayCandidate,
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, dict]],
    stats: ReplayRunStats,
) -> None:
    stats.cooldown_skipped_candidates += 1
    if trade is None:
        return
    trade.cooldown_blocked_count += 1
    forward_ret = _candidate_forward_return_pct(candidate, cache, horizon=5)
    if forward_ret is None or forward_ret <= 0:
        return
    harm = float(forward_ret)
    trade.cooldown_positive_blocked_count += 1
    trade.cooldown_harm_pct += harm
    stats.cooldown_positive_skips += 1
    stats.cooldown_harm_pct += harm


def _trade_to_example(trade: ReplayTrade) -> dict:
    return {
        "sym": trade.sym,
        "tf": trade.tf,
        "mode": trade.mode,
        "entry_ts": datetime.fromtimestamp(trade.entry_ts / 1000, tz=timezone.utc).isoformat(),
        "exit_ts": datetime.fromtimestamp(trade.exit_ts / 1000, tz=timezone.utc).isoformat(),
        "pnl_pct": round(trade.pnl_pct, 4),
        "reason": trade.exit_reason,
        "capture_ratio_at_entry": _finite_or_none(trade.capture_ratio_at_entry),
        "lead_time_to_final_top_min": _finite_or_none(trade.lead_time_to_final_top_min, digits=2),
        "exit_efficiency": _finite_or_none(trade.exit_efficiency),
        "giveback_pct": round(float(trade.giveback_pct), 4),
        "cooldown_harm_pct": round(float(trade.cooldown_harm_pct), 4),
    }


def _position_sort_key(trade: ReplayTrade) -> Tuple[float, int]:
    return (trade.entry_score, _signal_priority(trade.mode))


def _replacement_min_delta_for_candidate(mode: str) -> float:
    if _is_fresh_priority_mode(mode):
        return float(getattr(config, "PORTFOLIO_REPLACE_FRESH_MIN_DELTA", 4.0))
    return float(getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8.0))


def _count_open_trades(
    open_positions: Dict[Tuple[str, str], ReplayTrade],
    *,
    tf: Optional[str] = None,
    mode: Optional[str] = None,
) -> int:
    count = 0
    for trade in open_positions.values():
        if tf is not None and trade.tf != tf:
            continue
        if mode is not None and trade.mode != mode:
            continue
        count += 1
    return count


def _replacement_extra_delta(trade: ReplayTrade, current_price: Optional[float], entry_adx: float) -> Optional[float]:
    min_bars = int(getattr(config, "PORTFOLIO_REPLACE_MIN_BARS", 2))
    if trade.bars_held < min_bars:
        return None

    extra_delta = 0.0
    grace_bars = int(getattr(config, "PORTFOLIO_REPLACE_TREND_GRACE_BARS", 5))
    if trade.bars_held < grace_bars:
        extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_PROFIT_EXTRA_DELTA", 12.0))

    if trade.mode in ("strong_trend", "retest", "breakout", "impulse_speed"):
        extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA", 10.0))

    if entry_adx >= float(getattr(config, "PORTFOLIO_REPLACE_ADX_PROTECT_MIN", 30.0)):
        extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA", 10.0))

    if current_price is not None:
        pnl_pct = (current_price / trade.entry_price - 1.0) * 100.0 if trade.entry_price > 0 else 0.0
        if pnl_pct >= float(getattr(config, "PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT", 0.80)):
            return None
        if pnl_pct >= float(getattr(config, "PORTFOLIO_REPLACE_PROFIT_PROTECT_PCT", 0.35)):
            extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_PROFIT_EXTRA_DELTA", 12.0))

    return extra_delta


def _ranker_rotation_bonus(*, final_score: float, top_gainer_prob: float) -> float:
    return (
        float(final_score) * float(getattr(config, "PORTFOLIO_REPLACE_RANKER_FINAL_WEIGHT", 6.0))
        + float(top_gainer_prob) * float(getattr(config, "PORTFOLIO_REPLACE_TOP_GAINER_WEIGHT", 10.0))
    )


def _trade_rotation_score(trade: ReplayTrade) -> float:
    return float(trade.entry_score) + _ranker_rotation_bonus(
        final_score=float(getattr(trade, "ranker_final_score", 0.0)),
        top_gainer_prob=float(getattr(trade, "ranker_top_gainer_prob", 0.0)),
    )


def _candidate_rotation_score(candidate: ReplayCandidate) -> float:
    return float(candidate.score) + _ranker_rotation_bonus(
        final_score=float(getattr(candidate, "ranker_final_score", 0.0)),
        top_gainer_prob=float(getattr(candidate, "ranker_top_gainer_prob", 0.0)),
    )


def _candidate_ranker_rotation_ok(candidate: ReplayCandidate) -> bool:
    if not getattr(config, "PORTFOLIO_REPLACE_RANKER_ENABLED", True):
        return True
    return not (
        float(getattr(candidate, "ranker_final_score", 0.0))
        <= float(getattr(config, "PORTFOLIO_REPLACE_CANDIDATE_MIN_FINAL", -0.50))
        and float(getattr(candidate, "ranker_top_gainer_prob", 0.0))
        <= float(getattr(config, "PORTFOLIO_REPLACE_CANDIDATE_MIN_TOP_GAINER", 0.10))
    )


def _series_price_at_or_before(data: np.ndarray, ts_ms: int) -> Optional[Tuple[int, float]]:
    idx = _find_last_closed_index(data["t"], ts_ms)
    if idx is None:
        return None
    return idx, float(data["c"][idx])


def _update_trade_progress(
    trade: ReplayTrade,
    data: np.ndarray,
    feat: dict,
    idx: int,
    *,
    ts_ms: int,
    micro_pack: Optional[Tuple[np.ndarray, dict]] = None,
    protected_trailing_exit: bool = False,
    protected_weak_only: bool = False,
    exit_discriminator_shadow_policy: bool = False,
    partial_profit_take: bool = False,
) -> Optional[str]:
    close_now = float(data["c"][idx])
    _update_trade_extrema(trade, data, idx)
    atr_now = float(feat["atr"][idx]) if np.isfinite(feat["atr"][idx]) else 0.0
    trade.bars_held = idx - trade.entry_i
    for horizon in (3, 5, 10):
        if idx >= trade.entry_i + horizon:
            ret = (float(data["c"][trade.entry_i + horizon]) / trade.entry_price - 1.0) * 100.0
            if horizon == 3 and trade.ret_3 is None:
                trade.ret_3 = ret
            if horizon == 5 and trade.ret_5 is None:
                trade.ret_5 = ret
            if horizon == 10 and trade.ret_10 is None:
                trade.ret_10 = ret

    if trade.bars_held <= 0:
        return None
    current_pnl = (close_now / trade.entry_price - 1.0) * 100.0 if trade.entry_price > 0 else 0.0
    if partial_profit_take:
        _maybe_mark_partial_profit_take(
            trade=trade,
            close_now=close_now,
            ts_ms=ts_ms,
            current_pnl=current_pnl,
        )
    entry_rsi = float(feat["rsi"][trade.entry_i]) if np.isfinite(feat["rsi"][trade.entry_i]) else 50.0
    effective_trail_k = trade.trail_k
    continuation_profit_lock_active = _continuation_profit_lock_active(
        tf=trade.tf,
        mode=trade.mode,
        entry_rsi=entry_rsi,
        bars_elapsed=trade.bars_held,
        current_pnl=current_pnl,
        ret_3=trade.ret_3,
    )
    short_mode_profit_lock_active = _short_mode_profit_lock_active(
        tf=trade.tf,
        mode=trade.mode,
        bars_elapsed=trade.bars_held,
        current_pnl=current_pnl,
        ret_3=trade.ret_3,
    )
    if continuation_profit_lock_active or short_mode_profit_lock_active:
        trail_candidates = [trade.trail_k]
        floor_pcts: List[float] = []
        if continuation_profit_lock_active:
            trail_candidates.append(float(getattr(config, "CONTINUATION_PROFIT_LOCK_TRAIL_K", 1.4)))
            floor_pcts.append(float(getattr(config, "CONTINUATION_PROFIT_LOCK_FLOOR_PCT", 0.10)))
        if short_mode_profit_lock_active:
            trail_candidates.append(float(getattr(config, "SHORT_MODE_PROFIT_LOCK_TRAIL_K", 1.2)))
            floor_pcts.append(float(getattr(config, "SHORT_MODE_PROFIT_LOCK_FLOOR_PCT", 0.05)))
        effective_trail_k = min(trail_candidates)
        protect_floor_pct = max(floor_pcts) if floor_pcts else 0.0
        protect_floor = trade.entry_price * (1.0 + protect_floor_pct / 100.0)
        trade.trail_stop = max(trade.trail_stop, protect_floor)
        if continuation_profit_lock_active and micro_pack is not None:
            micro_data, micro_feat = micro_pack
            micro_signal = _continuation_micro_exit_signal(
                tf=trade.tf,
                mode=trade.mode,
                bars_elapsed=trade.bars_held,
                data_15m=micro_data,
                feat_15m=micro_feat,
                ts_ms=ts_ms,
            )
            if micro_signal is not None:
                trade.exit_ts, trade.exit_price, reason = micro_signal
                return reason
    if atr_now > 0:
        trade.trail_stop = max(trade.trail_stop, close_now - effective_trail_k * atr_now)
    if trade.trail_stop > 0 and close_now < trade.trail_stop:
        prefix = "protected trailing " if trade.protected_exit_active else ""
        return f"{prefix}ATR trail stop {trade.trail_stop:.6g}"
    if trade.bars_held >= trade.max_hold_bars and not _time_exit_should_wait(feat, idx, close_now):
        return f"time ({trade.max_hold_bars} bars)"

    fast_loss_reason = _fast_loss_ema_exit_reason(
        tf=trade.tf,
        mode=trade.mode,
        bars_elapsed=trade.bars_held,
        current_pnl=current_pnl,
        close_now=close_now,
        ema20=float(feat["ema_fast"][idx]) if np.isfinite(feat["ema_fast"][idx]) else np.nan,
        rsi=float(feat["rsi"][idx]) if np.isfinite(feat["rsi"][idx]) else np.nan,
    )
    if fast_loss_reason:
        return fast_loss_reason

    exit_reason = check_exit_conditions(
        feat,
        idx,
        data["c"].astype(float),
        mode=trade.mode,
        bars_elapsed=trade.bars_held,
        tf=trade.tf,
    )
    if _is_weak_exit_reason(exit_reason) and trade.bars_held < _min_weak_exit_bars(trade.mode):
        return None
    if _is_weak_exit_reason(exit_reason) and _trend_hold_weak_exit_active(
        trade=trade,
        feat=feat,
        idx=idx,
        close_now=close_now,
        current_pnl=current_pnl,
    ):
        if atr_now > 0:
            tight_k = min(
                float(trade.trail_k),
                float(getattr(config, "TREND_HOLD_WEAK_EXIT_TIGHTEN_ATR_K", 1.4)),
            )
            trade.trail_stop = max(trade.trail_stop, close_now - tight_k * atr_now)
        return None
    if (protected_trailing_exit or protected_weak_only) and _protected_trailing_exit_should_hold(
        trade=trade,
        reason=exit_reason,
        current_pnl=current_pnl,
        weak_only=protected_weak_only,
    ):
        max_bars = int(getattr(config, "PROTECTED_EXIT_MAX_HOLD_BARS", 4))
        if trade.protected_exit_active and trade.protected_exit_start_bar >= 0:
            if trade.bars_held - trade.protected_exit_start_bar >= max_bars:
                return f"protected trailing confirmed: {exit_reason}"
        trade.protected_exit_active = True
        trade.protected_exit_start_bar = trade.bars_held if trade.protected_exit_start_bar < 0 else trade.protected_exit_start_bar
        trade.protected_exit_original_reason = trade.protected_exit_original_reason or str(exit_reason)
        if atr_now > 0:
            tight_k = float(getattr(config, "PROTECTED_EXIT_TRAIL_ATR_K", 0.9))
            trade.trail_stop = max(trade.trail_stop, close_now - tight_k * atr_now)
        floor_pct = float(getattr(config, "PROTECTED_EXIT_PROFIT_FLOOR_PCT", 0.05))
        if trade.max_favorable_pct >= floor_pct:
            trade.trail_stop = max(trade.trail_stop, trade.entry_price * (1.0 + floor_pct / 100.0))
        return None
    if exit_discriminator_shadow_policy and _exit_discriminator_shadow_should_hold(
        trade=trade,
        reason=exit_reason,
        current_pnl=current_pnl,
    ):
        max_bars = int(getattr(config, "EXIT_DISCRIMINATOR_MAX_HOLD_BARS", 2))
        if trade.discriminator_exit_active and trade.discriminator_exit_start_bar >= 0:
            if trade.bars_held - trade.discriminator_exit_start_bar >= max_bars:
                return f"exit discriminator confirmed: {exit_reason}"
        trade.discriminator_exit_active = True
        trade.discriminator_exit_start_bar = (
            trade.bars_held if trade.discriminator_exit_start_bar < 0 else trade.discriminator_exit_start_bar
        )
        trade.discriminator_exit_original_reason = trade.discriminator_exit_original_reason or str(exit_reason)
        if atr_now > 0:
            tight_k = float(getattr(config, "EXIT_DISCRIMINATOR_TRAIL_ATR_K", 0.75))
            trade.trail_stop = max(trade.trail_stop, close_now - tight_k * atr_now)
        floor_pct = float(getattr(config, "EXIT_DISCRIMINATOR_PROFIT_FLOOR_PCT", 0.10))
        if trade.max_favorable_pct >= floor_pct:
            trade.trail_stop = max(trade.trail_stop, trade.entry_price * (1.0 + floor_pct / 100.0))
        return None
    return exit_reason

    # dead legacy branch removed; mode-aware exit is handled above
    if exit_reason and exit_reason.startswith("вљ пёЏ WEAK:") and trade.bars_held < getattr(config, "MIN_WEAK_EXIT_BARS", 2):
        return None
    return exit_reason


def summarize_trades(trades: List[ReplayTrade]) -> dict:
    by_mode: Dict[str, List[ReplayTrade]] = {}
    for trade in trades:
        by_mode.setdefault(trade.mode, []).append(trade)

    out = {}
    for mode, rows in sorted(by_mode.items()):
        out[mode] = {
            "count": len(rows),
            "pnl": _stats([t.pnl_pct for t in rows]),
            "ret_3": _stats([t.ret_3 for t in rows if t.ret_3 is not None]),
            "ret_5": _stats([t.ret_5 for t in rows if t.ret_5 is not None]),
            "ret_10": _stats([t.ret_10 for t in rows if t.ret_10 is not None]),
            "capture_ratio_at_entry": _metric_stats([
                t.capture_ratio_at_entry for t in rows if t.capture_ratio_at_entry is not None
            ]),
            "lead_time_to_final_top_min": _metric_stats([
                t.lead_time_to_final_top_min for t in rows if t.lead_time_to_final_top_min is not None
            ]),
            "exit_efficiency": _metric_stats([
                t.exit_efficiency for t in rows if t.exit_efficiency is not None
            ]),
            "giveback_pct": _metric_stats([t.giveback_pct for t in rows]),
            "cooldown_harm_pct": _metric_stats([t.cooldown_harm_pct for t in rows]),
        }
    return out


def summarize_totals(trades: List[ReplayTrade]) -> dict:
    pnl_values = [t.pnl_pct for t in trades]
    return {
        "closed_trades": len(trades),
        "pnl_total": round(sum(pnl_values), 4),
        "pnl_avg": round(mean(pnl_values), 4) if pnl_values else 0.0,
        "win_rate": round(sum(1 for v in pnl_values if v > 0) / len(pnl_values), 4) if pnl_values else 0.0,
        "ret_3": _stats([t.ret_3 for t in trades if t.ret_3 is not None]),
        "ret_5": _stats([t.ret_5 for t in trades if t.ret_5 is not None]),
        "ret_10": _stats([t.ret_10 for t in trades if t.ret_10 is not None]),
        "capture_ratio_at_entry": _metric_stats([
            t.capture_ratio_at_entry for t in trades if t.capture_ratio_at_entry is not None
        ]),
        "lead_time_to_final_top_min": _metric_stats([
            t.lead_time_to_final_top_min for t in trades if t.lead_time_to_final_top_min is not None
        ]),
        "exit_efficiency": _metric_stats([
            t.exit_efficiency for t in trades if t.exit_efficiency is not None
        ]),
        "giveback_pct": _metric_stats([t.giveback_pct for t in trades]),
        "cooldown_harm_pct": _metric_stats([t.cooldown_harm_pct for t in trades]),
    }


def build_suggestions(summary: dict) -> List[str]:
    suggestions: List[str] = []
    weak = []
    for mode, stats in summary.items():
        ret5 = stats.get("ret_5")
        if ret5 and ret5["n"] >= 3 and ret5["avg"] < 0:
            weak.append((mode, ret5["avg"]))
    weak.sort(key=lambda x: x[1])
    for mode, avg in weak[:4]:
        suggestions.append(
            f"Режим `{mode}` в replay имеет отрицательный ret_5 ({avg:+.3f}%). "
            f"Его стоит ужесточать по entry или быстрее выводить по времени/ATR."
        )
    if not suggestions:
        suggestions.append("Явно слабых режимов на текущем окне replay не найдено.")
    return suggestions


async def simulate_portfolio(
    symbols: List[str],
    timeframes: List[str],
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, dict]],
    cache_15m: Dict[str, Tuple[np.ndarray, dict]],
    cache_4h: Dict[str, Tuple[np.ndarray, dict]],
    market_ctx: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    max_open_positions: int,
    enable_replacement: bool,
    replace_min_delta: float,
    variant: str = "baseline",
    top_gainer_score_min: float = 0.0,
) -> Tuple[List[ReplayTrade], ReplayRunStats]:
    candidates_by_ts: Dict[int, List[ReplayCandidate]] = {}
    timestamps: set[int] = set()
    stats = ReplayRunStats()

    for sym in symbols:
        for tf in timeframes:
            pack = cache.get((sym, tf))
            if not pack:
                continue
            data, feat = pack
            timestamps.update(int(x) for x in data["t"][max(25, 5):])
            built = await _build_candidates_for_symbol(
                sym,
                tf,
                data,
                feat,
                cache_15m,
                cache_4h,
                market_ctx,
                include_trend_start=variant == "trend_start",
            )
            stats.candidates_total += len(built)
            for candidate in built:
                candidates_by_ts.setdefault(candidate.ts_ms, []).append(candidate)

    min_ts = min(timestamps) if timestamps else None
    max_ts = max(timestamps) if timestamps else None
    structural_events, wakeup_events = _load_temporal_scout_events(start_ms=min_ts, end_ms=max_ts)
    stats.temporal_structural_events_loaded = sum(len(rows) for rows in structural_events.values())
    stats.temporal_wakeup_events_loaded = sum(len(rows) for rows in wakeup_events.values())

    ordered_ts = sorted(timestamps)
    open_positions: Dict[Tuple[str, str], ReplayTrade] = {}
    cooldown_until: Dict[str, int] = {}
    last_closed_by_symbol: Dict[str, ReplayTrade] = {}
    suspicious_reentry_until: Dict[str, int] = {}
    trades: List[ReplayTrade] = []

    for ts_ms in ordered_ts:
        to_close: List[Tuple[Tuple[str, str], ReplayTrade]] = []
        for key, trade in list(open_positions.items()):
            data, feat = cache[(trade.sym, trade.tf)]
            idx = _find_last_closed_index(data["t"], ts_ms)
            if idx is None or idx <= trade.entry_i:
                continue
            micro_pack = cache_15m.get(trade.sym) if trade.tf == "1h" else None
            exit_reason = _update_trade_progress(
                trade,
                data,
                feat,
                idx,
                ts_ms=ts_ms,
                micro_pack=micro_pack,
                protected_trailing_exit=variant == "protected_trailing_exit",
                protected_weak_only=variant == "protected_weak_only",
                exit_discriminator_shadow_policy=variant == "exit_discriminator_shadow_policy",
                partial_profit_take=variant == "partial_profit_take",
            )
            if not exit_reason:
                continue
            if trade.exit_ts <= 0:
                trade.exit_ts = int(data["t"][idx])
            if trade.exit_price <= 0:
                trade.exit_price = float(data["c"][idx])
            trade.exit_reason = exit_reason
            to_close.append((key, trade))

        for key, trade in to_close:
            _finalize_trade_metrics(trade)
            trades.append(trade)
            cooldown_bars = _cooldown_bars_after_exit(trade.mode, trade.exit_reason)
            cooldown_until[trade.sym] = trade.exit_ts + cooldown_bars * BAR_MS[trade.tf]
            last_closed_by_symbol[trade.sym] = trade
            if _suspicious_reentry_enabled(variant):
                risk_score = _suspicious_exit_reentry_score(trade)
                if (
                    risk_score >= float(getattr(config, "SUSPICIOUS_REENTRY_EXIT_SCORE_MIN", 0.68))
                    and trade.max_favorable_pct >= float(getattr(config, "SUSPICIOUS_REENTRY_MIN_EXIT_MFE_PCT", 1.0))
                ):
                    window_bars = int(getattr(config, "SUSPICIOUS_REENTRY_WINDOW_BARS", 8))
                    suspicious_reentry_until[trade.sym] = trade.exit_ts + window_bars * BAR_MS[trade.tf]
                    stats.suspicious_reentry_windows += 1
            open_positions.pop(key, None)

        ts_candidates = candidates_by_ts.get(ts_ms, [])
        if not ts_candidates:
            continue

        use_top_gainer_score = variant in {
            "score",
            "score_replace",
            "score_replace_cluster",
            "trend_start",
            "agent_allowed",
            "agent_mode_rescue",
            "agent_wakeup_rescue",
            "protected_trailing_exit",
            "protected_weak_only",
            "exit_discriminator_shadow_policy",
            "suspicious_exit_reentry",
            "partial_profit_take",
        }
        use_cluster_cap = variant == "score_replace_cluster"
        ts_candidates.sort(
            key=lambda item: (
                item.top_gainer_score if use_top_gainer_score else item.score,
                item.score,
                _signal_priority(item.mode),
                item.price,
            ),
            reverse=True,
        )
        for candidate in ts_candidates:
            if variant == "agent_allowed" and not _agent_allowed_mode_replay(candidate.mode):
                continue
            if variant == "agent_mode_rescue" and not _agent_mode_rescue_replay_candidate_ok(
                candidate,
                top_gainer_score_min,
            ):
                continue
            if variant == "agent_wakeup_rescue":
                if not _agent_wakeup_rescue_replay_candidate_ok(
                    candidate,
                    top_gainer_score_min,
                    structural_events,
                    wakeup_events,
                ):
                    continue
                if not _agent_allowed_mode_replay(candidate.mode):
                    stats.temporal_rescue_admitted += 1
            candidate_top_gainer_score_min = _top_gainer_score_min_for_mode(
                candidate.mode,
                top_gainer_score_min,
            )
            if use_top_gainer_score and candidate.top_gainer_score < candidate_top_gainer_score_min:
                stats.skipped_top_gainer_score += 1
                continue
            sym_cooldown = cooldown_until.get(candidate.sym, 0)
            if ts_ms < sym_cooldown:
                reentry_allowed = (
                    _suspicious_reentry_enabled(variant)
                    and ts_ms <= suspicious_reentry_until.get(candidate.sym, 0)
                    and _suspicious_exit_reentry_candidate_ok(
                        candidate,
                        base_score_floor=candidate_top_gainer_score_min,
                    )
                )
                if not reentry_allowed:
                    _record_cooldown_skip(last_closed_by_symbol.get(candidate.sym), candidate, cache, stats)
                    continue
                stats.suspicious_reentry_admitted += 1
                cooldown_until[candidate.sym] = 0
                suspicious_reentry_until.pop(candidate.sym, None)
            if any(trade.sym == candidate.sym for trade in open_positions.values()):
                continue
            candidate_bucket = _signal_cluster_bucket_replay(candidate.tf, candidate.mode)
            if use_cluster_cap:
                same_cluster = [
                    key for key, trade in open_positions.items()
                    if _signal_cluster_bucket_replay(trade.tf, trade.mode) == candidate_bucket
                ]
                if len(same_cluster) >= _signal_cluster_cap_replay(candidate_bucket):
                    if not enable_replacement:
                        stats.skipped_cluster_cap += 1
                        continue
                    # A same-cluster replacement keeps exposure flat; cross-cluster replacement would exceed the cap.
                    cluster_replaceable_exists = any(
                        _signal_priority(candidate.mode) >= _signal_priority(open_positions[key].mode)
                        for key in same_cluster
                    )
                    if not cluster_replaceable_exists:
                        stats.skipped_cluster_cap += 1
                        continue
            if candidate.tf == "1h" and candidate.mode == "impulse_speed":
                impulse_cap = int(getattr(config, "MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS", 0))
                if impulse_cap > 0:
                    n_impulse_1h = _count_open_trades(open_positions, tf="1h", mode="impulse_speed")
                    if n_impulse_1h >= impulse_cap:
                        continue

            effective_max_positions = max_open_positions
            reserve_slots = int(getattr(config, "FRESH_SIGNAL_RESERVED_SLOTS", 0))
            if reserve_slots > 0 and not _is_fresh_priority_mode(candidate.mode):
                effective_max_positions = max(1, max_open_positions - reserve_slots)

            if len(open_positions) >= effective_max_positions:
                if not enable_replacement:
                    stats.skipped_portfolio_full += 1
                    continue
                if len(open_positions) >= max_open_positions:
                    late_rotation_reason = _late_impulse_speed_rotation_reason(
                        tf=candidate.tf,
                        mode=candidate.mode,
                        rsi=candidate.rsi,
                        daily_range=candidate.daily_range,
                    )
                    if late_rotation_reason is not None:
                        stats.skipped_portfolio_full += 1
                        continue
                replaceable: List[Tuple[float, Tuple[str, str], ReplayTrade, int, float]] = []
                candidate_min_delta = _replacement_min_delta_for_candidate(candidate.mode)
                use_ranker_rotation = bool(getattr(config, "PORTFOLIO_REPLACE_RANKER_ENABLED", True))
                if use_ranker_rotation and not _candidate_ranker_rotation_ok(candidate):
                    stats.skipped_portfolio_full += 1
                    continue
                candidate_rotation_score = (
                    float(candidate.top_gainer_score)
                    if use_top_gainer_score
                    else (_candidate_rotation_score(candidate) if use_ranker_rotation else float(candidate.score))
                )
                for open_key, open_trade in open_positions.items():
                    if use_cluster_cap and len([
                        trade for trade in open_positions.values()
                        if _signal_cluster_bucket_replay(trade.tf, trade.mode) == candidate_bucket
                    ]) >= _signal_cluster_cap_replay(candidate_bucket):
                        if _signal_cluster_bucket_replay(open_trade.tf, open_trade.mode) != candidate_bucket:
                            continue
                    if _signal_priority(candidate.mode) < _signal_priority(open_trade.mode):
                        continue
                    open_data, open_feat = cache[(open_trade.sym, open_trade.tf)]
                    current_point = _series_price_at_or_before(open_data, ts_ms)
                    if current_point is None:
                        continue
                    idx, price = current_point
                    open_trade.bars_held = idx - open_trade.entry_i
                    entry_adx = float(open_feat["adx"][open_trade.entry_i]) if np.isfinite(open_feat["adx"][open_trade.entry_i]) else 0.0
                    extra_delta = _replacement_extra_delta(open_trade, price, entry_adx)
                    if extra_delta is None:
                        continue
                    weakest_score = float(open_trade.entry_score)
                    weakest_rotation_score = weakest_score
                    if use_ranker_rotation:
                        if (
                            float(getattr(open_trade, "ranker_final_score", 0.0))
                            > float(getattr(config, "PORTFOLIO_REPLACE_POSITION_FINAL_MAX", 0.0))
                            and float(getattr(open_trade, "ranker_top_gainer_prob", 0.0))
                            > float(getattr(config, "PORTFOLIO_REPLACE_POSITION_TOP_GAINER_MAX", 0.20))
                        ):
                            continue
                        weakest_rotation_score = _trade_rotation_score(open_trade)
                    if use_top_gainer_score:
                        score_to_compare = candidate_rotation_score
                        baseline_score = float(getattr(open_trade, "top_gainer_score", 0.0))
                    else:
                        score_to_compare = candidate_rotation_score if use_ranker_rotation else float(candidate.score)
                        baseline_score = weakest_rotation_score if use_ranker_rotation else weakest_score
                    if score_to_compare < baseline_score + candidate_min_delta + extra_delta:
                        continue
                    replaceable.append((baseline_score, open_key, open_trade, idx, price))

                if not replaceable:
                    stats.skipped_portfolio_full += 1
                    continue

                replaceable.sort(key=lambda item: item[0])
                _, weakest_key, weakest_trade, idx, price = replaceable[0]
                weakest_data, _ = cache[(weakest_trade.sym, weakest_trade.tf)]
                _update_trade_extrema(weakest_trade, weakest_data, idx)
                weakest_trade.exit_ts = int(weakest_data["t"][idx])
                weakest_trade.exit_price = price
                weakest_trade.exit_reason = f"replaced_by_{candidate.sym}_{candidate.mode}"
                _finalize_trade_metrics(weakest_trade)
                trades.append(weakest_trade)
                stats.replacements_total += 1
                if weakest_trade.pnl_pct > 0:
                    stats.replacements_worsened += 1
                else:
                    stats.replacements_improved += 1
                cooldown_bars = _cooldown_bars_after_exit(weakest_trade.mode, weakest_trade.exit_reason)
                cooldown_until[weakest_trade.sym] = weakest_trade.exit_ts + cooldown_bars * BAR_MS[weakest_trade.tf]
                last_closed_by_symbol[weakest_trade.sym] = weakest_trade
                open_positions.pop(weakest_key, None)

            data, feat = cache[(candidate.sym, candidate.tf)]
            atr_val = float(feat["atr"][candidate.i]) if np.isfinite(feat["atr"][candidate.i]) else 0.0
            trail_stop = candidate.price - candidate.trail_k * atr_val if atr_val > 0 else 0.0
            day_metrics = _entry_day_metrics(data, candidate.i, candidate.price)
            open_positions[(candidate.sym, candidate.tf)] = ReplayTrade(
                sym=candidate.sym,
                tf=candidate.tf,
                mode=candidate.mode,
                entry_ts=candidate.ts_ms,
                entry_price=candidate.price,
                entry_i=candidate.i,
                trail_k=candidate.trail_k,
                max_hold_bars=candidate.max_hold_bars,
                trail_stop=trail_stop,
                entry_score=candidate.score,
                ranker_final_score=float(getattr(candidate, "ranker_final_score", 0.0)),
                ranker_top_gainer_prob=float(getattr(candidate, "ranker_top_gainer_prob", 0.0)),
                top_gainer_score=float(getattr(candidate, "top_gainer_score", 0.0)),
                entry_rsi=float(getattr(candidate, "rsi", 50.0)),
                entry_daily_range=float(getattr(candidate, "daily_range", 0.0)),
                entry_intraday_change_pct=float(getattr(candidate, "intraday_change_pct", 0.0)),
                capture_ratio_at_entry=day_metrics["capture_ratio_at_entry"],
                lead_time_to_final_top_min=day_metrics["lead_time_to_final_top_min"],
                day_open_price=float(day_metrics["day_open_price"]),
                day_final_price=float(day_metrics["day_final_price"]),
                max_price_since_entry=float(candidate.price),
                min_price_since_entry=float(candidate.price),
            )

    for key, trade in list(open_positions.items()):
        data, feat = cache[(trade.sym, trade.tf)]
        idx = len(data["c"]) - 2
        micro_pack = cache_15m.get(trade.sym) if trade.tf == "1h" else None
        _update_trade_progress(trade, data, feat, idx, ts_ms=int(data["t"][idx]), micro_pack=micro_pack)
        if trade.exit_ts <= 0:
            trade.exit_ts = int(data["t"][idx])
        if trade.exit_price <= 0:
            trade.exit_price = float(data["c"][idx])
        trade.exit_reason = "open_at_end"
        _finalize_trade_metrics(trade)
        trades.append(trade)
        open_positions.pop(key, None)

    return trades, stats


async def simulate_symbol(
    symbols: List[str],
    timeframes: List[str],
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, dict]],
    cache_15m: Dict[str, Tuple[np.ndarray, dict]],
    cache_4h: Optional[Dict[str, Tuple[np.ndarray, dict]]] = None,
    *,
    max_open_positions: int,
    enable_replacement: bool,
    replace_min_delta: float,
) -> List[ReplayTrade]:
    """Backward-compatible shim kept for tests and older callers."""
    trades, _ = await simulate_portfolio(
        symbols,
        timeframes,
        cache,
        cache_15m,
        cache_4h or {},
        None,
        max_open_positions=max_open_positions,
        enable_replacement=enable_replacement,
        replace_min_delta=replace_min_delta,
        variant="score_replace" if enable_replacement else "baseline",
        top_gainer_score_min=0.0,
    )
    return trades


def _make_report(
    *,
    start: datetime,
    end: datetime,
    symbols: List[str],
    timeframes: List[str],
    trades: List[ReplayTrade],
    run_stats: ReplayRunStats,
    label: str,
    final_top_symbols: Optional[set[str]] = None,
) -> dict:
    summary = summarize_trades(trades)
    final_top_symbols = final_top_symbols or set()
    traded_symbols = {t.sym for t in trades}
    captured_symbols = sorted(traded_symbols & final_top_symbols)
    objective = {
        "final_top_n": len(final_top_symbols),
        "captured_top_symbols": captured_symbols,
        "capture_rate": round(len(captured_symbols) / len(final_top_symbols), 4) if final_top_symbols else 0.0,
        "symbol_precision": round(len(captured_symbols) / len(traded_symbols), 4) if traded_symbols and final_top_symbols else 0.0,
        "trade_precision": round(
            sum(1 for t in trades if t.sym in final_top_symbols) / len(trades),
            4,
        ) if trades and final_top_symbols else 0.0,
        "non_objective_symbols": sorted(traded_symbols - final_top_symbols),
    }
    return {
        "label": label,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "symbols": symbols,
        "timeframes": timeframes,
        "trades_total": len(trades),
        "totals": summarize_totals(trades),
        "run_stats": {
            "candidates_total": run_stats.candidates_total,
            "skipped_portfolio_full": run_stats.skipped_portfolio_full,
            "replacements_total": run_stats.replacements_total,
            "replacements_improved": run_stats.replacements_improved,
            "replacements_worsened": run_stats.replacements_worsened,
            "skipped_top_gainer_score": run_stats.skipped_top_gainer_score,
            "skipped_cluster_cap": run_stats.skipped_cluster_cap,
            "cooldown_skipped_candidates": run_stats.cooldown_skipped_candidates,
            "cooldown_positive_skips": run_stats.cooldown_positive_skips,
            "cooldown_harm_pct": round(float(run_stats.cooldown_harm_pct), 4),
            "temporal_structural_events_loaded": run_stats.temporal_structural_events_loaded,
            "temporal_wakeup_events_loaded": run_stats.temporal_wakeup_events_loaded,
            "temporal_rescue_admitted": run_stats.temporal_rescue_admitted,
            "suspicious_reentry_windows": run_stats.suspicious_reentry_windows,
            "suspicious_reentry_admitted": run_stats.suspicious_reentry_admitted,
        },
        "summary": summary,
        "objective": objective,
        "suggestions": build_suggestions(summary),
        "examples": [_trade_to_example(t) for t in trades[:20]],
    }


def _final_top_symbols(
    symbols: List[str],
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, dict]],
    top_n: int = 10,
) -> set[str]:
    ranked: List[tuple[float, str]] = []
    for sym in symbols:
        item = cache.get((sym, "15m"))
        if not item:
            continue
        data, _ = item
        try:
            c_arr = data["c"]
        except Exception:
            c_arr = None
        if c_arr is None or len(c_arr) < 3:
            continue
        start_price = float(c_arr[0])
        end_price = float(c_arr[-2])
        if start_price <= 0 or not np.isfinite(start_price) or not np.isfinite(end_price):
            continue
        ranked.append(((end_price / start_price - 1.0) * 100.0, sym))
    ranked.sort(reverse=True)
    return {sym for _, sym in ranked[:top_n]}


async def run_replay(
    symbols: List[str],
    days: int,
    timeframes: List[str],
    *,
    max_open_positions: int,
    compare_baseline: bool,
    replace_min_delta: float,
    variant: str = "score_replace",
    top_gainer_score_min: float = 0.0,
    objective_top_n: int = 10,
) -> dict:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    async with aiohttp.ClientSession() as session:
        cache: Dict[Tuple[str, str], Tuple[np.ndarray, dict]] = {}
        fetch_symbols = list(dict.fromkeys(list(symbols) + ["BTCUSDT"]))
        for sym in fetch_symbols:
            for tf in sorted(set(timeframes) | {"15m", "4h"}):
                data = await fetch_klines(session, sym, tf, start_ms, end_ms)
                if data is None:
                    continue
                feat = compute_features(data["o"], data["h"], data["l"], data["c"].astype(float), data["v"])
                cache[(sym, tf)] = (data, feat)

    cache_15m = {sym: cache[(sym, "15m")] for sym in symbols if (sym, "15m") in cache}
    cache_4h = {sym: cache[(sym, "4h")] for sym in symbols if (sym, "4h") in cache}
    market_ctx = _build_bull_day_context(cache.get(("BTCUSDT", "1h"), (None, None))[0] if ("BTCUSDT", "1h") in cache else None)
    final_top_symbols = _final_top_symbols(symbols, cache, top_n=objective_top_n)
    enable_replacement = variant in {
        "score_replace",
        "score_replace_cluster",
        "protected_trailing_exit",
        "protected_weak_only",
        "exit_discriminator_shadow_policy",
        "suspicious_exit_reentry",
        "baseline_suspicious_reentry",
        "partial_profit_take",
    } or (
        variant == "baseline" and bool(getattr(config, "PORTFOLIO_REPLACE_ENABLED", True))
    )
    portfolio_trades, portfolio_stats = await simulate_portfolio(
        symbols,
        timeframes,
        cache,
        cache_15m,
        cache_4h,
        market_ctx,
        max_open_positions=max_open_positions,
        enable_replacement=enable_replacement,
        replace_min_delta=replace_min_delta,
        variant=variant,
        top_gainer_score_min=top_gainer_score_min,
    )
    reports = {
        "portfolio": _make_report(
            start=start,
            end=end,
            symbols=symbols,
            timeframes=timeframes,
            trades=portfolio_trades,
            run_stats=portfolio_stats,
            label="portfolio",
            final_top_symbols=final_top_symbols,
        )
    }

    if compare_baseline:
        baseline_trades, baseline_stats = await simulate_portfolio(
            symbols,
            timeframes,
            cache,
            cache_15m,
            cache_4h,
            market_ctx,
            max_open_positions=max_open_positions,
            enable_replacement=False,
            replace_min_delta=replace_min_delta,
            variant="baseline",
            top_gainer_score_min=0.0,
        )
        reports["baseline"] = _make_report(
            start=start,
            end=end,
            symbols=symbols,
            timeframes=timeframes,
            trades=baseline_trades,
            run_stats=baseline_stats,
            label="baseline",
            final_top_symbols=final_top_symbols,
        )

    comparison = {}
    if "baseline" in reports:
        portfolio_totals = reports["portfolio"]["totals"]
        baseline_totals = reports["baseline"]["totals"]
        portfolio_stats_dict = reports["portfolio"]["run_stats"]
        baseline_stats_dict = reports["baseline"]["run_stats"]
        comparison = {
            "trades_delta": reports["portfolio"]["trades_total"] - reports["baseline"]["trades_total"],
            "pnl_total_delta": round(portfolio_totals["pnl_total"] - baseline_totals["pnl_total"], 4),
            "pnl_avg_delta": round(portfolio_totals["pnl_avg"] - baseline_totals["pnl_avg"], 4),
            "win_rate_delta": round(portfolio_totals["win_rate"] - baseline_totals["win_rate"], 4),
            "ret_3_avg_delta": round(((portfolio_totals["ret_3"] or {}).get("avg", 0.0)) - ((baseline_totals["ret_3"] or {}).get("avg", 0.0)), 4),
            "ret_5_avg_delta": round(((portfolio_totals["ret_5"] or {}).get("avg", 0.0)) - ((baseline_totals["ret_5"] or {}).get("avg", 0.0)), 4),
            "ret_10_avg_delta": round(((portfolio_totals["ret_10"] or {}).get("avg", 0.0)) - ((baseline_totals["ret_10"] or {}).get("avg", 0.0)), 4),
            "skipped_full_delta": portfolio_stats_dict["skipped_portfolio_full"] - baseline_stats_dict["skipped_portfolio_full"],
            "replacements_total": portfolio_stats_dict["replacements_total"],
            "replacements_improved": portfolio_stats_dict["replacements_improved"],
            "replacements_worsened": portfolio_stats_dict["replacements_worsened"],
            "modes_only_in_portfolio": sorted(
                set(reports["portfolio"]["summary"]) - set(reports["baseline"]["summary"])
            ),
        }

    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "symbols": symbols,
        "timeframes": timeframes,
        "max_open_positions": max_open_positions,
        "replace_min_delta": replace_min_delta,
        "variant": variant,
        "top_gainer_score_min": top_gainer_score_min,
        "objective_top_n": objective_top_n,
        "reports": reports,
        "comparison": comparison,
    }


def render_text(report: dict) -> str:
    lines = [
        "Replay Backtest",
        f"Window: {report['window_start']} .. {report['window_end']}",
        f"Symbols: {len(report['symbols'])}, timeframes: {', '.join(report['timeframes'])}",
        f"Portfolio size: {report['max_open_positions']}, replace delta: {report['replace_min_delta']:.2f}",
        f"Variant: {report.get('variant', 'n/a')}, top-gainer min score: {report.get('top_gainer_score_min', 0.0):.2f}, objective top N: {report.get('objective_top_n', 10)}",
    ]
    for name, sub in report["reports"].items():
        totals = sub["totals"]
        run_stats = sub["run_stats"]
        lines.extend(
            [
                "",
                f"{name.title()}:",
                f"  Trades: {sub['trades_total']}",
                f"  Totals: pnl_total={totals['pnl_total']:+.4f}% pnl_avg={totals['pnl_avg']:+.4f}% win_rate={totals['win_rate']:.1%}",
                f"  Flow: candidates={run_stats['candidates_total']} skipped_full={run_stats['skipped_portfolio_full']} replacements={run_stats['replacements_total']}",
                f"  Objective: symbol_precision={sub['objective']['symbol_precision']:.1%} trade_precision={sub['objective']['trade_precision']:.1%} recall={sub['objective']['capture_rate']:.1%} captured={len(sub['objective']['captured_top_symbols'])}/{sub['objective']['final_top_n']}",
                f"  Score/cluster skips: score={run_stats.get('skipped_top_gainer_score', 0)} cluster={run_stats.get('skipped_cluster_cap', 0)}",
                f"  Cooldown harm: skips={run_stats.get('cooldown_skipped_candidates', 0)} positive={run_stats.get('cooldown_positive_skips', 0)} harm={run_stats.get('cooldown_harm_pct', 0.0):+.4f}%",
                "  By mode:",
            ]
        )
        for key in ("ret_3", "ret_5", "ret_10"):
            val = totals.get(key)
            if val:
                lines.append(f"    total_{key}: n={val['n']} avg={val['avg']:+.4f}% wr={val['win_rate']:.1%}")
        for key in ("capture_ratio_at_entry", "lead_time_to_final_top_min", "exit_efficiency", "giveback_pct"):
            val = totals.get(key)
            if val:
                lines.append(
                    f"    total_{key}: n={val['n']} avg={val['avg']:+.4f} median={val['median']:+.4f}"
                )
        for mode, stats in sub["summary"].items():
            pnl = stats.get("pnl")
            if not pnl:
                continue
            lines.append(f"    {mode}: trades={stats['count']} pnl_avg={pnl['avg']:+.4f}% wr={pnl['win_rate']:.1%}")
            for key in ("ret_3", "ret_5", "ret_10"):
                val = stats.get(key)
                if val:
                    lines.append(f"      {key}: n={val['n']} avg={val['avg']:+.4f}% wr={val['win_rate']:.1%}")
        lines.append("  Suggestions:")
        for item in sub["suggestions"]:
            lines.append(f"    - {item}")

    if report.get("comparison"):
        lines.extend(
            [
                "",
                "Comparison:",
                f"  trades_delta={report['comparison'].get('trades_delta', 0):+d}",
                f"  pnl_total_delta={report['comparison'].get('pnl_total_delta', 0.0):+.4f}%",
                f"  pnl_avg_delta={report['comparison'].get('pnl_avg_delta', 0.0):+.4f}%",
                f"  win_rate_delta={report['comparison'].get('win_rate_delta', 0.0):+.1%}",
                f"  ret_3_avg_delta={report['comparison'].get('ret_3_avg_delta', 0.0):+.4f}%",
                f"  ret_5_avg_delta={report['comparison'].get('ret_5_avg_delta', 0.0):+.4f}%",
                f"  ret_10_avg_delta={report['comparison'].get('ret_10_avg_delta', 0.0):+.4f}%",
                f"  skipped_full_delta={report['comparison'].get('skipped_full_delta', 0):+d}",
                f"  replacements_total={report['comparison'].get('replacements_total', 0)} improved={report['comparison'].get('replacements_improved', 0)} worsened={report['comparison'].get('replacements_worsened', 0)}",
            ]
        )
        modes_only = report["comparison"].get("modes_only_in_portfolio", [])
        if modes_only:
            lines.append(f"  modes_only_in_portfolio={', '.join(modes_only)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay strategy backtest on Binance historical candles")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframes", nargs="*", default=["15m", "1h"])
    parser.add_argument("--max-open-positions", type=int, default=getattr(config, "MAX_OPEN_POSITIONS", 6))
    parser.add_argument("--replace-min-delta", type=float, default=getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8.0))
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "score",
            "score_replace",
            "score_replace_cluster",
            "trend_start",
            "agent_allowed",
            "agent_mode_rescue",
            "agent_wakeup_rescue",
            "protected_trailing_exit",
            "protected_weak_only",
            "exit_discriminator_shadow_policy",
            "suspicious_exit_reentry",
            "baseline_suspicious_reentry",
            "partial_profit_take",
        ],
        default="score_replace",
    )
    parser.add_argument("--top-gainer-score-min", type=float, default=18.0)
    parser.add_argument("--objective-top-n", type=int, default=15)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    symbols = args.symbols or config.load_watchlist()
    report = asyncio.run(
        run_replay(
            symbols,
            args.days,
            args.timeframes,
            max_open_positions=args.max_open_positions,
            compare_baseline=not args.no_baseline,
            replace_min_delta=args.replace_min_delta,
            variant=args.variant,
            top_gainer_score_min=args.top_gainer_score_min,
            objective_top_n=args.objective_top_n,
        )
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
