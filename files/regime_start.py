from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FOUR_H_MS = 4 * 60 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class RegimeStartProfile:
    name: str = "base_recovery_v1"
    ema_period_4h: int = 20
    slope_lookback_4h: int = 3
    slope_min_pct_4h: float = 0.08
    rsi_min_4h: float = 50.0
    rsi_max_4h: float = 70.0
    adx_min_4h: float = 18.0
    vol_x_min_4h: float = 0.65
    price_edge_max_pct_4h: float = 3.0
    daily_ema_period: int = 7
    daily_rsi_min: float = 35.0
    daily_rsi_max: float = 65.0
    daily_price_edge_min_pct: float = -3.0
    daily_return_3d_max_pct: float = 12.0
    cooldown_4h_bars: int = 42
    min_4h_bars: int = 80
    min_daily_bars: int = 40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeStartSignal:
    profile: str
    bar_index_4h: int
    daily_index: int
    bar_open_ts_ms: int
    decision_ts_ms: int
    price: float
    ema20_4h: float
    slope_pct_4h: float
    rsi_4h: float
    adx_4h: float
    vol_x_4h: float
    macd_hist_4h: float
    daily_close: float
    daily_ema7: float
    daily_rsi: float
    daily_macd_hist: float
    daily_return_3d_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_from_config(config: Any) -> RegimeStartProfile:
    return RegimeStartProfile(
        name=str(getattr(config, "REGIME_START_PROFILE", "base_recovery_v1")),
        slope_min_pct_4h=float(getattr(config, "REGIME_START_4H_SLOPE_MIN_PCT", 0.08)),
        rsi_min_4h=float(getattr(config, "REGIME_START_4H_RSI_MIN", 50.0)),
        rsi_max_4h=float(getattr(config, "REGIME_START_4H_RSI_MAX", 70.0)),
        adx_min_4h=float(getattr(config, "REGIME_START_4H_ADX_MIN", 18.0)),
        vol_x_min_4h=float(getattr(config, "REGIME_START_4H_VOL_X_MIN", 0.65)),
        price_edge_max_pct_4h=float(getattr(config, "REGIME_START_4H_PRICE_EDGE_MAX_PCT", 3.0)),
        daily_rsi_min=float(getattr(config, "REGIME_START_1D_RSI_MIN", 35.0)),
        daily_rsi_max=float(getattr(config, "REGIME_START_1D_RSI_MAX", 65.0)),
        daily_price_edge_min_pct=float(getattr(config, "REGIME_START_1D_PRICE_EDGE_MIN_PCT", -3.0)),
        daily_return_3d_max_pct=float(getattr(config, "REGIME_START_1D_RETURN_3D_MAX_PCT", 12.0)),
        cooldown_4h_bars=int(getattr(config, "REGIME_START_DEDUP_4H_BARS", 42)),
    )


def _field(data: np.ndarray | Mapping[str, Sequence[float]], name: str) -> np.ndarray:
    try:
        values = data[name]
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"missing OHLCV field: {name}") from exc
    return np.asarray(values, dtype=float if name != "t" else np.int64)


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if not len(values):
        return out
    alpha = 2.0 / (float(period) + 1.0)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) <= period:
        return out
    delta = np.diff(values)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    out[period] = 100.0 if avg_loss == 0.0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(period + 1, len(values)):
        avg_gain = ((period - 1) * avg_gain + gains[i - 1]) / period
        avg_loss = ((period - 1) * avg_loss + losses[i - 1]) / period
        out[i] = 100.0 if avg_loss == 0.0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out


def _wilder(values: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < period:
        return out
    start = int(finite[0])
    seed_end = start + period
    if seed_end > len(values) or not np.all(np.isfinite(values[start:seed_end])):
        return out
    out[seed_end - 1] = float(np.mean(values[start:seed_end]))
    for i in range(seed_end, len(values)):
        if not np.isfinite(values[i]):
            continue
        out[i] = ((period - 1) * out[i - 1] + values[i]) / period
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    tr = np.full(n, np.nan, dtype=float)
    plus_dm = np.full(n, np.nan, dtype=float)
    minus_dm = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return tr
    tr[1:] = np.maximum.reduce((high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm[1:] = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm[1:] = np.where((down > up) & (down > 0.0), down, 0.0)
    atr = _wilder(tr, period)
    plus_smoothed = _wilder(plus_dm, period)
    minus_smoothed = _wilder(minus_dm, period)
    plus_di = np.divide(100.0 * plus_smoothed, atr, out=np.full(n, np.nan), where=atr > 0.0)
    minus_di = np.divide(100.0 * minus_smoothed, atr, out=np.full(n, np.nan), where=atr > 0.0)
    denom = plus_di + minus_di
    dx = np.divide(100.0 * np.abs(plus_di - minus_di), denom, out=np.full(n, np.nan), where=denom > 0.0)
    return _wilder(dx, period)


def _macd_hist(values: np.ndarray) -> np.ndarray:
    line = _ema(values, 12) - _ema(values, 26)
    return line - _ema(line, 9)


def _volume_ratio(values: np.ndarray, period: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(period - 1, len(values)):
        avg = float(np.mean(values[i - period + 1:i + 1]))
        if avg > 0.0:
            out[i] = values[i] / avg
    return out


def _features(data: np.ndarray | Mapping[str, Sequence[float]], *, ema_period: int) -> dict[str, np.ndarray]:
    close = _field(data, "c")
    high = _field(data, "h")
    low = _field(data, "l")
    volume = _field(data, "v")
    return {
        "ema": _ema(close, ema_period),
        "rsi": _rsi(close),
        "adx": _adx(high, low, close),
        "macd_hist": _macd_hist(close),
        "vol_x": _volume_ratio(volume),
    }


def _daily_index_for_decision(daily_t: np.ndarray, decision_ts_ms: int) -> int:
    daily_close_ts = daily_t + DAY_MS
    return int(np.searchsorted(daily_close_ts, decision_ts_ms, side="right") - 1)


def _snapshot(
    four_h: np.ndarray | Mapping[str, Sequence[float]],
    daily: np.ndarray | Mapping[str, Sequence[float]],
    i: int,
    four_h_feat: Mapping[str, np.ndarray],
    daily_feat: Mapping[str, np.ndarray],
    profile: RegimeStartProfile,
) -> RegimeStartSignal | None:
    t4 = _field(four_h, "t")
    c4 = _field(four_h, "c")
    td = _field(daily, "t")
    cd = _field(daily, "c")
    if i < max(profile.min_4h_bars - 1, profile.slope_lookback_4h, 1):
        return None
    decision_ts = int(t4[i]) + FOUR_H_MS
    di = _daily_index_for_decision(td, decision_ts)
    if di < max(profile.min_daily_bars - 1, 3, 1):
        return None

    ema4 = float(four_h_feat["ema"][i])
    ema4_prev = float(four_h_feat["ema"][i - profile.slope_lookback_4h])
    rsi4 = float(four_h_feat["rsi"][i])
    adx4 = float(four_h_feat["adx"][i])
    vol4 = float(four_h_feat["vol_x"][i])
    macd4 = float(four_h_feat["macd_hist"][i])
    macd4_prev = float(four_h_feat["macd_hist"][i - 1])
    ema_d = float(daily_feat["ema"][di])
    rsi_d = float(daily_feat["rsi"][di])
    macd_d = float(daily_feat["macd_hist"][di])
    macd_d_prev = float(daily_feat["macd_hist"][di - 1])
    values = (ema4, ema4_prev, rsi4, adx4, vol4, macd4, macd4_prev, ema_d, rsi_d, macd_d, macd_d_prev)
    if not all(np.isfinite(values)) or min(float(c4[i]), float(cd[di]), ema4, ema4_prev, ema_d) <= 0.0:
        return None

    slope4 = ((ema4 / ema4_prev) - 1.0) * 100.0
    price_edge4 = ((float(c4[i]) / ema4) - 1.0) * 100.0
    daily_edge = ((float(cd[di]) / ema_d) - 1.0) * 100.0
    daily_return3 = ((float(cd[di]) / float(cd[di - 3])) - 1.0) * 100.0
    if not (
        float(c4[i]) > ema4
        and slope4 >= profile.slope_min_pct_4h
        and profile.rsi_min_4h <= rsi4 <= profile.rsi_max_4h
        and adx4 >= profile.adx_min_4h
        and vol4 >= profile.vol_x_min_4h
        and 0.0 < macd4
        and macd4 > macd4_prev
        and price_edge4 <= profile.price_edge_max_pct_4h
        and profile.daily_rsi_min <= rsi_d <= profile.daily_rsi_max
        and macd_d > macd_d_prev
        and daily_edge >= profile.daily_price_edge_min_pct
        and daily_return3 <= profile.daily_return_3d_max_pct
    ):
        return None

    return RegimeStartSignal(
        profile=profile.name,
        bar_index_4h=i,
        daily_index=di,
        bar_open_ts_ms=int(t4[i]),
        decision_ts_ms=decision_ts,
        price=float(c4[i]),
        ema20_4h=ema4,
        slope_pct_4h=slope4,
        rsi_4h=rsi4,
        adx_4h=adx4,
        vol_x_4h=vol4,
        macd_hist_4h=macd4,
        daily_close=float(cd[di]),
        daily_ema7=ema_d,
        daily_rsi=rsi_d,
        daily_macd_hist=macd_d,
        daily_return_3d_pct=daily_return3,
    )


def detect_regime_starts(
    four_h: np.ndarray | Mapping[str, Sequence[float]],
    daily: np.ndarray | Mapping[str, Sequence[float]],
    profile: RegimeStartProfile | None = None,
) -> list[RegimeStartSignal]:
    profile = profile or RegimeStartProfile()
    if len(_field(four_h, "t")) < profile.min_4h_bars or len(_field(daily, "t")) < profile.min_daily_bars:
        return []
    four_h_feat = _features(four_h, ema_period=profile.ema_period_4h)
    daily_feat = _features(daily, ema_period=profile.daily_ema_period)
    out: list[RegimeStartSignal] = []
    last_signal_i = -profile.cooldown_4h_bars
    was_active = False
    for i in range(profile.min_4h_bars - 1, len(_field(four_h, "t"))):
        current = _snapshot(four_h, daily, i, four_h_feat, daily_feat, profile)
        active = current is not None
        if active and not was_active and i - last_signal_i >= profile.cooldown_4h_bars:
            out.append(current)
            last_signal_i = i
        was_active = active
    return out


def detect_latest_regime_start(
    four_h: np.ndarray | Mapping[str, Sequence[float]],
    daily: np.ndarray | Mapping[str, Sequence[float]],
    profile: RegimeStartProfile | None = None,
) -> RegimeStartSignal | None:
    starts = detect_regime_starts(four_h, daily, profile)
    if not starts:
        return None
    latest_i = len(_field(four_h, "t")) - 1
    return starts[-1] if starts[-1].bar_index_4h == latest_i else None
