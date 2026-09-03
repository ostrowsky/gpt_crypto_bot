from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import aiohttp
import numpy as np

import config
from regime_start import _adx, _ema, _macd_hist, _rsi, _volume_ratio, _wilder


ROOT = Path(__file__).resolve().parent.parent
HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
DEFAULT_START = datetime(2017, 8, 17, tzinfo=timezone.utc)
DEFAULT_CACHE = ROOT / ".runtime" / "slow_persistent_trend_history"
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "slow_persistent_trend_max_period_latest.json"
DEFAULT_SPEC = ROOT / "docs" / "specs" / "slow-persistent-trend-max-period.md"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DTYPE = [("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")]
ROUND_TRIP_COST_BPS = 20.0
EMBARGO_HOURS = 48
MIN_BARS = 200
FORWARD_HOURS = 36


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    swing_low_12h_min_pct: float
    swing_low_12h_max_pct: float
    return_12h_min_pct: float
    return_24h_min_pct: float
    ema7_slope_3h_min_pct: float
    ema25_slope_6h_min_pct: float
    rsi_min: float
    rsi_max: float
    adx_min: float
    adx_delta_3h_min: float
    volume_ratio_min: float
    ema25_edge_max_pct: float
    nonnegative_returns_12h_min: int
    atr_pct_max: float | None = None


PROFILES = (
    DetectorProfile(
        name="balanced_v1",
        swing_low_12h_min_pct=0.35,
        swing_low_12h_max_pct=2.50,
        return_12h_min_pct=0.15,
        return_24h_min_pct=-0.50,
        ema7_slope_3h_min_pct=0.06,
        ema25_slope_6h_min_pct=0.00,
        rsi_min=52.0,
        rsi_max=70.0,
        adx_min=18.0,
        adx_delta_3h_min=0.0,
        volume_ratio_min=0.50,
        ema25_edge_max_pct=1.50,
        nonnegative_returns_12h_min=7,
    ),
    DetectorProfile(
        name="strict_v1",
        swing_low_12h_min_pct=0.50,
        swing_low_12h_max_pct=2.00,
        return_12h_min_pct=0.25,
        return_24h_min_pct=0.00,
        ema7_slope_3h_min_pct=0.10,
        ema25_slope_6h_min_pct=0.02,
        rsi_min=55.0,
        rsi_max=68.0,
        adx_min=20.0,
        adx_delta_3h_min=0.0,
        volume_ratio_min=0.70,
        ema25_edge_max_pct=1.20,
        nonnegative_returns_12h_min=8,
    ),
    DetectorProfile(
        name="low_vol_persistence_v1",
        swing_low_12h_min_pct=0.30,
        swing_low_12h_max_pct=2.00,
        return_12h_min_pct=0.00,
        return_24h_min_pct=-1.00,
        ema7_slope_3h_min_pct=0.04,
        ema25_slope_6h_min_pct=0.00,
        rsi_min=50.0,
        rsi_max=70.0,
        adx_min=18.0,
        adx_delta_3h_min=-1.0,
        volume_ratio_min=0.40,
        ema25_edge_max_pct=1.20,
        nonnegative_returns_12h_min=7,
        atr_pct_max=1.00,
    ),
)


@dataclass(frozen=True)
class Signal:
    symbol: str
    profile: str
    bar_index: int
    bar_open_ts_ms: int
    decision_ts_ms: int
    close: float
    features: dict[str, float]


@dataclass(frozen=True)
class LabeledSignal:
    symbol: str
    profile: str
    decision_ts_ms: int
    entry_ts_ms: int
    entry_price_gross: float
    ret_12h_net_pct: float
    ret_24h_net_pct: float
    ret_36h_net_pct: float
    mfe_24h_net_pct: float
    mae_24h_net_pct: float
    useful: bool


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _closed_hour_ms(now_ms: int | None = None) -> int:
    value = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return (value // HOUR_MS) * HOUR_MS


def _rows_to_array(rows: Sequence[Sequence[Any]], *, end_ms: int) -> np.ndarray:
    usable = [row for row in rows if int(row[0]) + HOUR_MS <= end_ms]
    array = np.zeros(len(usable), dtype=DTYPE)
    if not usable:
        return array
    for name, pos, cast in (("t", 0, int), ("o", 1, float), ("h", 2, float), ("l", 3, float), ("c", 4, float), ("v", 5, float)):
        array[name] = [cast(row[pos]) for row in usable]
    return array


async def _request_batch(
    session: aiohttp.ClientSession,
    params: dict[str, Any],
    *,
    retries: int = 6,
) -> list[list[Any]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            async with session.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as response:
                if response.status in {418, 429}:
                    retry_after = float(response.headers.get("Retry-After", "1"))
                    await asyncio.sleep(max(retry_after, 1.0) * (attempt + 1))
                    continue
                response.raise_for_status()
                payload = await response.json()
                if not isinstance(payload, list):
                    raise ValueError(f"unexpected Binance payload: {type(payload).__name__}")
                return payload
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Binance request failed after {retries} attempts: {last_error}")


async def _fetch_history(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = await _request_batch(
            session,
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + HOUR_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        await asyncio.sleep(0.12)
    return _rows_to_array(rows, end_ms=end_ms)


def _cache_path(root: Path, symbol: str) -> Path:
    return root / symbol / "1h.npz"


def _load_cache(path: Path, *, start_ms: int, end_ms: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as payload:
            if int(payload["requested_start_ms"]) > start_ms or int(payload["requested_end_ms"]) < end_ms:
                return None
            array = np.zeros(len(payload["t"]), dtype=DTYPE)
            for field in ("t", "o", "h", "l", "c", "v"):
                array[field] = payload[field]
        return array[array["t"] + HOUR_MS <= end_ms]
    except Exception:
        return None


def _save_cache(path: Path, data: np.ndarray, *, start_ms: int, end_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        requested_start_ms=np.int64(start_ms),
        requested_end_ms=np.int64(end_ms),
        **{field: data[field] for field in ("t", "o", "h", "l", "c", "v")},
    )


async def _load_one_symbol(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    cache_root: Path,
    refresh: bool,
) -> tuple[str, np.ndarray | None, str | None]:
    try:
        path = _cache_path(cache_root, symbol)
        data = None if refresh else _load_cache(path, start_ms=start_ms, end_ms=end_ms)
        if data is None:
            async with semaphore:
                data = await _fetch_history(session, symbol, start_ms, end_ms)
            _save_cache(path, data, start_ms=start_ms, end_ms=end_ms)
        if len(data) < MIN_BARS + FORWARD_HOURS + 1:
            return symbol, None, f"insufficient closed 1h history: {len(data)} bars"
        return symbol, data, None
    except Exception as exc:
        return symbol, None, f"{type(exc).__name__}: {exc}"


async def load_histories(
    symbols: Sequence[str],
    *,
    start_ms: int,
    end_ms: int,
    cache_root: Path,
    refresh: bool,
    max_concurrency: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    histories: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    semaphore = asyncio.Semaphore(max_concurrency)
    connector = aiohttp.TCPConnector(limit=max(2, max_concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _load_one_symbol(
                session,
                semaphore,
                symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                cache_root=cache_root,
                refresh=refresh,
            )
            for symbol in symbols
        ]
        completed = 0
        for task in asyncio.as_completed(tasks):
            symbol, data, error = await task
            completed += 1
            if error is not None:
                errors[symbol] = error
            elif data is not None:
                histories[symbol] = data
            print(f"history {completed}/{len(tasks)} {symbol}: {'ERROR' if error else f'{len(data)} bars'}", flush=True)
    return histories, errors


def _pct_ratio(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return np.divide(
        current,
        previous,
        out=np.full(len(current), np.nan, dtype=float),
        where=np.isfinite(previous) & (previous > 0.0),
    ) * 100.0 - 100.0


def _lag_pct(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) > lag:
        out[lag:] = _pct_ratio(values[lag:], values[:-lag])
    return out


def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    indices: deque[int] = deque()
    for i, value in enumerate(values):
        while indices and indices[0] <= i - window:
            indices.popleft()
        while indices and values[indices[-1]] >= value:
            indices.pop()
        indices.append(i)
        if i >= window - 1:
            out[i] = values[indices[0]]
    return out


def _rolling_nonnegative_returns(close: np.ndarray, window: int) -> np.ndarray:
    flags = np.zeros(len(close), dtype=np.int64)
    if len(close) > 1:
        flags[1:] = close[1:] >= close[:-1]
    prefix = np.concatenate(([0], np.cumsum(flags)))
    out = np.full(len(close), np.nan, dtype=float)
    for i in range(window, len(close)):
        out[i] = float(prefix[i + 1] - prefix[i + 1 - window])
    return out


def build_features(data: np.ndarray) -> dict[str, np.ndarray]:
    close = np.asarray(data["c"], dtype=float)
    high = np.asarray(data["h"], dtype=float)
    low = np.asarray(data["l"], dtype=float)
    volume = np.asarray(data["v"], dtype=float)
    ema7 = _ema(close, 7)
    ema25 = _ema(close, 25)
    macd = _macd_hist(close)
    adx = _adx(high, low, close, 14)
    tr = np.full(len(close), np.nan, dtype=float)
    if len(close) > 1:
        tr[1:] = np.maximum.reduce(
            (high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
    atr = _wilder(tr, 14)
    low12 = _rolling_min(low, 12)
    return {
        "ema7": ema7,
        "ema25": ema25,
        "ema7_slope_3h_pct": _lag_pct(ema7, 3),
        "ema25_slope_6h_pct": _lag_pct(ema25, 6),
        "rsi": _rsi(close, 14),
        "adx": adx,
        "adx_delta_3h": np.concatenate((np.full(3, np.nan), adx[3:] - adx[:-3])),
        "macd_hist": macd,
        "macd_delta_3h": np.concatenate((np.full(3, np.nan), macd[3:] - macd[:-3])),
        "volume_ratio": _volume_ratio(volume, 20),
        "swing_low_12h_pct": _pct_ratio(close, low12),
        "return_12h_pct": _lag_pct(close, 12),
        "return_24h_pct": _lag_pct(close, 24),
        "ema25_edge_pct": _pct_ratio(close, ema25),
        "nonnegative_returns_12h": _rolling_nonnegative_returns(close, 12),
        "atr_pct": np.divide(atr, close, out=np.full(len(close), np.nan), where=close > 0.0) * 100.0,
    }


def broad_precursor_mask(data: np.ndarray, features: dict[str, np.ndarray]) -> np.ndarray:
    close = np.asarray(data["c"], dtype=float)
    mask = (
        (close > features["ema7"])
        & (features["ema7"] > features["ema25"])
        & (features["ema7_slope_3h_pct"] > 0.0)
        & (features["macd_hist"] > 0.0)
        & (features["rsi"] >= 45.0)
        & (features["rsi"] <= 75.0)
        & (features["swing_low_12h_pct"] > 0.0)
    )
    mask[:MIN_BARS] = False
    return mask & np.isfinite(features["atr_pct"])


def profile_mask(
    data: np.ndarray,
    features: dict[str, np.ndarray],
    profile: DetectorProfile,
) -> np.ndarray:
    mask = broad_precursor_mask(data, features)
    mask &= features["swing_low_12h_pct"] >= profile.swing_low_12h_min_pct
    mask &= features["swing_low_12h_pct"] <= profile.swing_low_12h_max_pct
    mask &= features["return_12h_pct"] >= profile.return_12h_min_pct
    mask &= features["return_24h_pct"] >= profile.return_24h_min_pct
    mask &= features["ema7_slope_3h_pct"] >= profile.ema7_slope_3h_min_pct
    mask &= features["ema25_slope_6h_pct"] >= profile.ema25_slope_6h_min_pct
    mask &= features["rsi"] >= profile.rsi_min
    mask &= features["rsi"] <= profile.rsi_max
    mask &= features["adx"] >= profile.adx_min
    mask &= features["adx_delta_3h"] >= profile.adx_delta_3h_min
    mask &= features["volume_ratio"] >= profile.volume_ratio_min
    mask &= features["ema25_edge_pct"] <= profile.ema25_edge_max_pct
    mask &= features["macd_delta_3h"] > 0.0
    mask &= features["nonnegative_returns_12h"] >= profile.nonnegative_returns_12h_min
    if profile.atr_pct_max is not None:
        mask &= features["atr_pct"] <= profile.atr_pct_max
    return mask


def detect_signals(
    symbol: str,
    data: np.ndarray,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    profile_name: str,
    cooldown_hours: int = 24,
) -> list[Signal]:
    output: list[Signal] = []
    was_active = False
    last_signal = -cooldown_hours
    feature_names = (
        "swing_low_12h_pct",
        "return_12h_pct",
        "return_24h_pct",
        "ema7_slope_3h_pct",
        "ema25_slope_6h_pct",
        "rsi",
        "adx",
        "adx_delta_3h",
        "volume_ratio",
        "ema25_edge_pct",
        "macd_hist",
        "macd_delta_3h",
        "nonnegative_returns_12h",
        "atr_pct",
    )
    for i in range(MIN_BARS, len(data)):
        active = bool(mask[i])
        if active and not was_active and i - last_signal >= cooldown_hours:
            output.append(
                Signal(
                    symbol=symbol,
                    profile=profile_name,
                    bar_index=i,
                    bar_open_ts_ms=int(data["t"][i]),
                    decision_ts_ms=int(data["t"][i]) + HOUR_MS,
                    close=float(data["c"][i]),
                    features={name: float(features[name][i]) for name in feature_names},
                )
            )
            last_signal = i
        was_active = active
    return output


def label_signal(signal: Signal, data: np.ndarray, *, cost_bps: float = ROUND_TRIP_COST_BPS) -> LabeledSignal | None:
    i = signal.bar_index
    if i + FORWARD_HOURS >= len(data):
        return None
    half_cost = cost_bps / 20_000.0
    entry = float(data["o"][i + 1]) * (1.0 + half_cost)

    def net_exit(price: float) -> float:
        return ((price * (1.0 - half_cost) / entry) - 1.0) * 100.0

    forward24 = data[i + 1 : i + 25]
    ret12 = net_exit(float(data["c"][i + 12]))
    ret24 = net_exit(float(data["c"][i + 24]))
    ret36 = net_exit(float(data["c"][i + 36]))
    mfe24 = net_exit(float(np.max(forward24["h"])))
    mae24 = net_exit(float(np.min(forward24["l"])))
    return LabeledSignal(
        symbol=signal.symbol,
        profile=signal.profile,
        decision_ts_ms=signal.decision_ts_ms,
        entry_ts_ms=int(data["t"][i + 1]),
        entry_price_gross=entry,
        ret_12h_net_pct=ret12,
        ret_24h_net_pct=ret24,
        ret_36h_net_pct=ret36,
        mfe_24h_net_pct=mfe24,
        mae_24h_net_pct=mae24,
        useful=bool(ret24 >= 0.50 and mfe24 >= 1.00 and mae24 > -1.00),
    )


def split_boundaries(start_ms: int, end_ms: int) -> dict[str, int]:
    span = end_ms - start_ms
    return {
        "train_validation_cut_ms": start_ms + int(span * 0.60),
        "validation_holdout_cut_ms": start_ms + int(span * 0.80),
        "embargo_ms": EMBARGO_HOURS * HOUR_MS,
    }


def split_name(decision_ts_ms: int, boundaries: dict[str, int]) -> str | None:
    first = boundaries["train_validation_cut_ms"]
    second = boundaries["validation_holdout_cut_ms"]
    embargo = boundaries["embargo_ms"]
    if decision_ts_ms < first - embargo:
        return "train"
    if first + embargo <= decision_ts_ms < second - embargo:
        return "validation"
    if decision_ts_ms >= second + embargo:
        return "holdout"
    return None


def partition(rows: Iterable[LabeledSignal], boundaries: dict[str, int]) -> dict[str, list[LabeledSignal]]:
    output: dict[str, list[LabeledSignal]] = {"train": [], "validation": [], "holdout": []}
    for row in rows:
        name = split_name(row.decision_ts_ms, boundaries)
        if name is not None:
            output[name].append(row)
    return output


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _safe_percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def metrics(
    rows: Sequence[LabeledSignal],
    *,
    period_start_ms: int,
    period_end_ms: int,
    base_rows: Sequence[LabeledSignal] | None = None,
) -> dict[str, Any]:
    denominator = len(rows)
    numerator = sum(row.useful for row in rows)
    useful_rate = numerator / denominator if denominator else None
    base_denominator = len(base_rows) if base_rows is not None else None
    base_numerator = sum(row.useful for row in base_rows) if base_rows is not None else None
    base_rate = (
        base_numerator / base_denominator
        if base_denominator is not None and base_denominator > 0 and base_numerator is not None
        else None
    )
    lift = useful_rate / base_rate if useful_rate is not None and base_rate not in {None, 0.0} else None
    lift_pp = (useful_rate - base_rate) * 100.0 if useful_rate is not None and base_rate is not None else None
    period_days = max(0.0, (period_end_ms - period_start_ms) / DAY_MS)
    day_counts = Counter(_iso(row.decision_ts_ms)[:10] for row in rows)
    ret12 = [row.ret_12h_net_pct for row in rows]
    ret24 = [row.ret_24h_net_pct for row in rows]
    ret36 = [row.ret_36h_net_pct for row in rows]
    mfe24 = [row.mfe_24h_net_pct for row in rows]
    mae24 = [row.mae_24h_net_pct for row in rows]
    return {
        "useful_numerator": int(numerator),
        "labeled_denominator": denominator,
        "useful_precision_pct": None if useful_rate is None else round(useful_rate * 100.0, 4),
        "base_useful_numerator": base_numerator,
        "base_labeled_denominator": base_denominator,
        "base_useful_precision_pct": None if base_rate is None else round(base_rate * 100.0, 4),
        "lift_x": None if lift is None else round(lift, 4),
        "lift_pp": None if lift_pp is None else round(lift_pp, 4),
        "mean_ret_12h_net_pct": None if not ret12 else round(float(_safe_mean(ret12)), 4),
        "mean_ret_24h_net_pct": None if not ret24 else round(float(_safe_mean(ret24)), 4),
        "median_ret_24h_net_pct": None if not ret24 else round(float(_safe_median(ret24)), 4),
        "p10_ret_24h_net_pct": None if not ret24 else round(float(_safe_percentile(ret24, 10.0)), 4),
        "mean_ret_36h_net_pct": None if not ret36 else round(float(_safe_mean(ret36)), 4),
        "median_mfe_24h_net_pct": None if not mfe24 else round(float(_safe_median(mfe24)), 4),
        "median_mae_24h_net_pct": None if not mae24 else round(float(_safe_median(mae24)), 4),
        "calendar_days": round(period_days, 2),
        "active_signal_days": len(day_counts),
        "signals_per_calendar_day": None if period_days <= 0 else round(denominator / period_days, 4),
        "max_signals_single_day": max(day_counts.values(), default=0),
    }


def choose_profile(validation_metrics: dict[str, dict[str, Any]]) -> str | None:
    eligible = [
        (name, value)
        for name, value in validation_metrics.items()
        if int(value.get("labeled_denominator") or 0) >= 100
        and value.get("useful_precision_pct") is not None
        and value.get("mean_ret_24h_net_pct") is not None
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            float(item[1]["useful_precision_pct"]),
            float(item[1]["mean_ret_24h_net_pct"]),
            int(item[1]["labeled_denominator"]),
            item[0],
        ),
        reverse=True,
    )
    return eligible[0][0]


def evaluate_shadow_acceptance(
    *,
    coverage_ratio: float,
    validation: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
    incident_detected_by_deadline: bool,
) -> dict[str, Any]:
    checks = {
        "coverage_at_least_95pct": coverage_ratio >= 0.95,
        "validation_labels_at_least_100": validation is not None and int(validation.get("labeled_denominator") or 0) >= 100,
        "holdout_labels_at_least_200": holdout is not None and int(holdout.get("labeled_denominator") or 0) >= 200,
        "holdout_calendar_days_at_least_180": holdout is not None and float(holdout.get("calendar_days") or 0.0) >= 180.0,
        "validation_mean_net24_positive": validation is not None and float(validation.get("mean_ret_24h_net_pct") or 0.0) > 0.0,
        "validation_median_net24_positive": validation is not None and float(validation.get("median_ret_24h_net_pct") or 0.0) > 0.0,
        "holdout_mean_net24_positive": holdout is not None and float(holdout.get("mean_ret_24h_net_pct") or 0.0) > 0.0,
        "holdout_median_net24_positive": holdout is not None and float(holdout.get("median_ret_24h_net_pct") or 0.0) > 0.0,
        "holdout_useful_precision_at_least_30pct": holdout is not None and float(holdout.get("useful_precision_pct") or 0.0) >= 30.0,
        "holdout_lift_at_least_1_25x": holdout is not None and float(holdout.get("lift_x") or 0.0) >= 1.25,
        "holdout_lift_at_least_5pp": holdout is not None and float(holdout.get("lift_pp") or 0.0) >= 5.0,
        "holdout_precision_drop_at_most_10pp": validation is not None
        and holdout is not None
        and holdout.get("useful_precision_pct") is not None
        and validation.get("useful_precision_pct") is not None
        and float(holdout["useful_precision_pct"]) >= float(validation["useful_precision_pct"]) - 10.0,
        "holdout_p10_net24_at_least_minus_2pct": holdout is not None
        and holdout.get("p10_ret_24h_net_pct") is not None
        and float(holdout["p10_ret_24h_net_pct"]) >= -2.0,
        "holdout_alerts_per_day_at_most_5": holdout is not None
        and holdout.get("signals_per_calendar_day") is not None
        and float(holdout["signals_per_calendar_day"]) <= 5.0,
        "trx_incident_detected_by_2026_09_02T10Z": bool(incident_detected_by_deadline),
    }
    sample_unknown = not (
        checks["coverage_at_least_95pct"]
        and checks["validation_labels_at_least_100"]
        and checks["holdout_labels_at_least_200"]
        and checks["holdout_calendar_days_at_least_180"]
    )
    passed = all(checks.values())
    decision = "eligible_for_shadow" if passed else ("inconclusive" if sample_unknown else "rejected")
    return {
        "decision": decision,
        "passed": passed,
        "checks": checks,
        "failed_checks": [name for name, passed_check in checks.items() if not passed_check],
        "production_buy_approved": False,
        "production_watch_approved": False,
        "actual_bot_candidate_population_validated": False,
        "unified_portfolio_alpha_validated": False,
    }


def _period_for_split(
    name: str,
    *,
    start_ms: int,
    end_ms: int,
    boundaries: dict[str, int],
) -> tuple[int, int]:
    first = boundaries["train_validation_cut_ms"]
    second = boundaries["validation_holdout_cut_ms"]
    embargo = boundaries["embargo_ms"]
    if name == "train":
        return start_ms, first - embargo
    if name == "validation":
        return first + embargo, second - embargo
    return second + embargo, end_ms


def _serialize_signal(signal: Signal | None) -> dict[str, Any] | None:
    if signal is None:
        return None
    payload = asdict(signal)
    payload["bar_open_time"] = _iso(signal.bar_open_ts_ms)
    payload["decision_time"] = _iso(signal.decision_ts_ms)
    return payload


def run_audit(
    histories: dict[str, np.ndarray],
    errors: dict[str, str],
    *,
    symbols: Sequence[str],
    start_ms: int,
    end_ms: int,
    spec_path: Path,
) -> dict[str, Any]:
    boundaries = split_boundaries(start_ms, end_ms)
    all_signals: dict[str, list[Signal]] = {"broad_precursor": []}
    all_signals.update({profile.name: [] for profile in PROFILES})
    labels: dict[str, list[LabeledSignal]] = {name: [] for name in all_signals}
    symbol_coverage: dict[str, dict[str, Any]] = {}

    for completed, (symbol, data) in enumerate(sorted(histories.items()), start=1):
        features = build_features(data)
        masks = {"broad_precursor": broad_precursor_mask(data, features)}
        masks.update({profile.name: profile_mask(data, features, profile) for profile in PROFILES})
        counts: dict[str, int] = {}
        for name, mask in masks.items():
            detected = detect_signals(symbol, data, features, mask, profile_name=name)
            all_signals[name].extend(detected)
            labels[name].extend(row for row in (label_signal(signal, data) for signal in detected) if row is not None)
            counts[name] = len(detected)
        symbol_coverage[symbol] = {
            "bars": len(data),
            "first_closed_bar_open_time": _iso(int(data["t"][0])),
            "last_closed_bar_open_time": _iso(int(data["t"][-1])),
            "signals": counts,
        }
        print(f"features {completed}/{len(histories)} {symbol}", flush=True)

    partitions = {name: partition(rows, boundaries) for name, rows in labels.items()}
    base_parts = partitions["broad_precursor"]
    result_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for profile in PROFILES:
        split_metrics: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "holdout"):
            period_start, period_end = _period_for_split(
                split,
                start_ms=start_ms,
                end_ms=end_ms,
                boundaries=boundaries,
            )
            split_metrics[split] = metrics(
                partitions[profile.name][split],
                period_start_ms=period_start,
                period_end_ms=period_end,
                base_rows=base_parts[split],
            )
        result_metrics[profile.name] = split_metrics

    validation_metrics = {name: values["validation"] for name, values in result_metrics.items()}
    selected = choose_profile(validation_metrics)
    incident_start = int(_parse_utc("2026-09-02T04:00:00Z").timestamp() * 1000)
    incident_deadline = int(_parse_utc("2026-09-02T10:00:00Z").timestamp() * 1000)
    incident_end = int(_parse_utc("2026-09-03T12:00:00Z").timestamp() * 1000)
    incident_by_profile: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        candidates = sorted(
            (
                signal
                for signal in all_signals[profile.name]
                if signal.symbol == "TRXUSDT" and incident_start <= signal.decision_ts_ms <= incident_end
            ),
            key=lambda signal: signal.decision_ts_ms,
        )
        first = candidates[0] if candidates else None
        incident_by_profile[profile.name] = {
            "first_signal": _serialize_signal(first),
            "detected_by_deadline": first is not None and first.decision_ts_ms <= incident_deadline,
            "signals_in_window": len(candidates),
        }

    coverage_ratio = len(histories) / len(symbols) if symbols else 0.0
    selected_validation = result_metrics[selected]["validation"] if selected else None
    selected_holdout = result_metrics[selected]["holdout"] if selected else None
    selected_incident = incident_by_profile.get(selected or "", {})
    acceptance = evaluate_shadow_acceptance(
        coverage_ratio=coverage_ratio,
        validation=selected_validation,
        holdout=selected_holdout,
        incident_detected_by_deadline=bool(selected_incident.get("detected_by_deadline", False)),
    )

    spec_bytes = spec_path.read_bytes()
    return {
        "schema": "slow_persistent_trend_max_period_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hypothesis": "closed-1h rolling 12-36h persistence can identify slow multi-session rises earlier than the daily-reset gate",
        "evidence_class": "market_wide_research_proxy",
        "objective_metric": False,
        "spec": {
            "path": str(spec_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "registered_before_first_result_run": True,
        },
        "population": {
            "source": "current config.load_watchlist()",
            "requested_symbols": len(symbols),
            "usable_symbols": len(histories),
            "coverage_ratio": round(coverage_ratio, 6),
            "errors": errors,
            "symbol_coverage": symbol_coverage,
        },
        "period": {
            "requested_start": _iso(start_ms),
            "requested_end_last_closed_boundary": _iso(end_ms),
            "actual_earliest_bar": min((_iso(int(data["t"][0])) for data in histories.values()), default=None),
            "actual_latest_bar": max((_iso(int(data["t"][-1])) for data in histories.values()), default=None),
            "timeframe": "1h",
        },
        "causal_contract": {
            "features_through": "closed candle i",
            "decision_time": "close of candle i",
            "entry_time": "open of candle i+1",
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "label_time_hours": [12, 24, 36],
            "useful_label": "net_ret24>=0.50% and net_mfe24>=1.00% and net_mae24>-1.00%",
            "same_bar_entry": False,
        },
        "split": {
            "method": "global chronological 60/20/20",
            "train_validation_cut": _iso(boundaries["train_validation_cut_ms"]),
            "validation_holdout_cut": _iso(boundaries["validation_holdout_cut_ms"]),
            "embargo_hours": EMBARGO_HOURS,
        },
        "profiles": {profile.name: asdict(profile) for profile in PROFILES},
        "selection": {
            "uses_split": "validation_only",
            "minimum_validation_labels": 100,
            "order": ["useful_precision", "mean_net_ret24", "sample_count", "name"],
            "selected_profile": selected,
        },
        "metrics": result_metrics,
        "broad_precursor": {
            split: metrics(
                base_parts[split],
                period_start_ms=_period_for_split(split, start_ms=start_ms, end_ms=end_ms, boundaries=boundaries)[0],
                period_end_ms=_period_for_split(split, start_ms=start_ms, end_ms=end_ms, boundaries=boundaries)[1],
            )
            for split in ("train", "validation", "holdout")
        },
        "trx_incident_diagnostic": {
            "window_start": _iso(incident_start),
            "deadline": _iso(incident_deadline),
            "window_end": _iso(incident_end),
            "not_used_for_variant_selection": True,
            "forward_label_used": False,
            "by_profile": incident_by_profile,
        },
        "acceptance": acceptance,
        "limitations": [
            "This is a market-wide proxy population, not the bot's actual historically logged candidate population (TH-06).",
            "The useful label is not immutable top-mover ground truth and is not realized portfolio PnL (TH-02/TH-11).",
            "Passing can authorize only a default-off shadow implementation, never BUY/WATCH or score-gate relaxation.",
        ],
    }


def _parse_symbols(value: str | None) -> list[str]:
    if value:
        raw = value.replace(",", " ").split()
    else:
        raw = config.load_watchlist()
    return sorted({str(symbol).strip().upper() for symbol in raw if str(symbol).strip()})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the pre-registered slow persistent trend hypothesis.")
    parser.add_argument("--symbols", help="comma/space-separated override; default is current watchlist")
    parser.add_argument("--start", default=DEFAULT_START.isoformat(), help="UTC maximum-period start")
    parser.add_argument("--end", help="UTC boundary; default is the last fully closed hour")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    start_ms = int(_parse_utc(args.start).timestamp() * 1000)
    end_ms = int(_parse_utc(args.end).timestamp() * 1000) if args.end else _closed_hour_ms()
    if start_ms >= end_ms:
        raise SystemExit("start must precede end")
    if not args.spec.exists():
        raise SystemExit(f"missing pre-registered spec: {args.spec}")
    print(
        f"loading maximum closed-1h history: symbols={len(symbols)} start={_iso(start_ms)} end={_iso(end_ms)}",
        flush=True,
    )
    histories, errors = asyncio.run(
        load_histories(
            symbols,
            start_ms=start_ms,
            end_ms=end_ms,
            cache_root=args.cache_dir,
            refresh=args.refresh,
            max_concurrency=max(1, args.max_concurrency),
        )
    )
    result = run_audit(
        histories,
        errors,
        symbols=symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        spec_path=args.spec.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = result["selection"]["selected_profile"]
    print(f"result={result['acceptance']['decision']} selected={selected} output={args.output}", flush=True)
    if selected:
        print(json.dumps(result["metrics"][selected], ensure_ascii=False, indent=2), flush=True)
    print("failed_checks=" + ",".join(result["acceptance"]["failed_checks"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
