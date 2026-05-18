from __future__ import annotations

from .entry_admission_dataset import V1ProjectedStructuralFeatures
from .state_reconstruction import FEATURE_NAMES, ReconstructionRow


def project_v1_structural_features(row: ReconstructionRow, *, today_change_pct: float) -> V1ProjectedStructuralFeatures:
    values = dict(zip(FEATURE_NAMES, row.features))
    slope = float(values["slope"])
    adx = float(values["adx"])
    rsi = float(values["rsi"])
    vol_x = float(values["vol_x"])
    daily_range = float(values["daily_range_pct"])
    price_vs_ema20 = float(values["price_vs_ema20"])
    forecast = _forecast_proxy_pct(
        today_change_pct=today_change_pct,
        slope=slope,
        adx=adx,
        vol_x=vol_x,
        rsi=rsi,
    )
    trend_score = _entry_signal_score(
        mode="trend",
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    impulse_score = _entry_signal_score(
        mode="impulse_speed",
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    leader_trend = _leader_score_trend(
        today_change_pct=today_change_pct,
        forecast_proxy_pct=forecast,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    return V1ProjectedStructuralFeatures(
        projected_today_change_pct=round(today_change_pct, 6),
        projected_forecast_proxy_pct=round(forecast, 6),
        projected_candidate_score_trend=round(trend_score, 6),
        projected_candidate_score_impulse_speed=round(impulse_score, 6),
        projected_leader_score_trend=round(leader_trend, 6),
        slope=round(slope, 6),
        adx=round(adx, 6),
        rsi=round(rsi, 6),
        vol_x=round(vol_x, 6),
        daily_range_pct=round(daily_range, 6),
        price_vs_ema20_pct=round(price_vs_ema20, 6),
    )


def _forecast_proxy_pct(*, today_change_pct: float, slope: float, adx: float, vol_x: float, rsi: float) -> float:
    upside = (
        max(0.0, today_change_pct) * 0.35
        + max(0.0, slope) * 1.8
        + max(0.0, adx - 18.0) * 0.06
        + max(0.0, vol_x - 1.0) * 0.45
    )
    downside = max(0.0, rsi - 70.0) * 0.20
    return max(-2.0, min(12.0, upside - downside))


def _entry_signal_score(*, mode: str, slope: float, adx: float, rsi: float, vol_x: float, daily_range: float) -> float:
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
    if mode == "impulse_speed":
        score += 0.3
    return score


def _leader_score_trend(
    *,
    today_change_pct: float,
    forecast_proxy_pct: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> float:
    score = 0.0
    score += max(0.0, min(today_change_pct, 15.0)) * 2.4
    score += max(0.0, forecast_proxy_pct) * 2.0
    score += max(0.0, slope) * 0.9
    score += max(0.0, adx - 18.0) * 0.10
    score += max(0.0, vol_x - 1.0) * 1.4
    score += 0.9
    if rsi > 73.0:
        score -= (rsi - 73.0) * 0.30
    if daily_range > 12.0:
        score -= (daily_range - 12.0) * 0.45
    return score
