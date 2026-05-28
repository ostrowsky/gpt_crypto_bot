from __future__ import annotations

"""
Intraday analysis logic.

Алгоритм для каждой монеты:
  1. Берём данные с начала сегодняшнего дня (00:00 UTC) + историю для прогрева индикаторов
  2. На сегодняшних барах находим все сигналы стратегии
  3. Для каждого сигнала у которого уже прошло T+3/5/10 свечей — проверяем был ли он верным
  4. Если точность сегодня ≥ порога → стратегия подтверждена для этой монеты сегодня
  5. Проверяем активен ли сигнал прямо сейчас (последняя закрытая свеча)

Если сегодня ещё мало данных (утро, мало сигналов) — помечаем как "не подтверждено"
и не добавляем в мониторинг.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np

import config
from runtime_executors import run_cpu
from indicators import compute_features, _ema
import botlog


# ── Market Regime ──────────────────────────────────────────────────────────────

class MarketRegime:
    """
    Определяет режим рынка по BTC и применяет адаптивные пороги.

    Режимы:
      bull_trend    — BTC > EMA20 > EMA50, ADX сильный → мягче RSI, range
      bear_trend    — BTC < EMA50, ADX сильный → только ретесты
      consolidation — ADX слабый → ждём пробоя, строже vol
      recovery      — BTC только что пробил EMA50 снизу → агрессивный вход
      neutral       — базовые параметры config

    Параметры хранятся в config.REGIME_PARAMS.
    """

    def __init__(self, regime_name: str = "neutral"):
        self.name = regime_name
        params = getattr(config, "REGIME_PARAMS", {})
        self._p = params.get(regime_name, params.get("neutral", {}))

    def get(self, key: str, fallback):
        """Возвращает параметр режима или fallback из config."""
        val = self._p.get(key)
        return fallback if val is None else val

    @property
    def rsi_hi(self)        -> float: return self.get("rsi_hi",    config.RSI_BUY_HI)
    @property
    def vol_mult(self)      -> float: return self.get("vol_mult",  config.VOL_MULT)
    @property
    def range_max(self)     -> float: return self.get("range_max", config.DAILY_RANGE_MAX)
    @property
    def adx_min(self)       -> float: return self.get("adx_min",   config.ADX_MIN)
    @property
    def slope_min(self)     -> float: return self.get("slope_min", config.EMA_SLOPE_MIN)
    @property
    def allow_new_buy(self) -> bool:  return self.name != "bear_trend"

    def __str__(self) -> str:
        icons = {
            "bull_trend":    "🐂",
            "bear_trend":    "🐻",
            "consolidation": "↔️",
            "recovery":      "🌱",
            "neutral":       "➡️",
        }
        return f"{icons.get(self.name, '?')} {self.name}"


def _get_coin_regime(feat: dict, i: int) -> MarketRegime:
    """Режим конкретной монеты по её собственным индикаторам."""
    regime_arr = feat.get("regime")
    if regime_arr is not None and i < len(regime_arr):
        name = str(regime_arr[i])
        return MarketRegime(name)
    return MarketRegime("neutral")


def _get_effective_range_max(feat: dict, i: int, regime: "MarketRegime") -> float:
    """
    Итоговый порог daily_range_max с учётом:
    1. Динамического (по волатильности монеты) — если DYNAMIC_RANGE_ENABLED
    2. Режима рынка
    3. Бычьего дня (_effective_range_max от market_scan)
    Берём наибольший из применимых порогов.
    """
    base = getattr(config, "_effective_range_max", config.DAILY_RANGE_MAX)

    # Динамический порог
    if getattr(config, "DYNAMIC_RANGE_ENABLED", True):
        dyn_arr = feat.get("dyn_range_max")
        if dyn_arr is not None and i < len(dyn_arr) and np.isfinite(dyn_arr[i]):
            dyn = float(dyn_arr[i])
            base = max(base, dyn)

    # Режим рынка
    return max(base, regime.range_max)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class HorizonAccuracy:
    horizon: int
    total:   int
    correct: int

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"T+{self.horizon}: {self.pct:.0f}% ({self.correct}/{self.total})"


@dataclass
class CoinReport:
    symbol:        str
    tf:            str

    # Сегодняшние данные форвард-теста
    today_signals:  int                        # сигналов найдено сегодня
    today_accuracy: Dict[int, HorizonAccuracy] # точность на сегодняшних данных
    today_confirmed: bool                      # стратегия подтверждена сегодня

    best_horizon:  int
    best_accuracy: float
    in_play:       bool

    note:          str   = ""
    from_scan:     bool  = False

    # Текущий статус сигнала (последняя закрытая свеча)
    signal_now:    bool  = False
    current_price: float = 0.0
    current_slope: float = 0.0
    current_rsi:   float = 0.0
    current_adx:   float = 0.0
    current_vol_x: float = 0.0
    current_macd:  float = 0.0

    # Почему сигнала нет прямо сейчас (для диагностики)
    no_signal_reason: str = ""

    # SETUP: структура бычья, но не хватает 1 жёсткого фильтра
    setup_now:          bool = False
    setup_reason:       str  = ""  # что именно не дотягивает до BUY
    setup_missing_count: int = 99  # П8: кол-во недостающих условий (меньше = ближе к BUY)

    # П7: тип активного сигнала: "trend"/"strong_trend"/"retest"/"breakout"
    signal_mode:  str  = ""
    wakeup_shadow: bool = False
    wakeup_ts: int = 0
    wakeup_priority_bonus: float = 0.0

    def summary(self) -> str:
        scan  = " 🔍" if self.from_scan else ""

        # Заголовок с точностью сегодня
        acc_parts = "  ".join(
            str(self.today_accuracy[h])
            for h in config.FORWARD_BARS
            if h in self.today_accuracy
        )

        if self.today_confirmed:
            conf_icon = "✅"
            conf_note = f"Подтверждено сегодня: {acc_parts}"
        else:
            conf_icon = "⚠️"
            conf_note = f"Мало данных сегодня ({self.today_signals} сигн.) — {acc_parts or 'нет оценки'}"

        # П7: тип активного сигнала
        _mode_label = {
            "trend":          "📈 BUY",
            "strong_trend":   "💪 BUY strong",
            "impulse_speed":  "⚡ Быстрое движение",
            "retest":         "🔄 RETEST",
            "breakout":       "⚡ BREAKOUT",
            "impulse_cross":  "🚀 IMPULSE cross",
            "alignment":      "🌊 ALIGNMENT",
        }.get(self.signal_mode, "📈 BUY")

        # Текущий статус
        if self.signal_now:
            now_line = (
                f"   🟢 *{_mode_label}*  "
                f"`{self.current_price:.6g}`  "
                f"slope:`{self.current_slope:+.2f}%`  "
                f"RSI:`{self.current_rsi:.1f}`  "
                f"ADX:`{self.current_adx:.1f}`  "
                f"vol×:`{self.current_vol_x:.2f}`"
            )
        else:
            _r = (self.no_signal_reason
                  .replace("_",  r"\_")
                  .replace("[",  r"\[")
                  .replace("]",  r"\]")
                  .replace("(",  r"\(")
                  .replace(")",  r"\)"))
            now_line = f"   ⏸ Сигнала нет сейчас: {_r}"

        return (
            f"{conf_icon} {self.symbol}{scan} [{self.tf}]\n"
            f"   {conf_note}\n"
            f"{now_line}"
        )


# ── Binance fetch ──────────────────────────────────────────────────────────────

async def fetch_klines(
    session:  aiohttp.ClientSession,
    symbol:   str,
    interval: str,
    limit:    int = config.HISTORY_LIMIT,
) -> Optional[np.ndarray]:
    url = f"{config.BINANCE_REST}/api/v3/klines"
    try:
        async with session.get(
            url,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            r.raise_for_status()
            js = await r.json()
    except Exception:
        return None

    if not isinstance(js, list) or len(js) < 60:
        return None

    arr = np.zeros(len(js), dtype=[
        ("t", "i8"), ("o", "f8"), ("h", "f8"),
        ("l", "f8"), ("c", "f8"), ("v", "f8"),
    ])
    arr["t"] = [int(x[0])   for x in js]
    arr["o"] = [float(x[1]) for x in js]
    arr["h"] = [float(x[2]) for x in js]
    arr["l"] = [float(x[3]) for x in js]
    arr["c"] = [float(x[4]) for x in js]
    arr["v"] = [float(x[5]) for x in js]
    return arr


async def fetch_top_symbols(session: aiohttp.ClientSession) -> List[str]:
    url = f"{config.BINANCE_REST}/api/v3/ticker/24hr"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            data = await r.json()
    except Exception:
        return []

    rows: List[Tuple[float, str]] = []
    for it in data:
        sym = it.get("symbol", "")
        if not sym.endswith(config.SCAN_QUOTE):
            continue
        if any(x in sym for x in config.SCAN_EXCLUDE):
            continue
        try:
            qv = float(it.get("quoteVolume", 0))
        except Exception:
            qv = 0.0
        rows.append((qv, sym))

    rows.sort(reverse=True)
    return [s for _, s in rows[:config.SCAN_TOP_N]]


# ── Signal detection ───────────────────────────────────────────────────────────

def check_entry_conditions(
    feat: Dict, i: int, c: np.ndarray,
    tf: str = "",
    regime: "MarketRegime" = None,
) -> Tuple[bool, str]:
    """
    Проверяет все условия входа.
    v2: адаптивные пороги через MarketRegime + slope acceleration + squeeze bypass.
    """
    _ = tf
    if regime is None:
        regime = _get_coin_regime(feat, i)

    ef  = feat["ema_fast"][i]
    es  = feat["ema_slow"][i]
    rsi = feat["rsi"][i]
    adx = feat["adx"][i]
    slp = feat["slope"][i]
    vx  = feat["vol_x"][i]

    if not all(np.isfinite([ef, es, rsi, adx, slp, vx])):
        return False, "нет данных индикаторов"

    # Медвежий режим: запрет новых BUY — проверяем в первую очередь
    if not regime.allow_new_buy:
        return False, f"режим {regime.name} — новые BUY запрещены"

    if not (float(c[i]) > ef > es):
        return False, f"цена {float(c[i]):.6g} не выше EMA20 {ef:.6g} > EMA50 {es:.6g}"

    # Slope: режимный порог, снижается при наличии slope acceleration
    slope_min = regime.slope_min
    accel_arr = feat.get("slope_accel")
    has_accel = (
        accel_arr is not None and i < len(accel_arr) and
        np.isfinite(accel_arr[i]) and
        accel_arr[i] >= getattr(config, "SLOPE_ACCEL_MIN", 0.05)
    )
    if has_accel:
        slope_min = slope_min * 0.7  # slope acceleration снижает порог на 30%
    if slp < slope_min:
        return False, f"наклон EMA20 {slp:+.2f}% < {slope_min:.2f}%"

    # ADX: режимный порог, снижается при squeeze breakout
    adx_min = regime.adx_min
    sq_arr = feat.get("squeeze_breakout")
    squeeze_active = (
        sq_arr is not None and i < len(sq_arr) and sq_arr[i] == 1.0
    )
    if squeeze_active:
        adx_min = max(adx_min * 0.7, 15.0)
    if adx < adx_min:
        return False, f"ADX {adx:.1f} < {adx_min:.1f}"

    adx_sma = feat["adx_sma"][i]
    if np.isfinite(adx_sma) and adx <= adx_sma:
        bypass_threshold = getattr(config, "ADX_SMA_BYPASS", 35.0)
        if adx < bypass_threshold and not squeeze_active:
            return False, f"ADX {adx:.1f} ≤ SMA(ADX,10) {adx_sma:.1f} — тренд слабый"

    # Volume: режимный порог, снижается при squeeze
    vol_min = regime.vol_mult
    if squeeze_active:
        vol_min = vol_min * 0.85
    if vx < vol_min:
        return False, f"объём {vx:.2f}× < {vol_min:.2f}×"

    # RSI: адаптивная верхняя граница (режим + strong_trend + squeeze)
    strong_trend = (
        np.isfinite(adx) and adx >= config.STRONG_ADX_MIN and
        np.isfinite(vx)  and vx  >= config.STRONG_VOL_MIN
    )
    rsi_hi = config.RSI_BUY_HI_STRONG if strong_trend else regime.rsi_hi
    if squeeze_active:
        rsi_hi = min(rsi_hi + 3, 85.0)
    if not (config.RSI_BUY_LO <= rsi <= rsi_hi):
        mode = " [сильный тренд]" if strong_trend else ""
        sq_note = " [squeeze]" if squeeze_active else ""
        return False, f"RSI {rsi:.1f} вне зоны [{config.RSI_BUY_LO}-{rsi_hi:.0f}]{mode}{sq_note}"

    macd_h = feat["macd_hist"][i]
    if np.isfinite(macd_h) and macd_h < 0:
        return False, f"MACD гистограмма отрицательная ({macd_h:.6g})"

    # Range max: динамический + режим
    _range_max = _get_effective_range_max(feat, i, regime)
    dr_pct = feat["daily_range_pct"][i]
    if np.isfinite(dr_pct) and dr_pct > _range_max:
        return False, f"монета уже +{dr_pct:.1f}% от дна дня (> {_range_max:.1f}%)"

    if not regime.allow_new_buy:
        return False, f"режим {regime.name} — новые BUY запрещены"

    return True, ""


def check_setup_conditions(
    feat: Dict, i: int, c: np.ndarray
) -> Tuple[bool, str, int]:
    """
    Мягкие условия "зарождающегося тренда".
    Срабатывает когда структура бычья но не хватает 1 жёсткого фильтра BUY.
    НЕ используется для форвард-теста — только для UI-секции 🟡.
    Возвращает (ok, reason, missing_count) — П8: missing_count для сортировки.
    """
    ef  = feat["ema_fast"][i]
    es  = feat["ema_slow"][i]
    rsi = feat["rsi"][i]
    slp = feat["slope"][i]
    vx  = feat["vol_x"][i]
    macd_h  = feat["macd_hist"][i]
    dr_pct  = feat["daily_range_pct"][i]
    adx     = feat["adx"][i]

    if not all(np.isfinite([ef, rsi, slp])):
        return False, "нет данных", 99

    price = float(c[i])

    # Базовые условия — обязательны даже для SETUP
    if not (price > ef):
        return False, f"цена {price:.6g} ниже EMA20 {ef:.6g}", 99
    if slp <= 0:
        return False, f"EMA20 не растёт (slope {slp:+.2f}%)", 99
    if np.isfinite(rsi) and rsi < 50:
        return False, f"RSI {rsi:.1f} < 50", 99
    if np.isfinite(macd_h) and macd_h < 0:
        return False, f"MACD отрицательный ({macd_h:.6g})", 99
    _range_max = getattr(config, "_effective_range_max", config.DAILY_RANGE_MAX)
    if np.isfinite(dr_pct) and dr_pct > _range_max:
        return False, f"перегрета +{dr_pct:.1f}% от дна (> {_range_max}%)", 99
    if np.isfinite(adx) and adx < 15:
        return False, f"ADX {adx:.1f} < 15 — нет тренда", 99
    if np.isfinite(vx) and vx < 0.8:
        return False, f"объём {vx:.2f}× — слишком слабый", 99

    # П8: считаем сколько условий не хватает до BUY
    missing = []
    if np.isfinite(es) and ef <= es:
        missing.append(f"EMA20 {ef:.4g} ≤ EMA50 {es:.4g}")
    if slp < config.EMA_SLOPE_MIN:
        missing.append(f"slope {slp:+.2f}% < {config.EMA_SLOPE_MIN}%")
    if np.isfinite(vx) and vx < config.VOL_MULT:
        missing.append(f"vol× {vx:.2f} < {config.VOL_MULT}")
    if np.isfinite(adx) and adx < config.ADX_MIN:
        missing.append(f"ADX {adx:.1f} < {config.ADX_MIN}")
    _strong = (
        np.isfinite(adx) and adx >= config.STRONG_ADX_MIN and
        np.isfinite(vx) and vx >= config.STRONG_VOL_MIN
    )
    _rsi_hi = config.RSI_BUY_HI_STRONG if _strong else config.RSI_BUY_HI
    if np.isfinite(rsi) and not (config.RSI_BUY_LO <= rsi <= _rsi_hi):
        missing.append(f"RSI {rsi:.1f} вне [{config.RSI_BUY_LO}-{_rsi_hi}]")

    reason = "не хватает: " + ", ".join(missing) if missing else "почти BUY"
    return True, reason, len(missing)


def check_exit_conditions(
    feat: Dict,
    i: int,
    c: np.ndarray,
    *,
    mode: str | None = None,
    bars_elapsed: int | None = None,
    tf: str | None = None,
) -> Optional[str]:
    """
    Проверяет условия выхода из позиции.
    v2: добавлены RSI дивергенция, volume exhaustion, EMA fan collapse.

    Логика ужесточения трейлинг-стопа при обнаружении слабости:
    Эти сигналы не вызывают немедленный выход, но помечают слабость
    через возвращение строки с prefix "⚠️ WEAK:" — мониторинг может
    ужесточить ATR_TRAIL_K до ATR_TRAIL_K * RSI_DIV_TRAIL_MULT.
    """
    close = float(c[i])
    ef    = feat["ema_fast"][i]
    rsi   = feat["rsi"][i]
    adx   = feat["adx"][i]
    slp   = feat["slope"][i]

    # ── Жёсткие условия выхода ────────────────────────────────────────────────

    # П6: 2 закрытия подряд ниже EMA20 → ранний разворот
    if i >= 1 and np.isfinite(ef):
        prev_ef    = feat["ema_fast"][i - 1]
        prev_close = float(c[i - 1])
        if np.isfinite(prev_ef) and prev_close < prev_ef and close < ef:
            return f"2 закрытия подряд ниже EMA20 ({ef:.6g}) — ранний разворот"

    # Одиночное закрытие ниже EMA20
    if np.isfinite(ef) and close < ef:
        return f"Цена ниже EMA20 ({ef:.6g})"

    if np.isfinite(rsi) and rsi > config.RSI_OVERBOUGHT:
        return f"RSI перекуплен ({rsi:.1f})"
    if np.isfinite(slp) and slp < 0:
        return f"EMA20 разворачивается вниз (slope {slp:+.2f}%)"
    j = i - config.ADX_GROW_BARS
    if j >= 0 and np.isfinite(adx) and np.isfinite(feat["adx"][j]):
        if adx < feat["adx"][j] * config.ADX_DROP_RATIO:
            return f"ADX ослабевает ({adx:.1f} ← {feat['adx'][j]:.1f})"

    # ── Ранние сигналы слабости (не выход, но ужесточение стопа) ─────────────
    # Возвращаем строку с префиксом "⚠️ WEAK:" — монитор реагирует на это.

    # v2.A: RSI дивергенция — цена выше, RSI нет
    rsi_div_arr = feat.get("rsi_divergence")
    if rsi_div_arr is not None and i < len(rsi_div_arr) and rsi_div_arr[i] == 1.0:
        return f"⚠️ WEAK: RSI дивергенция — momentum ослабевает (стоп ужесточён)"

    # v2.B: Volume exhaustion — объём убывает при росте цены
    vol_ex_arr = feat.get("vol_exhaustion")
    if vol_ex_arr is not None and i < len(vol_ex_arr) and vol_ex_arr[i] == 1.0:
        return f"⚠️ WEAK: объёмное истощение — покупатели заканчиваются"

    # v2.C: EMA Fan Collapse — веер EMA сужается
    fan_arr = feat.get("ema_fan_spread")
    fan_threshold = getattr(config, "EMA_FAN_DECAY_THRESHOLD", 0.30)
    if fan_arr is not None and i < len(fan_arr) and np.isfinite(fan_arr[i]):
        if float(fan_arr[i]) >= fan_threshold:
            decay_pct = float(fan_arr[i]) * 100
            return f"⚠️ WEAK: EMA-веер сузился на {decay_pct:.0f}% — тренд слабеет"

    return None


# ── Entry mode helper ──────────────────────────────────────────────────────────

def get_entry_mode(feat: Dict, i: int) -> str:
    """
    П1: режим входа — выбирает ATR_TRAIL_K и MAX_HOLD_BARS.
      'strong_trend' → ATR_TRAIL_K_STRONG (шире, держим дольше)
      'trend'        → ATR_TRAIL_K (стандартный)

    Улучшение: считаем также скорость роста цены за 3 бара.
    Если за 3 бара цена выросла ≥ 1.5% — это импульс даже при низком ADX
    (ADX лагует 10+ баров, не успевает подтвердить быстрое движение).
    """
    adx = feat["adx"][i]
    vx  = feat["vol_x"][i]

    # Скорость за 3 бара из feat["close"]
    price_speed = 0.0
    c_arr = feat.get("close")
    if c_arr is not None and i >= 3:
        c0 = float(c_arr[i - 3])
        ci = float(c_arr[i])
        if c0 > 0:
            price_speed = (ci - c0) / c0 * 100.0

    # ADX + vol ≥ порогов → настоящий сильный тренд
    if (np.isfinite(adx) and adx >= config.STRONG_ADX_MIN
            and np.isfinite(vx) and vx >= config.STRONG_VOL_MIN):
        return "strong_trend"
    # Быстрое ценовое движение при слабом ADX (ADX лагует ~10 баров после импульса).
    # Стоп такой же широкий как у strong_trend, но метка честная — не «сильный тренд».
    if price_speed >= 1.5:
        return "impulse_speed"
    return "trend"


def _early_15m_continuation_entry_ok(feat: Dict, i: int, c: np.ndarray, tf: str = "") -> bool:
    if tf != "15m":
        return False
    if i <= 0:
        return False
    entry_now, _ = check_entry_conditions(feat, i, c)
    if not entry_now:
        return False
    entry_prev, _ = check_entry_conditions(feat, i - 1, c)
    return entry_prev


def get_effective_entry_mode(
    feat: Dict, i: int, c: np.ndarray, tf: str = ""
) -> tuple[str, bool]:
    mode = get_entry_mode(feat, i)
    early_15m_continuation = _early_15m_continuation_entry_ok(feat, i, c, tf=tf)
    return mode, early_15m_continuation





# ── П5: Bull Day detector ───────────────────────────────────────────────────────

async def is_bull_day(session: aiohttp.ClientSession) -> tuple:
    """
    П5: бычий день = BTC выше EMA50 на 1h и EMA50 наклонена вверх.
    Возвращает (bool, btc_price, btc_ema50) для логирования.
    """
    try:
        data = await fetch_klines(session, "BTCUSDT", "1h", limit=60)
        if data is None or len(data) < 55:
            return False, 0.0, 0.0
        c_btc = data["c"].astype(float)
        ema50 = _ema(c_btc, 50)
        if ema50[-6] > 0:
            slope = (ema50[-1] - ema50[-6]) / ema50[-6] * 100
        else:
            slope = 0.0
        btc_price = float(c_btc[-1])
        btc_ema50 = float(ema50[-1])
        is_bull   = btc_price > btc_ema50 and slope > 0
        return is_bull, btc_price, btc_ema50
    except Exception:
        return False, 0.0, 0.0


async def detect_market_regime(session: aiohttp.ClientSession) -> MarketRegime:
    """
    Определяет глобальный режим рынка по BTC 1h.
    Использует данные за последние 60 часов для расчёта ADX и EMA.

    Возвращает MarketRegime с именем режима и адаптивными параметрами.
    """
    try:
        data = await fetch_klines(session, "BTCUSDT", "1h", limit=100)
        if data is None or len(data) < 60:
            return MarketRegime("neutral")

        from indicators import (
            _ema as ema_fn, _adx as adx_fn, _ema_slope as slope_fn
        )

        c = data["c"].astype(float)
        h = data["h"].astype(float)
        l = data["l"].astype(float)

        ema20  = ema_fn(c, 20)
        ema50  = ema_fn(c, 50)
        adx_v  = adx_fn(h, l, c, 14)
        slope  = slope_fn(ema20, 5)

        i = len(c) - 2  # последний закрытый бар
        cur_c   = float(c[i])
        cur_ef  = float(ema20[i])
        cur_es  = float(ema50[i])
        cur_adx = float(adx_v[i]) if np.isfinite(adx_v[i]) else 0.0
        cur_slp = float(slope[i]) if np.isfinite(slope[i]) else 0.0

        adx_trend = getattr(config, "REGIME_BTC_ADX_TREND", 22.0)
        adx_flat  = getattr(config, "REGIME_BTC_ADX_FLAT",  18.0)

        # Проверяем был ли BTC под EMA50 предыдущий бар (recovery детектор)
        prev_c  = float(c[i - 1]) if i >= 1 else cur_c
        prev_es = float(ema50[i - 1]) if i >= 1 else cur_es
        was_below = prev_c < prev_es

        if cur_adx >= adx_trend:
            if cur_c > cur_ef > cur_es:
                name = "recovery" if was_below else "bull_trend"
            elif cur_c < cur_es:
                name = "bear_trend"
            else:
                name = "neutral"
        elif cur_adx < adx_flat:
            name = "consolidation"
        else:
            if cur_c > cur_ef and cur_slp > 0:
                name = "bull_trend"
            elif cur_c < cur_es:
                name = "bear_trend"
            else:
                name = "consolidation"

        return MarketRegime(name)

    except Exception:
        return MarketRegime("neutral")


# ── RETEST: откат к EMA20 в существующем тренде ────────────────────────────────

def check_retest_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    RETEST — вход не в начало тренда а в его продолжение после отката к EMA20.
    Условия:
      1. Тренд существовал RETEST_LOOKBACK баров назад (close > EMA20)
      2. В последние RETEST_TOUCH_BARS баров: low касалось EMA20 (откат)
      3. Текущий бар: close > EMA20 и close > prev close (отскок подтверждён)
      4. EMA20 slope > 0 (тренд не сломан)
      5. RSI < RETEST_RSI_MAX (не перегрет)
      6. ADX > 20 (тренд существует)
      7. vol_x > RETEST_VOL_MIN (объём необязательно высокий)
    """
    lb      = getattr(config, "RETEST_LOOKBACK",   12)
    tb      = getattr(config, "RETEST_TOUCH_BARS",  5)
    rsi_mx  = getattr(config, "RETEST_RSI_MAX",    65.0)
    vol_mn  = getattr(config, "RETEST_VOL_MIN",     0.8)

    if i < lb + 2:
        return False, "недостаточно баров"

    c_arr    = feat.get("close")
    lo       = feat.get("low")
    ema_fast = feat["ema_fast"]
    slope    = feat["slope"]
    rsi      = feat["rsi"]
    adx      = feat["adx"]
    vol_x    = feat["vol_x"]

    if c_arr is None or lo is None:
        return False, "нет ценовых рядов в feat"

    # 1. Тренд существовал lb баров назад
    lb_idx = i - lb
    if not (np.isfinite(float(c_arr[lb_idx])) and np.isfinite(float(ema_fast[lb_idx]))):
        return False, "нет данных для проверки тренда"
    if float(c_arr[lb_idx]) <= float(ema_fast[lb_idx]):
        return False, f"тренда не было {lb} баров назад (close ≤ EMA20)"

    # 2. Откат: low касалось EMA20 в последние tb баров (исключая текущий)
    touched = False
    for k in range(1, tb + 1):
        ki = i - k
        if ki < 0:
            break
        if not (np.isfinite(float(lo[ki])) and np.isfinite(float(ema_fast[ki]))):
            continue
        # Касание: low <= EMA20 × 1.005 (допуск 0.5%)
        if float(lo[ki]) <= float(ema_fast[ki]) * 1.005:
            touched = True
            break
    if not touched:
        return False, f"нет касания EMA20 за {tb} баров"

    # 3. Отскок: текущий бар выше EMA20 и выше предыдущего закрытия
    if not (np.isfinite(float(c_arr[i])) and np.isfinite(float(ema_fast[i]))):
        return False, "нет данных текущего бара"
    if float(c_arr[i]) <= float(ema_fast[i]):
        return False, f"close {float(c_arr[i]):.6g} ≤ EMA20 {float(ema_fast[i]):.6g}"
    if float(c_arr[i]) <= float(c_arr[i - 1]):
        return False, "нет отскока (close не выше предыдущего бара)"

    # 4. Slope EMA20 > 0
    if not np.isfinite(float(slope[i])) or float(slope[i]) <= 0:
        return False, f"slope EMA20 {float(slope[i]):.2f}% ≤ 0"

    # 5. RSI не перегрет
    if not np.isfinite(float(rsi[i])) or float(rsi[i]) >= rsi_mx:
        return False, f"RSI {float(rsi[i]):.1f} ≥ {rsi_mx} (перегрет для ретеста)"

    # 6. ADX подтверждает существование тренда
    if not np.isfinite(float(adx[i])) or float(adx[i]) < 20:
        return False, f"ADX {float(adx[i]):.1f} < 20"

    # 7. Минимальный объём
    if not np.isfinite(float(vol_x[i])) or float(vol_x[i]) < vol_mn:
        return False, f"vol× {float(vol_x[i]):.2f} < {vol_mn}"

    return True, ""


# ── BREAKOUT: пробой флэта с объёмом ───────────────────────────────────────────

def check_breakout_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    BREAKOUT — пробой диапазона флэта с высоким объёмом.
    Условия:
      1. Последние BREAKOUT_FLAT_BARS баров: диапазон (max-min)/min < BREAKOUT_FLAT_MAX_PCT
      2. Текущий close > max(high флэта) — реальный пробой уровня
      3. vol_x ≥ BREAKOUT_VOL_MIN (сильный объём подтверждает пробой)
      4. MACD_hist > 0 и вырос vs предыдущий бар (импульс)
      5. daily_range < BREAKOUT_RANGE_MAX (движение только началось, не поздно)
      6. RSI < 75
      7. close > EMA20 (минимальная бычья структура)
    """
    flat_bars = getattr(config, "BREAKOUT_FLAT_BARS",    8)
    flat_pct  = getattr(config, "BREAKOUT_FLAT_MAX_PCT", 2.0)
    vol_mn    = getattr(config, "BREAKOUT_VOL_MIN",      2.0)
    rng_mx    = getattr(config, "BREAKOUT_RANGE_MAX",    4.0)

    if i < flat_bars + 2:
        return False, "недостаточно баров"

    c_arr      = feat.get("close")
    hi         = feat.get("high")
    lo         = feat.get("low")
    ema_fast   = feat["ema_fast"]
    vol_x      = feat["vol_x"]
    rsi        = feat["rsi"]
    macd_hist  = feat["macd_hist"]
    daily_rng  = feat["daily_range_pct"]

    if c_arr is None or hi is None or lo is None:
        return False, "нет ценовых рядов в feat"

    # 1. Флэт: диапазон последних flat_bars баров (не включая текущий)
    flat_hi = float(np.max(hi[i - flat_bars: i]))
    flat_lo = float(np.min(lo[i - flat_bars: i]))
    if flat_lo <= 0:
        return False, "некорректные данные low"
    flat_rng = (flat_hi - flat_lo) / flat_lo * 100
    if flat_rng > flat_pct:
        return False, f"нет флэта: диапазон {flat_rng:.1f}% > {flat_pct}%"

    # 2. Пробой: текущий close выше максимума флэта
    ci = float(c_arr[i])
    if ci <= flat_hi:
        return False, f"нет пробоя: close {ci:.6g} ≤ max флэта {flat_hi:.6g}"

    # 3. Объём
    vx = float(vol_x[i])
    if not np.isfinite(vx) or vx < vol_mn:
        return False, f"vol× {vx:.2f} < {vol_mn} (нужен сильный объём на пробое)"

    # 4. MACD растёт
    mh = float(macd_hist[i])
    mh_prev = float(macd_hist[i - 1]) if i > 0 else float("nan")
    if not (np.isfinite(mh) and mh > 0 and np.isfinite(mh_prev) and mh > mh_prev):
        return False, "MACD_hist ≤ 0 или не растёт"

    # 5. Движение только началось
    dr = float(daily_rng[i])
    if np.isfinite(dr) and dr > rng_mx:
        return False, f"daily_range {dr:.1f}% > {rng_mx}% (поздно входить)"

    # 6. RSI
    ri = float(rsi[i])
    if not np.isfinite(ri) or ri >= 75:
        return False, f"RSI {ri:.1f} ≥ 75"

    # 7. Структура
    ef = float(ema_fast[i])
    if not np.isfinite(ef) or ci <= ef:
        return False, f"close {ci:.6g} ≤ EMA20 {ef:.6g}"

    return True, ""


def check_impulse_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    IMPULSE — детектор самого начала тренда, на 1-2 бара раньше BUY/STRONG_TREND.

    ADX лагует 10+ баров и не успевает подтвердить быстрый импульс.
    IMPULSE обходит ADX, вместо этого смотрит на скорость ценового движения.

    Откалиброван по данным 04.03.2026:
      ETH 15:15 — r1=+2.37% r3=+3.50% RSI=78.8 vol×2.63 → поймал за 1 бар до BUY
      SOL 15:15 — r1=+2.11% r3=+3.27% RSI=76.4 vol×2.35
      XRP 15:15 — r1=+2.05% r3=+2.69% RSI=79.5 vol×3.77

    Условия:
      1. close > EMA20 (бычья структура)
      2. EMA20 slope > 0 (направление правильное)
      3. r1 >= IMPULSE_R1_MIN (текущий бар вырос %)
      4. r3 >= IMPULSE_R3_MIN (за 3 бара выросло %)
      5. body >= IMPULSE_BODY_MIN (реальное тело, не wick)
      6. vol_x >= IMPULSE_VOL_MIN (объём подтверждает)
      7. RSI в [IMPULSE_RSI_LO, IMPULSE_RSI_HI]
      8. daily_range <= effective_range_max (не слишком поздно)
    """
    if i < 5:
        return False, "недостаточно баров"

    c_arr     = feat.get("close")
    o_arr     = feat.get("open")
    ema_fast  = feat["ema_fast"]
    slope     = feat["slope"]
    rsi_arr   = feat["rsi"]
    vol_x_arr = feat["vol_x"]
    dr_arr    = feat["daily_range_pct"]

    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    ef  = float(ema_fast[i])
    slp = float(slope[i])
    ri  = float(rsi_arr[i])
    vx  = float(vol_x_arr[i])
    dr  = float(dr_arr[i])

    if not np.isfinite(ci) or not np.isfinite(ef):
        return False, "нет данных"

    # 1. Цена выше EMA20
    if ci <= ef:
        return False, f"close {ci:.4g} ≤ EMA20 {ef:.4g}"

    # 2. EMA20 растёт
    if not np.isfinite(slp) or slp <= 0:
        return False, f"slope EMA20 {slp:+.3f}% ≤ 0"

    # 3. r1 — рост текущего бара
    prev = float(c_arr[i - 1])
    if prev <= 0:
        return False, "нет предыдущего закрытия"
    r1 = (ci / prev - 1) * 100
    if r1 < config.IMPULSE_R1_MIN:
        return False, f"r1 {r1:+.2f}% < {config.IMPULSE_R1_MIN}%"

    # 4. r3 — рост за 3 бара
    if i < 3:
        return False, "мало баров для r3"
    base3 = float(c_arr[i - 3])
    if base3 <= 0:
        return False, "нет базы для r3"
    r3 = (ci / base3 - 1) * 100
    if r3 < config.IMPULSE_R3_MIN:
        return False, f"r3 {r3:+.2f}% < {config.IMPULSE_R3_MIN}% (импульс слабый)"

    # 5. Тело свечи (если есть данные open)
    if o_arr is not None:
        oi = float(o_arr[i])
        if oi > 0:
            body = (ci - oi) / oi * 100
            if body < config.IMPULSE_BODY_MIN:
                return False, f"тело свечи {body:+.2f}% < {config.IMPULSE_BODY_MIN}% (wick)"

    # 6. Объём
    if not np.isfinite(vx) or vx < config.IMPULSE_VOL_MIN:
        return False, f"vol× {vx:.2f} < {config.IMPULSE_VOL_MIN}"

    # 7. RSI
    if not np.isfinite(ri) or not (config.IMPULSE_RSI_LO <= ri <= config.IMPULSE_RSI_HI):
        return False, f"RSI {ri:.1f} вне [{config.IMPULSE_RSI_LO:.0f}–{config.IMPULSE_RSI_HI:.0f}]"

    # 8. daily_range — не слишком поздно
    _range_max = getattr(config, "_effective_range_max", config.DAILY_RANGE_MAX)
    if np.isfinite(dr) and dr > _range_max:
        return False, f"daily_range {dr:.1f}% > {_range_max}% (поздно)"

    return True, f"r1={r1:+.2f}% r3={r3:+.2f}% vol×{vx:.1f} RSI={ri:.0f}"


# ── ALIGNMENT: плавный бычий тренд, ADX не требуется ─────────────────────────

def check_alignment_conditions(feat: Dict, i: int, tf: str = "") -> Tuple[bool, str]:
    _ = tf
    """
    ALIGNMENT — устойчивый бычий тренд без требования к ADX и скорости.

    Назначение: ловить медленные альт-тренды где ADX лагует 28+ баров (7ч на 15m)
    и не успевает подтвердить, но структура чётко и устойчиво бычья.

    Пример: CHZ 08.03.2026 09:00-18:00 UTC — цена +8% за 9 часов по +0.1-0.3% за свечу.
    IMPULSE требовал r1>=1.5% — ни один бар не дотянул.
    ALIGNMENT поймал бы с первых баров выше EMA20.

    Условия:
      1. close > EMA20 > EMA50 (полная бычья структура)
      2. EMA20 slope >= ALIGNMENT_SLOPE_MIN (тренд растёт, пусть и плавно)
      3. MACD hist > 0 последние ALIGNMENT_MACD_BARS баров подряд
         (устойчивый импульс, не случайный всплеск)
      4. RSI в [ALIGNMENT_RSI_LO, ALIGNMENT_RSI_HI] (не перегрет, не слаб)
      5. vol_x > ALIGNMENT_VOL_MIN (минимальная активность)
      6. daily_range < ALIGNMENT_RANGE_MAX (не слишком поздно)

    ADX не проверяется — это принципиальное отличие от BUY.
    """
    if i < 5:\
        return False, "недостаточно баров"

    ef       = feat["ema_fast"]
    es       = feat["ema_slow"]
    slope    = feat["slope"]
    rsi_arr  = feat["rsi"]
    vol_x    = feat["vol_x"]
    mh_arr   = feat["macd_hist"]
    dr_arr   = feat["daily_range_pct"]
    c_arr    = feat.get("close")

    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    efv = float(ef[i])
    esv = float(es[i])
    slp = float(slope[i])
    ri  = float(rsi_arr[i])
    vx  = float(vol_x[i])
    dr  = float(dr_arr[i])

    if not all(np.isfinite([ci, efv, esv, slp, ri, vx])):
        return False, "нет данных индикаторов"

    # 1. Полная бычья структура: цена > EMA20 > EMA50
    if not (ci > efv > esv):
        return False, f"цена/EMA структура нарушена"

    # 2. EMA20 растёт (мягкий порог)
    slope_min = getattr(config, "ALIGNMENT_SLOPE_MIN", 0.05)
    if slp < slope_min:
        return False, f"slope {slp:+.2f}% < {slope_min}%"

    # 3. MACD hist > 0 последние N баров подряд
    macd_bars = getattr(config, "ALIGNMENT_MACD_BARS", 3)
    for k in range(macd_bars):
        ki = i - k
        if ki < 0:
            return False, "мало баров для MACD проверки"
        mh = float(mh_arr[ki])
        if not np.isfinite(mh) or mh <= 0:
            return False, f"MACD hist ≤ 0 на баре -{k} (нужно {macd_bars} подряд)"

    # 4. RSI в рабочей зоне
    rsi_lo = getattr(config, "ALIGNMENT_RSI_LO", 45.0)
    rsi_hi = getattr(config, "ALIGNMENT_RSI_HI", 72.0)
    if not (rsi_lo <= ri <= rsi_hi):
        return False, f"RSI {ri:.1f} вне [{rsi_lo:.0f}–{rsi_hi:.0f}]"

    # 5. Минимальный объём (не требует спайка)
    vol_min = getattr(config, "ALIGNMENT_VOL_MIN", 0.8)
    if not np.isfinite(vx) or vx < vol_min:
        return False, f"vol× {vx:.2f} < {vol_min}"

    # 6. Не слишком поздно входить
    range_max = getattr(config, "ALIGNMENT_RANGE_MAX", 9.0)
    if np.isfinite(dr) and dr > range_max:
        return False, f"daily_range {dr:.1f}% > {range_max}% (поздно)"

    return True, f"EMA↑ slope={slp:+.2f}% MACD×{macd_bars}б RSI={ri:.0f} vol×{vx:.1f}"


# ── TREND_SURGE: детектор начала устойчивого тренда ──────────────────────────

def check_trend_surge_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    TREND_SURGE — фиксирует момент когда тренд «включается».

    Отличие от существующих сигналов:
      - IMPULSE: смотрит на r1/r3 (прыжок цены за 1-3 бара). Работает при быстрых
                 пробоях, но не ловит плавное начало многочасового тренда.
      - ALIGNMENT: требует 3 бара подряд MACD > 0. Пропускает первый бар разворота.
      - BUY: блокируется если ADX < 20 или ADX <= SMA(ADX,10). Лагует 10+ баров.

    TREND_SURGE фокусируется на СТРУКТУРНОМ ускорении:
      1. close > EMA20 > EMA50 — бычья структура
      2. EMA20 slope резко выросло по сравнению с 3 барами назад (ускорение)
      3. MACD hist > 0 И растёт (минимум 2 бара подряд)
      4. vol_x ≥ SURGE_VOL_MIN (объём подтверждает)
      5. RSI в [SURGE_RSI_LO, SURGE_RSI_HI] — импульс есть, не перегрет
      6. daily_range НЕ проверяется — специально для монет где BUY уже заблокирован

    Кулдаун: SURGE_COOLDOWN_BARS (20 × 15m = 5 часов) — один сигнал на тренд.

    Примеры: JASMY 09.03 03:00 UTC, BONK 09.03 12:00 UTC.
    """
    if i < 10:
        return False, "недостаточно баров"

    ef      = feat["ema_fast"]
    es      = feat["ema_slow"]
    slope   = feat["slope"]
    rsi_arr = feat["rsi"]
    vol_x   = feat["vol_x"]
    mh_arr  = feat["macd_hist"]
    c_arr   = feat.get("close")

    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    efv = float(ef[i])
    esv = float(es[i])
    slp = float(slope[i])
    ri  = float(rsi_arr[i])
    vx  = float(vol_x[i])

    if not all(np.isfinite([ci, efv, esv, slp, ri, vx])):
        return False, "нет данных индикаторов"

    # 1. Бычья структура: close > EMA20 > EMA50
    if not (ci > efv > esv):
        return False, f"структура нарушена"

    # 2. EMA20 slope ускорился: текущий slope > slope 3 бара назад + абс. порог
    slope_min = getattr(config, "SURGE_SLOPE_MIN", 0.15)
    if slp < slope_min:
        return False, f"slope {slp:+.2f}% < {slope_min}%"

    slp_prev = float(slope[i - 3]) if i >= 3 and np.isfinite(slope[i - 3]) else -999.0
    if slp <= slp_prev:
        return False, f"slope не ускоряется ({slp:+.2f}% ≤ {slp_prev:+.2f}% 3б назад)"

    # 3. MACD hist > 0 И растёт (2 бара подряд)
    mh_now  = float(mh_arr[i])
    mh_prev = float(mh_arr[i - 1]) if i >= 1 and np.isfinite(mh_arr[i - 1]) else 0.0
    if not np.isfinite(mh_now) or mh_now <= 0:
        return False, f"MACD hist ≤ 0"
    if mh_now <= mh_prev:
        return False, f"MACD hist не растёт"

    # 4. Объём
    vol_min = getattr(config, "SURGE_VOL_MIN", 1.5)
    if not np.isfinite(vx) or vx < vol_min:
        return False, f"vol× {vx:.2f} < {vol_min}"

    # 5. RSI в рабочей зоне
    rsi_lo = getattr(config, "SURGE_RSI_LO", 50.0)
    rsi_hi = getattr(config, "SURGE_RSI_HI", 80.0)
    if not np.isfinite(ri) or not (rsi_lo <= ri <= rsi_hi):
        return False, f"RSI {ri:.1f} вне [{rsi_lo:.0f}–{rsi_hi:.0f}]"

    pct_above = (ci / efv - 1) * 100
    return True, f"slope={slp:+.2f}%(↑{slp-slp_prev:+.2f}%) MACD↑ vol×{vx:.1f} RSI={ri:.0f} +{pct_above:.1f}%>EMA"



def check_ema_cross_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    EMA_CROSS — самый ранний сигнал: пробой EMA20 снизу вверх с объёмом.

    Срабатывает на 3-5 баров (45-75 мин) РАНЬШЕ стандартного BUY, потому что:
      - BUY требует: slope > EMA_SLOPE_MIN, ADX > 20, ADX > SMA(ADX)
        ADX лагует 10+ баров, slope только разворачивается
      - EMA_CROSS требует только: пробой EMA20 + объём + RSI не перекуплен

    Паттерн (GLM/ORDI/AXS 11.03.2026 ~10:00 UTC):
      - 8:00-9:45: цена ниже EMA20 (боковик или слабый даун)
      - 10:00: объёмный бар закрывается выше EMA20
      - 10:15: бот даёт обычный IMPULSE/BUY
      - Цель EMA_CROSS: дать сигнал на баре 10:00

    Условия:
      1. ema_cross[i] == 1.0  (индикатор: пробой EMA20 + объём + RSI фильтр)
      2. regime != bear_trend
    """
    if i < 5:
        return False, "недостаточно баров"

    cross_arr = feat.get("ema_cross")
    if cross_arr is None:
        return False, "ema_cross не вычислен"
    if not np.isfinite(float(cross_arr[i])) or float(cross_arr[i]) < 1.0:
        return False, "нет пробоя EMA20 с объёмом"

    # Блокируем только если ГЛОБАЛЬНЫЙ рынок (BTC) в bear_trend.
    # Режим самой монеты намеренно игнорируется — EMA_CROSS ловит начало
    # восстановления именно из медвежьего/бокового периода монеты.
    global_regime = getattr(config, "_current_regime", "neutral")
    if global_regime == "bear_trend":
        return False, "BTC в режиме bear_trend — EMA_CROSS запрещён"

    ef_arr  = feat["ema_fast"]
    es_arr  = feat["ema_slow"]
    rsi_arr = feat["rsi"]
    vol_arr = feat["vol_x"]
    dr_arr  = feat["daily_range_pct"]
    c_arr   = feat.get("close")
    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    ef  = float(ef_arr[i])
    es  = float(es_arr[i])
    ri  = float(rsi_arr[i]) if np.isfinite(rsi_arr[i]) else 0.0
    vx  = float(vol_arr[i]) if np.isfinite(vol_arr[i]) else 0.0
    dr  = float(dr_arr[i])  if np.isfinite(dr_arr[i])  else 0.0

    pct_above = (ci / ef - 1) * 100 if ef > 0 else 0.0
    above50   = "✓" if ci > es else "–"

    return (
        True,
        f"пробой EMA20 +{pct_above:.2f}% EMA50:{above50} "
        f"vol×{vx:.1f} RSI={ri:.0f} DR={dr:.1f}%"
    )


def _forward_accuracy(
    signals: List[int], c: np.ndarray
) -> Dict[int, HorizonAccuracy]:
    result: Dict[int, HorizonAccuracy] = {}
    for h in config.FORWARD_BARS:
        correct = total = 0
        for idx in signals:
            if idx + h >= len(c):
                continue  # форвард-бар ещё не закрылся
            total += 1
            if float(c[idx + h]) > float(c[idx]):
                correct += 1
        result[h] = HorizonAccuracy(horizon=h, total=total, correct=correct)
    return result


# ── Today start index ──────────────────────────────────────────────────────────

def _today_start_ms() -> int:
    """Unix ms начало окна форвард-теста.

    Скользящее окно 24ч вместо UTC-midnight:
    - в 03:00 UTC не теряем ночные сигналы (Азия)
    - в 06:00 UTC всегда достаточно данных для подтверждения
    - confirmed работает круглосуточно без провала в 00:00-06:00 UTC
    """
    now = datetime.now(timezone.utc)
    window_hours = getattr(config, "FORWARD_TEST_WINDOW_HOURS", 24)
    window_start = now - timedelta(hours=window_hours)
    return int(window_start.timestamp() * 1000)


def _find_today_start(timestamps: np.ndarray) -> int:
    """Индекс первого бара начиная с 00:00 UTC сегодня."""
    today_ms = _today_start_ms()
    for i, t in enumerate(timestamps):
        if int(t) >= today_ms:
            return i
    return len(timestamps) - 1  # fallback: последний бар


# ── Main analysis ──────────────────────────────────────────────────────────────


def _live_entry_signal_mode(feat: Dict, i: int, c: np.ndarray, *, tf: str = "") -> Tuple[bool, str]:
    """Return the V1 live-admissible entry mode for a bar.

    Keep today's forward-accuracy gate aligned with the live monitor.  If an
    early mode is omitted here, `today_confirmed` can lag real candidate
    admission and create false accuracy-gate blocks.
    """
    entry_ok, _ = check_entry_conditions(feat, i, c, tf=tf)
    if entry_ok:
        return True, get_entry_mode(feat, i)

    brk_ok, _ = check_breakout_conditions(feat, i)
    if brk_ok:
        return True, "breakout"

    retest_ok, _ = check_retest_conditions(feat, i)
    if retest_ok:
        return True, "retest"

    surge_ok, _ = check_trend_surge_conditions(feat, i)
    if surge_ok:
        return True, "impulse_speed"

    imp_ok, _ = check_impulse_conditions(feat, i)
    if imp_ok:
        return True, "impulse"

    aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)
    if aln_ok and bool(getattr(config, "ALIGNMENT_BUY_ENABLED", False)):
        return True, "alignment"

    return False, ""

def analyze_coin(
    symbol:    str,
    tf:        str,
    data:      np.ndarray,
    from_scan: bool = False,
) -> CoinReport:
    c    = data["c"].astype(float)
    feat = compute_features(data["o"], data["h"], data["l"], c, data["v"])

    # Минимальный прогрев индикаторов
    warmup = max(
        config.EMA_SLOW + config.SLOPE_LOOKBACK + 2,
        config.ADX_PERIOD * 2 + config.ADX_GROW_BARS + 2,
        config.VOL_LOOKBACK + 2,
        60,
    )

    # ── Найти начало сегодняшнего дня ─────────────────────────────────────────
    today_start = _find_today_start(data["t"])
    # Убедиться что прогрев индикаторов уже завершён к началу дня
    today_start = max(today_start, warmup)

    # ── Сигналы сегодня (только оцениваемые: есть все форвард-бары) ──────────
    # Последняя закрытая свеча
    i_now = len(c) - 2

    # Сигналы для которых уже прошло T+3 бара (45 мин) → можно оценить точность.
    # Используем min(FORWARD_BARS)=3 вместо max=10:
    # _forward_accuracy пропускает недоступные бары (if idx+h >= len(c): continue),
    # поэтому T+10 вычислится там где уже есть данные, остальные игнорируются.
    # Старая логика (i_now - 10) задерживала подтверждение на 150 мин — ранние
    # тренды (TON 08:30) не могли подтвердиться раньше 11:30.
    min_fwd  = min(config.FORWARD_BARS)
    max_fwd  = max(config.FORWARD_BARS)
    eval_end = i_now - min_fwd  # последний бар у которого T+3 уже прошёл

    today_eval_signals: List[int] = []
    if eval_end >= today_start:
        today_eval_signals = [
            i for i in range(today_start, eval_end + 1)
            if _live_entry_signal_mode(feat, i, c, tf=tf)[0]
        ]

    # Все сигналы сегодня (включая последние, ещё не оценимые)
    today_all_signals: List[int] = [
        i for i in range(today_start, i_now + 1)  # включаем последнюю закрытую свечу
        if _live_entry_signal_mode(feat, i, c, tf=tf)[0]
    ]

    # ── Форвард-тест на сегодняшних данных ───────────────────────────────────
    empty_acc = {h: HorizonAccuracy(h, 0, 0) for h in config.FORWARD_BARS}

    if len(today_eval_signals) >= config.TODAY_MIN_SIGNALS:
        today_acc = _forward_accuracy(today_eval_signals, c)
        acc_pct   = {h: fa.pct for h, fa in today_acc.items()
                     if fa.total > 0}
        if acc_pct:
            best_h   = max(acc_pct, key=acc_pct.get)
            best_acc = acc_pct[best_h]
            # Требуем:
            # 1. Лучший горизонт ≥ MIN_ACCURACY (60%)
            # 2. T+3 ≥ TODAY_T3_MIN (60%) — вход краткосрочно работает
            # 3. T+10 ≥ TODAY_T10_MIN (40%) — стратегия не убыточна на длинном горизонте
            t3  = today_acc.get(3)
            t10 = today_acc.get(10)
            t3_ok  = t3.pct  >= config.TODAY_T3_MIN  if t3  and t3.total  > 0 else False
            # T+10: если оценок < 2 — не блокируем (рано утром ещё нет данных)
            # если оценок ≥ 2 — требуем ≥ TODAY_T10_MIN (защита от DYDX-проблемы)
            if t10 and t10.total >= 2:
                t10_ok = t10.pct >= config.TODAY_T10_MIN
            else:
                t10_ok = True  # слишком мало данных чтобы судить
            confirmed = best_acc >= config.MIN_ACCURACY and t10_ok
        else:
            best_h = best_acc = 0
            confirmed = False

        note = (
            f"Сегодня {len(today_eval_signals)} сигн., точность подтверждена"
            if confirmed else
            f"Сегодня {len(today_eval_signals)} сигн., точность недостаточна"
        )
    else:
        today_acc = empty_acc
        best_h    = 0
        best_acc  = 0.0
        confirmed = False
        note = (
            f"Сегодня {len(today_eval_signals)} оцен. сигн. "
            f"(нужно ≥ {config.TODAY_MIN_SIGNALS}) — не подтверждено"
        )

    # Монета в игре только если стратегия подтверждена сегодняшними данными
    in_play = confirmed

    # ── Текущий сигнал (последняя закрытая свеча) ─────────────────────────────
    signal_now     = False
    no_signal_reason = "недостаточно баров"
    signal_mode    = ""

    if i_now >= warmup:
        mode_ok, mode_name = _live_entry_signal_mode(feat, i_now, c, tf=tf)
        if mode_ok:
            signal_now = True
            signal_mode = mode_name
            no_signal_reason = ""
        else:
            buy_ok, buy_reason = check_entry_conditions(feat, i_now, c, tf=tf)
            aln_ok, aln_reason = check_alignment_conditions(feat, i_now, tf=tf)
            no_signal_reason = f"ALIGNMENT: {aln_reason}  |  BUY: {buy_reason}"

    # SETUP: только если нет ни одного сигнала — не влияет на форвард-тест
    setup_now, setup_reason, setup_missing = (False, "", 99)
    if not signal_now and i_now >= warmup:
        setup_now, setup_reason, setup_missing = check_setup_conditions(feat, i_now, c)

    def _safe(arr, idx):
        v = arr[idx] if idx < len(arr) else np.nan
        return float(v) if np.isfinite(v) else 0.0

    return CoinReport(
        symbol=symbol, tf=tf,
        today_signals=len(today_all_signals),
        today_accuracy=today_acc,
        today_confirmed=confirmed,
        best_horizon=best_h,
        best_accuracy=best_acc,
        in_play=in_play,
        note=note,
        from_scan=from_scan,
        signal_now=signal_now,
        current_price=_safe(c, i_now),
        current_slope=_safe(feat["slope"], i_now),
        current_rsi=_safe(feat["rsi"], i_now),
        current_adx=_safe(feat["adx"], i_now),
        current_vol_x=_safe(feat["vol_x"], i_now),
        current_macd=_safe(feat["macd_hist"], i_now),
        no_signal_reason=no_signal_reason,
        setup_now=setup_now,
        setup_reason=setup_reason,
        setup_missing_count=setup_missing,
        signal_mode=signal_mode,
    )


# ── Batch runner ───────────────────────────────────────────────────────────────

async def _run_analysis(
    symbols:   List[str],
    from_scan: bool = False,
) -> Tuple[List[CoinReport], List[CoinReport]]:
    all_reports: List[CoinReport] = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            (sym, tf, asyncio.create_task(fetch_klines(session, sym, tf)))
            for sym in symbols
            for tf in config.TIMEFRAMES
        ]
        for idx, (sym, tf, task) in enumerate(tasks):
            data = await task
            if data is None:
                continue
            all_reports.append(await run_cpu(analyze_coin, sym, tf, data, from_scan))
            if idx % 12 == 0:
                await asyncio.sleep(0)

    # Лучший таймфрейм на монету.
    # Приоритет (по убыванию важности):
    #   1. today_confirmed — стратегия подтверждена сегодня
    #   2. signal_now      — активный сигнал прямо сейчас (важно для ⚡ секции)
    #   3. best_accuracy   — наибольшая точность среди оставшихся
    def _report_key(r: "CoinReport"):
        return (r.today_confirmed, r.signal_now, r.today_signals, r.best_accuracy)

    best: Dict[str, CoinReport] = {}
    for r in all_reports:
        prev = best.get(r.symbol)
        if prev is None or _report_key(r) > _report_key(prev):
            best[r.symbol] = r

    ranked   = sorted(best.values(), key=_report_key, reverse=True)
    in_play  = [r for r in ranked if r.in_play]
    skipped  = [r for r in ranked if not r.in_play]
    return in_play, skipped


async def morning_analysis(symbols: List[str]) -> Tuple[List[CoinReport], List[CoinReport]]:
    """Анализ вашего рабочего списка монет."""
    return await _run_analysis(symbols, from_scan=False)


async def market_scan() -> Tuple[List[CoinReport], List[CoinReport]]:
    """
    Полный скан всех монет из вотчлиста (load_watchlist).
    П5: перед анализом определяет характер дня (BTC 1h EMA50).
    v2: определяет Market Regime и применяет адаптивные пороги.
    """
    async with aiohttp.ClientSession() as _sess:
        bull, btc_price, btc_ema50 = await is_bull_day(_sess)
        regime = await detect_market_regime(_sess)

    config._bull_day_active       = bull
    config._effective_range_max   = getattr(config, "BULL_DAY_RANGE_MAX", 10.0) if bull else config.DAILY_RANGE_MAX
    config._effective_rsi_hi      = getattr(config, "BULL_DAY_RSI_HI",    75.0) if bull else config.RSI_BUY_HI
    config._btc_vs_ema50          = round((btc_price / btc_ema50 - 1) * 100, 4) if btc_ema50 > 0 else 0.0

    # v2: сохраняем глобальный режим рынка для использования в анализе
    config._current_regime        = regime.name
    config._regime_params_active  = getattr(config, "REGIME_PARAMS", {}).get(regime.name, {})

    # В бычьем тренде или recovery — расширяем диапазон дополнительно
    if regime.name in ("bull_trend", "recovery") and not bull:
        # BTC в хорошем состоянии, но не официальный бычий день → небольшое расширение
        config._effective_range_max = max(
            config._effective_range_max, regime.range_max
        )

    botlog.log_bull_day(
        is_bull=bull, btc_price=btc_price, btc_ema50=btc_ema50,
        eff_range_max=config._effective_range_max,
        eff_rsi_hi=config._effective_rsi_hi,
    )

    watchlist = config.load_watchlist()
    in_play, skipped = await _run_analysis(watchlist, from_scan=False)

    regime_str = str(regime)
    botlog.log_analysis(
        n_scanned=len(watchlist),
        n_confirmed=len(in_play),
        n_signal_now=sum(1 for r in in_play if r.signal_now),
        n_setup=sum(1 for r in skipped if r.setup_now),
        n_early=sum(1 for r in skipped if r.signal_now),
        is_bull=bull,
        confirmed_symbols=[r.symbol for r in in_play],
    )

    # Добавляем информацию о режиме в первый отчёт (используется в bot.py для заголовка)
    config._regime_display = regime_str

    return in_play, skipped
