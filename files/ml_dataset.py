from __future__ import annotations

"""
ML Dataset Logger — ml_dataset.jsonl

Отдельный от bot_events.jsonl файл для накопления датасета под обучение моделей.

Ключевые принципы:
  1. Одна запись = один кандидат на вход (сигнал любого типа).
  2. Запись содержит ПОЛНЫЙ контекст: окно последних SEQ_LEN баров
     с нормализованными признаками — для LSTM, Transformer, GBM.
  3. Метки (labels) заполняются позже, когда прошло T+N баров.
     До этого labels = null. Скрипт fill_labels.py довзаключает их.
  4. Все признаки ОТНОСИТЕЛЬНЫЕ (% от текущей цены, z-score и т.д.)
     чтобы модель обобщалась на любую монету и ценовой диапазон.

Формат одной записи (JSONL):
{
  // ── Идентификация ─────────────────────────────────────────────────
  "id":          "COTIUSDT_15m_20260303T200016Z",   // уникальный ключ
  "sym":         "COTIUSDT",
  "tf":          "15m",
  "ts_signal":   "2026-03-03T20:00:16Z",            // UTC время сигнала
  "bar_ts":      1741032000000,                     // unix ms входного бара
  "signal_type": "retest",                          // trend/strong_trend/retest/breakout
  "is_bull_day": true,
  "hour_utc":    20,                                // 0-23
  "dow":         1,                                 // 0=Mon .. 6=Sun

  // ── Скалярные признаки на баре входа (нормализованные) ────────────
  "f": {
    "close_vs_ema20":   1.55,   // (close/ema20 - 1) * 100, %
    "close_vs_ema50":   3.80,
    "close_vs_ema200":  5.92,
    "ema20_vs_ema50":   2.21,   // (ema20/ema50 - 1) * 100
    "ema50_vs_ema200":  1.74,
    "slope":            0.607,  // EMA20 slope %, уже нормализован
    "rsi":              62.1,
    "adx":              48.4,
    "vol_x":            1.09,
    "macd_hist_norm":  -0.010,  // macd_hist / close * 100
    "atr_pct":          0.82,   // atr / close * 100
    "daily_range":      7.92,   // % от дня
    "body_pct":         0.40,   // |close-open|/close*100 — размер тела свечи
    "upper_wick_pct":   0.20,   // (high-max(open,close))/close*100
    "lower_wick_pct":   0.15,   // (min(open,close)-low)/close*100
    "btc_vs_ema50":     1.20    // BTC относительно своей EMA50
  },

  // ── Последовательный контекст: окно SEQ_LEN баров ─────────────────
  // Каждый элемент = нормализованный вектор одного бара.
  // Порядок: от старого к новому, последний = бар сигнала.
  // Для нормализации: все цены делятся на close[signal_bar].
  // seq[i] = список из SEQ_FEATURES значений (см. SEQ_FEATURE_NAMES).
  "seq": [
    [1.003, 1.005, 0.998, 1.001, 1.8, 0.6, 48.1, 62.0, 0.0, 0.4],
    // ... SEQ_LEN строк
  ],

  // ── Метки (заполняются fill_labels.py) ───────────────────────────
  "labels": {
    "ret_3":    null,   // (close[+3] / entry_price - 1) * 100
    "ret_5":    null,
    "ret_10":   null,
    "label_3":  null,   // bool: ret_3 > 0
    "label_5":  null,
    "label_10": null,
    "exit_pnl": null,   // реальный PnL бота (с учётом стопа)
    "exit_reason": null,
    "bars_held": null
  }
}

Имена признаков в seq (SEQ_FEATURE_NAMES):
  0: close_norm    — close / signal_close
  1: high_norm     — high / signal_close
  2: low_norm      — low / signal_close
  3: open_norm     — open / signal_close
  4: vol_x         — объём / SMA(объём, 20)
  5: slope         — EMA20 slope %
  6: adx           — ADX
  7: rsi           — RSI
  8: macd_hist_norm — macd_hist / signal_close * 100
  9: atr_pct       — ATR / signal_close * 100
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows production path
    msvcrt = None

ROOT = Path(__file__).resolve().parent
ML_FILE   = ROOT / "ml_dataset.jsonl"
SEQ_LEN   = 20   # баров контекста (20 × 15m = 5 часов)
_pylog    = logging.getLogger("ml_dataset")
_DATASET_THREAD_LOCK = threading.RLock()
_LOCK_TIMEOUT_SEC = 120.0
_LOCK_POLL_SEC = 0.05
_REPLACE_RETRIES = 8
_REPLACE_RETRY_SEC = 0.1

SEQ_FEATURE_NAMES = [
    "close_norm", "high_norm", "low_norm", "open_norm",
    "vol_x", "slope", "adx", "rsi", "macd_hist_norm", "atr_pct",
]


# ── JSON encoder ───────────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


@contextmanager
def _dataset_io_lock():
    with _DATASET_THREAD_LOCK:
        lock_path = ML_FILE.with_suffix(ML_FILE.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if msvcrt is None:
            yield
            return
        start = time.monotonic()
        lock_handle = None
        while True:
            try:
                lock_handle = lock_path.open("a+b")
                if lock_handle.tell() == 0:
                    lock_handle.write(b"0")
                    lock_handle.flush()
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if lock_handle is not None:
                    lock_handle.close()
                    lock_handle = None
                if time.monotonic() - start >= _LOCK_TIMEOUT_SEC:
                    raise TimeoutError(f"timeout acquiring ml_dataset lock: {lock_path}")
                time.sleep(_LOCK_POLL_SEC)
        try:
            yield
        finally:
            try:
                if lock_handle is not None:
                    lock_handle.seek(0)
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                if lock_handle is not None:
                    lock_handle.close()


def _atomic_replace_with_retry(tmp: Path, target: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, _REPLACE_RETRIES + 1):
        try:
            tmp.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt >= _REPLACE_RETRIES:
                raise
            time.sleep(_REPLACE_RETRY_SEC * attempt)
    if last_error is not None:
        raise last_error


def _rewrite_records(
    mutator,
    *,
    raw_required_tokens: tuple[str, ...] = (),
) -> tuple[int, int]:
    if not ML_FILE.exists():
        return 0, 0
    changed_rows = 0
    bad_lines = 0
    with _dataset_io_lock():
        needs_rewrite = False
        with ML_FILE.open("r", encoding="utf-8", errors="ignore") as source:
            for line in source:
                if raw_required_tokens and not all(token in line for token in raw_required_tokens):
                    continue
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and bool(mutator(rec)):
                    needs_rewrite = True
                    break
        if not needs_rewrite:
            return 0, 0

        tmp = ML_FILE.with_name(f"{ML_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with ML_FILE.open("r", encoding="utf-8", errors="ignore") as source, tmp.open(
                "w", encoding="utf-8"
            ) as destination:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue
                    if not isinstance(rec, dict):
                        bad_lines += 1
                        continue
                    if bool(mutator(rec)):
                        changed_rows += 1
                    destination.write(json.dumps(rec, ensure_ascii=False, cls=_Enc) + "\n")
            if changed_rows or bad_lines:
                _atomic_replace_with_retry(tmp, ML_FILE)
            else:
                tmp.unlink(missing_ok=True)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    return changed_rows, bad_lines


def _w(record: Dict[str, Any]) -> None:
    try:
        with _dataset_io_lock():
            with ML_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, cls=_Enc) + "\n")
    except Exception as e:
        _pylog.warning("ml_dataset write error: %s", e)


def _safe(v) -> float:
    """Конвертирует numpy scalar в float, nan → 0.0."""
    try:
        f = float(v)
        return 0.0 if (f != f) else f  # nan check
    except Exception:
        return 0.0


# ── Основная функция ───────────────────────────────────────────────────────────

def log_bar_snapshot(
    sym:             str,
    tf:              str,
    bar_ts:          int,
    rule_signal:     str,
    is_bull_day:     bool,
    feat:            Dict,
    i:               int,
    data:            Any,
    btc_vs_ema50:    float = 0.0,
    btc_momentum_4h: float = 0.0,
    market_vol_24h:  float = 0.0,
) -> str:
    """
    Логирует КАЖДЫЙ бар каждой мониторируемой монеты — независимо от того
    сработал ли сигнал. Это основа несмещённого датасета для ML.

    rule_signal = "none" если стратегия не нашла входа.
    rule_signal = "trend"/"retest"/etc если нашла.

    Labels (ret_3/ret_5/ret_10) заполняются позже через fill_forward_label_by_ts()
    когда пройдут нужные бары — бот сам вычисляет их в следующих poll-циклах.
    """
    # Дедупликация: один и тот же бар монеты логируем только раз
    # (poll идёт каждые 60с, но бар закрывается каждые 15м)
    dedup_key = f"{sym}_{tf}_{bar_ts}"
    if _is_logged(dedup_key):
        return ""
    _mark_logged(dedup_key)

    ts_str    = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record_id = f"{sym}_{tf}_{ts_str.replace(':', '').replace('-', '')}"
    return _write_candidate(
        record_id=record_id, sym=sym, tf=tf, bar_ts=bar_ts, ts_str=ts_str,
        signal_type=rule_signal, is_bull_day=is_bull_day,
        feat=feat, i=i, data=data, btc_vs_ema50=btc_vs_ema50,
    )


# ── Дедупликация баров (в памяти, сбрасывается при перезапуске) ────────────────
from collections import OrderedDict as _OD
_logged_bars: _OD = _OD()
_MAX_LOGGED  = 50_000   # не держим больше N ключей в памяти

def _is_logged(key: str) -> bool:
    return key in _logged_bars

def _mark_logged(key: str) -> None:
    if key in _logged_bars:
        return  # уже есть, не трогаем
    if len(_logged_bars) >= _MAX_LOGGED:
        # Выбрасываем 10% самых старых записей — не сбрасываем всё сразу
        evict = _MAX_LOGGED // 10
        for _ in range(evict):
            _logged_bars.popitem(last=False)
    _logged_bars[key] = True


def log_signal_candidate(
    sym:         str,
    tf:          str,
    bar_ts:      int,
    signal_type: str,
    is_bull_day: bool,
    feat:        Dict,
    i:           int,
    data:        Any,
    btc_vs_ema50: float = 0.0,
) -> str:
    """
    Записывает подтверждённый сигнал (бот входит в позицию).
    Используется для связки с реальным PnL через fill_labels().
    """
    ts_str    = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record_id = f"{sym}_{tf}_{ts_str.replace(':', '').replace('-', '')}"
    return _write_candidate(
        record_id=record_id, sym=sym, tf=tf, bar_ts=bar_ts, ts_str=ts_str,
        signal_type=signal_type, is_bull_day=is_bull_day,
        feat=feat, i=i, data=data, btc_vs_ema50=btc_vs_ema50,
    )


def _write_candidate(
    record_id: str, sym: str, tf: str, bar_ts: int, ts_str: str,
    signal_type: str, is_bull_day: bool,
    feat: Dict, i: int, data: Any, btc_vs_ema50: float,
    btc_momentum_4h: float = 0.0,
    market_vol_24h:  float = 0.0,
) -> str:
    """Общая логика записи — используется и log_bar_snapshot и log_signal_candidate."""
    c_arr = data["c"].astype(float)
    o_arr = data["o"].astype(float)
    h_arr = data["h"].astype(float)
    l_arr = data["l"].astype(float)
    v_arr = data["v"].astype(float)

    close_sig = _safe(c_arr[i])
    if close_sig <= 0:
        return record_id

    # ── Скалярные признаки на баре входа ──────────────────────────────────────
    ema20 = _safe(feat["ema_fast"][i])
    ema50 = _safe(feat["ema_slow"][i])
    # EMA200 может отсутствовать — считаем из данных если есть
    ema200 = 0.0
    if "ema200" in feat:
        ema200 = _safe(feat["ema200"][i])

    atr   = _safe(feat["atr"][i])
    o_sig = _safe(o_arr[i])
    h_sig = _safe(h_arr[i])
    l_sig = _safe(l_arr[i])

    body        = abs(close_sig - o_sig) / close_sig * 100 if close_sig > 0 else 0.0
    upper_wick  = (h_sig - max(o_sig, close_sig)) / close_sig * 100 if close_sig > 0 else 0.0
    lower_wick  = (min(o_sig, close_sig) - l_sig) / close_sig * 100 if close_sig > 0 else 0.0
    macd_norm   = _safe(feat["macd_hist"][i]) / close_sig * 100 if close_sig > 0 else 0.0
    atr_pct     = atr / close_sig * 100 if close_sig > 0 else 0.0

    scalar_f = {
        "close_vs_ema20":  round((close_sig / ema20 - 1) * 100, 4) if ema20 > 0 else 0.0,
        "close_vs_ema50":  round((close_sig / ema50 - 1) * 100, 4) if ema50 > 0 else 0.0,
        "close_vs_ema200": round((close_sig / ema200 - 1) * 100, 4) if ema200 > 0 else 0.0,
        "ema20_vs_ema50":  round((ema20 / ema50 - 1) * 100, 4) if ema50 > 0 else 0.0,
        "ema50_vs_ema200": round((ema50 / ema200 - 1) * 100, 4) if ema200 > 0 else 0.0,
        "slope":           round(_safe(feat["slope"][i]), 4),
        "rsi":             round(_safe(feat["rsi"][i]), 2),
        "adx":             round(_safe(feat["adx"][i]), 2),
        "vol_x":           round(_safe(feat["vol_x"][i]), 4),
        "macd_hist_norm":  round(macd_norm, 6),
        "atr_pct":         round(atr_pct, 4),
        "daily_range":     round(_safe(feat["daily_range_pct"][i]), 4),
        "body_pct":        round(body, 4),
        "upper_wick_pct":  round(upper_wick, 4),
        "lower_wick_pct":  round(lower_wick, 4),
        "btc_vs_ema50":    round(btc_vs_ema50, 4),
        "btc_momentum_4h": round(btc_momentum_4h, 4),
        "market_vol_24h":  round(market_vol_24h, 4),
        # Возвраты на нескольких лагах — явная динамика цены
        "r1":  round((float(c_arr[i]) / float(c_arr[i-1])  - 1) * 100, 4) if i >= 1  and c_arr[i-1]  > 0 else 0.0,
        "r3":  round((float(c_arr[i]) / float(c_arr[i-3])  - 1) * 100, 4) if i >= 3  and c_arr[i-3]  > 0 else 0.0,
        "r5":  round((float(c_arr[i]) / float(c_arr[i-5])  - 1) * 100, 4) if i >= 5  and c_arr[i-5]  > 0 else 0.0,
        "r10": round((float(c_arr[i]) / float(c_arr[i-10]) - 1) * 100, 4) if i >= 10 and c_arr[i-10] > 0 else 0.0,
    }

    # ── Последовательный контекст SEQ_LEN баров ───────────────────────────────
    seq: List[List[float]] = []
    start = max(0, i - SEQ_LEN + 1)
    for k in range(start, i + 1):
        c_k   = _safe(c_arr[k])
        c_n   = c_k / close_sig if close_sig > 0 else 1.0   # нормализация к цене сигнала
        h_n   = _safe(h_arr[k]) / close_sig if close_sig > 0 else 1.0
        l_n   = _safe(l_arr[k]) / close_sig if close_sig > 0 else 1.0
        o_n   = _safe(o_arr[k]) / close_sig if close_sig > 0 else 1.0
        vx_k  = _safe(feat["vol_x"][k])
        slp_k = _safe(feat["slope"][k])
        adx_k = _safe(feat["adx"][k])
        rsi_k = _safe(feat["rsi"][k])
        mh_k  = _safe(feat["macd_hist"][k]) / close_sig * 100 if close_sig > 0 else 0.0
        atr_k = _safe(feat["atr"][k]) / close_sig * 100 if close_sig > 0 else 0.0
        seq.append([
            round(c_n, 6), round(h_n, 6), round(l_n, 6), round(o_n, 6),
            round(vx_k, 4), round(slp_k, 4), round(adx_k, 2), round(rsi_k, 2),
            round(mh_k, 6), round(atr_k, 4),
        ])

    # Если баров меньше SEQ_LEN — дополняем нулями в начале (O(1))
    if len(seq) < SEQ_LEN:
        padding = [[0.0] * len(SEQ_FEATURE_NAMES)] * (SEQ_LEN - len(seq))
        seq = padding + seq

    # ── Временны́е признаки ────────────────────────────────────────────────────
    dt = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)

    record = {
        "id":          record_id,
        "sym":         sym,
        "tf":          tf,
        "ts_signal":   ts_str,
        "bar_ts":      bar_ts,
        "signal_type": signal_type,
        "is_bull_day": bool(is_bull_day),
        "hour_utc":    dt.hour,
        "dow":         dt.weekday(),  # 0=Mon
        "f":           scalar_f,
        "seq":         seq,
        "seq_feature_names": SEQ_FEATURE_NAMES,
        "labels": {
            "ret_3":       None,
            "ret_5":       None,
            "ret_10":      None,
            "label_3":     None,
            "label_5":     None,
            "label_10":    None,
            "exit_pnl":    None,
            "exit_reason": None,
            "bars_held":   None,
        },
    }
    _w(record)
    return record_id


# ── Заполнение меток по timestamp (для log_bar_snapshot) ───────────────────────

def fill_pending_from_data(
    sym:    str,
    tf:     str,
    t_arr:  Any,   # numpy int array с unix ms timestamps
    c_arr:  Any,   # numpy float array с close ценами
    bar_ms: int,   # длина бара в мс (900000 для 15m)
) -> None:
    """
    Заполняет незаполненные ret_3/ret_5/ret_10 для всех записей монеты sym+tf,
    используя переданные массивы данных.
    Вызывается из analyze_coin — работает для ВСЕХ монет вотчлиста,
    не только тех что в мониторинге.
    Эффективно: читает файл один раз, обновляет все нужные записи, пишет один раз.
    """
    if not ML_FILE.exists():
        return
    try:
        def _mutate(rec: Dict[str, Any]) -> bool:
            if rec.get("sym") != sym or rec.get("tf") != tf:
                return False
            lab = rec.setdefault("labels", {})
            rec_bar_ts = rec.get("bar_ts", 0)
            # Цена входного бара хранится в seq[-1][0] * (ema20 * (1 + close_vs_ema20/100))
            # Проще: используем bar_ts для поиска в t_arr и берём close напрямую
            idx_arr = np.where(t_arr == rec_bar_ts)[0]
            if len(idx_arr) == 0:
                return False
            entry_close = float(c_arr[idx_arr[0]])
            if entry_close <= 0:
                return False
            changed = False
            for h in (3, 5, 10):
                key_ret   = f"ret_{h}"
                key_label = f"label_{h}"
                if lab.get(key_ret) is not None:
                    continue  # уже заполнено
                future_ts  = rec_bar_ts + h * bar_ms
                fut_idx    = np.where(t_arr >= future_ts)[0]
                if len(fut_idx) == 0:
                    continue  # данных ещё нет
                future_close = float(c_arr[fut_idx[0]])
                ret_pct = (future_close / entry_close - 1) * 100
                rec["labels"][key_ret]   = round(ret_pct, 4)
                rec["labels"][key_label] = ret_pct > 0
                changed = True
            return changed

        _, bad_lines = _rewrite_records(
            _mutate,
            raw_required_tokens=(f'"sym": "{sym}"', f'"tf": "{tf}"'),
        )
        if bad_lines:
            _pylog.warning("fill_pending_from_data skipped %d malformed jsonl lines", bad_lines)
    except Exception as e:
        _pylog.warning("fill_pending_from_data error: %s", e)


def fill_forward_label_by_ts(
    sym: str, tf: str, bar_ts: int,
    horizon: int, ret_pct: float,
) -> None:
    """
    Заполняет ret_N/label_N для всех записей с данным sym+tf+bar_ts.
    Используется для заполнения меток баров которые были залогированы
    через log_bar_snapshot (не через log_signal_candidate).
    """
    if not ML_FILE.exists():
        return
    ts_str    = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record_id = f"{sym}_{tf}_{ts_str.replace(':', '').replace('-', '')}"
    fill_forward_label(record_id, horizon, ret_pct)


def fill_labels(
    record_id:   str,
    exit_pnl:    float,
    exit_reason: str,
    bars_held:   int,
) -> None:
    """
    Обновляет поле labels в существующей записи по record_id.
    Вызывается из monitor.py при выходе из позиции.
    Читает весь файл, обновляет нужную запись, перезаписывает.
    Медленно, но файл небольшой и это делается редко.
    """
    if not ML_FILE.exists():
        return
    try:
        def _mutate(rec: Dict[str, Any]) -> bool:
            if rec.get("id") == record_id:
                labels = rec.setdefault("labels", {})
                values = {
                    "exit_pnl": round(exit_pnl, 4),
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                }
                changed = any(labels.get(key) != value for key, value in values.items())
                labels.update(values)
                return changed
            return False

        _, bad_lines = _rewrite_records(
            _mutate,
            raw_required_tokens=(f'"id": "{record_id}"',),
        )
        if bad_lines:
            _pylog.warning("fill_labels skipped %d malformed jsonl lines", bad_lines)
    except Exception as e:
        _pylog.warning("fill_labels error: %s", e)


def fill_learning_labels(record_id: str, labels: Dict[str, Any]) -> None:
    if not record_id or not labels or not ML_FILE.exists():
        return
    try:
        def _mutate(rec: Dict[str, Any]) -> bool:
            if rec.get("id") == record_id:
                rec.setdefault("labels", {})
                changed = False
                for key, value in labels.items():
                    if rec["labels"].get(key) != value:
                        rec["labels"][key] = value
                        changed = True
                return changed
            return False

        _, bad_lines = _rewrite_records(
            _mutate,
            raw_required_tokens=(f'"id": "{record_id}"',),
        )
        if bad_lines:
            _pylog.warning("fill_learning_labels skipped %d malformed jsonl lines", bad_lines)
    except Exception as e:
        _pylog.warning("fill_learning_labels error: %s", e)


def fill_forward_label(
    record_id: str,
    horizon:   int,       # 3, 5 или 10
    ret_pct:   float,     # (forward_price / entry_price - 1) * 100
) -> None:
    """
    Заполняет ret_N и label_N по горизонту.
    Вызывается из monitor.py при оценке форвард-прогноза.
    """
    if not ML_FILE.exists():
        return
    key_ret   = f"ret_{horizon}"
    key_label = f"label_{horizon}"
    try:
        def _mutate(rec: Dict[str, Any]) -> bool:
            if rec.get("id") == record_id:
                labels = rec.setdefault("labels", {})
                values = {key_ret: round(ret_pct, 4), key_label: ret_pct > 0}
                changed = any(labels.get(key) != value for key, value in values.items())
                labels.update(values)
                return changed
            return False

        _, bad_lines = _rewrite_records(
            _mutate,
            raw_required_tokens=(f'"id": "{record_id}"',),
        )
        if bad_lines:
            _pylog.warning("fill_forward_label skipped %d malformed jsonl lines", bad_lines)
    except Exception as e:
        _pylog.warning("fill_forward_label error: %s", e)
