from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_regime_start_max_period as audit
import monitor
import regime_start


DTYPE = [("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")]


def bars(count: int, step_ms: int, *, start: float = 100.0) -> np.ndarray:
    data = np.zeros(count, dtype=DTYPE)
    close = start * np.power(1.001, np.arange(count))
    data["t"] = np.arange(count, dtype=np.int64) * step_ms
    data["o"] = close * 0.999
    data["h"] = close * 1.003
    data["l"] = close * 0.997
    data["c"] = close
    data["v"] = 1000.0
    return data


def features(data: np.ndarray, *, active_from: int | None = None) -> dict[str, np.ndarray]:
    n = len(data)
    ema = data["c"] * 0.995
    macd = np.full(n, -0.1)
    if active_from is None:
        macd = np.linspace(-0.2, 0.2, n)
    else:
        macd[active_from:] = 0.1 + np.arange(n - active_from) * 0.001
    return {
        "ema": ema,
        "rsi": np.full(n, 55.0),
        "adx": np.full(n, 25.0),
        "vol_x": np.full(n, 1.0),
        "macd_hist": macd,
    }


def signal(index: int = 100) -> regime_start.RegimeStartSignal:
    return regime_start.RegimeStartSignal(
        profile="base_recovery_v1",
        bar_index_4h=index,
        daily_index=40,
        bar_open_ts_ms=index * regime_start.FOUR_H_MS,
        decision_ts_ms=(index + 1) * regime_start.FOUR_H_MS,
        price=100.0,
        ema20_4h=99.0,
        slope_pct_4h=0.2,
        rsi_4h=55.0,
        adx_4h=25.0,
        vol_x_4h=1.0,
        macd_hist_4h=0.1,
        daily_close=99.0,
        daily_ema7=99.5,
        daily_rsi=45.0,
        daily_macd_hist=-0.01,
        daily_return_3d_pct=1.0,
    )


class RegimeStartDetectorTests(unittest.TestCase):
    def test_daily_alignment_uses_only_candles_closed_by_decision(self) -> None:
        daily_t = np.array([0, regime_start.DAY_MS, 2 * regime_start.DAY_MS], dtype=np.int64)
        before_second_close = 2 * regime_start.DAY_MS - 1
        at_second_close = 2 * regime_start.DAY_MS
        self.assertEqual(regime_start._daily_index_for_decision(daily_t, before_second_close), 0)
        self.assertEqual(regime_start._daily_index_for_decision(daily_t, at_second_close), 1)

    def test_detector_emits_only_false_to_true_transition(self) -> None:
        four_h = bars(270, regime_start.FOUR_H_MS)
        daily = bars(80, regime_start.DAY_MS)
        active_from = 245
        f4 = features(four_h, active_from=active_from)
        fd = features(daily)
        with patch.object(regime_start, "_features", side_effect=[f4, fd]):
            starts = regime_start.detect_regime_starts(four_h, daily)
        self.assertEqual([row.bar_index_4h for row in starts], [active_from])

    def test_latest_detector_rejects_old_transition(self) -> None:
        four_h = bars(270, regime_start.FOUR_H_MS)
        daily = bars(80, regime_start.DAY_MS)
        f4 = features(four_h, active_from=245)
        fd = features(daily)
        with patch.object(regime_start, "_features", side_effect=[f4, fd]):
            latest = regime_start.detect_latest_regime_start(four_h, daily)
        self.assertIsNone(latest)

    def test_closed_candles_excludes_incomplete_bar(self) -> None:
        data = bars(3, regime_start.FOUR_H_MS)
        now_ms = int(data["t"][2]) + regime_start.FOUR_H_MS - 1
        closed = monitor._closed_candles(data, regime_start.FOUR_H_MS, now_ms)
        self.assertEqual(list(closed["t"]), list(data["t"][:2]))


class RegimeStartAuditTests(unittest.TestCase):
    def test_label_uses_strictly_forward_4h_bars(self) -> None:
        data = bars(170, regime_start.FOUR_H_MS)
        start = signal(100)
        data["h"][101:131] = 109.0
        data["l"][101:131] = 96.0
        label = audit.label_signal("AAAUSDT", start, data)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertTrue(label.useful)
        self.assertAlmostEqual(label.mfe_5d_pct, 9.0)
        self.assertAlmostEqual(label.mae_5d_pct, -4.0)

    def test_promotion_fails_closed_when_pol_case_is_missing(self) -> None:
        train = {"useful_precision_pct": 40.0}
        holdout = {
            "signals": 200,
            "calendar_days": 100.0,
            "useful_precision_pct": 38.0,
            "median_ret_5d_pct": 1.0,
            "median_mfe_5d_pct": 7.0,
            "signals_per_calendar_day": 2.0,
        }
        decision = audit.evaluate_promotion(train, holdout, pol_detected_by_deadline=False)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["decision"], "shadow_only")
        self.assertIn("pol_detected_by_2026_07_02_close", decision["failed_checks"])

    def test_promotion_can_only_approve_watch(self) -> None:
        train = {"useful_precision_pct": 40.0}
        holdout = {
            "signals": 200,
            "calendar_days": 100.0,
            "useful_precision_pct": 38.0,
            "median_ret_5d_pct": 1.0,
            "median_mfe_5d_pct": 7.0,
            "signals_per_calendar_day": 2.0,
        }
        decision = audit.evaluate_promotion(train, holdout, pol_detected_by_deadline=True)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["decision"], "eligible_for_watch_only")
        self.assertFalse(decision["buy_policy_changed"])


class RegimeStartMonitorTests(unittest.TestCase):
    def test_shadow_scan_logs_once_and_does_not_send_when_telegram_disabled(self) -> None:
        state = monitor.MonitorState()
        send = AsyncMock()
        snapshot = signal()
        with (
            patch.object(monitor.config, "REGIME_START_SHADOW_ENABLED", True),
            patch.object(monitor.config, "REGIME_START_TELEGRAM_ENABLED", False),
            patch.object(monitor.config, "load_watchlist", return_value=["AAAUSDT"]),
            patch.object(monitor, "_regime_start_snapshot", new=AsyncMock(return_value=snapshot)),
            patch.object(monitor.botlog, "log_regime_start_shadow") as log_event,
        ):
            asyncio.run(monitor._run_regime_start_scan(AsyncMock(), state, send))
            asyncio.run(monitor._run_regime_start_scan(AsyncMock(), state, send))
        log_event.assert_called_once()
        send.assert_not_awaited()

    def test_watch_message_is_explicitly_not_a_buy(self) -> None:
        state = monitor.MonitorState()
        send = AsyncMock()
        with (
            patch.object(monitor.config, "REGIME_START_SHADOW_ENABLED", True),
            patch.object(monitor.config, "REGIME_START_TELEGRAM_ENABLED", True),
            patch.object(monitor.config, "load_watchlist", return_value=["AAAUSDT"]),
            patch.object(monitor, "_regime_start_snapshot", new=AsyncMock(return_value=signal())),
            patch.object(monitor.botlog, "log_regime_start_shadow"),
        ):
            asyncio.run(monitor._run_regime_start_scan(AsyncMock(), state, send))
        message = send.await_args.args[0]
        self.assertIn("WATCH, не BUY", message)
        self.assertIn("Позиция не открыта", message)


if __name__ == "__main__":
    unittest.main()
