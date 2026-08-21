from __future__ import annotations

import unittest
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np

import replay_backtest as rb


class ReplayDailyObjectiveTest(unittest.TestCase):
    def _series(self, local_day: date, open_price: float, final_price: float) -> np.ndarray:
        tz = ZoneInfo(rb.OBJECTIVE_TZ)
        starts = [
            datetime.combine(local_day, time(0, 0), tzinfo=tz),
            datetime.combine(local_day, time(21, 45), tzinfo=tz),
        ]
        rows = np.zeros(2, dtype=rb._KLINE_DTYPE)
        rows["t"] = [int(row.astimezone(timezone.utc).timestamp() * 1000) for row in starts]
        rows["o"] = [open_price, final_price]
        rows["h"] = np.maximum(rows["o"], [open_price, final_price])
        rows["l"] = np.minimum(rows["o"], [open_price, final_price])
        rows["c"] = [open_price, final_price]
        rows["v"] = 1.0
        return rows

    def _trade(
        self,
        symbol: str,
        local_day: date,
        *,
        capture: float,
        minute: int = 60,
    ) -> rb.ReplayTrade:
        tz = ZoneInfo(rb.OBJECTIVE_TZ)
        entry = datetime.combine(local_day, time(0, 0), tzinfo=tz).timestamp() * 1000 + minute * 60_000
        return rb.ReplayTrade(
            sym=symbol,
            tf="15m",
            mode="trend",
            entry_ts=int(entry),
            entry_price=100.0,
            entry_i=0,
            trail_k=2.0,
            max_hold_bars=10,
            trail_stop=90.0,
            exit_ts=int(entry + 900_000),
            exit_price=101.0,
            capture_ratio_at_entry=capture,
            lead_time_to_final_top_min=120.0,
        )

    def test_daily_labels_and_report_use_day_symbol_denominators(self) -> None:
        tz = ZoneInfo(rb.OBJECTIVE_TZ)
        day1 = date(2026, 8, 18)
        day2 = date(2026, 8, 19)
        symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        cache = {
            ("AAAUSDT", "15m"): (np.concatenate([self._series(day1, 100, 110), self._series(day2, 100, 101)]), {}),
            ("BBBUSDT", "15m"): (np.concatenate([self._series(day1, 100, 105), self._series(day2, 100, 120)]), {}),
            ("CCCUSDT", "15m"): (np.concatenate([self._series(day1, 100, 99), self._series(day2, 100, 115)]), {}),
        }
        start = datetime.combine(day1, time(0, 0), tzinfo=tz).astimezone(timezone.utc)
        end = datetime.combine(day2, time(22, 0), tzinfo=tz).astimezone(timezone.utc)
        objective = rb._daily_top_objective(
            symbols,
            cache,
            start_ms=int(start.timestamp() * 1000),
            end_ms=int(end.timestamp() * 1000),
            top_n=2,
        )

        self.assertEqual(objective["label_pairs"], {
            ("2026-08-18", "AAAUSDT"),
            ("2026-08-18", "BBBUSDT"),
            ("2026-08-19", "BBBUSDT"),
            ("2026-08-19", "CCCUSDT"),
        })

        trades = [
            self._trade("AAAUSDT", day1, capture=0.80),
            self._trade("BBBUSDT", day1, capture=0.10),
            self._trade("BBBUSDT", day2, capture=0.75),
            self._trade("AAAUSDT", day2, capture=0.90),
        ]
        report = rb._make_report(
            start=start,
            end=end,
            symbols=symbols,
            timeframes=["15m"],
            trades=trades,
            run_stats=rb.ReplayRunStats(),
            label="test",
            daily_objective=objective,
        )["objective"]

        self.assertEqual(report["captured_pair_count"], 3)
        self.assertEqual(report["label_pair_count"], 4)
        self.assertEqual(report["expected_label_pair_count"], 4)
        self.assertTrue(report["decision_grade"])
        self.assertEqual(report["capture_rate"], 0.75)
        self.assertEqual(report["early_pair_count"], 2)
        self.assertEqual(report["early_capture_rate"], 0.5)
        self.assertEqual(report["objective_trade_count"], 3)
        self.assertEqual(report["eligible_trade_count"], 4)
        self.assertEqual(report["trade_precision"], 0.75)
        self.assertEqual(report["captured_pair_capture_ratio"]["n"], 3)
        self.assertEqual(report["captured_pair_capture_ratio"]["median"], 0.75)

    def test_partial_boundary_days_are_excluded_and_empty_rates_are_unknown(self) -> None:
        tz = ZoneInfo(rb.OBJECTIVE_TZ)
        local_day = date(2026, 8, 18)
        cache = {("AAAUSDT", "15m"): (self._series(local_day, 100, 110), {})}
        start = datetime.combine(local_day, time(12, 0), tzinfo=tz).astimezone(timezone.utc)
        end = datetime.combine(local_day, time(21, 0), tzinfo=tz).astimezone(timezone.utc)
        objective = rb._daily_top_objective(
            ["AAAUSDT"], cache,
            start_ms=int(start.timestamp() * 1000), end_ms=int(end.timestamp() * 1000), top_n=1,
        )
        report = rb._make_report(
            start=start, end=end, symbols=["AAAUSDT"], timeframes=["15m"], trades=[],
            run_stats=rb.ReplayRunStats(), label="test", daily_objective=objective,
        )["objective"]

        self.assertEqual(report["eligible_day_count"], 0)
        self.assertFalse(report["decision_grade"])
        self.assertGreaterEqual(report["excluded_boundary_day_count"], 1)
        self.assertIsNone(report["capture_rate"])
        self.assertIsNone(report["early_capture_rate"])
        self.assertIsNone(report["trade_precision"])


if __name__ == "__main__":
    unittest.main()
