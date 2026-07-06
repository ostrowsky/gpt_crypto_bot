from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import backfill_binance_topn_critic as backfill


def _bar(ts: datetime, open_price: float, close_price: float, quote_volume: float) -> list:
    ts_ms = int(ts.timestamp() * 1000)
    return [ts_ms, str(open_price), str(max(open_price, close_price)), str(min(open_price, close_price)), str(close_price), "0", ts_ms + 1, str(quote_volume)]


class BackfillTopNTests(unittest.TestCase):
    def test_day_performance_uses_local_day_and_volume_filter(self) -> None:
        tz = ZoneInfo("Europe/Budapest")
        history = {
            "AAAUSDT": [
                _bar(datetime(2026, 6, 1, 21, tzinfo=timezone.utc), 99.0, 100.0, 500.0),
                _bar(datetime(2026, 6, 1, 22, tzinfo=timezone.utc), 100.0, 105.0, 600.0),
                _bar(datetime(2026, 6, 2, 21, tzinfo=timezone.utc), 105.0, 110.0, 600.0),
            ],
            "LOWUSDT": [
                _bar(datetime(2026, 6, 1, 22, tzinfo=timezone.utc), 1.0, 2.0, 10.0),
            ],
        }

        rows = backfill._day_performance(history, date(2026, 6, 2), tz, {"AAAUSDT"}, 1_000.0)

        self.assertEqual([row.symbol for row in rows], ["AAAUSDT"])
        self.assertAlmostEqual(rows[0].day_change_pct, 10.0)
        self.assertTrue(rows[0].in_watchlist)


if __name__ == "__main__":
    unittest.main()
