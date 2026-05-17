from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


class TestV2HistoryPopulation(unittest.IsolatedAsyncioTestCase):
    async def test_population_report_marks_valid_symbols(self) -> None:
        from v2.history import CanonicalBar
        from v2.history_population import build_requests, populate_requests
        from v2.history_store import LocalHistoryStore

        end = datetime(2026, 5, 17, tzinfo=timezone.utc)
        requests = build_requests(["AAAUSDT"], ["15m", "1h"], days=1, end=end)

        async def fake_fetcher(_session, request):
            step = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}[request.timeframe]
            return tuple(
                CanonicalBar(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open_ts_ms=ts,
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                )
                for ts in range(request.start_ms, request.end_ms, step)
            )

        with tempfile.TemporaryDirectory() as tmp:
            report = await populate_requests(
                LocalHistoryStore(Path(tmp)),
                requests,
                target_days=1,
                fetcher=fake_fetcher,
            )
        self.assertEqual(report.valid_symbols, ("AAAUSDT",))
        self.assertTrue(report.coverage_passed)

    async def test_partial_window_is_not_valid_coverage(self) -> None:
        from v2.history import CanonicalBar
        from v2.history_population import build_requests, populate_requests
        from v2.history_store import LocalHistoryStore

        end = datetime(2026, 5, 17, tzinfo=timezone.utc)
        requests = build_requests(["AAAUSDT"], ["15m"], days=1, end=end)

        async def fake_fetcher(_session, request):
            return (
                CanonicalBar(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open_ts_ms=request.start_ms,
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            report = await populate_requests(
                LocalHistoryStore(Path(tmp)),
                requests,
                target_days=1,
                fetcher=fake_fetcher,
            )
        self.assertEqual(report.valid_symbols, tuple())
        self.assertFalse(report.coverage_passed)

    def test_build_requests_covers_symbol_timeframe_grid(self) -> None:
        from v2.history_population import build_requests

        end = datetime(2026, 5, 17, tzinfo=timezone.utc)
        requests = build_requests(["AAAUSDT", "BBBUSDT"], ["15m", "1h"], days=3, end=end)
        self.assertEqual(len(requests), 4)


if __name__ == "__main__":
    unittest.main()
