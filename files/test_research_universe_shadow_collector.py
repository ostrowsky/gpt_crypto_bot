from __future__ import annotations

import json
import tempfile
import unittest
import asyncio
from pathlib import Path

import numpy as np

import research_universe_shadow_collector as collector


class ResearchUniverseShadowCollectorTest(unittest.TestCase):
    def test_research_symbol_filter_rejects_non_tradeable_noise(self) -> None:
        self.assertTrue(collector._is_research_symbol("RIFUSDT"))
        self.assertFalse(collector._is_research_symbol("币安人生USDT"))
        self.assertFalse(collector._is_research_symbol("USDCUSDT"))
        self.assertFalse(collector._is_research_symbol("ETHBTC"))
        self.assertFalse(collector._is_research_symbol("ABCUPUSDT"))

    def test_fill_mature_labels_updates_only_available_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "research.jsonl"
            row = {
                "id": "AAAUSDT_15m_1000",
                "sym": "AAAUSDT",
                "tf": "15m",
                "bar_ts": 1000,
                "labels": {"ret_3": None, "ret_5": None, "ret_10": None},
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            data = {
                "t": np.array([1000, 2000, 3000, 4000, 5000, 6000], dtype=np.int64),
                "c": np.array([10.0, 10.2, 10.3, 11.0, 10.8, 12.0], dtype=float),
            }

            changed = collector._fill_mature_labels_for_symbol(dataset, "AAAUSDT", "15m", data)
            updated = json.loads(dataset.read_text(encoding="utf-8").strip())

        self.assertEqual(changed, 2)
        self.assertEqual(updated["labels"]["ret_3"], 10.0)
        self.assertEqual(updated["labels"]["ret_5"], 20.0)
        self.assertIsNone(updated["labels"]["ret_10"])

    def test_build_research_universe_ranks_by_quote_volume(self) -> None:
        async def fake_symbols(_session):
            return ["AAAUSDT", "BBBUSDT", "USDCUSDT", "币安人生USDT"]

        async def fake_tickers(_session):
            return {
                "AAAUSDT": {"quoteVolume": "1000000", "priceChangePercent": "5"},
                "BBBUSDT": {"quoteVolume": "3000000", "priceChangePercent": "1"},
                "USDCUSDT": {"quoteVolume": "999999999", "priceChangePercent": "0"},
                "币安人生USDT": {"quoteVolume": "999999999", "priceChangePercent": "100"},
            }

        original_symbols = collector._fetch_exchange_symbols
        original_tickers = collector._fetch_all_tickers
        try:
            collector._fetch_exchange_symbols = fake_symbols
            collector._fetch_all_tickers = fake_tickers
            universe = asyncio.run(
                collector.build_research_universe(None, max_symbols=10, min_quote_volume=1_000_000)
            )
        finally:
            collector._fetch_exchange_symbols = original_symbols
            collector._fetch_all_tickers = original_tickers

        self.assertEqual([item.symbol for item in universe], ["BBBUSDT", "AAAUSDT"])
        self.assertEqual(universe[0].rank_24h, 1)
        self.assertEqual(universe[0].quote_volume_24h, 3_000_000)


if __name__ == "__main__":
    unittest.main()
