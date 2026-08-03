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

    def test_batch_labeling_quarantines_bad_row_without_blocking_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "research.jsonl"
            quarantine = Path(td) / "bad.jsonl"
            first = {
                "id": "AAAUSDT_15m_1000",
                "sym": "AAAUSDT",
                "tf": "15m",
                "bar_ts": 1000,
                "labels": {"ret_3": None, "ret_5": None, "ret_10": None},
            }
            second = {
                "id": "BBBUSDT_15m_1000",
                "sym": "BBBUSDT",
                "tf": "15m",
                "bar_ts": 1000,
                "labels": {"ret_3": None, "ret_5": None, "ret_10": None},
            }
            dataset.write_text(
                json.dumps(first) + "\n{not valid json\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            market_data = {
                ("AAAUSDT", "15m"): {
                    "t": np.array([1000, 2000, 3000, 4000], dtype=np.int64),
                    "c": np.array([10.0, 10.1, 10.2, 11.0], dtype=float),
                },
                ("BBBUSDT", "15m"): {
                    "t": np.array([1000, 2000, 3000, 4000], dtype=np.int64),
                    "c": np.array([20.0, 20.2, 20.4, 18.0], dtype=float),
                },
            }

            result = collector._fill_mature_labels_batch(
                dataset,
                market_data,
                quarantine_file=quarantine,
            )
            updated = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
            quarantined = json.loads(quarantine.read_text(encoding="utf-8").strip())

        self.assertEqual(result["labels_updated"], 2)
        self.assertEqual(result["malformed_rows_quarantined"], 1)
        self.assertEqual([row["id"] for row in updated], [first["id"], second["id"]])
        self.assertEqual(updated[0]["labels"]["ret_3"], 10.0)
        self.assertEqual(updated[1]["labels"]["ret_3"], -10.0)
        self.assertEqual(quarantined["line_number"], 2)
        self.assertEqual(quarantined["raw"], "{not valid json")

    def test_batch_labeling_preserves_dataset_when_nothing_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "research.jsonl"
            original = '{"id":"done","sym":"AAAUSDT","tf":"15m","bar_ts":1000,"labels":{"ret_3":1}}\n'
            dataset.write_text(original, encoding="utf-8")

            result = collector._fill_mature_labels_batch(dataset, {})

            self.assertEqual(result["labels_updated"], 0)
            self.assertEqual(result["malformed_rows_quarantined"], 0)
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)
            self.assertFalse(dataset.with_name("research.jsonl.labels.tmp").exists())

    def test_incomplete_label_ranges_cover_maximum_pending_period(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "research.jsonl"
            rows = [
                {"sym": "AAAUSDT", "tf": "15m", "bar_ts": 3000, "labels": {"ret_10": None}},
                {"sym": "AAAUSDT", "tf": "15m", "bar_ts": 1000, "labels": {"ret_10": None}},
                {
                    "sym": "AAAUSDT",
                    "tf": "15m",
                    "bar_ts": 5000,
                    "labels": {"ret_3": 1, "ret_5": 1, "ret_10": 1},
                },
                {"sym": "BBBUSDT", "tf": "1h", "bar_ts": 2000, "labels": {}},
            ]
            dataset.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\nnot-json\n",
                encoding="utf-8",
            )

            ranges = collector._incomplete_label_ranges(dataset)

        self.assertEqual(
            ranges[("AAAUSDT", "15m")],
            {"min_bar_ts": 1000, "max_bar_ts": 3000, "rows": 2},
        )
        self.assertEqual(ranges[("BBBUSDT", "1h")]["rows"], 1)
        self.assertEqual(collector._timeframe_ms("15m"), 900_000)
        self.assertEqual(collector._timeframe_ms("1h"), 3_600_000)

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
