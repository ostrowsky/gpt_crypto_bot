from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestV2HistoryStore(unittest.TestCase):
    def _bar(self, ts: int, close: float = 1.05):
        from v2.history import CanonicalBar

        return CanonicalBar(
            symbol="AAAUSDT",
            timeframe="15m",
            open_ts_ms=ts,
            open=1.0,
            high=1.1,
            low=0.9,
            close=close,
            volume=100.0,
        )

    def test_roundtrip_and_metadata(self) -> None:
        from v2.history_store import LocalHistoryStore

        base = 1_779_019_200_000
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalHistoryStore(Path(tmp))
            store.upsert(
                "AAAUSDT",
                "15m",
                [self._bar(base), self._bar(base + 15 * 60 * 1000)],
                source="test_source",
            )
            loaded = store.load("AAAUSDT", "15m")
            meta = store.metadata("AAAUSDT", "15m")
            self.assertTrue(loaded.continuity.is_contiguous)
            self.assertEqual(len(loaded.bars), 2)
            self.assertEqual(meta.source, "test_source")
            self.assertEqual(store.keys(), (("AAAUSDT", "15m"),))

    def test_upsert_dedupes_by_timestamp(self) -> None:
        from v2.history_store import LocalHistoryStore

        base = 1_779_019_200_000
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalHistoryStore(Path(tmp))
            store.upsert("AAAUSDT", "15m", [self._bar(base)], source="first")
            store.upsert("AAAUSDT", "15m", [self._bar(base, close=1.2)], source="second")
            loaded = store.load("AAAUSDT", "15m")
            self.assertEqual(len(loaded.bars), 1)
            self.assertAlmostEqual(loaded.bars[0].close, 1.2)


if __name__ == "__main__":
    unittest.main()
