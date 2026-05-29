from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report_signal_quality_coverage import build_report


class SignalQualityCoverageReportTest(unittest.TestCase):
    def test_reports_missing_series_by_timeframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchlist = root / "watchlist.json"
            cache = root / "cache"
            out_json = root / "out.json"
            out_txt = root / "out.txt"
            watchlist.write_text(json.dumps(["AAAUSDT", "BBBUSDT"]), encoding="utf-8")
            signal = root / "signal_quality_2026-05-28_final.json"
            signal.write_text(
                json.dumps(
                    {
                        "window": {"start": "2026-05-27T22:00:00Z", "end": "2026-05-28T22:00:00Z"},
                        "scope": {"timeframes": ["15m", "1h"], "requested_symbol_filter": None},
                        "coverage": {"status": "partial", "events_loaded": 3, "paired_trades": 1},
                    }
                ),
                encoding="utf-8",
            )
            # max tf is 1h, so fetch window is start-4h to end+32h
            # because the evaluator uses the maximum configured horizon bars.
            start = 1779919200000 - 4 * 3_600_000
            end = 1780005600000 + 32 * 3_600_000
            cache.mkdir()
            (cache / f"AAAUSDT_15m_{start}_{end}.json").write_text("[{\"t\":1}]", encoding="utf-8")
            (cache / f"AAAUSDT_1h_{start}_{end}.json").write_text("[{\"t\":1}]", encoding="utf-8")
            (cache / f"BBBUSDT_15m_{start}_{end}.json").write_text("[]", encoding="utf-8")

            report = build_report(
                signal_report=signal,
                watchlist_path=watchlist,
                cache_dir=cache,
                output_json=out_json,
                output_txt=out_txt,
            )

            self.assertEqual(report["requested_series"], 4)
            self.assertEqual(report["loaded_series_from_cache"], 2)
            self.assertEqual(report["missing_series_count"], 2)
            self.assertEqual(report["by_timeframe"]["15m"]["missing"], 1)
            self.assertEqual(report["by_timeframe"]["1h"]["missing"], 1)
            self.assertIn("metric_affecting_possible", report["assessment"])
            self.assertTrue(out_json.exists())
            self.assertTrue(out_txt.exists())


if __name__ == "__main__":
    unittest.main()
