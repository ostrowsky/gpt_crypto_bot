from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from report_signal_quality_coverage import _exchange_statuses, build_report


class SignalQualityCoverageReportTest(unittest.TestCase):
    def test_reports_missing_series_by_timeframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchlist = root / "watchlist.json"
            cache = root / "cache"
            exchange_status = root / "exchange_status.json"
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
            exchange_status.write_text(json.dumps({"BBBUSDT": "BREAK"}), encoding="utf-8")

            report = build_report(
                signal_report=signal,
                watchlist_path=watchlist,
                cache_dir=cache,
                exchange_status_cache=exchange_status,
                check_exchange_status=False,
                output_json=out_json,
                output_txt=out_txt,
            )

            self.assertEqual(report["requested_series"], 4)
            self.assertEqual(report["loaded_series_from_cache"], 2)
            self.assertEqual(report["missing_series_count"], 2)
            self.assertEqual(report["active_requested_series"], 2)
            self.assertEqual(report["active_loaded_series"], 2)
            self.assertEqual(report["active_missing_series_count"], 0)
            self.assertEqual(report["inactive_excluded_series_count"], 2)
            self.assertEqual(report["by_timeframe"]["15m"]["missing"], 1)
            self.assertEqual(report["by_timeframe"]["1h"]["missing"], 1)
            self.assertEqual(report["missing_symbol_status_counts"], {"BREAK": 2})
            self.assertEqual(report["assessment"], "partial_safe_inactive_symbols_only")
            self.assertTrue(out_json.exists())
            self.assertTrue(out_txt.exists())

    def test_refreshes_legacy_status_cache_before_trusting_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "exchange_status.json"
            cache.write_text(json.dumps({"TONUSDT": "TRADING"}), encoding="utf-8")
            now = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)

            with patch(
                "report_signal_quality_coverage._fetch_exchange_statuses",
                return_value=({"TONUSDT": "BREAK"}, ""),
            ) as fetch:
                statuses, provenance = _exchange_statuses(
                    ["TONUSDT"], cache, enabled=True, now=now
                )

            self.assertEqual(statuses, {"TONUSDT": "BREAK"})
            self.assertEqual(provenance["freshness"], "fresh")
            self.assertTrue(provenance["trusted"])
            fetch.assert_called_once_with(["TONUSDT"])
            stored = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(stored["schema_version"], 2)
            self.assertEqual(stored["statuses"]["TONUSDT"], "BREAK")

    def test_failed_refresh_does_not_trust_stale_break_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchlist = root / "watchlist.json"
            candle_cache = root / "cache"
            status_cache = root / "exchange_status.json"
            watchlist.write_text(json.dumps(["TONUSDT"]), encoding="utf-8")
            status_cache.write_text(json.dumps({"TONUSDT": "BREAK"}), encoding="utf-8")
            signal = root / "signal_quality_2026-08-02_final.json"
            signal.write_text(json.dumps({
                "window": {"start": "2026-08-01T22:00:00Z", "end": "2026-08-02T22:00:00Z"},
                "scope": {"timeframes": ["15m"]},
                "coverage": {"status": "partial", "events_loaded": 3, "paired_trades": 1},
            }), encoding="utf-8")

            with patch(
                "report_signal_quality_coverage._fetch_exchange_statuses",
                return_value=({}, "exchange_info_failed: timeout"),
            ):
                report = build_report(
                    signal_report=signal,
                    watchlist_path=watchlist,
                    cache_dir=candle_cache,
                    exchange_status_cache=status_cache,
                    save=False,
                )

            self.assertEqual(
                report["assessment"],
                "metric_affecting_possible: candle series missing while events/trades exist",
            )
            self.assertEqual(
                report["exchange_status_provenance"]["freshness"],
                "stale_refresh_failed",
            )
            self.assertFalse(report["exchange_status_provenance"]["trusted"])


if __name__ == "__main__":
    unittest.main()
