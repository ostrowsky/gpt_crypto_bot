from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_research_universe_impact as audit


class ResearchUniverseImpactReportTest(unittest.TestCase):
    def test_build_report_separates_trade_watchlist_from_research_universe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"
            reports.mkdir()
            watchlist = root / "watchlist.json"
            watchlist.write_text(json.dumps(["AAAUSDT", "BBBUSDT"]), encoding="utf-8")
            (reports / "top_gainer_critic_2026-06-01_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-06-01",
                        "summary": {
                            "exchange_top_count": 4,
                            "exchange_top_in_watchlist": 1,
                            "watchlist_top_count": 1,
                            "watchlist_top_bought": 1,
                            "watchlist_top_early_captured": 1,
                            "bot_false_positive_buys": 0,
                        },
                        "exchange_top_gainers": [
                            {"symbol": "AAAUSDT", "day_change_pct": 10, "quote_volume_24h": 10_000_000},
                            {"symbol": "币安人生USDT", "day_change_pct": 9.5, "quote_volume_24h": 9_000_000},
                            {"symbol": "XXXUSDT", "day_change_pct": 9, "quote_volume_24h": 8_000_000},
                            {"symbol": "YYYUSDT", "day_change_pct": 8, "quote_volume_24h": 7_000_000},
                        ],
                        "watchlist_top_gainers": [{"symbol": "AAAUSDT", "status": "bought"}],
                    }
                ),
                encoding="utf-8",
            )
            (reports / "top_gainer_critic_2026-06-02_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-06-02",
                        "summary": {
                            "exchange_top_count": 4,
                            "exchange_top_in_watchlist": 1,
                            "watchlist_top_count": 1,
                            "watchlist_top_bought": 0,
                            "watchlist_top_early_captured": 0,
                            "bot_false_positive_buys": 1,
                        },
                        "exchange_top_gainers": [
                            {"symbol": "BBBUSDT", "day_change_pct": 11, "quote_volume_24h": 9_000_000},
                            {"symbol": "币安人生USDT", "day_change_pct": 10.5, "quote_volume_24h": 9_000_000},
                            {"symbol": "XXXUSDT", "day_change_pct": 10, "quote_volume_24h": 8_000_000},
                            {"symbol": "YYYUSDT", "day_change_pct": 9, "quote_volume_24h": 7_000_000},
                        ],
                        "watchlist_top_gainers": [{"symbol": "BBBUSDT", "status": "no_signal"}],
                    }
                ),
                encoding="utf-8",
            )

            report = audit.build_report(
                reports_dir=reports,
                watchlist_file=watchlist,
                min_repeats_for_promotion=2,
                output_json=root / "out.json",
                output_txt=root / "out.txt",
            )

        summary = report["summary"]
        self.assertEqual(summary["exchange_top_events"], 8)
        self.assertEqual(summary["exchange_top_in_watchlist_events"], 2)
        self.assertEqual(summary["exchange_top_outside_watchlist_events"], 6)
        self.assertEqual(summary["watchlist_capture_pct"], 50.0)
        self.assertEqual(summary["watchlist_early_capture_pct"], 50.0)
        self.assertEqual(summary["exchange_diagnostic_capture_pct"], 12.5)
        self.assertEqual(report["positive_label_expansion_factor"], 4.0)
        self.assertEqual(report["recommendation"]["decision"], "add_research_universe_shadow_layer")
        self.assertEqual([row["symbol"] for row in report["promotion_candidates"]], ["XXXUSDT", "YYYUSDT"])
        self.assertIn("币安人生USDT", [row["symbol"] for row in report["top_outside_watchlist_symbols"]])
        self.assertNotIn("币安人生USDT", [row["symbol"] for row in report["promotion_candidates"]])
        self.assertIn("live watchlist unchanged", audit.render_text(report))


if __name__ == "__main__":
    unittest.main()
