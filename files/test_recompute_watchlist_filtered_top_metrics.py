from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestRecomputeWatchlistFilteredTopMetrics(unittest.TestCase):
    def test_recompute_report_uses_exchange_top_filtered_to_watchlist(self) -> None:
        import recompute_watchlist_filtered_top_metrics as mod

        exchange = []
        for idx in range(10):
            sym = {1: "W1USDT", 3: "W2USDT", 6: "W3USDT", 8: "W4USDT"}.get(idx, f"X{idx}USDT")
            exchange.append({
                "symbol": sym,
                "day_change_pct": 20 - idx,
                "in_watchlist": sym.startswith("W"),
                "status": "bought" if sym in {"W1USDT", "W3USDT"} else ("not_in_watchlist" if sym.startswith("X") else "blocked_rule"),
                "blocked_count": 1 if sym in {"W2USDT", "W4USDT"} else 0,
                "blocked_reason_counts": {"top_gainer_score_gate": 1} if sym in {"W2USDT", "W4USDT"} else {},
                "capture_ratio": 0.4 if sym == "W1USDT" else 0.1,
            })
        old_watchlist = [row for row in exchange if row["in_watchlist"]] + [{
            "symbol": "W5USDT",
            "day_change_pct": -5,
            "in_watchlist": True,
            "status": "blocked_rule",
            "blocked_count": 1,
            "blocked_reason_counts": {"agent_mode_disabled": 1},
        }]
        report = {
            "target_day_local": "2026-05-27",
            "phase": "final",
            "summary": {"watchlist_top_count": 5, "watchlist_top_bought": 2},
            "exchange_top_gainers": exchange,
            "watchlist_top_gainers": old_watchlist,
        }
        new_report, delta = mod.recompute_report(report)
        summary = new_report["summary"]
        self.assertEqual(summary["watchlist_top_denominator"], "exchange_top_filtered_to_watchlist")
        self.assertEqual(summary["watchlist_top_count"], 4)
        self.assertEqual(summary["watchlist_top_bought"], 2)
        self.assertEqual(summary["watchlist_top_missed"], 2)
        self.assertEqual(summary["watchlist_top_capture_rate_pct"], 50.0)
        self.assertEqual(summary["watchlist_universe_top_count"], 5)
        self.assertEqual([row["symbol"] for row in new_report["watchlist_top_gainers"]], ["W1USDT", "W2USDT", "W3USDT", "W4USDT"])
        self.assertEqual(new_report["watchlist_universe_top_gainers"][-1]["symbol"], "W5USDT")
        self.assertEqual(delta["before"]["watchlist_top_count"], 5)
        self.assertEqual(delta["after"]["watchlist_top_count"], 4)

    def test_recompute_reports_updates_history_when_write_enabled(self) -> None:
        import recompute_watchlist_filtered_top_metrics as mod

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            report_path = reports / "top_gainer_critic_2026-05-27_final.json"
            report = {
                "target_day_local": "2026-05-27",
                "phase": "final",
                "summary": {"watchlist_top_count": 2, "watchlist_top_bought": 0},
                "exchange_top_gainers": [
                    {"symbol": "XUSDT", "in_watchlist": False, "status": "not_in_watchlist"},
                    {"symbol": "WUSDT", "in_watchlist": True, "status": "bought", "capture_ratio": 0.5},
                ],
                "watchlist_top_gainers": [
                    {"symbol": "WUSDT", "in_watchlist": True, "status": "bought", "capture_ratio": 0.5},
                    {"symbol": "OLDUSDT", "in_watchlist": True, "status": "blocked_rule", "blocked_count": 1},
                ],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            history = reports / "top_gainer_critic_history.jsonl"
            history.write_text(json.dumps({"target_day_local": "2026-05-27", "phase": "final", "summary": report["summary"]}) + "\n", encoding="utf-8")
            result = mod.recompute_reports(reports, write=True)
            updated = json.loads(report_path.read_text(encoding="utf-8"))
            hist = json.loads(history.read_text(encoding="utf-8").strip())
        self.assertEqual(result["reports_changed"], 1)
        self.assertEqual(updated["summary"]["watchlist_top_count"], 1)
        self.assertEqual(updated["summary"]["watchlist_top_capture_rate_pct"], 100.0)
        self.assertEqual(hist["summary"]["watchlist_top_count"], 1)


if __name__ == "__main__":
    unittest.main()
