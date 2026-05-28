from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestV2EarlyAdmissionFullBacktest(unittest.TestCase):
    def test_backtest_counts_false_favorable_and_top_recall(self) -> None:
        import backtest_v2_early_admission as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"; reports.mkdir()
            hist = root / "hist"; events = root / "events.jsonl"; watch = root / "watchlist.json"
            watch.write_text(json.dumps(["AAAUSDT", "BBBUSDT"]), encoding="utf-8")
            report = {
                "target_day_local": "2026-05-01",
                "summary": {"watchlist_top_denominator": "exchange_top_filtered_to_watchlist"},
                "watchlist_top_gainers": [
                    {"symbol": "AAAUSDT", "status": "bought", "day_open": 100, "day_close": 110, "day_high": 112, "first_entry_price": 108, "latest_exit_pnl_pct": 1.0},
                ],
            }
            (reports / "top_gainer_critic_2026-05-01_final.json").write_text(json.dumps(report), encoding="utf-8")
            rows = [
                {"ts": "2026-05-01T01:00:00Z", "sym": "AAAUSDT", "state": "emerging_move", "bootstrap": False, "features": {"price": 101}},
                {"ts": "2026-05-01T02:00:00Z", "sym": "AAAUSDT", "state": "confirmed_trend", "bootstrap": False, "features": {"price": 103}},
                {"ts": "2026-05-01T01:00:00Z", "sym": "BBBUSDT", "state": "emerging_move", "bootstrap": False, "features": {"price": 50}},
            ]
            events.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            for sym, close in [("AAAUSDT", 110), ("BBBUSDT", 49)]:
                d = hist / sym; d.mkdir(parents=True)
                (d / "15m.jsonl").write_text(json.dumps({"open_ts_ms": 1777597200000, "open": close, "high": close + 2, "low": close - 1, "close": close}) + "\n", encoding="utf-8")
            payload = mod.run_backtest(reports_dir=reports, events_file=events, watchlist_file=watch, history_root=hist, save=False)
        first = payload["policies"]["v2_first_upside"]
        confirmed = payload["policies"]["v2_first_confirmed"]
        self.assertEqual(first["n"], 2)
        self.assertEqual(first["top_count"], 1)
        self.assertEqual(first["top_precision_pct"], 50.0)
        self.assertEqual(first["false_favorable_count"], 1)
        self.assertEqual(confirmed["n"], 1)
        self.assertEqual(confirmed["top_precision_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
