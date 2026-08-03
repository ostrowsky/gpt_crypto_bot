from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import audit_score_gate_near_miss_band as audit


class ScoreGateNearMissBandAuditTests(unittest.TestCase):
    def test_scan_deduplicates_repeated_band_events_and_keeps_first(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            files = Path(td)
            rows = [
                {"event": "blocked", "sym": "AAAUSDT", "price": 1.0, "signal_type": "top_gainer_score_gate", "reason": "top-gainer score gate: score 32.50 < 34.00 for 15m trend", "ts": "2026-08-02T10:05:00Z"},
                {"event": "blocked", "sym": "AAAUSDT", "price": 0.9, "signal_type": "top_gainer_score_gate", "reason": "top-gainer score gate: score 32.25 < 34.00 for 15m trend", "ts": "2026-08-02T10:00:00Z"},
                {"event": "blocked", "sym": "BBBUSDT", "price": 1.0, "signal_type": "top_gainer_score_gate", "reason": "top-gainer score gate: score 31.99 < 34.00 for 15m trend", "ts": "2026-08-02T10:00:00Z"},
            ]
            (files / "bot_events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (files / "agent_events.jsonl").write_text("", encoding="utf-8")

            candidates, stats = audit._scan_candidates(files, {"2026-08-02"}, audit.AuditConfig())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["ts"], "2026-08-02T10:00:00Z")
        self.assertEqual(candidates[0]["repeat_count"], 2)
        self.assertEqual(stats["band_events"], 2)

    def test_watch_gate_requires_positive_holdout_and_recent_top_opportunity(self) -> None:
        passing = {
            "admission_eligible": 12,
            "earlier_top_opportunities": 1,
            "avg_ret10_net_pct": 0.5,
            "median_ret10_net_pct": 0.2,
            "ret10_positive_rate_pct": 58.0,
        }
        segments = {name: dict(passing) for name in ("all_mature", "holdout", "recent_stability")}

        self.assertEqual(
            audit._decision(segments, audit.AuditConfig()),
            "advance_score_32_33_to_watch_shadow",
        )
        segments["recent_stability"]["median_ret10_net_pct"] = -0.1
        self.assertEqual(
            audit._decision(segments, audit.AuditConfig()),
            "reject_score_32_33_watch_shadow_gate",
        )

    def test_summary_excludes_capacity_full_and_already_bought_candidates(self) -> None:
        rows = [
            {"capacity_available": True, "ret10_net_pct": 1.0, "mae10_pct": -0.5, "is_watchlist_top": True, "already_bought_before_band": False},
            {"capacity_available": False, "ret10_net_pct": 10.0, "mae10_pct": -0.1, "is_watchlist_top": False, "already_bought_before_band": False},
            {"capacity_available": True, "ret10_net_pct": 5.0, "mae10_pct": -0.1, "is_watchlist_top": True, "already_bought_before_band": True},
        ]

        summary = audit._summary(rows, audit.AuditConfig())

        self.assertEqual(summary["capacity_eligible"], 2)
        self.assertEqual(summary["admission_eligible"], 1)
        self.assertEqual(summary["avg_ret10_net_pct"], 1.0)
        self.assertEqual(summary["earlier_top_opportunities"], 1)

    def test_candle_cache_skips_null_horizon_close(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            start = 1_000_000
            bar_ms = audit.BAR_MS["15m"]
            candles = [
                {"t": start + index * bar_ms, "c": 1.0, "h": 1.0, "l": 1.0}
                for index in range(11)
            ]
            candles[10]["c"] = None
            end = candles[-1]["t"]
            path = cache_dir / f"AAAUSDT_15m_{start}_{end}.json"
            path.write_text(json.dumps(candles), encoding="utf-8")
            candidate = {
                "symbol": "AAAUSDT",
                "tf": "15m",
                "ts_ms": start,
                "price": 1.0,
            }

            result = audit.CandleCache(cache_dir).forward_metrics(candidate, audit.AuditConfig())

        self.assertEqual(result["label_status"], "missing_candles")
        self.assertIsNone(result["ret10_net_pct"])


if __name__ == "__main__":
    unittest.main()
