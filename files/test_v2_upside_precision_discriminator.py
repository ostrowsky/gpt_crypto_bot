from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path


class TestV2UpsidePrecisionDiscriminator(unittest.TestCase):
    def test_build_dataset_uses_first_upside_and_joins_outcome(self) -> None:
        import audit_v2_upside_precision_discriminator as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            reports = root / "reports"
            reports.mkdir()
            rows = [
                {"ts": "2026-05-26T07:45:00Z", "sym": "AAAUSDT", "state": "noise", "bootstrap": False},
                {"ts": "2026-05-26T08:00:00Z", "sym": "AAAUSDT", "previous_state": "noise", "state": "emerging_move", "action": "elevate_priority", "confidence": 0.64, "bootstrap": False, "features": {"adx": 22, "rsi": 61, "slope": 0.2, "vol_x": 2.1, "daily_range": 4.0, "price": 101, "ema20": 100}},
                {"ts": "2026-05-26T08:15:00Z", "sym": "AAAUSDT", "previous_state": "emerging_move", "state": "confirmed_trend", "bootstrap": False, "features": {}},
                {"ts": "2026-05-26T09:00:00Z", "sym": "CCCUSDT", "previous_state": "noise", "state": "emerging_move", "action": "elevate_priority", "confidence": 0.64, "bootstrap": False, "features": {"adx": 10, "rsi": 52, "slope": 0.01, "vol_x": 0.5, "daily_range": 1.0, "price": 99, "ema20": 100}},
            ]
            events.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            top = {"watchlist_top_gainers": [{"symbol": "AAAUSDT", "status": "bought", "day_change_pct": 9.0, "capture_ratio_at_entry": 0.4}]}
            (reports / "top_gainer_critic_2026-05-26_final.json").write_text(json.dumps(top), encoding="utf-8")

            dataset, coverage = mod.build_dataset(date(2026, 5, 26), date(2026, 5, 26), events_file=events, reports_dir=reports)

        self.assertEqual(coverage["rows"], 2)
        aaa = next(row for row in dataset if row.symbol == "AAAUSDT")
        ccc = next(row for row in dataset if row.symbol == "CCCUSDT")
        self.assertTrue(aaa.top_mover)
        self.assertTrue(aaa.bought_by_v1)
        self.assertEqual(aaa.later_confirmed_events, 1)
        self.assertAlmostEqual(aaa.price_vs_ema20_pct or 0, 1.0)
        self.assertFalse(ccc.top_mover)

    def test_rank_slices_prefers_precision_with_support(self) -> None:
        import audit_v2_upside_precision_discriminator as mod

        rows = [
            mod.FirstUpsideRow(day="2026-05-26", symbol="A", ts="", state="emerging_move", previous_state="noise", action="", confidence=0.64, reason="", slope=0.2, vol_x=2.0, top_mover=True),
            mod.FirstUpsideRow(day="2026-05-26", symbol="B", ts="", state="emerging_move", previous_state="noise", action="", confidence=0.64, reason="", slope=0.3, vol_x=2.5, top_mover=True),
            mod.FirstUpsideRow(day="2026-05-26", symbol="C", ts="", state="emerging_move", previous_state="noise", action="", confidence=0.64, reason="", slope=0.4, vol_x=3.0, top_mover=True),
            mod.FirstUpsideRow(day="2026-05-26", symbol="D", ts="", state="emerging_move", previous_state="noise", action="", confidence=0.64, reason="", slope=0.0, vol_x=0.5, top_mover=False),
            mod.FirstUpsideRow(day="2026-05-26", symbol="E", ts="", state="emerging_move", previous_state="noise", action="", confidence=0.64, reason="", slope=0.0, vol_x=0.5, top_mover=False),
        ]
        slices = mod.rank_slices(rows)
        by_name = {item.name: item for item in slices}
        self.assertEqual(by_name["slope_and_volume"].precision_pct, 100.0)
        self.assertEqual(by_name["slope_and_volume"].recall_pct, 100.0)
        self.assertLess(slices.index(by_name["slope_and_volume"]), slices.index(by_name["confidence_ge_0_64"]))

    def test_run_audit_marks_missing_days_partial(self) -> None:
        import audit_v2_upside_precision_discriminator as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"ts": "2026-05-26T08:00:00Z", "sym": "AAAUSDT", "state": "emerging_move", "features": {}}), encoding="utf-8")
            report = mod.run_audit(
                target_day=date(2026, 5, 26),
                days=1,
                events_file=events,
                reports_dir=root / "reports",
                output_json=root / "out.json",
                output_txt=root / "out.txt",
            )
        self.assertEqual(report["status"], "partial")
        self.assertIn("partial_only", report["decision"])


if __name__ == "__main__":
    unittest.main()
