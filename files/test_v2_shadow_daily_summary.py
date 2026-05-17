from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path


class TestV2ShadowDailySummary(unittest.TestCase):
    def test_counts_non_bootstrap_events(self) -> None:
        import v2_shadow_daily_summary as mod

        rows = [
            {"ts": "2026-05-17T10:00:00Z", "sym": "AAAUSDT", "state": "emerging_move", "bootstrap": True},
            {"ts": "2026-05-17T10:15:00Z", "sym": "AAAUSDT", "state": "emerging_move", "bootstrap": False},
            {"ts": "2026-05-17T10:30:00Z", "sym": "AAAUSDT", "state": "confirmed_trend", "bootstrap": False},
            {"ts": "2026-05-17T10:45:00Z", "sym": "BBBUSDT", "state": "noise", "bootstrap": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            summary = mod.build_summary(date(2026, 5, 17), path)
        self.assertEqual(summary["events_total"], 3)
        self.assertEqual(summary["upside_discovery_events"], 2)
        self.assertEqual(summary["confirmed_trend_events"], 1)
        self.assertEqual(summary["unique_upside_symbols"], 1)
        self.assertEqual(summary["deescalation_to_noise_events"], 1)
        self.assertEqual(summary["confirmation_ratio"], 0.5)

    def test_empty_day_is_explicit(self) -> None:
        import v2_shadow_daily_summary as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text("", encoding="utf-8")
            summary = mod.build_summary(date(2026, 5, 17), path)
        self.assertEqual(summary["events_total"], 0)
        self.assertEqual(summary["upside_discovery_events"], 0)
        self.assertEqual(summary["confirmation_ratio"], 0.0)
