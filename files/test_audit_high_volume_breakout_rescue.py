from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import audit_high_volume_breakout_rescue as audit


class HighVolumeBreakoutRescueAuditTests(unittest.TestCase):
    def test_scanner_keeps_only_frozen_profile_and_deduplicates_day_symbol(self) -> None:
        base = {
            "event": "blocked_learning_label",
            "sym": "AAAUSDT",
            "tf": "15m",
            "mode": "breakout",
            "label_type": "blocked_strong_score_gate",
            "reason": "top-gainer score gate: score 28.50 < 34.00 for 15m breakout",
            "price": 1.0,
            "candidate_score": 125.0,
            "live_score": 28.5,
            "vol_x": 5.5,
        }
        with tempfile.TemporaryDirectory() as td:
            files = Path(td)
            rows = [
                {**base, "ts": "2026-08-02T10:05:00Z"},
                {**base, "ts": "2026-08-02T10:00:00Z"},
                {**base, "sym": "LOWUSDT", "candidate_score": 119.9, "ts": "2026-08-02T10:00:00Z"},
                {**base, "sym": "SLOWUSDT", "vol_x": 4.99, "ts": "2026-08-02T10:00:00Z"},
                {**base, "sym": "TRENDUSDT", "mode": "trend", "ts": "2026-08-02T10:00:00Z"},
            ]
            (files / "bot_events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            (files / "agent_events.jsonl").write_text("", encoding="utf-8")

            candidates, stats = audit._scan_candidates(files, {"2026-08-02"}, audit.AuditConfig())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["symbol"], "AAAUSDT")
        self.assertEqual(candidates[0]["ts"], "2026-08-02T10:00:00Z")
        self.assertEqual(candidates[0]["repeat_count"], 2)
        self.assertEqual(stats["blocked_learning_events"], 5)
        self.assertEqual(stats["raw_profile_events"], 2)

    def test_watch_shadow_gate_rejects_negative_holdout(self) -> None:
        cfg = audit.AuditConfig()
        summary_cfg = audit.common.AuditConfig(
            min_holdout_cases=cfg.min_holdout_cases,
            min_top_opportunities=cfg.min_top_opportunities,
        )
        row = {
            "admission_eligible": 12,
            "earlier_top_opportunities": 1,
            "avg_ret10_net_pct": 0.5,
            "median_ret10_net_pct": 0.2,
            "ret10_positive_rate_pct": 58.0,
        }
        self.assertTrue(audit.common._segment_passes(row, summary_cfg))
        row["median_ret10_net_pct"] = -0.1
        self.assertFalse(audit.common._segment_passes(row, summary_cfg))


if __name__ == "__main__":
    unittest.main()
