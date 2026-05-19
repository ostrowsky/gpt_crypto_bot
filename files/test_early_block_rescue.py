from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_early_block_rescue import _decision, build


class EarlyBlockRescueAuditTest(unittest.TestCase):
    def test_build_selects_early_blocked_missed_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "top_gainer_critic_2026-05-01_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-05-01",
                        "watchlist_top_gainers": [
                            {
                                "symbol": "AAAUSDT",
                                "status": "blocked_rule",
                                "first_block_time": "01:15",
                                "first_block_reason_code": "agent_leader_filter",
                                "blocked_count": 99,
                                "opportunity_from_first_block_pct": 12.5,
                            },
                            {
                                "symbol": "BBBUSDT",
                                "status": "bought",
                                "first_block_time": "02:00",
                                "first_block_reason_code": "agent_mode_disabled",
                                "blocked_count": 60,
                                "capture_ratio_at_entry": 0.0,
                                "opportunity_from_first_block_pct": 8.0,
                                "opportunity_from_entry_pct": -2.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = build(root, root / "out.json")
            self.assertEqual(payload["source_rows"], 2)
            self.assertGreater(payload["best_variant"]["proxy_gain_pct"], 0)
            self.assertTrue(payload["best_variant"]["top_examples"])

    def test_decision_requires_enough_missed_winners(self) -> None:
        self.assertEqual(_decision(None), "no_candidate")
        weak = {"proxy_gain_pct": 10.0, "rescued_missed_winners": 1, "non_positive_cases": 0, "selected_cases": 1}
        self.assertEqual(_decision(weak), "diagnostic_only_insufficient_proxy_gain")


if __name__ == "__main__":
    unittest.main()
