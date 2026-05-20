from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import blocked_winner_focus_audit as audit


class BlockedWinnerFocusAuditTest(unittest.TestCase):
    def test_builds_compact_focus_rows_from_critic_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "top_gainer_critic_2026-05-20_midday.json"
            path.write_text(json.dumps({
                "target_day_local": "2026-05-20",
                "phase": "midday",
                "summary": {"watchlist_top_bought": 1, "watchlist_top_count": 2},
                "watchlist_top_gainers": [
                    {
                        "symbol": "STRKUSDT",
                        "status": "bought",
                        "day_change_pct": 5.25,
                        "entries_count": 1,
                        "blocked_count": 3,
                        "blocked_reason_counts": {"symbol_cooldown": 2, "agent_mode_disabled": 1},
                        "first_entry_time": "09:30",
                        "first_entry_mode": "strong_trend",
                        "first_entry_price": 0.0417,
                        "capture_ratio_at_entry": 0.19,
                        "latest_exit_time": "10:30",
                        "latest_exit_pnl_pct": 0.48,
                    },
                    {
                        "symbol": "TIAUSDT",
                        "status": "blocked_rule",
                        "day_change_pct": 3.06,
                        "entries_count": 0,
                        "blocked_count": 5,
                        "blocked_reason_counts": {"agent_mode_disabled": 4, "top_gainer_score_gate": 1},
                        "first_block_time": "08:15",
                        "first_block_reason_code": "agent_mode_disabled",
                        "latest_block_reason_code": "agent_mode_disabled",
                        "latest_block_reason": "agent mode disabled: alignment",
                    },
                ],
            }), encoding="utf-8")

            result = audit.build_focus_audit(path, ["STRK/USDT", "TIAUSDT", "DOGSUSDT"])
            self.assertEqual(result["focus_symbols"][0]["symbol"], "STRKUSDT")
            self.assertEqual(result["focus_symbols"][0]["dominant_block_reason"], "symbol_cooldown")
            self.assertEqual(result["focus_symbols"][1]["dominant_block_reason"], "agent_mode_disabled")
            self.assertEqual(result["focus_symbols"][2]["status"], "not_in_latest_top_report")
            text = audit.render_text(result)
            self.assertIn("STRKUSDT: bought", text)
            self.assertIn("TIAUSDT: blocked_rule", text)
            self.assertIn("DOGSUSDT: not in latest critic top report", text)


if __name__ == "__main__":
    unittest.main()
