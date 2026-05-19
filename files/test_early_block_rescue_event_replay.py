from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_early_block_rescue_event_replay import build


class EarlyBlockRescueEventReplayTest(unittest.TestCase):
    def test_event_replay_counts_top_and_false_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            files = root / "files"
            reports.mkdir()
            files.mkdir()
            (reports / "top_gainer_critic_2026-05-01_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-05-01",
                        "watchlist_top_gainers": [
                            {"symbol": "AAAUSDT", "status": "blocked_rule", "opportunity_from_first_block_pct": 5.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            events = []
            for i in range(5):
                events.append({"event": "blocked", "sym": "AAAUSDT", "signal_type": "top_gainer_score_gate", "reason": "top_gainer_score", "ts": f"2026-05-01T0{i}:00:00Z"})
                events.append({"event": "blocked", "sym": "ZZZUSDT", "signal_type": "top_gainer_score_gate", "reason": "top_gainer_score", "ts": f"2026-05-01T0{i}:10:00Z"})
            (files / "bot_events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            (files / "agent_events.jsonl").write_text("", encoding="utf-8")
            payload = build(reports, files, root / "out.json")
            self.assertEqual(payload["label_days"], 1)
            best = payload["best_variant"]
            self.assertGreaterEqual(best["candidate_count"], 1)
            self.assertGreaterEqual(best["top15_candidates"], 1)
            self.assertGreaterEqual(best["false_positive_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
