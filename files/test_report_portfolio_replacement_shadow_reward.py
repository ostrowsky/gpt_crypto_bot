from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_portfolio_replacement_shadow_reward as rpr


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class PortfolioReplacementShadowRewardTests(unittest.TestCase):
    def test_matches_replacement_entry_exit_and_computes_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            reports = root / ".runtime" / "reports"
            files.mkdir(parents=True)
            reports.mkdir(parents=True)
            _write_jsonl(
                files / "agent_events.jsonl",
                [
                    {
                        "event": "exit",
                        "ts": "2026-05-31T10:00:00Z",
                        "sym": "OLDUSDT",
                        "pnl_pct": -0.4,
                        "reason": "portfolio replacement: NEWUSDT leader 45.5 > OLDUSDT leader 22.0",
                    },
                    {"event": "entry", "ts": "2026-05-31T10:05:00Z", "sym": "NEWUSDT", "price": 1.0},
                    {"event": "exit", "ts": "2026-05-31T11:00:00Z", "sym": "NEWUSDT", "pnl_pct": 1.2, "reason": "regular exit"},
                ],
            )
            (reports / "top_gainer_critic_2026-05-31_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-05-31",
                        "watchlist_top_gainers": [
                            {"symbol": "NEWUSDT", "status": "bought", "capture_ratio_at_entry": 0.35}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = rpr.build_report(
                files_dir=files,
                reports_dir=reports,
                cfg=rpr.ReplacementConfig(min_closed_cases=1, min_avg_delta_pct=0.1),
                save=False,
            )

        self.assertEqual(report["coverage"]["replacement_events"], 1)
        self.assertEqual(report["coverage"]["closed_incoming"], 1)
        self.assertEqual(report["summary"]["avg_replacement_delta_pct"], 1.6)
        self.assertEqual(report["summary"]["incoming_watchlist_top_count"], 1)
        self.assertEqual(report["decision"], "advance_replacement_policy_to_counterfactual_replay")

    def test_collects_more_when_closed_cases_are_insufficient(self) -> None:
        summary = {
            "closed_incoming_count": 1,
            "avg_replacement_delta_pct": 5.0,
            "median_replacement_delta_pct": 5.0,
        }
        decision = rpr._decision(summary, rpr.ReplacementConfig(min_closed_cases=10))
        self.assertEqual(decision, "collect_more_replacement_outcomes")

    def test_marks_policy_hurting_when_enough_cases_have_negative_average(self) -> None:
        summary = {
            "closed_incoming_count": 10,
            "avg_replacement_delta_pct": -0.25,
            "median_replacement_delta_pct": 0.0,
        }
        decision = rpr._decision(summary, rpr.ReplacementConfig(min_closed_cases=10))
        self.assertEqual(decision, "replacement_policy_hurting_in_shadow_monitor")

    def test_render_text_contains_key_metrics(self) -> None:
        text = rpr.render_text(
            {
                "coverage": {"events_loaded": 3, "replacement_events": 1, "closed_incoming": 1},
                "decision": "collect_more_replacement_outcomes",
                "summary": {
                    "avg_replacement_delta_pct": 0.1,
                    "median_replacement_delta_pct": 0.0,
                    "positive_delta_rate_pct": 50.0,
                    "avg_replaced_exit_pnl_pct": -0.1,
                    "avg_incoming_exit_pnl_pct": 0.0,
                    "incoming_watchlist_top_count": 1,
                },
            }
        )
        self.assertIn("Portfolio replacement shadow reward", text)
        self.assertIn("avg_delta=0.1%", text)


if __name__ == "__main__":
    unittest.main()
